"""Which name points at which stream.

Resolution mirrors block resolution: the catalog holds the standard streams,
an investigation registers its own, and a later registration of the same name
shadows an earlier one -- so an investigation overrides the catalog simply by
registering after import.
"""
from dataclasses import dataclass

from harness.streams.spec import (RESERVED, Stream, StreamInvalid,
                                  check_receipt_map)
from harness.streams.validate import column_names, validate_stream


class StreamNotRegistered(KeyError):
    """A name was requested that nothing has registered."""


@dataclass(frozen=True)
class RegisteredStream:
    name: str
    path: str
    adapter: Stream | None
    meta: dict
    causal: bool
    values: tuple

    @property
    def time_col(self):
        return self.adapter.time_col if self.adapter else "recv_ns"

    @property
    def time_unit(self):
        return self.adapter.time_unit if self.adapter else "ns"

    @property
    def value_time_cols(self):
        """(value_col, receipt_col) pairs, or () when one receipt governs all.

        `grid_stream` buckets a mapped column on its own receipt. See
        `harness.streams.spec` for why a row with several receipts is several
        arrivals rather than one.
        """
        return self.adapter.value_time_cols if self.adapter else ()

    @property
    def pre_gridded(self):
        """True for panels already keyed on (open_ts, t_ms).

        `grid_stream` treats its time column as an ABSOLUTE epoch time. The
        book and spot panels key on `t_ms`, milliseconds SINCE THE MARKET
        OPEN, so routing them through it would compute `ts - open_ts` as a
        hugely negative number and silently drop every observation. They are
        also already on the decision grid, so there is nothing to grid. Such a
        panel is read column-wise and never gridded.
        """
        return bool(self.adapter and self.adapter.pre_gridded)


_REGISTRY = {}


def register(name, path, adapter=None):
    """Register `path` under the local alias `name`.

    A conforming file needs no adapter. A non-conforming one needs an explicit
    `Stream(...)`, whose `time_kind` has no default -- see harness.streams.spec.

    Registration is also where a per-field-receipt file is caught: if a value
    column has its own receipt in the file and the declaration governs it by
    some other column, this raises rather than gridding it at the wrong
    arrival. That is the point of the standard -- the alternative is a real
    number, from a real row, that the host had not been told yet.

    An existing name is OVERWRITTEN, unconditionally. That is the shadowing
    feature: an investigation overrides the catalog by registering after
    import. `catalog.install()` is the one caller that must not do this, and
    it does not -- see its `_install`.
    """
    if adapter is None:
        try:
            meta = validate_stream(path)
        except StreamInvalid as exc:
            raise StreamInvalid(
                f"{name}: {exc}. Pass adapter=Stream(...) to register a "
                f"non-conforming file explicitly.") from exc
        causal, values = meta["causal"], meta["values"]
        columns = meta["columns"]
    else:
        meta = {"name": adapter.name}
        if adapter.causal is None:
            raise StreamInvalid(f"{name}: adapter must state causal=")
        causal = adapter.causal
        # Mapping a value column to its receipt also DECLARES it. A file with
        # a receipt and a copy count per field carries three real values among
        # thirteen columns; without this the fallback ("everything not
        # RESERVED") grids all thirteen, including receipt columns as float
        # epochs and the source stamp `oracle_ms` as an alpha-shaped series.
        values = tuple(v for v, _ in adapter.value_time_cols)
        # Best effort: an adapter may name a file that does not exist yet, and
        # a declaration check must not be what fails such a run first.
        columns = column_names(path)

    if columns is not None and not (adapter and adapter.pre_gridded):
        check_receipt_map(name, columns, values or
                          tuple(c for c in columns if c not in RESERVED),
                          adapter.time_col if adapter else "recv_ns",
                          adapter.value_time_cols if adapter else ())

    _REGISTRY[name] = RegisteredStream(name, path, adapter, meta, causal,
                                       values)


def resolve(name):
    if name not in _REGISTRY:
        raise StreamNotRegistered(
            f"{name!r} is not registered. Registered: {registered()}")
    return _REGISTRY[name]


def registered():
    return tuple(sorted(_REGISTRY))


def clear_registry():
    """Tests only."""
    _REGISTRY.clear()
