"""Which name points at which stream.

Resolution mirrors block resolution: the catalog holds the standard streams,
an investigation registers its own, and a later registration of the same name
shadows an earlier one -- so an investigation overrides the catalog simply by
registering after import.
"""
from dataclasses import dataclass

from harness.streams.spec import Stream, StreamInvalid
from harness.streams.validate import validate_stream


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
    """
    if adapter is None:
        try:
            meta = validate_stream(path)
        except StreamInvalid as exc:
            raise StreamInvalid(
                f"{name}: {exc}. Pass adapter=Stream(...) to register a "
                f"non-conforming file explicitly.") from exc
        causal, values = meta["causal"], meta["values"]
    else:
        meta = {"name": adapter.name}
        if adapter.causal is None:
            raise StreamInvalid(f"{name}: adapter must state causal=")
        causal, values = adapter.causal, ()

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
