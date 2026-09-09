"""What a stream is, and the one rule that makes lookahead unwritable.

RESERVED COLUMNS
    recv_ns   int64, REQUIRED. Epoch nanoseconds, when THIS HOST received it.
    src_ns    int64, required when the source provides one. The source's own
              stamp. Recorded, never used for alignment.
    venue     optional, for multi-venue streams.
    seq       optional, gap detection.

Everything else is a value column. There is deliberately no metadata map of
value columns: the parquet schema is authoritative and cannot drift from
itself.

ALIGNMENT IS ALWAYS `recv_ns`, NEVER `src_ns`.

A source stamp can LEAD receipt. `rtds_btc`'s `oracle_ms` leads its own
`px_first_recv_ns` by roughly 1.5 s, so aligning on it hands a model prices
before it was told them -- lookahead, silently, in the alpha column. That bug
was found and fixed by hand on 2026-09-09. Making receipt the only alignable
column means it cannot be written again.

ONE RECEIPT PER VALUE, NEVER ONE RECEIPT FOR ALL OF THEM.

The same argument applies a second time, one level down. A recorder may pack
several fields into one row and stamp EACH with its own arrival, as `rtds_btc`
does: `px`/`px_first_recv_ns`, `twap30`/`twap30_first_recv_ns`,
`twap60`/`twap60_first_recv_ns`. Those receipts disagree --
`twap60_first_recv_ns` runs later than `px_first_recv_ns` on 70.8 % of rows,
median +46.6 ms, above one whole bucket on 27.5 % -- because they are
different arrivals that share a row, not one arrival with three names. Bucket
`twap60` by the price's receipt and 4.6 % of buckets carry a `twap60` the host
had not been told yet; and `twap60` is the settlement variable, so that is not
optimism, it is the answer. A row is a filing convenience. The arrival is the
fact.

So `Stream.value_time_cols` names the receipt that governs each value column,
and a value column whose sibling receipt is present in the file but mapped by
nothing is a registration error rather than a silent 4.6 %: see
`check_receipt_map`. The rule is the same rule in both cases -- align on when
you were told, never on when it happened -- and this file is where it is
written down once.
"""
from dataclasses import dataclass
from enum import Enum

RESERVED = ("recv_ns", "src_ns", "venue", "seq")
META_PREFIX = "stream."
SCHEMA_VERSION = "1"

#: How a recorder spells "when field X arrived". Checked in this order, so a
#: file carrying both `X_first_recv_ns` and `X_last_recv_ns` is reported
#: against the earlier one -- the arrival, not the last duplicate of it.
RECEIPT_SUFFIXES = ("_first_recv_ns", "_recv_ns", "_last_recv_ns")


class StreamInvalid(ValueError):
    """A stream file or declaration does not meet the standard."""


class TimeKind(Enum):
    RECEIPT = "receipt"
    SOURCE_STAMP = "source_stamp"


@dataclass(frozen=True)
class Stream:
    """An adapter for a NON-conforming file. Conforming files need none.

    `time_kind` has no default on purpose. Declaring RECEIPT is free;
    declaring SOURCE_STAMP additionally requires a transport offset, which the
    run records as an assumption -- the same treatment the clock offset gets.
    A stream that cannot say which it is cannot be registered.
    """
    name: str
    time_col: str = "recv_ns"
    time_kind: TimeKind | None = None
    time_unit: str = "ns"
    causal: bool | None = None
    max_age_ms: float | None = None
    transport_offset_ms: float | None = None

    #: Value column -> the receipt column that governs IT, as a tuple of
    #: pairs. A dict would be the obvious shape and is the wrong one: this
    #: dataclass is frozen and must stay hashable. Unmapped value columns fall
    #: back to `time_col`.
    #:
    #: Declaring any pair also DECLARES THE VALUE COLUMNS: `register()` takes
    #: the mapped names as the stream's `values`, so a file that carries ten
    #: bookkeeping columns beside three real ones puts three on the grid. See
    #: the module docstring for why one receipt cannot govern three arrivals.
    value_time_cols: tuple = ()

    #: True for a panel already keyed on (open_ts, t_ms) rather than an
    #: absolute receipt time -- the book and spot panels. Such a file is
    #: already on the decision grid and must NOT be routed through
    #: `grid_stream`, whose time column is an absolute epoch: `t_ms` is
    #: milliseconds since the market open, so the subtraction would go hugely
    #: negative and drop every row silently.
    pre_gridded: bool = False

    def __post_init__(self):
        if self.time_kind is None:
            raise StreamInvalid(
                f"{self.name}: time_kind is required and has no default. Use "
                f"TimeKind.RECEIPT when the column records arrival on this "
                f"host, or TimeKind.SOURCE_STAMP with a transport_offset_ms.")
        if (self.time_kind is TimeKind.SOURCE_STAMP
                and self.transport_offset_ms is None):
            raise StreamInvalid(
                f"{self.name}: time_kind=SOURCE_STAMP requires "
                f"transport_offset_ms. A source stamp can lead receipt, so "
                f"aligning on it without an offset is lookahead.")
        if self.time_unit not in ("ns", "us", "ms", "s"):
            raise StreamInvalid(f"{self.name}: bad time_unit {self.time_unit!r}")

        # Normalised to a tuple of tuples so the frozen dataclass stays
        # hashable however the caller spelt the pairs.
        pairs = tuple(tuple(p) if isinstance(p, (list, tuple)) else p
                      for p in self.value_time_cols)
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2
                    and all(isinstance(x, str) for x in pair)):
                raise StreamInvalid(
                    f"{self.name}: value_time_cols is a tuple of "
                    f"(value_col, time_col) pairs, not {pair!r}. A dict would "
                    f"make this dataclass unhashable.")
        seen = [v for v, _ in pairs]
        dupes = sorted({v for v in seen if seen.count(v) > 1})
        if dupes:
            raise StreamInvalid(
                f"{self.name}: value_time_cols maps {dupes} more than once; "
                f"a value column has exactly one receipt.")
        object.__setattr__(self, "value_time_cols", pairs)


def receipt_siblings(col, columns):
    """The receipt columns in `columns` that belong to value column `col`."""
    return tuple(col + s for s in RECEIPT_SUFFIXES if col + s in columns)


def check_receipt_map(name, columns, values, time_col, value_time_cols=()):
    """Raise unless every value column with its own receipt is governed by it.

    The general case of the `rtds_btc` defect: a file carries a receipt per
    field, the stream declares one `time_col`, and every other field is
    bucketed at the wrong arrival -- correct-looking, silently early, and
    exactly the lookahead this module exists to make unwritable. There is no
    safe default here, because guessing wrong is invisible: the values are
    real numbers from real rows, just not the ones the host had yet.

    So a value column `X` passes only if the stream maps it explicitly, or if
    the stream-level `time_col` already IS one of X's own receipts. Otherwise
    this raises, naming X and the receipt being ignored. Loud at registration
    beats a 4.6 % lookahead nobody reads.
    """
    columns = set(columns)
    mapped = {v for v, _ in value_time_cols}
    for col in values:
        if col in mapped:
            continue
        siblings = receipt_siblings(col, columns)
        if not siblings or time_col in siblings:
            continue
        raise StreamInvalid(
            f"{name}: value column {col!r} has its own receipt "
            f"{siblings[0]!r} in the file, but the stream buckets it on "
            f"{time_col!r}. Two receipts in one row are two arrivals, so "
            f"bucketing {col!r} on another field's receipt puts it on the "
            f"grid before it arrived. Declare it: "
            f"value_time_cols=(({col!r}, {siblings[0]!r}), ...).")
