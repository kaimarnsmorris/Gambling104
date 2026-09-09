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
"""
from dataclasses import dataclass
from enum import Enum

RESERVED = ("recv_ns", "src_ns", "venue", "seq")
META_PREFIX = "stream."
SCHEMA_VERSION = "1"


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
