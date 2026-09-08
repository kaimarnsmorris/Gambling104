"""US/Eastern DST arithmetic, table-driven so the streaming module never needs tzdata.

The autotrader has to evaluate the New-York time-of-week coordinate on every tick. We
therefore precompute the UTC epochs of the US DST transitions (post-2007 rule: forward on
the 2nd Sunday of March at 02:00 local standard, back on the 1st Sunday of November at
02:00 local daylight) and ship them in params.json. `tests/test_dst.py` checks every
transition in 2015-2040 against the stdlib zoneinfo database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

STD_OFFSET = -5 * 3600  # EST
DST_OFFSET = -4 * 3600  # EDT

SEC_WEEK = 7 * 86400
# 1970-01-01 was a Thursday; NY local Monday 00:00 is the origin of time-of-week.
# offset such that (t_local + TOW_ORIGIN) % SEC_WEEK == seconds since local Monday 00:00
_TOW_SHIFT = 3 * 86400  # epoch 0 was a Thursday; shift so Monday 00:00 local is the origin


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """n-th (1-based) `weekday` (Mon=0) of `month` in `year`, naive."""
    d = datetime(year, month, 1)
    delta = (weekday - d.weekday()) % 7
    return d + timedelta(days=delta + 7 * (n - 1))


def dst_transitions(y0: int = 2015, y1: int = 2040) -> np.ndarray:
    """UTC epoch seconds of every transition, ascending; alternating start/end.

    Returns an (n, 2) int64 array of (epoch, offset_after) pairs.
    """
    rows = []
    for y in range(y0, y1 + 1):
        # forward: 2nd Sunday March, 02:00 EST -> 03:00 EDT  => 07:00 UTC
        f = _nth_weekday(y, 3, 6, 2).replace(hour=2)
        f_epoch = int(f.replace(tzinfo=timezone.utc).timestamp()) - STD_OFFSET
        rows.append((f_epoch, DST_OFFSET))
        # back: 1st Sunday November, 02:00 EDT -> 01:00 EST  => 06:00 UTC
        b = _nth_weekday(y, 11, 6, 1).replace(hour=2)
        b_epoch = int(b.replace(tzinfo=timezone.utc).timestamp()) - DST_OFFSET
        rows.append((b_epoch, STD_OFFSET))
    return np.array(sorted(rows), dtype=np.int64)


_TRANS = dst_transitions()


def ny_offset(ts: np.ndarray | int) -> np.ndarray | int:
    """UTC offset in seconds of America/New_York at UTC epoch second(s) `ts`."""
    scalar = np.isscalar(ts)
    a = np.atleast_1d(np.asarray(ts, dtype=np.int64))
    i = np.searchsorted(_TRANS[:, 0], a, side="right") - 1
    off = np.where(i < 0, STD_OFFSET, _TRANS[np.clip(i, 0, None), 1])
    return int(off[0]) if scalar else off.astype(np.int64)


def ny_tow(ts: np.ndarray | int) -> np.ndarray | int:
    """Seconds since NY-local Monday 00:00, in [0, 604800)."""
    a = np.asarray(ts, dtype=np.int64)
    local = a + ny_offset(a)
    return (local + _TOW_SHIFT) % SEC_WEEK


def utc_tod(ts: np.ndarray | int) -> np.ndarray | int:
    """Seconds since 00:00 UTC, in [0, 86400)."""
    return np.asarray(ts, dtype=np.int64) % 86400


def is_dst(ts: np.ndarray | int) -> np.ndarray | bool:
    return ny_offset(ts) == DST_OFFSET


def transitions_payload() -> dict:
    """Serialisable form for params.json."""
    return {
        "rule": "US post-2007: 2nd Sun Mar 02:00 local standard -> 1st Sun Nov 02:00 local daylight",
        "std_offset_s": STD_OFFSET,
        "dst_offset_s": DST_OFFSET,
        "tow_origin": "NY-local Monday 00:00",
        "transitions": [[int(a), int(b)] for a, b in _TRANS],
    }
