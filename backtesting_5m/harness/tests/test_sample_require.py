"""Pins require-by-stream.

`require_spot` ignored Chainlink and silently scored 237 unpriceable markets as
$0.00, which entered the mean as real observations. Excluded must mean ABSENT
-- not present-with-NaN, and never zero.
"""
import numpy as np
import pytest

from harness.core.config import Sample
from harness.core.run import select_episodes


def _eps(flat_episode, n=4, with_cl=2):
    from dataclasses import replace
    out = []
    for k in range(n):
        streams = {}
        if k < with_cl:
            streams["chainlink"] = {
                "px": np.full(len(flat_episode), 1.0),
                "age_ms": np.zeros(len(flat_episode)),
                "has": np.ones(len(flat_episode), dtype=bool)}
        out.append(replace(flat_episode, market_id=f"m{k}",
                           open_ts=1786665600 + 300 * k, streams=streams))
    return out


def test_require_excludes_markets_missing_the_stream(flat_episode):
    kept, dropped = select_episodes(_eps(flat_episode), Sample(
        require=("chainlink",)))
    assert [e.market_id for e in kept] == ["m0", "m1"]
    assert dropped["chainlink"] == 2


def test_excluded_markets_are_absent_not_zeroed(flat_episode):
    kept, _ = select_episodes(_eps(flat_episode), Sample(require=("chainlink",)))
    assert all("chainlink" in e.streams for e in kept)


def test_no_require_keeps_everything(flat_episode):
    kept, dropped = select_episodes(_eps(flat_episode), Sample())
    assert len(kept) == 4 and dropped == {}


def test_require_spot_still_works_for_existing_callers(flat_episode):
    kept, _ = select_episodes(_eps(flat_episode), Sample(require_spot=True))
    assert len(kept) == 4          # flat_episode has spot
