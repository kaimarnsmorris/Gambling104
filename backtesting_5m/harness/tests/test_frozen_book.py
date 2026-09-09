"""Pins the venue-maintenance filter.

On 2026-08-19 the venue published bid 0.50 / ask 0.51 unchanged for all 2,940
buckets of 37 consecutive markets, from a single quote source, while BTC moved
normally. Quoting against a book that cannot move prints money that was never
available: those markets returned +$44.94 each against -$0.48 on the other
1,066, turning a losing strategy into a winning one on the strength of an
outage.

The filter is ON by default because including such a market does not degrade
a result, it manufactures one -- and it is a conjunction because neither half
is safe alone. Across 7,111 markets no two-source market has fewer than 11
distinct mids, and 32 single-source markets have books that move perfectly
well, up to 144 distinct mids.
"""
from dataclasses import replace

import numpy as np
import pytest

from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.run import _frozen_book, run


def _pinned(ep, bid=0.50, ask=0.51, n_src=1.0):
    ep.bid[:] = bid
    ep.ask[:] = ask
    ep.mid[:] = (bid + ask) / 2.0
    ep.n_src[:] = n_src
    return ep


def test_a_pinned_single_source_book_is_frozen(flat_episode):
    assert _frozen_book(_pinned(flat_episode), 10)


def test_a_pinned_book_with_two_sources_is_not(flat_episode):
    """The conjunction, and the reason for it. No two-source market in the
    panel has fewer than 11 distinct mids, so this branch cannot be reached
    by real data -- which is exactly what makes the threshold safe."""
    assert not _frozen_book(_pinned(flat_episode, n_src=2.0), 10)


def test_a_single_source_book_that_moves_is_kept(flat_episode):
    """One venue quoting is not the same as nobody quoting: 32 markets are
    single-source with books moving across up to 144 distinct mids."""
    ep = _pinned(flat_episode)
    ep.mid[:] = np.linspace(0.30, 0.70, len(ep))
    assert not _frozen_book(ep, 10)


def test_a_book_that_is_entirely_absent_is_not_called_frozen(flat_episode):
    """Coverage is `require`'s question. Answering it here as well, and
    differently, is how a market gets dropped for the wrong reason and the
    drop attribution lies about why."""
    ep = _pinned(flat_episode)
    ep.mid[:] = np.nan
    assert not _frozen_book(ep, 10)


def test_the_threshold_is_the_count_of_distinct_mids(flat_episode):
    ep = _pinned(flat_episode)
    ep.mid[:] = 0.50
    ep.mid[:5] = np.linspace(0.40, 0.60, 5)      # 6 distinct
    assert _frozen_book(ep, 10)
    assert not _frozen_book(ep, 3)


@pytest.fixture
def episodes(flat_episode):
    """Four markets, one of them the outage.

    Every array field is COPIED. `dataclasses.replace` carries the same
    ndarray objects into the new instance, so without this, pinning one
    episode's book silently pins all four -- which is how the first version
    of this fixture reported every market frozen.
    """
    out = []
    for k in range(4):
        ep = replace(flat_episode, market_id=f"m{k}",
                     open_ts=1786665600 + 300 * k, day="2026-08-19",
                     bid=np.array(flat_episode.bid, copy=True),
                     ask=np.array(flat_episode.ask, copy=True),
                     mid=np.array(flat_episode.mid, copy=True),
                     n_src=np.array(flat_episode.n_src, copy=True))
        ep.ask[:] = 0.20
        ep.mid[:] = np.linspace(0.30, 0.70, len(ep))     # a book that moves
        ep.n_src[:] = 2.0
        out.append(ep)
    _pinned(out[2])                               # the outage market
    return out


def _run(tmp_path, episodes, **kw):
    kw.setdefault("quote", QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0))
    kw.setdefault("execn", ExecConfig())
    kw.setdefault("output", Output())
    return run(str(tmp_path), episodes=episodes, **kw)


def test_a_run_drops_the_frozen_market_by_default(tmp_path, episodes):
    out = _run(tmp_path, episodes, sample=Sample())
    s = out["summary"]["sample"]
    assert s["n_selected"] == 3
    assert s["dropped"]["frozen_book"] == 1


def test_the_drop_is_attributed_to_the_right_reason(tmp_path, episodes):
    """`dropped` is the only record of why a market is missing. Attributing
    an outage to 'spot' would send the next person to look at the panel."""
    out = _run(tmp_path, episodes, sample=Sample(require=("spot",)))
    assert "frozen_book" in out["summary"]["sample"]["dropped"]


def test_the_filter_can_be_turned_off_deliberately(tmp_path, episodes):
    """Off is a choice someone might make to study the outage itself; it
    should be possible and it should be recorded in the config."""
    out = _run(tmp_path, episodes, sample=Sample(drop_frozen_book=False))
    assert out["summary"]["sample"]["n_selected"] == 4


def test_the_filter_setting_is_in_the_config_hash(tmp_path, episodes):
    """Two runs differing only in whether the outage was included are two
    different runs and must not share a folder."""
    a = _run(tmp_path, episodes, sample=Sample())
    b = _run(tmp_path, episodes, sample=Sample(drop_frozen_book=False))
    assert a["run_dir"] != b["run_dir"]
