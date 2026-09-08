"""Pins the feed loop end to end on episodes whose answer is arithmetic.

The loop is where causality, latency, policy, fills and fees meet. If any of
them is wrong, these numbers move.
"""
import numpy as np
import pytest

from harness.blocks.defaults import execution, f, fees, fill, link, quote
from harness.core.config import ExecConfig, QuoteParams
from harness.core.latency import LatencyModel
from harness.core.loop import run_episode

BLOCKS = {"f": f.standardise, "link": link.link, "quote": quote.quotes,
          "execution": execution.decide, "fill": fill.resolve,
          "fees": fees.FeeSchedule()}


def _run(ep, params, execn, emit_ticks=False):
    blocks = dict(BLOCKS)
    blocks["s"] = ep.s
    blocks["sigma"] = np.full(len(ep), 50.0)
    return run_episode(ep, blocks, params, execn, seed=0, emit_ticks=emit_ticks)


def test_a_policy_that_never_quotes_trades_nothing(flat_episode):
    out = _run(flat_episode, QuoteParams(e_p=0.9, shares=10.0),
               ExecConfig(mode="maker"))
    assert out["n_fills"] == 0
    assert out["pnl_net"] == pytest.approx(0.0)


def test_a_flat_book_and_a_flat_fair_produce_no_taker_trades(flat_episode):
    out = _run(flat_episode, QuoteParams(e_p=0.0, shares=10.0),
               ExecConfig(mode="taker"))
    assert out["n_fills"] == 0


def test_a_taker_that_lifts_a_cheap_book_pays_the_book_price(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["n_fills"] == 1
    fills = out["fills"]
    assert fills[0]["price"] == pytest.approx(0.20)
    assert fills[0]["liquidity"] == int(fees.Liquidity.TAKER)


def test_settlement_pays_one_for_a_winning_long(flat_episode):
    """Buy 10 shares at 0.20, settle UP: gross = 10 * (1 - 0.20) = 8.00."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["pnl_gross"] == pytest.approx(8.0)


def test_fees_are_subtracted_from_gross(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    expected = 10.0 * 0.07 * 0.20 * 0.80 * (1.0 - 0.0833)
    assert out["fees"] == pytest.approx(expected)
    assert out["pnl_net"] == pytest.approx(out["pnl_gross"] - expected)


def test_a_maker_fill_is_paid_a_rebate(flat_episode):
    """Maker fees are negative, so net beats gross."""
    ep = flat_episode
    ep.ask[500:] = 0.30
    out = _run(ep, QuoteParams(e_p=0.15, shares=10.0, max_pos=10.0),
               ExecConfig(mode="maker"))
    assert out["n_fills"] >= 1
    assert out["fees"] < 0.0
    assert out["pnl_net"] > out["pnl_gross"]


def test_the_position_cap_is_never_exceeded(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=30.0),
               ExecConfig(mode="taker"))
    assert out["max_abs_q"] <= 30.0


def test_the_cap_holds_when_requoting_faster_than_the_taker_lock(flat_episode):
    """The in-flight breach: max_pos=1 lot, requoted every index.

    A taker cross cannot be cancelled while the venue holds it, so a cap
    tested against realised inventory alone re-emits the same cross on every
    requote until the first fill lands. At requote_every=1 against an always
    crossable book that bought 9 lots against a cap of 1. The default
    requote_every=10 hides it, which is why this test uses 1.
    """
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=1.0, max_pos=1.0),
               ExecConfig(mode="taker", requote_every=1))
    assert out["max_abs_q"] <= 1.0, "position cap breached by in-flight orders"


def test_place_latency_delays_the_first_possible_fill(flat_episode):
    """With a one-second placement delay nothing can trade in the first second."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker",
                          latency=LatencyModel(take_ms=1000.0)))
    assert out["fills"][0]["t_ms"] >= 1000


def test_an_unsettled_market_reports_no_pnl(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, winner_up=None, settle=None)
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["settled"] is False
    assert np.isnan(out["pnl_net"])


def test_tick_output_is_off_by_default_and_complete_when_on(flat_episode):
    params = QuoteParams(e_p=0.05, shares=10.0)
    assert _run(flat_episode, params, ExecConfig())["ticks"] == []
    ticks = _run(flat_episode, params, ExecConfig(), emit_ticks=True)["ticks"]
    assert len(ticks) == 3000
    assert {"t_ms", "eff_bid", "eff_ask", "q", "cum_pnl"} <= set(ticks[0])


def test_results_are_reproducible_across_runs(flat_episode):
    """Jitter is seeded per episode, so two identical runs agree exactly."""
    ep = flat_episode
    ep.ask[500:] = 0.30
    execn = ExecConfig(mode="maker", latency=LatencyModel(jitter_frac=0.4))
    params = QuoteParams(e_p=0.15, shares=10.0, max_pos=20.0)
    a = _run(ep, params, execn)
    b = _run(ep, params, execn)
    assert a["pnl_net"] == pytest.approx(b["pnl_net"])
    assert a["n_fills"] == b["n_fills"]
