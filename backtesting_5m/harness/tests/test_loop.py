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
from harness.core.types import Side

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
               ExecConfig())
    assert out["n_fills"] == 0
    assert out["pnl_net"] == pytest.approx(0.0)


def test_a_flat_book_and_a_flat_fair_produce_no_taker_trades(flat_episode):
    out = _run(flat_episode, QuoteParams(e_p=0.0, shares=10.0),
               ExecConfig())
    assert out["n_fills"] == 0


def test_a_taker_that_lifts_a_cheap_book_pays_the_book_price(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig())
    assert out["n_fills"] == 1
    fills = out["fills"]
    assert fills[0]["price"] == pytest.approx(0.20)
    assert fills[0]["liquidity"] == int(fees.Liquidity.TAKER)


def test_settlement_pays_one_for_a_winning_long(flat_episode):
    """Buy 10 shares at 0.20, settle UP: gross = 10 * (1 - 0.20) = 8.00."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig())
    assert out["pnl_gross"] == pytest.approx(8.0)


def test_fees_are_subtracted_from_gross(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig())
    expected = 10.0 * 0.07 * 0.20 * 0.80 * (1.0 - 0.0833)
    assert out["fees"] == pytest.approx(expected)
    assert out["pnl_net"] == pytest.approx(out["pnl_gross"] - expected)


def test_a_maker_fill_is_paid_a_rebate(flat_episode):
    """Maker fees are negative, so net beats gross."""
    ep = flat_episode
    ep.ask[500:] = 0.30
    out = _run(ep, QuoteParams(e_p=0.15, shares=10.0, max_pos=10.0),
               ExecConfig())
    assert out["n_fills"] >= 1
    assert out["fees"] < 0.0
    assert out["pnl_net"] > out["pnl_gross"]


def test_the_position_cap_is_never_exceeded(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=30.0),
               ExecConfig())
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
               ExecConfig(requote_every=1))
    assert out["max_abs_q"] <= 1.0, "position cap breached by in-flight orders"


def test_place_latency_delays_the_first_possible_fill(flat_episode):
    """With a one-second placement delay nothing can trade in the first second."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(latency=LatencyModel(take_ms=1000.0)))
    assert out["fills"][0]["t_ms"] >= 1000


def test_an_unsettled_market_reports_no_pnl(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, winner_up=None, settle=None)
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig())
    assert out["settled"] is False
    assert np.isnan(out["pnl_net"])


def test_a_model_outage_pulls_the_resting_orders(flat_episode):
    """NaN fair is a reason to stop quoting, never a reason to stop cancelling.

    The quote was gated on a finite eff_bid, and so was the whole decide()
    call -- so when the model went NaN no cancels were issued and the resting
    buy stayed live with no policy and no book-age gate behind it. `vol.py`
    returns NaN during EWMA warm-up, so this is reachable, and here the book
    crashes to 0.30 two indices after the outage begins.
    """
    ep = flat_episode
    ep.ask[1002:] = 0.30          # only crossable AFTER the model goes dark

    s = ep.s.copy()
    s[1000:] = np.nan             # the outage
    blocks = dict(BLOCKS)
    blocks["s"] = s
    blocks["sigma"] = np.full(len(ep), 50.0)

    out = run_episode(ep, blocks, QuoteParams(e_p=0.15, shares=10.0,
                                              max_pos=10.0),
                      ExecConfig(), seed=0)
    assert out["n_fills"] == 0, (
        "a resting order survived the model outage and filled into the crash")


def test_a_cross_inside_the_venue_lock_survives_a_stale_book(flat_episode):
    """A marketable order the venue is still holding cannot be pulled.

    The cross goes out at index 0 and the venue releases it at index 3 (the
    250 ms lock, a floor on the taker path). The book goes stale at index 1,
    and the not-tradable branch cancels -- but a cancel does not reach an
    order inside the lock, so the cross still arrives and still fills.
    Retracting it would be the simulation dodging the crosses a moving book
    picks off, which flatters every number downstream.
    """
    ep = flat_episode
    ep.ask[:] = 0.20
    ep.book_age_ms[1:] = 5000.0

    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(requote_every=1))
    assert out["n_fills"] == 1, "the venue released a cross it does not release"
    assert out["fills"][0]["liquidity"] == int(fees.Liquidity.TAKER)
    assert out["fills"][0]["t_ms"] == 300


def test_a_cross_inside_the_venue_lock_survives_a_model_outage(flat_episode):
    """The same, on the other cancel path.

    The model goes dark at index 1, so the engine pulls everything it can --
    which is every resting order and NOT the cross the venue is still holding.
    Both paths go through the one predicate for exactly this reason.
    """
    ep = flat_episode
    ep.ask[:] = 0.20

    s = ep.s.copy()
    s[1:] = np.nan
    blocks = dict(BLOCKS)
    blocks["s"] = s
    blocks["sigma"] = np.full(len(ep), 50.0)

    out = run_episode(ep, blocks, QuoteParams(e_p=0.0, shares=10.0,
                                              max_pos=10.0),
                      ExecConfig(), seed=0)
    assert out["n_fills"] == 1, "an in-flight cross was cancelled by an outage"
    assert out["fills"][0]["liquidity"] == int(fees.Liquidity.TAKER)
    assert out["fills"][0]["t_ms"] == 300


def test_an_unfilled_cross_expires_and_gives_its_cap_room_back(flat_episode):
    """A cross is live for the tick it arrives on, and not one index longer.

    The book is crossable at index 0 only. The venue releases the order at
    index 3, by which time the ask is back at 0.51 and there is nothing to
    lift -- 300 ms is ample for a book to move away, which is the whole reason
    the lock is modelled. With no expiry that order stays live forever: it
    holds 10 of the 10 shares of buy-side room the cap allows, so the RESTING
    bid can never be posted either -- and then it trades anyway, 50 seconds
    later, against a book nobody ever decided to lift. Here it expires, the
    room comes back, and the resting bid is touched at index 500 instead.
    """
    ep = flat_episode
    ep.ask[0] = 0.20          # crossable, once
    ep.ask[500:] = 0.30       # later, a touch on the resting bid

    out = _run(ep, QuoteParams(e_p=0.15, shares=10.0, max_pos=10.0),
               ExecConfig())
    assert out["n_fills"] == 1, "the expired cross went on holding cap room"
    fill = out["fills"][0]
    assert fill["liquidity"] == int(fees.Liquidity.MAKER)
    assert fill["side"] == int(Side.BUY)
    assert fill["t_ms"] == 50_000


def test_one_run_both_makes_and_takes(flat_episode):
    """The unified policy, end to end. There is no arm to switch between.

    The book comes to our resting bid at index 500 (a maker fill at our own
    price), and only at 1000 does it fall far enough through the fee-adjusted
    threshold to be worth lifting. Both happen in one episode, with the cap
    still holding.
    """
    ep = flat_episode
    ep.ask[500:] = 0.34       # touches the resting bid, but inside the fee
    ep.ask[1000:] = 0.20      # now genuinely worth crossing for

    out = _run(ep, QuoteParams(e_p=0.15, shares=10.0, max_pos=50.0),
               ExecConfig())
    kinds = {int(fl["liquidity"]) for fl in out["fills"]}
    assert kinds == {int(fees.Liquidity.MAKER), int(fees.Liquidity.TAKER)}
    assert out["max_abs_q"] <= 50.0


def test_a_one_cent_edge_is_not_crossed_into_a_larger_fee(flat_episode):
    """The whole point of putting fees in the threshold.

    eff_bid is 0.51 against an ask of 0.50: a 1 c edge, and the taker fee at
    0.50 is 1.6 c. The old policy crossed this every requote and lost by
    construction.
    """
    ep = flat_episode
    ep.ask[:] = 0.50
    params = QuoteParams(e_p=-0.01, shares=10.0, max_pos=10.0)

    charged = _run(ep, params, ExecConfig(requote_every=1))
    assert charged["n_fills"] == 0, "crossed a 1 c edge into a 1.6 c fee"

    # the same book, the same forecast, with the fee waived: now it crosses,
    # which is what proves the fee -- and nothing else -- was the brake
    waived = _run(ep, params,
                  ExecConfig(requote_every=1,
                             fees=fees.FeeSchedule(base_fee_rate=0.0)))
    assert waived["n_fills"] >= 1
    assert all(fl["liquidity"] == int(fees.Liquidity.TAKER)
               for fl in waived["fills"])


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
    execn = ExecConfig(latency=LatencyModel(jitter_frac=0.4))
    params = QuoteParams(e_p=0.15, shares=10.0, max_pos=20.0)
    a = _run(ep, params, execn)
    b = _run(ep, params, execn)
    assert a["pnl_net"] == pytest.approx(b["pnl_net"])
    assert a["n_fills"] == b["n_fills"]
