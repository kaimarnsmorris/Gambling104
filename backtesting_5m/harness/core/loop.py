"""The feed loop: one market, 3,000 decisions, in order.

Latency lives HERE, not in the policy block. The policy says what orders
should exist; this converts that into live_from / cancel_at / expires_at
indices. That is why latency can be swept without touching a line of policy.

Policy is consulted on EVERY index, not on the requote cadence. `decide` owns
that cadence itself (it throttles quoting and nothing else) because the gates
it applies first -- book present, book age, time to expiry -- have to be
evaluated every index or a book that goes stale between requotes keeps our
orders resting on it for up to a second.
"""
import math
from dataclasses import replace

import numpy as np

from harness.core.latency import episode_rng
from harness.core.ledger import Ledger
from harness.core.types import Liquidity, Order, cancellable_ids


def run_episode(ep, blocks, params, execn, seed=0, emit_ticks=False):
    standardise = blocks["f"]
    link = blocks["link"]
    quotes = blocks["quote"]
    decide = blocks["execution"]
    resolve = blocks["fill"]
    fee_schedule = blocks["fees"]
    fill_params = blocks.get("fill_params", {})

    rng = episode_rng(ep.market_id, seed)
    latency = execn.latency
    ledger = Ledger()

    q = 0.0
    cash = 0.0
    fees_paid = 0.0
    max_abs_q = 0.0
    next_id = 1
    live = []

    s_arr = blocks["s"]
    sigma_arr = blocks["sigma"]

    for i in range(len(ep)):
        # Orders that can never trade again are gone BEFORE policy sees them:
        # a cancel that has landed, or a marketable order past its bounded
        # life. Leaving an expired cross in `live` would go on consuming
        # per-side cap room -- and so suppress the resting quote on that side
        # -- for the rest of the episode.
        live = [o for o in live if not o.is_dead(i)]

        s_i = float(s_arr[i])
        sigma_i = float(sigma_arr[i])

        if math.isfinite(s_i) and math.isfinite(sigma_i):
            # Theoretical, off the cent grid. Policy adjusts for the maker fee
            # and snaps once, so the price sent to the venue is not this one.
            eff_bid, eff_ask = quotes(s_i, q, ep.strike, sigma_i, params,
                                      standardise, link)
            z_i = standardise(s_i, ep.strike, sigma_i)
            fair_p = link(z_i)
        else:
            eff_bid = eff_ask = float("nan")
            z_i = fair_p = float("nan")

        # --- fills against orders that were already live -------------------
        if live:
            for fl in resolve(live, ep, i, fill_params):
                fee = fee_schedule.charge(fl.liquidity, fl.shares, fl.price)
                q_before = q
                q += fl.shares * int(fl.side)
                cash -= fl.shares * fl.price * int(fl.side)
                fees_paid += fee
                max_abs_q = max(max_abs_q, abs(q))
                order = next(o for o in live if o.order_id == fl.order_id)
                ledger.record_fill(
                    ep, fl, fee, q_before, q, eff_bid, eff_ask, s_i, sigma_i,
                    z_i, fair_p, order.latency_ms,
                    (i - order.live_from) * 100.0, seed)
                live = [o for o in live if o.order_id != fl.order_id]

        # --- policy --------------------------------------------------------
        # Cancels are NOT conditional on having a quote. `fair` or `vol` can
        # return NaN -- an EWMA warm-up alone does it -- and a model outage is
        # no reason to leave orders resting unmanaged, with no policy and no
        # book-age gate, filling into whatever the book does next. Placing
        # needs a quote; pulling never does.
        # A cancel also only reaches what the venue will release: a
        # marketable order inside its lock window is beyond recall, on this
        # path exactly as on policy's own. Both go through `cancellable_ids`.
        quoting = math.isfinite(eff_bid) and math.isfinite(eff_ask)
        to_place, to_cancel = [], []

        if quoting:
            to_place, to_cancel = decide(i, eff_bid, eff_ask, q, ep, live,
                                         execn, params)
        elif live:
            to_cancel = cancellable_ids(live, i)

        if to_cancel:
            in_move = ep.book_age_ms[i] == 0.0
            lands = i + latency.delay_idx(
                latency.draw(rng, "cancel", in_move=in_move))
            live = [replace(o, cancel_at=min(lands, o.cancel_at or lands))
                    if o.order_id in to_cancel else o
                    for o in live]

        for req in to_place:
            taker = req.liquidity == Liquidity.TAKER
            kind = "take" if taker else "place"
            drawn = latency.draw(rng, kind)
            live_from = i + latency.delay_idx(drawn)
            # A marketable order is live for the tick it arrives on and no
            # longer: the venue held it through the lock, and by `live_from`
            # the book has had that long to move away. If it is not marketable
            # then, it is over -- it does not become a resting order, and it
            # does not go on holding cap room until the market closes.
            expires_at = live_from + 1 if taker else None
            live.append(Order(next_id, req.side, req.price, req.shares,
                              req.liquidity, live_from, None, req.reason,
                              latency_ms=drawn, expires_at=expires_at))
            next_id += 1

        if emit_ticks:
            ledger.record_tick(ep, i, s_i, sigma_i, z_i, fair_p,
                               eff_bid, eff_ask, q, cash,
                               cash + q * float(ep.mid[i])
                               if np.isfinite(ep.mid[i]) else cash,
                               len(live), seed)

    # --- settlement --------------------------------------------------------
    if ep.winner_up is None:
        pnl_gross = pnl_net = float("nan")
        settled = False
    else:
        pnl_gross = cash + q * (1.0 if ep.winner_up else 0.0)
        pnl_net = pnl_gross - fees_paid
        settled = True

    return {
        "market_id": ep.market_id, "open_ts": ep.open_ts, "day": ep.day,
        "fills": ledger.fills, "ticks": ledger.ticks,
        "pnl_gross": pnl_gross, "pnl_net": pnl_net, "fees": fees_paid,
        "shares": sum(f["shares"] for f in ledger.fills),
        "n_fills": len(ledger.fills), "max_abs_q": max_abs_q,
        "settled": settled, "has_spot": bool(ep.has_spot.any()),
        "winner_up": ep.winner_up,
    }
