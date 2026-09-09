"""Recording what happened, and scoring it 10 s later.

Every fill carries its own fee (negative for makers) and its 10 s markout, so
gross and net are always both recoverable and adverse selection is visible per
fill rather than only in the aggregate.

`eff_bid`/`eff_ask` in both frames are the THEORETICAL quote off the cent grid,
not an order price. The order price is the fee-adjusted, snapped one that
`execution.resting_quote` produces, and it is already recorded where it is
meaningful: a maker fill trades at its own limit, so `price` on the fill IS the
order price, and the gap `price - eff_bid` is exactly the rebate the snap let
through. The tick frame carries no order price on purpose -- ticks are emitted
every 100 ms and decisions only every `requote_every`, so a per-tick "price we
would send" would be a number no order was ever placed at.
"""
import numpy as np
import pandas as pd

from harness.core.types import Side

MARKOUT_IDX = 100          # 100 buckets x 100 ms = 10 s


def markout(ep, idx, side, price):
    """(mid_t10, delta_quality_c, markout_settled) for a fill at `idx`.

    Past the end of the window there is no mid, so the settlement outcome is
    the reference instead -- which is the honest comparison, not a null.
    """
    j = idx + MARKOUT_IDX
    if j < len(ep):
        ref, settled = float(ep.mid[j]), False
    elif ep.winner_up is None:
        return float("nan"), float("nan"), True
    else:
        ref, settled = (1.0 if ep.winner_up else 0.0), True

    if not np.isfinite(ref):
        return float("nan"), float("nan"), settled

    signed = (ref - price) if side == Side.BUY else (price - ref)
    return ref, 100.0 * signed, settled


class Ledger:
    """Accumulates fills and (optionally) per-tick diagnostics."""

    def __init__(self):
        self.fills = []
        self.ticks = []

    def record_fill(self, ep, fill, fee_usd, q_before, q_after,
                    eff_bid, eff_ask, s_i, sigma_i, z_i, fair_p, latency_ms,
                    order_age_ms, seed):
        mid_t10, dq, settled = markout(ep, fill.idx, fill.side, fill.price)
        self.fills.append({
            "market_id": ep.market_id, "open_ts": ep.open_ts, "day": ep.day,
            "t_ms": fill.idx * 100,
            "side": int(fill.side), "liquidity": int(fill.liquidity),
            "shares": fill.shares, "price": fill.price, "fee_usd": fee_usd,
            "s": s_i, "sigma": sigma_i, "z": z_i, "fair_p": fair_p,
            "eff_bid": eff_bid, "eff_ask": eff_ask,
            "book_bid": float(ep.bid[fill.idx]),
            "book_ask": float(ep.ask[fill.idx]),
            "mid_at_fill": float(ep.mid[fill.idx]),
            "q_before": q_before, "q_after": q_after,
            "latency_ms": latency_ms, "order_age_ms": order_age_ms,
            "mid_t10": mid_t10, "delta_quality_c": dq,
            "markout_settled": settled,
            "seed": seed, "order_id": fill.order_id, "reason": fill.reason,
        })

    def record_tick(self, ep, i, s_i, sigma_i, z_i, fair_p, eff_bid, eff_ask,
                    q, cash, cum_pnl, orders_live, seed):
        self.ticks.append({
            # `seed` for the same reason a fill carries it: every seed
            # replays the SAME market, so without it a multi-seed run's
            # ticks.parquet holds several rows per (market_id, t_ms) with
            # nothing to tell them apart, and any per-market plot silently
            # overlays three paths.
            "market_id": ep.market_id, "seed": seed, "t_ms": i * 100,
            "s": s_i, "sigma": sigma_i, "z": z_i, "fair_p": fair_p,
            "eff_bid": eff_bid, "eff_ask": eff_ask,
            "book_bid": float(ep.bid[i]), "book_ask": float(ep.ask[i]),
            "mid": float(ep.mid[i]), "book_age_ms": float(ep.book_age_ms[i]),
            # BTC space, so a market can be read in the units the model
            # actually forecasts: the venue mid it sees, the oracle it is
            # forecasting, and `s`, its own estimate of where that oracle
            # will settle. `chainlink` is NaN where no oracle was loaded.
            "spot": float(ep.spot[i]),
            "chainlink": float(ep.chainlink[i]),
            "q": q, "cash": cash, "cum_pnl": cum_pnl,
            "orders_live": orders_live,
        })

    def fills_frame(self):
        return pd.DataFrame(self.fills)

    def ticks_frame(self):
        return pd.DataFrame(self.ticks)
