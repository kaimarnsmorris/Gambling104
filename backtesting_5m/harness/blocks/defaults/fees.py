"""Polymarket fee model for crypto up/down markets.

Verified on-chain 2026-07-07: exact to 0.0 % median error across 9,073 fills,
constant since at least 2026-05-20.

    fee = BASE_FEE_RATE * p * (1 - p) * shares

It peaks at 1.75 c/share at p = 0.50 and vanishes at the bounds, which is why
taking is only ever economic in the tails. Makers pay nothing and are PAID a
fifth of the same curve, so a maker fill carries a NEGATIVE fee.
"""
from dataclasses import dataclass
from enum import IntEnum

BASE_FEE_RATE = 0.07
MAKER_REBATE_PHI = 0.20      # measured 0.199-0.209, both wallets, every period
TAKER_REBATE_RHO = 0.0833    # Silver tier; Gold is not worth buying


class Liquidity(IntEnum):
    MAKER = 0
    TAKER = 1


@dataclass(frozen=True)
class FeeSchedule:
    base_fee_rate: float = BASE_FEE_RATE
    maker_rebate_phi: float = MAKER_REBATE_PHI
    taker_rebate_rho: float = TAKER_REBATE_RHO

    def charge(self, liquidity: Liquidity, shares: float, p: float) -> float:
        """USD owed on a fill. Negative means we are paid."""
        base = self.base_fee_rate * p * (1.0 - p) * shares
        if liquidity == Liquidity.TAKER:
            return base * (1.0 - self.taker_rebate_rho)
        return -base * self.maker_rebate_phi


def apply_daily_minimum(ledger, minimum_usd: float = 1.0):
    """Zero out maker rebates on days that never reached the payout floor.

    Measured with perfect separation over 131 earn-days: the smallest paid was
    $1.0359 and the largest skipped $0.7209. Dust days are simply not paid, so
    a backtest that books them is overstating maker economics.

    The floor is per day PER REPLAY. A seed is an independent replay of the
    same calendar, so an n-seed ledger holds every day n times; grouping on
    `day` alone would sum one day's rebate across all n and lift genuine dust
    days over the floor -- roughly n times too generous, and generous is the
    one direction this harness must never be. Group on (day, seed) whenever
    the ledger carries a seed, and on day alone when it does not.
    """
    import pandas as pd  # local: the fee model itself stays dependency-free

    if not len(ledger):
        return ledger

    out = ledger.copy()
    maker = out["liquidity"] == int(Liquidity.MAKER)
    keys = ["day", "seed"] if "seed" in out.columns else ["day"]
    earned = -out.loc[maker].groupby(keys)["fee_usd"].sum()
    dust = set(earned[earned < minimum_usd].index)
    if len(keys) == 1:
        is_dust = out["day"].isin(dust)
    else:
        is_dust = pd.Series(
            [k in dust for k in zip(*(out[c] for c in keys))],
            index=out.index)
    out.loc[maker & is_dust, "fee_usd"] = 0.0
    return out
