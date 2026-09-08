"""One row per market: `market_id`, `open_ts`, `strike`.

`open_ts` is carried so the settlement chain closes without another file:
`settle(N)` is the strike of the market opening at `open_ts + 300`. Without it
the chain only resolves for markets whose successor survived into the panel
(98.95 %), and the rest would need the gamma intermediate.

The strike is the 60 s Chainlink TWAP stamped at the market's own open. Gamma
publishes it as `eventMetadata.priceToBeat`, but that field goes null
intermittently, so two other things fill and check it:

  the chain property   `settle(N) == strike(N+1)` for back-to-back 5 m
                       windows. Verified here, not assumed: it holds EXACTLY,
                       7,330 of 7,330 testable pairs. So a market whose own
                       `priceToBeat` is null takes the previous window's
                       settlement.
  the signed report    Polygon `ReportVerified` for the 60 s feed at the
                       window boundary -- Chainlink's own signed value, and
                       the settlement source of truth. Available for part of
                       the era, and used as an independent gate on the rest.

The gate is the point of the second source. Gamma is one publisher's view of a
number it did not compute; agreeing with the signed report is what makes it
trustworthy.
"""
import os
import sys

import numpy as np
import pandas as pd

import common

CHAIN = os.path.join(
    common.GAMBLING102, "research", "btc", "backtesting",
    "2026-08-23_backtesting_5m", "data", "london", "chain_reports.parquet")

#: The signed report and gamma should agree to the cent; a dollar of slack
#: would hide a feed mix-up (30 s vs 60 s differ by ~$2 at the median).
AGREE_USD = 0.01
MIN_AGREE = 0.99


def build(g):
    g = g.sort_values("open_ts", kind="stable").reset_index(drop=True)
    strike = pd.to_numeric(g.gamma_strike, errors="coerce")
    settle = pd.to_numeric(g.gamma_settle, errors="coerce")

    back_to_back = g.open_ts.diff() == common.H
    d = (strike - settle.shift()).abs()[back_to_back]
    d = d[d.notna()]
    exact = float((d == 0).mean()) if len(d) else float("nan")
    print(f"  chain property strike(N)==settle(N-1): {exact:.4%} exact "
          f"on {len(d):,} pairs", flush=True)
    if exact < 1.0:
        print("  ⚠ the property is not exact -- the fill below inherits that",
              flush=True)

    filled = strike.where(strike.notna(), settle.shift().where(back_to_back))
    n_fill = int(strike.isna().sum() - filled.isna().sum())
    print(f"  {int(strike.isna().sum())} markets had no priceToBeat; "
          f"{n_fill} filled from the previous window's settlement", flush=True)
    g["strike"] = filled
    return g


def gate_onchain(g):
    """Gamma's strike against Chainlink's own signed 60 s report."""
    if not os.path.exists(CHAIN):
        print("  (no chain_reports.parquet -- no independent strike check)",
              flush=True)
        return pd.DataFrame()
    c = pd.read_parquet(CHAIN)
    c = c[c.feed == "60s"].drop_duplicates("boundary_ts")
    j = g.merge(c[["boundary_ts", "price"]], left_on="open_ts",
                right_on="boundary_ts", how="inner")
    j = j[j.strike.notna()]
    if not len(j):
        return pd.DataFrame()
    d = (j.strike - j.price).abs()
    return pd.DataFrame([{
        "markets_checked": len(j),
        "agree_1c": round(float((d <= AGREE_USD).mean()), 4),
        "median_abs_usd": round(float(d.median()), 6),
        "max_abs_usd": round(float(d.max()), 4),
    }])


def main():
    g = pd.read_parquet(os.path.join(common.DATA, "gamma_5m.parquet"))
    g = build(g)

    chk = gate_onchain(g)
    if not chk.empty:
        print("\n-- gate: gamma strike vs the signed on-chain 60 s report")
        print(chk.to_string(index=False))
        if float(chk.agree_1c.iloc[0]) < MIN_AGREE:
            raise SystemExit(
                f"REFUSING TO WRITE: gamma's strike matches the signed report "
                f"on only {float(chk.agree_1c.iloc[0]):.2%} of markets")

    miss = int(g.strike.isna().sum())
    if miss:
        print(f"  dropping {miss} markets with no strike from any source",
              flush=True)
    out = (g[g.strike.notna()][["market_id", "open_ts", "strike"]]
           .sort_values("open_ts")
           .reset_index(drop=True))

    os.makedirs(common.RESULTS, exist_ok=True)
    if not chk.empty:
        chk.to_csv(os.path.join(common.RESULTS, "gate_strike.tsv"),
                   sep="\t", index=False)
    out.to_parquet(common.STRIKES, index=False)
    print(f"\nwrote {common.STRIKES}: {len(out):,} markets, "
          f"strike ${out.strike.min():,.0f}..${out.strike.max():,.0f}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
