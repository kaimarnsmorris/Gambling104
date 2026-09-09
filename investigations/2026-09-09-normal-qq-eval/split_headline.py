"""Split the unified run's headline economics by liquidity (maker/taker).

The harness's `stats.headline()` operates on the per-market `pnl_net`
column, which is the whole market's mark-to-market/settlement PnL and mixes
maker and taker fills together -- it cannot be cut by liquidity after the
fact. This script instead derives fee-inclusive, per-fill net economics
straight from the ledger (fee-adjusted markout: `delta_quality_c` is the
gross markout in cents/share; `fee_usd` is the USD owed on the fill,
negative when we are paid), and sums that by liquidity. This is an
economically fair split -- it is not the same quantity as the headline
`c/share`/`$/market` in `summary.json`, but it agrees with it in total.

Usage: python split_headline.py <run_dir>
"""
import json
import os
import sys

from harness.io import read_parquet
from harness.blocks.defaults.fees import Liquidity


def split(run_dir, seed=None):
    ledger = read_parquet(os.path.join(run_dir, "ledger.parquet"))
    markets = read_parquet(os.path.join(run_dir, "markets.parquet"))
    if seed is not None:
        ledger = ledger[ledger["seed"] == seed]
        markets = markets[markets["seed"] == seed]
    n_markets_total = int(markets["market_id"].nunique())

    out = {"n_markets_total": n_markets_total}
    for name, liq in (("maker", int(Liquidity.MAKER)),
                      ("taker", int(Liquidity.TAKER)),
                      ("overall", None)):
        l = ledger if liq is None else ledger[ledger["liquidity"] == liq]
        n_fills = int(len(l))
        n_markets = int(l["market_id"].nunique())
        shares = float(l["shares"].sum())
        gross_usd = float((l["delta_quality_c"] / 100.0 * l["shares"]).sum())
        fees_usd = float(l["fee_usd"].sum())
        net_usd = gross_usd - fees_usd
        out[name] = {
            "n_fills": n_fills,
            "n_markets_touched": n_markets,
            "shares": shares,
            # MARKOUT, before fees. `delta_quality_c` is already cents per
            # share, so the share-weighted mean is the honest per-share
            # number and the plain mean is the per-FILL one; they differ
            # whenever fill sizes differ, and both are reported because the
            # first says what the strategy earned and the second says what a
            # typical fill was worth.
            "markout_c_per_share": (100.0 * gross_usd / shares
                                    if shares else float("nan")),
            "markout_c_per_fill": (float(l["delta_quality_c"].mean())
                                   if n_fills else float("nan")),
            "gross_usd": gross_usd,
            "fees_usd": fees_usd,
            "net_usd": net_usd,
            "c_per_share": 100.0 * net_usd / shares if shares else float("nan"),
            "usd_per_market_touched": net_usd / n_markets if n_markets else float("nan"),
            "usd_per_market_all": net_usd / n_markets_total if n_markets_total else float("nan"),
        }
    return out


if __name__ == "__main__":
    run_dir = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    result = split(run_dir, seed=seed)
    print(json.dumps(result, indent=2))
