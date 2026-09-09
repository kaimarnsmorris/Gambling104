"""How good a forecast of the SETTLE is `s`, in BTC dollars?

The probability-space diagnosis says the model is overconfident. That has two
possible sources and they need different fixes:

  * `s` is a poor estimate of where the oracle will settle -- a location
    problem, fixable only by a better fair;
  * `s` is fine and `sigma` understates its own error -- a scale problem,
    fixable with a multiplier.

Probability space cannot separate them, because `link(z)` folds both into one
number. BTC space can: measure the realised error of `s` against the settle,
per time to expiry, and compare it with what `sigma` claims that error is.

    settle(N) == strike(N+1)

is the venue's own rule for this era, so the settle level comes from the
strikes file rather than needing a second oracle read.

Benchmarks, because "is `s` good" is meaningless alone:
  * carry the last Chainlink print -- the naive forecast of a martingale,
    and what `s` must beat to have earned the basis machinery;
  * the venue mid, uncorrected BTC/USDT, which is `s` without the basis;
  * the book's own mid, converted back to a BTC level, is not comparable in
    dollars, so probability space is left to the other probes.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "backtesting_5m"))
from harness import paths                # noqa: E402
from harness.io import read_parquet      # noqa: E402


def main():
    with open(os.path.join(HERE, "last_run.txt")) as fh:
        run_dir = fh.read().strip()

    t = read_parquet(os.path.join(run_dir, "ticks.parquet"),
                     columns=["market_id", "t_ms", "s", "sigma", "spot",
                              "chainlink"])
    t = t.drop_duplicates(subset=["market_id", "t_ms"])

    # settle(N) = strike(N+1), by open time.
    k = read_parquet(paths.STRIKES).sort_values("open_ts")
    k["settle"] = k["strike"].shift(-1)
    k["next_open"] = k["open_ts"].shift(-1)
    # only where the next market really is the next window
    k = k[(k["next_open"] - k["open_ts"]) == 300]
    t = t.merge(k[["market_id", "strike", "settle"]], on="market_id",
                how="inner")
    t = t[np.isfinite(t["s"]) & np.isfinite(t["settle"])]
    print(f"{len(t):,} ticks over {t['market_id'].nunique():,} markets "
          "with a known settle")

    t["tte_s"] = (300_000 - t["t_ms"]) / 1000.0
    t["err_s"] = t["settle"] - t["s"]
    t["err_cl"] = t["settle"] - t["chainlink"]
    t["err_spot"] = t["settle"] - t["spot"]
    # sigma is a FRACTION of price over the remaining window; put it in
    # dollars so it is comparable with the realised error above.
    t["sigma_usd"] = t["sigma"] * t["s"]

    t["bucket"] = pd.cut(t["tte_s"], [0, 15, 30, 60, 120, 240, 300],
                         right=False)
    g = t.groupby("bucket", observed=True)
    tab = pd.DataFrame({
        "n": g.size(),
        "bias_s": g["err_s"].mean(),
        "rmse_s": g["err_s"].apply(lambda v: float(np.sqrt((v ** 2).mean()))),
        "rmse_chainlink": g["err_cl"].apply(
            lambda v: float(np.sqrt((v ** 2).mean()))),
        "rmse_spot": g["err_spot"].apply(
            lambda v: float(np.sqrt((v ** 2).mean()))),
        "sigma_claimed": g["sigma_usd"].mean(),
    })
    tab["ratio_realised_over_claimed"] = tab["rmse_s"] / tab["sigma_claimed"]
    tab["s_beats_carry"] = tab["rmse_chainlink"] - tab["rmse_s"]

    print("\n--- forecast error of the settle, USD, by time to expiry ---")
    print(tab.to_string(float_format=lambda v: f"{v:10.2f}"))

    print("\nWhat the last column means: sigma is the model's own claim about")
    print("the size of `rmse_s`. A ratio above 1 is the model being")
    print("overconfident by that factor, measured in dollars rather than")
    print("inferred from Brier.")

    overall = float(np.sqrt((t["err_s"] ** 2).mean())
                    / t["sigma_usd"].mean())
    print(f"\nshare-weighted ratio over all ticks: {overall:.3f}")

    # Does the basis correction earn its keep at all?
    print(f"\nrmse(s)         = {np.sqrt((t['err_s'] ** 2).mean()):9.2f}")
    print(f"rmse(chainlink) = {np.sqrt((t['err_cl'] ** 2).mean()):9.2f}")
    print(f"rmse(spot,USDT) = {np.sqrt((t['err_spot'] ** 2).mean()):9.2f}")
    print(f"mean bias of s  = {t['err_s'].mean():9.2f}  "
          f"(sd {t['err_s'].std():.2f})")


if __name__ == "__main__":
    main()
