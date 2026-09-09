"""Where do the cents go?

The model is calibrated to within ~0.018 Brier of the book and still loses
about 1.7 c/share, and fee drag on the wide arm is only ~0.11 c/share -- so
fees are no longer the story and the loss is in the fills themselves. Two
very different worlds produce that:

  ADVERSE SELECTION -- the fair is fine, but we are filled when it is about
  to move against us. Signature: maker markout strongly negative, taker
  markout near zero or positive, loss concentrated where the book moves
  fastest (low tte, stale book, large |z| moves).

  MISPRICING -- the fair is simply wrong, so both sides of the book lose.
  Signature: maker AND taker markout both negative by similar amounts, and a
  conditional calibration error that unconditional Brier hides.

They call for opposite fixes -- quote wider / cancel faster versus fix the
model -- so this decides which before anything is changed.

`delta_quality_c` is the signed markout at +10 s in cents per share, already
on the fill row: positive is good, and where the 10 s reference runs past the
market's end it is the settlement outcome instead, flagged `markout_settled`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "backtesting_5m"))
from harness.blocks.defaults.fees import Liquidity   # noqa: E402
from harness.io import read_parquet                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAKER = int(Liquidity.MAKER)


def _load():
    with open(os.path.join(HERE, "last_run.txt")) as fh:
        run_dir = fh.read().strip()
    led = read_parquet(os.path.join(run_dir, "ledger.parquet"))
    mkt = read_parquet(os.path.join(run_dir, "markets.parquet"))
    tks = read_parquet(os.path.join(run_dir, "ticks.parquet"))
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    return run_dir, led, mkt, tks, summary


def _by(df, key, label):
    g = df.groupby(key)
    out = pd.DataFrame({
        "fills": g.size(),
        "shares": g["shares"].sum(),
        "markout_c": g["delta_quality_c"].mean(),
        "fee_c_per_share": 100 * g["fee_usd"].sum() / g["shares"].sum(),
    })
    out["net_c"] = out["markout_c"] - out["fee_c_per_share"]
    print(f"\n--- by {label} ---")
    print(out.to_string(float_format=lambda v: f"{v:8.3f}"))
    return out


def main():
    run_dir, led, mkt, tks, summary = _load()
    print(f"run: {os.path.basename(run_dir)}")
    h = summary["headline"]
    print(f"markets={h['n_markets']} fills={h['n_fills']} "
          f"pnl/market={h['pnl_per_market']:.3f} c/share={h['c_per_share']:.3f}")

    led = led[led["seed"] == led["seed"].min()].copy()
    led["is_maker"] = led["liquidity"] == MAKER
    led["side_name"] = np.where(led["side"] > 0, "buy", "sell")
    led["tte_s"] = (300_000 - led["t_ms"]) / 1000.0
    led["liq"] = np.where(led["is_maker"], "maker", "taker")

    print(f"\nfill mix: {led['is_maker'].mean():.1%} maker, "
          f"{(~led['is_maker']).mean():.1%} taker")
    print(f"gross markout over all fills: "
          f"{led['delta_quality_c'].mean():.3f} c/share")
    print(f"fee:                          "
          f"{100 * led['fee_usd'].sum() / led['shares'].sum():.3f} c/share")

    # THE decisive cut. Maker much worse than taker means the cancel is
    # losing the race; both similar means the fair itself is wrong.
    _by(led, "liq", "liquidity")
    _by(led, ["liq", "side_name"], "liquidity x side")

    led["tte_bucket"] = pd.cut(led["tte_s"], [0, 30, 60, 120, 240, 300],
                               right=False)
    _by(led, ["liq", "tte_bucket"], "liquidity x time to expiry (s)")

    led["absz"] = pd.cut(led["z"].abs(), [0, 0.5, 1.0, 2.0, 4.0, np.inf],
                         right=False)
    _by(led, ["liq", "absz"], "liquidity x |z| at fill")

    led["age"] = pd.cut(led["t_ms"] * 0 + led.get("order_age_ms", 0),
                        [0, 100, 500, 2000, np.inf], right=False)
    _by(led, ["liq", "age"], "liquidity x order age at fill (ms)")

    # Settled markouts use the outcome, not a mid: keep them apart or the
    # last 10 s of every market silently dominates a different quantity.
    print(f"\nmarkout_settled share: {led['markout_settled'].mean():.1%}")
    _by(led, ["liq", "markout_settled"], "liquidity x markout reference")

    # Calibration, from the ticks rather than the fills: the fills are a
    # selected sample by construction, and a model is not calibrated "where
    # it traded", it is calibrated or it is not.
    t = tks[tks["seed"] == tks["seed"].min()] if "seed" in tks else tks
    t = t[np.isfinite(t["fair_p"]) & np.isfinite(t["mid"])]
    w = mkt[["market_id", "winner_up"]].drop_duplicates()
    t = t.merge(w, on="market_id", how="left")
    t = t[t["winner_up"].notna()]
    y = t["winner_up"].astype(float).to_numpy()
    print(f"\n--- calibration over {len(t):,} quoting ticks ---")
    print(f"Brier(model fair_p) = {np.mean((t['fair_p'] - y) ** 2):.5f}")
    print(f"Brier(book mid)     = {np.mean((t['mid'] - y) ** 2):.5f}")
    print(f"Brier(0.5)          = {np.mean((0.5 - y) ** 2):.5f}")

    t = t.assign(bucket=pd.cut(t["fair_p"], np.linspace(0, 1, 11)))
    cal = t.groupby("bucket", observed=True).agg(
        n=("fair_p", "size"), predicted=("fair_p", "mean"),
        realised=("winner_up", "mean"), book=("mid", "mean"))
    cal["model_err"] = cal["predicted"] - cal["realised"]
    cal["book_err"] = cal["book"] - cal["realised"]
    print("\n--- reliability by predicted probability ---")
    print(cal.to_string(float_format=lambda v: f"{v:8.4f}"))

    t2 = t.assign(tte=pd.cut((300_000 - t["t_ms"]) / 1000.0,
                             [0, 30, 60, 120, 240, 300], right=False))
    bt = t2.groupby("tte", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "brier_model": np.mean((g["fair_p"] - g["winner_up"]) ** 2),
            "brier_book": np.mean((g["mid"] - g["winner_up"]) ** 2),
            "mean_edge_c": 100 * (g["fair_p"] - g["mid"]).mean(),
            "sd_edge_c": 100 * (g["fair_p"] - g["mid"]).std(),
        }), include_groups=False)
    bt["gap"] = bt["brier_model"] - bt["brier_book"]
    print("\n--- model vs book by time to expiry ---")
    print(bt.to_string(float_format=lambda v: f"{v:9.4f}"))


if __name__ == "__main__":
    main()
