"""Unconditional calibration of the headline model against the book's own mid.

Five decision indices per market (tte 270, 210, 150, 90 and 30 s), the model's
p and the book's mid scored on EXACTLY the same rows against the realised
outcome. The book is the benchmark that says how much of the residual Brier is
irreducible; scoring it on its own larger sample would make it a different
question.

Run against the same episodes the backtest uses -- `paths.SPOT_ORACLE_WINDOW`,
the Chainlink line attached, and 900 s of pre-open warm-up. All three matter:
without the oracle `fair.py` returns NaN everywhere, and without the warm-up
its basis runs a 180 s halflife from a cold seed, which is a different model
from the one `run.py` prices with.

Saturation (p >= 0.999) and the worst decile gap are printed alongside Brier,
because a model can look respectable on Brier while pinning 40 % of its
observations at p = 1.000 and settling in the money on only 77 % of them.
"""
import json
import os

import numpy as np
import pandas as pd

from harness.build.episodes import load_episodes
from harness.core import provenance
from harness import paths

HERE = os.path.dirname(os.path.abspath(__file__))
#: `fair.py`/`vol.py`/`f.py`/`link.py` were byte-identical to the canonical
#: model and have been deleted from this folder; resolve_slots below is
#: given MODEL so every slot still resolves to the file the run priced with.
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
EVAL_INDICES = (300, 900, 1500, 2100, 2700)     # tte 270, 210, 150, 90, 30 s
SATURATED = 0.999
OUT_JSON = "calibration_uncond.json"


def main():
    manifest_path = os.path.join(HERE, "last_run_manifest.json")
    spot_path, days, warmup_s = paths.SPOT_ORACLE_WINDOW, None, 900.0
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            m = json.load(fh)
        spot_path = m.get("spot_path", spot_path)
        days = tuple(m.get("spot_days") or ()) or None
        warmup_s = float(m.get("warmup_s", warmup_s))

    mods = {s: provenance.load_slot(p, s)
            for s, p in provenance.resolve_slots(HERE, model_dir=MODEL).items()}

    eps = load_episodes(spot_path=spot_path, days=days,
                        rtds_path=paths.RTDS_BTC,
                        warmup=warmup_s > 0, warmup_s=warmup_s or 900.0)
    eps = [e for e in eps if e.has_spot.any() and e.winner_up is not None]
    print(f"episodes: {len(eps)}  (panel {os.path.basename(spot_path)}, "
          f"warm-up {warmup_s:.0f} s)", flush=True)

    rows = []
    for ep in eps:
        s = mods["fair"].precompute(ep)
        sig = mods["vol"].precompute(ep)
        for i in EVAL_INDICES:
            if not (np.isfinite(s[i]) and np.isfinite(sig[i])):
                continue
            z = mods["f"].standardise(float(s[i]), ep.strike, float(sig[i]))
            rows.append((ep.day, ep.tte_s(i), mods["link"].link(z),
                         float(ep.mid[i]), 1.0 if ep.winner_up else 0.0))

    d = pd.DataFrame(rows, columns=["day", "tte", "p", "book_mid", "y"]).dropna()
    print(f"observations: {len(d)} over "
          f"{d['day'].nunique()} days", flush=True)

    out = {"n": int(len(d)), "spot_path": spot_path, "warmup_s": warmup_s}
    for col, label in (("p", "MODEL fair_p"), ("book_mid", "BOOK mid")):
        out[col] = curve(d, col, label)
    with open(os.path.join(HERE, OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_JSON}")


def curve(df, col, label):
    q = pd.qcut(df[col], 10, duplicates="drop")
    g = df.groupby(q, observed=True).agg(pred=(col, "mean"),
                                         real=("y", "mean"), n=("y", "size"))
    g["gap"] = g["real"] - g["pred"]
    brier = float(((df[col] - df["y"]) ** 2).mean())
    hi = df[col] >= SATURATED
    lo = df[col] <= 1.0 - SATURATED
    worst = g.loc[g["gap"].abs().idxmax()]

    print(f"\n--- {label}, unconditional ---")
    print(f"{'pred':>8}{'real':>8}{'n':>8}{'gap':>8}")
    for _, r in g.iterrows():
        print(f"{r.pred:>8.3f}{r.real:>8.3f}{int(r.n):>8d}{r.gap:>8.3f}")
    print(f"Brier {brier:.4f}   p>=0.999 {float(hi.mean()):.4f} "
          f"(realised {float(df.loc[hi,'y'].mean()) if hi.any() else float('nan'):.4f})"
          f"   p<=0.001 {float(lo.mean()):.4f}"
          f"   worst decile gap {float(worst['gap']):+.3f}")
    return {
        "brier": brier,
        "sat_hi_frac": float(hi.mean()),
        "sat_hi_realised": float(df.loc[hi, "y"].mean()) if hi.any() else None,
        "sat_lo_frac": float(lo.mean()),
        "sat_lo_realised": float(df.loc[lo, "y"].mean()) if lo.any() else None,
        "worst_decile_gap": float(worst["gap"]),
        "worst_decile_pred": float(worst["pred"]),
        "worst_decile_real": float(worst["real"]),
        "mean_p": float(df[col].mean()),
        "mean_y": float(df["y"].mean()),
        "deciles": g.reset_index(drop=True).to_dict("records"),
    }


if __name__ == "__main__":
    main()
