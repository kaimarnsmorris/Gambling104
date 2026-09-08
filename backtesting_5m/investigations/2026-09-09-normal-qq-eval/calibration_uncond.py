import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kaima/Github2/Gambling104/backtesting_5m")
from harness.build.episodes import load_episodes
from harness.core import provenance
from harness import paths

INV = r"C:/Users/kaima/Github2/Gambling104/backtesting_5m/investigations/2026-09-09-normal-qq-eval"
res = provenance.resolve_slots(INV)
mods = {s: provenance.load_slot(p, s) for s, p in res.items()}

eps = load_episodes(spot_path=paths.SPOT)
eps = [e for e in eps if e.has_spot.any() and e.winner_up is not None]
print("episodes:", len(eps), flush=True)

rows = []
for ep in eps:
    s = mods["fair"].precompute(ep)
    sig = mods["vol"].precompute(ep)
    for i in (300, 900, 1500, 2100, 2700):        # tte 270,210,150,90,30 s
        if not (np.isfinite(s[i]) and np.isfinite(sig[i])):
            continue
        z = mods["f"].standardise(float(s[i]), ep.strike, float(sig[i]))
        rows.append((ep.tte_s(i), mods["link"].link(z), float(ep.mid[i]),
                     1.0 if ep.winner_up else 0.0))

d = pd.DataFrame(rows, columns=["tte", "p", "book_mid", "y"]).dropna()
print("observations:", len(d), flush=True)

def curve(df, col, label):
    q = pd.qcut(df[col], 10, duplicates="drop")
    g = df.groupby(q, observed=True).agg(pred=(col, "mean"), real=("y", "mean"), n=("y", "size"))
    print(f"\n--- {label} ---")
    print(g.assign(gap=(g.real - g.pred).round(3)).round(3).to_string(index=False))
    brier = ((df[col] - df.y) ** 2).mean()
    print(f"Brier: {brier:.4f}")

curve(d, "p", "MODEL fair_p, unconditional")
curve(d, "book_mid", "BOOK mid, same rows (benchmark)")
