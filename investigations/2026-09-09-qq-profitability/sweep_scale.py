"""Does fixing the overconfidence convert into PnL?

`scale_probe.py` says the model's distribution is ~60 % too narrow and that
widening it removes 70 % of the Brier gap to the book. Better calibration is
not the same as money, so this runs it: sigma scale against half-spread, on
the oracle window, one seed, ranking only.

Read the caveat before reading the table. This is a 30-point in-sample grid
over 1,102 markets of one four-day window. Its best arm will look better than
it is -- that is what grids do -- so the arm that wins here is a hypothesis
for out-of-sample testing, not a result. `summary["gates"]` is what would
make it one.
"""
import os
import sys

from harness import (Arm, ExecConfig, Output, QuoteParams, Sample,
                     default_workers, grid_search, paths)
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21")

#: Measured in BTC dollars, not inferred: realised |settle - s| runs
#: 2.2-2.5x the sigma the model claims, at every tte past 30 s. Brier-optimal
#: widening is only 1.6x, which is the fat tails showing -- RMSE is inflated
#: by the tails, so matching it over-widens the body.
SCALES = (1.0, 1.3, 1.6, 1.9, 2.2, 2.5)

#: Capped at 0.02 sigma deliberately. Wider arms "win" by not trading, which
#: is not a strategy: the earlier 0.1-0.8 sweep improved PnL monotonically
#: with width while c/share got WORSE, i.e. every gain came from fewer fills
#: rather than better ones. Held near zero, the only way to profit is a fair
#: value that is actually right.
E_Z = (0.0, 0.005, 0.01, 0.02)

if __name__ == "__main__":
    catalog.install()
    eps = load_episodes(days=DAYS, streams=("chainlink",),
                        spot_path=paths.SPOT_ORACLE_WINDOW,
                        rtds_path=paths.RTDS_BTC, warmup=True)

    grid = [
        Arm(label=f"scale={k:g} e_z={e:g}",
            quote=QuoteParams(e_p=0.02, rpl_p=0.002, e_z=e, rpl_z=e / 10.0,
                              max_pos=50.0, shares=10.0),
            signal_params={"vol": {"scale": k}})
        for k in SCALES for e in E_Z
    ]

    res = grid_search(
        grid=grid, investigation_dir=HERE, model=MODEL,
        streams=("chainlink",), execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        output=Output(seeds=(0,)), episodes=eps,
        workers=default_workers(),
        inputs=(paths.PANEL, paths.STRIKES, paths.SPOT_ORACLE_WINDOW,
                paths.RTDS_BTC))

    rows = []
    for label, r in res.items():
        h = r["summary"]["headline"]
        rows.append((label, h["n_fills"], h["pnl_per_market"],
                     h["c_per_share"], r["run_dir"]))
    rows.sort(key=lambda t: -t[2])
    print(f"\n{'arm':26s} {'fills':>7s} {'pnl/mkt':>9s} {'c/share':>9s}")
    for label, n, pnl, cps, _ in rows:
        print(f"{label:26s} {n:7d} {pnl:9.3f} {cps:9.3f}")
    with open(os.path.join(HERE, "best_run.txt"), "w") as fh:
        fh.write(rows[0][4])
    print("\nbest:", rows[0][0])
