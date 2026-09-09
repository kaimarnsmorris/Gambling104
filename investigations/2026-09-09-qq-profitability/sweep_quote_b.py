"""A second quote configuration, against the first, across the same windows.

e_p 0.04 / rpl_p 0.0035 / max_pos 100 versus 0.02 / 0.002 / 50. Both arms of
each pair share everything else, so the comparison is the quote rule alone.

Note what max_pos does to the retreat. `q` in the quote algebra is the raw
position in SHARES, clamped to max_pos, so the lean at full inventory is
max_pos * rpl_p:

    old:  50 * 0.0020 = 0.10   ->  10 c of retreat at the cap
    new: 100 * 0.0035 = 0.35   ->  35 c

35 c is past the point where one side clips to the [0, 1] bound, so at full
inventory the new rule stops quoting that side rather than merely leaning.
That may be exactly what is wanted -- it is a hard inventory brake -- but it
means the two rules differ in kind near the cap, not only in degree.
"""
import os

from harness import (Arm, ExecConfig, Output, QuoteParams, Sample,
                     default_workers, grid_search, paths)
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21")

QUOTES = {
    "A": dict(e_p=0.02, rpl_p=0.002, max_pos=50.0),
    "B": dict(e_p=0.04, rpl_p=0.0035, max_pos=100.0),
}
WINDOWS = [(0, 300), (0, 120), (0, 90), (0, 60), (20, 75), (30, 60)]
E_Z = (0.0, 0.02)

if __name__ == "__main__":
    catalog.install()
    eps = load_episodes(days=DAYS, streams=("chainlink",),
                        spot_path=paths.SPOT_ORACLE_WINDOW,
                        rtds_path=paths.RTDS_BTC, warmup=True)

    grid = []
    for name, kw in QUOTES.items():
        for (a, b) in WINDOWS:
            for e_z in E_Z:
                grid.append(Arm(
                    label=f"{name} tte[{a},{b}) e_z={e_z:g}",
                    quote=QuoteParams(e_z=e_z, rpl_z=e_z / 10.0, shares=10.0,
                                      **kw),
                    execn=ExecConfig(min_tte_s=a, max_tte_s=b)))

    res = grid_search(
        grid=grid, investigation_dir=HERE, model=MODEL,
        streams=("chainlink",), execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        output=Output(seeds=(0,)), episodes=eps, workers=default_workers(),
        inputs=(paths.PANEL, paths.STRIKES, paths.SPOT_ORACLE_WINDOW,
                paths.RTDS_BTC))

    rows = []
    for label, r in res.items():
        h, g = r["summary"]["headline"], r["summary"]["gates"]
        rows.append((label, h["n_fills"], h["max_abs_q"] if "max_abs_q" in h
                     else float("nan"), h["pnl_per_market"], h["c_per_share"],
                     g, r["run_dir"]))
    rows.sort(key=lambda t: -t[3])
    print(f"\n{'arm':26s} {'fills':>7s} {'pnl/mkt':>9s} {'c/share':>9s}")
    for label, n, _, pnl, cps, _, _ in rows:
        print(f"{label:26s} {n:7d} {pnl:9.3f} {cps:9.3f}")

    best = rows[0]
    print(f"\nbest: {best[0]}")
    gs = best[5]["gates"]
    print(f"  sign_survives_periods {gs['sign_survives_periods']['passed']}  "
          f"{[round(v, 3) for v in gs['sign_survives_periods']['per_period']]}")
    print(f"  ci_excludes_zero      {gs['ci_excludes_zero']['passed']}  "
          f"{[round(v, 3) for v in gs['ci_excludes_zero']['ci']]}")
    dt = gs["delete_top_10"]
    print(f"  delete_top_10         {dt['passed']}  "
          f"{dt['mean_full']:.3f} -> {dt['mean_without_top']:.3f}")
    with open(os.path.join(HERE, "best_run.txt"), "w") as fh:
        fh.write(best[6])
