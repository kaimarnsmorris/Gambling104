"""Trade only where the model actually beats the book.

Everything so far says the model has no LOCATION edge on average -- rmse(s)
is 62.77 against 62.68 for the raw venue mid -- and that recalibrating its
width therefore cannot pay. But "on average" hides one regime. By time to
expiry, model Brier against book Brier:

    [0, 30)    0.0159   0.0145    +0.0014   book wins
    [30, 60)   0.0329   0.0354    -0.0025   MODEL WINS
    [60, 120)  0.0880   0.0851    +0.0028   book wins
    [120, 240) 0.1678   0.1575    +0.0103   book wins, badly
    [240, 300) 0.2260   0.2167    +0.0093   book wins

There is exactly one window where the model forecasts better than the market,
and it is the window where the fair block starts blending realised Chainlink
prints into its estimate -- i.e. where it knows something the book is slower
to price. The strategy currently quotes across all 300 s, so whatever edge
lives in [30,60) is diluted by 240 s of trading at a disadvantage.

`ExecConfig(min_tte_s=, max_tte_s=)` gates quoting by time to expiry, so this
is a one-parameter test of that reading.
"""
import os

from harness import (Arm, ExecConfig, Output, QuoteParams, Sample,
                     default_workers, grid_search, paths)
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21")

WINDOWS = [(0, 300), (0, 120), (15, 90), (20, 75), (30, 60), (25, 65),
           (30, 90), (0, 60)]
SCALES = (1.0, 1.6, 2.2)

if __name__ == "__main__":
    catalog.install()
    eps = load_episodes(days=DAYS, streams=("chainlink",),
                        spot_path=paths.SPOT_ORACLE_WINDOW,
                        rtds_path=paths.RTDS_BTC, warmup=True)

    quote = QuoteParams(e_p=0.02, rpl_p=0.002, e_z=0.02, rpl_z=0.002,
                        max_pos=50.0, shares=10.0)
    grid = [
        Arm(label=f"tte[{a},{b}) scale={k:g}", quote=quote,
            signal_params={"vol": {"scale": k}},
            execn=ExecConfig(min_tte_s=a, max_tte_s=b))
        for (a, b) in WINDOWS for k in SCALES
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
        g = r["summary"].get("gates", {})
        rows.append((label, h["n_fills"], h["pnl_per_market"],
                     h["c_per_share"], g, r["run_dir"]))
    rows.sort(key=lambda t: -t[2])
    print(f"\n{'arm':26s} {'fills':>7s} {'pnl/mkt':>9s} {'c/share':>9s}")
    for label, n, pnl, cps, _, _ in rows:
        print(f"{label:26s} {n:7d} {pnl:9.3f} {cps:9.3f}")
    print("\nbest:", rows[0][0])
    print("gates on best:", rows[0][4])
    with open(os.path.join(HERE, "best_run.txt"), "w") as fh:
        fh.write(rows[0][5])
