"""Baseline the benchmark model over the whole oracle window, for diagnosis.

Not a sweep. One arm, one seed, the widest sample the Chainlink capture can
price (2026-08-17..08-21, versus the single day the template uses), so the
per-fill decomposition in `diagnose.py` has enough fills to condition on.

`SPOT_ORACLE_WINDOW` rather than `SPOT`: the default panel runs 08-19..24 but
the oracle stops 08-21 01:59, so on the default panel a basis-learning fair
block has an oracle for only ~2 of its 6 days. This build is the three-way
overlap.
"""
import os

from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")

DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21")

#: The template's widest arm. Chosen because it was the least-bad of the
#: three, not because it is good -- the point here is to find out WHY it
#: loses, so the arm only has to be representative.
QUOTE = QuoteParams(e_p=0.02, rpl_p=0.002, e_z=0.3, rpl_z=0.03,
                    max_pos=50.0, shares=10.0)

if __name__ == "__main__":
    catalog.install()
    eps = load_episodes(days=DAYS, streams=("chainlink",),
                        spot_path=paths.SPOT_ORACLE_WINDOW,
                        rtds_path=paths.RTDS_BTC, warmup=True)
    print(f"loaded {len(eps)} episodes")

    r = backtest(
        investigation_dir=HERE, model=MODEL, streams=("chainlink",),
        quote=QUOTE, execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        # Ticks on for every market: the decomposition needs the quotes and
        # the fair at moments we did NOT trade, which the fill ledger by
        # construction cannot show.
        output=Output(seeds=(0,), emit_ticks=True),
        episodes=eps,
        inputs=(paths.PANEL, paths.STRIKES, paths.SPOT_ORACLE_WINDOW,
                paths.RTDS_BTC),
    )
    s, h = r.summary["sample"], r.summary["headline"]
    print(f"n={s['n_selected']} dropped={s['dropped']} fills={h['n_fills']} "
          f"pnl/market={h['pnl_per_market']:.3f} c/share={h['c_per_share']:.3f}")
    print("run_dir:", r.run_dir)
    with open(os.path.join(HERE, "last_run.txt"), "w") as fh:
        fh.write(r.run_dir)
