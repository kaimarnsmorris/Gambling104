"""Copy this folder, rename it, drop in the blocks you want to change."""
import os

from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")

def one(e_p):
    return backtest(
        investigation_dir=HERE,
        model=MODEL,
        streams=("chainlink",),
        quote=QuoteParams(e_p=e_p, rpl_p=0.0005, max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        output=Output(seeds=(0, 1, 2)),
        episodes=EPISODES,
    )


if __name__ == "__main__":
    catalog.install()
    EPISODES = load_episodes(days=("2026-08-20",), streams=("chainlink",),
                             spot_path=paths.SPOT)

    # Running and reporting are separate steps. The harness writes artefacts;
    # the figures are drawn afterwards, across as many runs as you like.
    first = one(0.01)

    # `Sample(require=(...))` drops episodes silently -- a run that selects
    # zero markets still writes an empty PNG and exits 0. Fail loudly here
    # rather than let that pass for a result.
    n_selected = first.summary["sample"]["n_selected"]
    if n_selected == 0:
        raise SystemExit(
            "template selected 0 markets -- "
            f"dropped: {first.summary['sample']['dropped']!r}")
    print(f"n_selected={n_selected}")

    runs = {"e_p=0.01": first.run_dir}
    runs.update({f"e_p={e}": one(e).run_dir for e in (0.02, 0.03)})

    from harness.report import cumulative_pnl, load_runs
    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"))
