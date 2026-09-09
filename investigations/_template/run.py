"""Copy this folder, rename it, drop in the blocks you want to change."""
import os

from harness import ExecConfig, Output, QuoteParams, Sample, backtest
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
    EPISODES = load_episodes(days=("2026-08-20",), streams=("chainlink",))

    # Running and reporting are separate steps. The harness writes artefacts;
    # the figures are drawn afterwards, across as many runs as you like.
    runs = {f"e_p={e}": one(e).run_dir for e in (0.01, 0.02, 0.03)}

    from harness.report import cumulative_pnl, load_runs
    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"))
