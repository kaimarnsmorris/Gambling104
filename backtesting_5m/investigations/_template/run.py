"""Copy this folder, rename it, drop in the block files you want to change."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness.build.episodes import load_episodes          # noqa: E402
from harness.core.config import ExecConfig, Output, QuoteParams, Sample  # noqa: E402
from harness.core.latency import LatencyModel             # noqa: E402
from harness.core.run import run                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    episodes = load_episodes(days=("2026-08-20",))
    result = run(
        HERE,
        quote=QuoteParams(e_p=0.02, rpl_p=0.001, max_pos=50.0, shares=10.0),
        execn=ExecConfig(mode="maker", latency=LatencyModel()),
        sample=Sample(),
        output=Output(seeds=(0, 1, 2)),
        episodes=episodes,
    )
    print(result["run_dir"])
    print(result["summary"]["headline"])
