"""End-to-end smoke run on one real day. Not a result -- a proof of plumbing."""
import os

from harness import paths
from harness.build.episodes import load_episodes
from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.latency import LatencyModel
from harness.core.run import run

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    episodes = load_episodes(days=("2026-08-20",), max_markets=50)
    print(f"loaded {len(episodes)} episodes")

    result = run(
        HERE,
        quote=QuoteParams(e_p=0.03, rpl_p=0.0005, max_pos=50.0, shares=10.0),
        execn=ExecConfig(latency=LatencyModel(jitter_frac=0.2)),
        sample=Sample(),
        output=Output(emit_ticks=True, seeds=(0, 1, 2)),
        episodes=episodes,
        # fingerprinted into manifest.json, so the run folder
        # still says which data it read
        inputs=(paths.PANEL, paths.STRIKES),
    )
    print(result["run_dir"])
    print(result["summary"]["headline"])
    print(result["summary"]["gates"]["passed"])
