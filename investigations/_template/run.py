"""Copy this folder, rename it, drop in the blocks you want to change.

Runs the benchmark model in `models/normal_qq` over one day, three times with
a different sigma-layer edge, and draws the three cumulative-PnL curves
against each other. That is the shape of an investigation: one question, a
parameter you vary, and a comparison you can look at.

To change a block, drop the file here. `link.py` in this folder shadows the
model's `link.py` and nothing else -- slots resolve THIS FOLDER first, then
`model=`, then `harness/blocks/defaults/`. Nothing is registered and nothing
is named.
"""
import os

from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")


#: The price-layer edge, held FIXED across the arms below. 2 c of half-spread,
#: and a retreat of 0.2 c per lot -- so an inventory of 5 pushes both quotes
#: 1 c away from the position, without widening the pair. The 10:1 ratio
#: between the two is the shape worth keeping when you rescale: the edge sets
#: where you start quoting, the retreat sets how fast inventory moves you.
E_P, RPL_P = 0.02, 0.002

#: The same shape one layer down, in sigma rather than probability. At the
#: money the link's slope is about 0.4 probability per sigma, so 0.05 sigma of
#: edge is worth roughly the 2 c above, and 0.005 sigma per lot puts an
#: inventory of 5 at roughly the same 1 c. The two layers are NOT redundant:
#: pass-through runs 0.79 c/$ at tau=300 s against 14.5 c/$ at tau=30 s, so a
#: sigma-layer edge tightens into the close where a probability-layer one does
#: not. That is what these three arms are for.
E_Z, RPL_Z = 0.05, 0.005

#: Both z-layer parameters scale together, so the ratio holds and the only
#: thing varying is how much of the edge lives in sigma space.
FACTORS = (0.5, 1.0, 2.0)


def one(factor):
    return backtest(
        investigation_dir=HERE,
        model=MODEL,
        # Named streams reach a block as `ep.stream("chainlink")`, and their
        # paths are fingerprinted into the run manifest.
        streams=("chainlink",),
        quote=QuoteParams(e_p=E_P, rpl_p=RPL_P,
                          e_z=E_Z * factor, rpl_z=RPL_Z * factor,
                          max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        output=Output(seeds=(0, 1, 2)),
        episodes=EPISODES,
        inputs=(paths.PANEL, paths.STRIKES, paths.SPOT, paths.RTDS_BTC),
    )


if __name__ == "__main__":
    catalog.install()

    # `rtds_path` is what the benchmark fair block actually reads: it learns
    # the BTC/USDT-to-BTC/USD basis against the oracle, so it takes
    # `ep.spot_usdt` and `ep.chainlink` and never `ep.spot`. Passing the
    # stream by NAME as well is not the same thing -- that populates
    # `ep.streams["chainlink"]`, which is the general accessor, not the
    # built-in array `fair.py` reads. Omit `rtds_path` and every `s` is NaN,
    # so the run completes, selects every market, and fills nothing.
    #
    # `warmup=True` matters as much: sigma is an EWMA, and a market that
    # starts its variance from scratch quotes badly for the first seconds of
    # every window. The warm-up runs the recursion over the PRIOR market so
    # the window opens with history behind it.
    EPISODES = load_episodes(days=("2026-08-20",), streams=("chainlink",),
                             spot_path=paths.SPOT, rtds_path=paths.RTDS_BTC,
                             warmup=True)

    # Running and reporting are separate steps. The harness writes artefacts;
    # the figures are drawn afterwards, across as many runs as you like.
    first = one(FACTORS[0])

    # Two ways to get a clean-looking nothing, both of which exit 0 and draw
    # an empty plot. `require=(...)` drops episodes silently, and a model
    # whose `s` is all NaN quotes nothing on every market it kept. Neither is
    # a result; fail on both.
    sample = first.summary["sample"]
    head = first.summary["headline"]
    if sample["n_selected"] == 0:
        raise SystemExit(f"selected 0 markets -- dropped: {sample['dropped']!r}")
    if head["n_fills"] == 0:
        raise SystemExit(
            f"selected {sample['n_selected']} markets and filled none. The "
            "model quoted nothing: check that `s` is finite, i.e. that the "
            "inputs its fair block reads were actually loaded.")
    print(f"n_selected={sample['n_selected']} n_fills={head['n_fills']} "
          f"pnl_per_market={head['pnl_per_market']:.4f}")

    def label(factor):
        return f"e_z={E_Z * factor:.3g}, rpl_z={RPL_Z * factor:.3g}"

    runs = {label(FACTORS[0]): first.run_dir}
    runs.update({label(f): one(f).run_dir for f in FACTORS[1:]})

    from harness.report import cumulative_pnl, load_runs
    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"))
    print(f"wrote {os.path.join(HERE, 'cum_pnl.png')}")
