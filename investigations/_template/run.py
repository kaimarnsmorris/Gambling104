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

#: The same shape one layer down, in sigma rather than probability, and the
#: one thing these three arms vary. `rpl_z` holds the same 10:1 ratio to
#: `e_z`, so the arms rescale the sigma-layer edge without changing its
#: shape. The two layers are NOT redundant: pass-through runs 0.79 c/$ at
#: tau=300 s against 14.5 c/$ at tau=30 s, so a sigma-layer edge tightens
#: into the close where a probability-layer one does not.
E_Z_ARMS = (0.1, 0.2, 0.3)

#: Ticks are ~3,000 rows per market, so they are emitted for a handful of
#: markets rather than the day. Chosen before the runs and held the same
#: across them, so the arms can be read against each other on one market.
N_DETAIL = 3


def one(e_z, tick_markets=()):
    return backtest(
        investigation_dir=HERE,
        model=MODEL,
        # Named streams reach a block as `ep.stream("chainlink")`, and their
        # paths are fingerprinted into the run manifest.
        streams=("chainlink",),
        quote=QuoteParams(e_p=E_P, rpl_p=RPL_P,
                          e_z=e_z, rpl_z=e_z / 10.0,
                          max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        # ONE seed. A seed changes only the latency draws, so extra seeds
        # buy a dispersion estimate, not a better estimate -- worth having
        # on a finalist, wasteful on every arm of a comparison. Re-run the
        # arm you settle on with seeds=(0, 1, 2) and read the spread there.
        output=Output(seeds=(0,), emit_ticks=bool(tick_markets),
                      tick_markets=tick_markets),
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

    # Ticks are per-market and heavy, so pick the sample BEFORE running:
    # three markets evenly spaced across the day, so the detail plots are not
    # all from one stretch of the session.
    step = max(1, len(EPISODES) // (N_DETAIL + 1))
    detail_ids = tuple(EPISODES[i * step].market_id
                       for i in range(1, N_DETAIL + 1))

    # Running and reporting are separate steps. The harness writes artefacts;
    # the figures are drawn afterwards, across as many runs as you like.
    results = {}
    for e_z in E_Z_ARMS:
        r = one(e_z, tick_markets=detail_ids)
        results[f"e_z={e_z:g}, rpl_z={e_z / 10.0:g}"] = r

        # Two ways to get a clean-looking nothing, both of which exit 0 and
        # draw an empty plot. `require=(...)` drops episodes silently, and a
        # model whose `s` is all NaN quotes nothing on every market it kept.
        # Neither is a result; fail on both.
        sample, head = r.summary["sample"], r.summary["headline"]
        if sample["n_selected"] == 0:
            raise SystemExit(
                f"e_z={e_z:g} selected 0 markets -- {sample['dropped']!r}")
        if head["n_fills"] == 0:
            raise SystemExit(
                f"e_z={e_z:g} selected {sample['n_selected']} markets and "
                "filled none. The model quoted nothing: check that `s` is "
                "finite, i.e. that the inputs its fair block reads loaded.")
        print(f"e_z={e_z:g}  n={sample['n_selected']}  "
              f"fills={head['n_fills']}  "
              f"pnl/market={head['pnl_per_market']:.3f}  "
              f"c/share={head['c_per_share']:.3f}")

    from harness.report import (cumulative_pnl, load_runs, load_ticks,
                                load_ledgers, market_detail)

    runs = {label: r.run_dir for label, r in results.items()}
    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"))

    # Detail plots come from the arm that actually did best, chosen from the
    # results rather than assumed -- an example that hardcodes its own winner
    # stops being an example the first time the data changes.
    best = max(results,
               key=lambda k: results[k].summary["headline"]["pnl_per_market"])
    print(f"best arm: {best}")

    ticks = load_ticks({best: results[best].run_dir})
    ledger = load_ledgers({best: results[best].run_dir})
    for n, market_id in enumerate(detail_ids, start=1):
        market_detail(ticks, ledger, market_id,
                      os.path.join(HERE, f"market_{n}.png"),
                      title=f"{best} -- market {n} of {len(detail_ids)}")
    print(f"wrote cum_pnl.png and {len(detail_ids)} market detail plots")
