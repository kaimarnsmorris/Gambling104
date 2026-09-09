"""Figures for the arms worth looking at.

Draws the four arms that carry the argument -- the quote rule (A vs B) and
the quoting window (last minute vs all 300 s) -- plus per-market detail for
the best one, with ticks emitted for a handful of markets so the quotes,
the fair and the inventory path can be read against the book.

Detail markets are chosen BEFORE the run and spread across the sample, not
picked from the winners: three markets from the best arm chosen by PnL would
show what a good market looks like and teach nothing about the strategy.
"""
import os

from harness import (Arm, ExecConfig, Output, QuoteParams, Sample,
                     grid_search, paths)
from harness.build.episodes import load_episodes
from harness.report import (cumulative_pnl, load_ledgers, load_runs,
                            load_ticks, market_detail)
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21")

A = dict(e_p=0.02, rpl_p=0.002, max_pos=50.0)
B = dict(e_p=0.04, rpl_p=0.0035, max_pos=100.0)
N_DETAIL = 3

if __name__ == "__main__":
    catalog.install()
    eps = load_episodes(days=DAYS, streams=("chainlink",),
                        spot_path=paths.SPOT_ORACLE_WINDOW,
                        rtds_path=paths.RTDS_BTC, warmup=True)

    step = max(1, len(eps) // (N_DETAIL + 1))
    detail_ids = tuple(eps[i * step].market_id for i in range(1, N_DETAIL + 1))

    def arm(label, kw, lo, hi, ticks=()):
        return Arm(label=label,
                   quote=QuoteParams(e_z=0.02, rpl_z=0.002, shares=10.0, **kw),
                   execn=ExecConfig(min_tte_s=lo, max_tte_s=hi))

    grid = [
        arm("B, last 60 s", B, 0, 60),
        arm("B, 30-60 s", B, 30, 60),
        arm("A, last 60 s", A, 0, 60),
        arm("B, all 300 s", B, 0, 300),
        arm("A, all 300 s", A, 0, 300),
    ]
    res = grid_search(
        grid=grid, investigation_dir=HERE, model=MODEL,
        streams=("chainlink",), execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        # ticks only for the three detail markets: ~3,000 rows each
        output=Output(seeds=(0,), emit_ticks=True, tick_markets=detail_ids),
        episodes=eps, workers=1,
        inputs=(paths.PANEL, paths.STRIKES, paths.SPOT_ORACLE_WINDOW,
                paths.RTDS_BTC))

    runs = {k: v["run_dir"] for k, v in res.items()}
    for label, r in res.items():
        h = r["summary"]["headline"]
        print(f"{label:16s} fills={h['n_fills']:6d} "
              f"pnl/mkt={h['pnl_per_market']:7.3f} "
              f"c/share={h['c_per_share']:7.3f}")

    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"),
                   title="QQ model: quote rule and quoting window")
    print("wrote cum_pnl.png")

    best = "B, last 60 s"
    ticks = load_ticks({best: runs[best]})
    ledger = load_ledgers({best: runs[best]})
    for n, mid in enumerate(detail_ids, start=1):
        market_detail(ticks, ledger, mid,
                      os.path.join(HERE, f"market_{n}.png"),
                      title=f"{best} -- market {n} of {len(detail_ids)}")
    print(f"wrote {len(detail_ids)} market detail plots")
