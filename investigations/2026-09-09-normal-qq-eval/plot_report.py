"""The full plot set for the normal-QQ model evaluation.

Reads the run folders recorded in `last_run_manifest.json` (written by
`run.py`) and writes every figure as a PNG into the headline run folder,
which is treated as the report's home directory.

The execution policy is unified -- one run makes and takes -- so maker and
taker are separated by the ledger's `liquidity` column, not by running the
harness twice.

    1. cumulative PnL over the ordered market sequence, day boundaries
       marked -- NET in blue over GROSS (pre-fee) in grey, with the fee drag
       shaded between them and annotated in dollars and in c/share
    2. per-market detail for 4 traded markets, chosen from the days with
       Chainlink coverage: book/fair/eff quotes with fill markers, position,
       mark-to-market PnL, and a fourth panel in BTC dollar space (venue
       spot, the model's fair BTC estimate, the real Chainlink 60 s TWAP
       where it is covered, and the Chainlink strike/settlement levels)
    3. markout (delta_quality_c) distribution, maker vs taker
    4. PnL-proxy and fill count by time-to-expiry bucket
    5. calibration: fair_p at fill time vs realised outcome frequency

Note on the BTC-space panel's Chainlink content: `twap60` from the RTDS
capture (see `harness.paths.RTDS_BTC`) is a real, continuous, 1 s-cadence
Chainlink series and IS plotted as a line where it covers the market -- it is
the exact settlement variable, so it sits on the same axis as the model's
E[A] and the gap between them is the model's forecast error. That feed's
coverage window ends at 01:59 UTC on 2026-08-21, so the four detail markets
are drawn only from the days it covers in full (08-17..08-20); a market
outside the window would show no Chainlink line and the panel says so rather
than interpolating across the gap. The strike and settlement horizontals are independent of that coverage
-- they come from the boundary reports this project already verified against
signed on-chain `ReportVerified` reports at median and max difference $0.00
across 1,992 markets -- and are drawn regardless.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from harness.blocks.defaults.fees import Liquidity
LIQ_MAKER = int(Liquidity.MAKER)
from harness.io import read_parquet
from harness.core.provenance import load_slot, resolve_slots
from harness.build.episodes import load_episodes
from harness import paths

HERE = os.path.dirname(os.path.abspath(__file__))
#: `fair.py`/`vol.py`/`f.py`/`link.py` were byte-identical to the canonical
#: model and have been deleted from this folder; resolve_slots below is
#: given MODEL so `link` still resolves to the same file the run used.
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")

BUY_MARKER = dict(marker="^", color="tab:green", s=28, zorder=5, label="buy")
SELL_MARKER = dict(marker="v", color="tab:red", s=28, zorder=5, label="sell")

#: Fallback only. The days, the spot panel and the warm-up length are all
#: read from `last_run_manifest.json`, so these figures are drawn against the
#: same episodes the run priced; this constant is what an old manifest that
#: predates those keys falls back to.
RTDS_COVERED_DAYS = ("2026-08-19", "2026-08-20")


def _rtds_for_market(open_ts):
    """(tte_s, px, twap60) for one market's window, or Nones if uncovered.

    RTDS is 1 Hz; `oracle_ms` is epoch ms, `open_ts` is epoch seconds. No
    interpolation -- only the actual 1 s samples inside [open_ts, open_ts+300)
    are returned, converted to seconds-to-expiry the same way the tick axis
    is.

    Both series are returned. `twap60` is the settlement variable but lags by
    construction, being a trailing 60 s average; the RAW oracle price `px` is
    where Chainlink actually is at each instant. `px` is also the comparator
    for the venue spot line, and WHAT THE GAP BETWEEN THEM MEANS DEPENDS ON
    WHICH SPOT PANEL THE RUN USED. On `paths.SPOT` / `SPOT_ORACLE_WINDOW` the
    panel is BTC/USD, so the two should very nearly coincide and a visible gap
    is a basis problem. On `paths.SPOT_LONDON` the panel is BTC/USDT and
    uncorrected, so a gap of roughly +$50 is EXPECTED and is the USDT premium
    itself -- the model's own `B_t` is what removes it, which is why the blue
    E[A] line, not the grey spot line, is the one that should sit on the
    oracle. `main` labels the spot line accordingly.

    ON THE ORACLE'S OWN CLOCK, DELIBERATELY. This selects on `oracle_ms`, the
    stamp Chainlink writes, whereas `harness.build.episodes.load_chainlink`
    grids by `px_first_recv_ns`, when the price reached this vantage ~1.5 s
    later. That difference is the whole reason the model reads receipts: a
    decision may not use a price it has not been told. Here it is the right
    choice for the opposite reason -- this line is not an input, it is the
    ground truth being plotted, and the honest place to draw the settlement
    variable is where it actually was. So the Chainlink line on these figures
    runs ~1.5 s to the LEFT of what the model could see, by construction.
    """
    lo_ms, hi_ms = open_ts * 1000, (open_ts + 300) * 1000
    rtds = read_parquet(paths.RTDS_BTC,
                        columns=["oracle_ms", "px", "twap60"],
                        filters=[("oracle_ms", ">=", lo_ms),
                                ("oracle_ms", "<", hi_ms)])
    if not len(rtds):
        return None, None, None
    rtds = rtds.dropna(subset=["px", "twap60"]).sort_values("oracle_ms")
    if not len(rtds):
        return None, None, None
    tte_s = 300.0 - (rtds["oracle_ms"].to_numpy() / 1000.0 - open_ts)
    return tte_s, rtds["px"].to_numpy(), rtds["twap60"].to_numpy()


def _load_run(run_dir):
    out = {"run_dir": run_dir}
    out["markets"] = read_parquet(os.path.join(run_dir, "markets.parquet"))
    out["ledger"] = read_parquet(os.path.join(run_dir, "ledger.parquet"))
    tick_path = os.path.join(run_dir, "ticks.parquet")
    out["ticks"] = (read_parquet(tick_path) if os.path.exists(tick_path)
                    else pd.DataFrame())
    with open(os.path.join(run_dir, "summary.json")) as fh:
        out["summary"] = json.load(fh)
    return out


# -- 1. cumulative net PnL, day boundaries marked ---------------------------

def plot_cumulative_pnl(markets, path, title):
    """Cumulative NET PnL, with GROSS (pre-fee) greyed in behind it.

    The two lines answer a question the net line alone cannot: how much of the
    loss is a bad forecast and how much is simply the cost of trading. Taker
    fees have been running ~1.23 c/share against a total loss of ~1.30, so the
    vertical gap between grey and blue is very nearly the whole story -- and if
    grey ends near zero, the model is roughly break-even before fees and the
    finding is "the edge does not cover the fee", not "the forecast is wrong".
    Those are materially different conclusions, so the split is annotated in
    dollars AND in c/share rather than left to be eyeballed off the axis.

    Net is drawn last, thicker and in colour, so it stays the dominant series;
    gross is a light grey line underneath it at a lower z-order.
    """
    m = markets[markets["seed"] == markets["seed"].iloc[0]].sort_values(
        "open_ts").reset_index(drop=True)
    net = m["pnl_net"].fillna(0.0)
    gross = m["pnl_gross"].fillna(0.0)
    cum_net = net.cumsum()
    cum_gross = gross.cumsum()

    # `shares` is the traded volume the c/share denominator uses everywhere
    # else in this report (summary.json's `c_per_share` = 100 * pnl / shares),
    # so the same denominator is used here and the three numbers add up.
    shares = float(m["shares"].fillna(0.0).sum())
    fee_drag = float(gross.sum() - net.sum())        # >= 0 when fees are paid

    def cps(usd):
        return 100.0 * usd / shares if shares else float("nan")

    def usd(v):
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = range(len(m))
    ax.plot(x, cum_gross, lw=1.0, color="0.62", zorder=2,
            label=(f"gross, before fees: {usd(gross.sum())} "
                   f"({cps(gross.sum()):+.3f} c/share)"))
    ax.fill_between(x, cum_net, cum_gross, color="0.62", alpha=0.16, zorder=1,
                    label=(f"fee drag: {usd(-fee_drag)} "
                           f"({cps(-fee_drag):+.3f} c/share)"))
    ax.plot(x, cum_net, lw=1.4, color="tab:blue", zorder=3,
            label=(f"net, after fees: {usd(net.sum())} "
                   f"({cps(net.sum()):+.3f} c/share)"))
    ax.axhline(0.0, color="0.6", lw=0.8)

    day_changes = m.index[m["day"] != m["day"].shift(1)].tolist()
    for pos, i in enumerate(day_changes):
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)
        if pos < len(day_changes):
            ax.text(i, ax.get_ylim()[1], m.loc[i, "day"], rotation=90,
                    fontsize=7, va="top", ha="right", color="0.4")

    ax.set_xlabel("market (chronological)")
    ax.set_ylabel("cumulative PnL, USD")
    ax.set_title(title)
    # The split goes in the legend TITLE rather than a floating text box, so
    # it can never land on top of the very lines it is describing.
    ax.legend(fontsize=8, loc="lower left", framealpha=0.92,
              title=(f"{len(m):,} markets, {shares:,.0f} shares   |   "
                     f"gross {cps(gross.sum()):+.3f} "
                     f"- fees {cps(fee_drag):.3f} "
                     f"= net {cps(net.sum()):+.3f} c/share"),
              title_fontsize=8)
    ax.get_legend().get_title().set_color("0.25")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- 2. per-market detail ----------------------------------------------------

def plot_market_detail(ticks, ledger, ep, path, spot_label="venue spot"):
    market_id = ep.market_id
    t = ticks[ticks["market_id"] == market_id].sort_values("t_ms")
    if not len(t):
        return False
    tte_s = 300.0 - t["t_ms"].to_numpy() / 1000.0

    fills = ledger[ledger["market_id"] == market_id].sort_values("t_ms")
    fill_tte = 300.0 - fills["t_ms"].to_numpy() / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1.6]})

    ax = axes[0]
    ax.plot(tte_s, t["book_bid"], color="0.6", lw=0.8, label="book bid")
    ax.plot(tte_s, t["book_ask"], color="0.6", lw=0.8, ls="--",
           label="book ask")
    ax.plot(tte_s, t["mid"], color="0.3", lw=0.8, label="mid")
    ax.plot(tte_s, t["fair_p"], color="tab:blue", lw=1.3, label="fair_p")
    ax.plot(tte_s, t["eff_bid"], color="tab:orange", lw=1.0, ls=":",
           label="eff_bid (theoretical, not an order price)")
    ax.plot(tte_s, t["eff_ask"], color="tab:purple", lw=1.0, ls=":",
           label="eff_ask (theoretical, not an order price)")

    # `price` on a fill IS the actual placed/executed order price: for a
    # maker fill it is the fee-adjusted, snapped resting limit; for a taker
    # fill it is the crossing price paid. eff_bid/eff_ask above are the
    # unsnapped theoretical valuation that price is derived from -- they are
    # deliberately not the same series (see harness/core/ledger.py).
    is_maker = fills["liquidity"] == LIQ_MAKER
    buys_maker = fills[(fills["side"] == 1) & is_maker]
    buys_taker = fills[(fills["side"] == 1) & ~is_maker]
    sells_maker = fills[(fills["side"] == -1) & is_maker]
    sells_taker = fills[(fills["side"] == -1) & ~is_maker]
    if len(buys_maker):
        ax.scatter(300.0 - buys_maker["t_ms"] / 1000.0, buys_maker["price"],
                  **{**BUY_MARKER, "label": "buy (maker, placed price)"})
    if len(buys_taker):
        ax.scatter(300.0 - buys_taker["t_ms"] / 1000.0, buys_taker["price"],
                  marker="^", facecolors="none", edgecolors="tab:green",
                  s=40, zorder=5, linewidths=1.3,
                  label="buy (taker, placed price)")
    if len(sells_maker):
        ax.scatter(300.0 - sells_maker["t_ms"] / 1000.0, sells_maker["price"],
                  **{**SELL_MARKER, "label": "sell (maker, placed price)"})
    if len(sells_taker):
        ax.scatter(300.0 - sells_taker["t_ms"] / 1000.0, sells_taker["price"],
                  marker="v", facecolors="none", edgecolors="tab:red",
                  s=40, zorder=5, linewidths=1.3,
                  label="sell (taker, placed price)")

    ax.set_ylabel("probability")
    ax.set_title(f"...{market_id[-12:]}: quotes, fair value and fills",
                fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")

    axes[1].plot(tte_s, t["q"], color="tab:blue", lw=1.0)
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].set_ylabel("position q")

    axes[2].plot(tte_s, t["cum_pnl"], color="tab:green", lw=1.0)
    axes[2].axhline(0.0, color="0.6", lw=0.6)
    axes[2].set_ylabel("mark-to-market PnL, USD")

    # -- panel 4: BTC dollar space -------------------------------------
    ax4 = axes[3]
    ax4.plot(tte_s, t["spot"], color="0.4", lw=0.9, label=spot_label)
    ax4.plot(tte_s, t["s"], color="tab:blue", lw=1.2,
            label="model E[A] (fair.precompute)")

    rtds_tte, rtds_px, rtds_twap60 = _rtds_for_market(ep.open_ts)
    if rtds_tte is not None:
        ax4.plot(rtds_tte, rtds_px, color="tab:green", lw=0.9, alpha=0.85,
                label="Chainlink px (raw oracle)")
        ax4.plot(rtds_tte, rtds_twap60, color="tab:red", lw=1.1,
                label="Chainlink twap60 (settlement variable)")
    else:
        ax4.text(0.02, 0.05, "Chainlink px/twap60: no coverage this market",
                fontsize=7, color="tab:red", transform=ax4.transAxes)

    ax4.axhline(ep.strike, color="0.2", lw=1.2, ls="-",
               label="Chainlink strike (60s TWAP @ open)")
    if ep.settle is not None:
        ax4.axhline(ep.settle, color="0.2", lw=1.2, ls="--",
                   label="Chainlink settle (60s TWAP @ expiry)")
    ax4.set_ylabel("BTC, USD")
    ax4.set_xlabel("seconds to expiry")
    ax4.legend(fontsize=6.5, ncol=2, loc="upper left")

    # The four panels share one x-axis, so invert it exactly ONCE. Looping
    # over `axes` toggles the same shared axis four times -- an even number,
    # which lands back on the default and makes time run right-to-left. That
    # is what happened when this figure grew from three panels to four.
    axes[0].invert_xaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# -- 3. markout distribution --------------------------------------------------

def plot_markout(ledger_maker, ledger_taker, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-30, 30, 61)
    for label, ledger, color in (("maker", ledger_maker, "tab:blue"),
                                 ("taker", ledger_taker, "tab:orange")):
        dq = ledger["delta_quality_c"].dropna()
        if not len(dq):
            continue
        ax.hist(dq.clip(-30, 30), bins=bins, alpha=0.5, density=True,
               color=color, label=f"{label} (n={len(dq)})")
        ax.axvline(dq.mean(), color=color, lw=1.6, ls="--")

    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel("markout, delta_quality_c (cents/share, +10s or settlement)")
    ax.set_ylabel("density")
    ax.set_title("Markout distribution by liquidity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- 4. PnL-proxy and fill count by time-to-expiry bucket --------------------

def plot_by_tte_bucket(ledger, path, bucket_s=30.0):
    if not len(ledger):
        return
    tte = 300.0 - ledger["t_ms"] / 1000.0
    bucket = (tte // bucket_s * bucket_s).astype(int)
    ledger = ledger.assign(tte_bucket=bucket,
                          pnl_proxy_usd=ledger["delta_quality_c"] / 100.0
                          * ledger["shares"])
    grp = ledger.groupby("tte_bucket").agg(
        pnl_proxy=("pnl_proxy_usd", "sum"), n_fills=("shares", "size"))
    grp = grp.sort_index(ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(grp.index, grp["pnl_proxy"], width=bucket_s * 0.9,
               color="tab:blue")
    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel("markout PnL proxy, USD")
    axes[0].set_title("Markout-based PnL proxy and fills, by time-to-expiry "
                      f"bucket ({bucket_s:.0f}s)")

    axes[1].bar(grp.index, grp["n_fills"], width=bucket_s * 0.9,
               color="tab:gray")
    axes[1].set_ylabel("fill count")
    axes[1].set_xlabel("time to expiry at fill, s (bucket start)")
    axes[1].invert_xaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -- 5. calibration -----------------------------------------------------------

def plot_calibration(ledger, winner_up_by_market, path, n_buckets=10):
    l = ledger.copy()
    l["outcome"] = l["market_id"].map(winner_up_by_market).astype("float64")
    l = l.dropna(subset=["z", "outcome"])
    if not len(l):
        return

    resolved = resolve_slots(HERE, model_dir=MODEL)
    link_mod = load_slot(resolved["link"], "link")
    l["fair_p"] = l["z"].apply(link_mod.link)

    l["decile"] = pd.qcut(l["fair_p"], n_buckets, duplicates="drop")
    grp = l.groupby("decile", observed=True).agg(
        predicted=("fair_p", "mean"), realised=("outcome", "mean"),
        n=("outcome", "size"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")
    ax.scatter(grp["predicted"], grp["realised"], s=grp["n"] / grp["n"].max()
              * 200 + 20, color="tab:blue", zorder=5)
    for _, row in grp.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["predicted"], row["realised"]),
                   fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted fair_p at fill time (decile mean)")
    ax.set_ylabel("realised settlement frequency")
    ax.set_title("Calibration: fills, maker+taker pooled")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    with open(os.path.join(HERE, "last_run_manifest.json")) as fh:
        manifest = json.load(fh)

    headline = _load_run(manifest["run_dirs"]["headline"])
    detail = _load_run(manifest["run_dirs"]["detail"])

    out_dir = headline["run_dir"]
    print("plots ->", out_dir)

    plot_cumulative_pnl(headline["markets"],
                        os.path.join(out_dir, "cum_pnl_days.png"),
                        "Cumulative PnL, gross vs net, unified making "
                        "and taking")

    chosen = manifest["chosen_detail_markets"]
    # NOTE: deliberately NOT `markets=tuple(chosen)` here. That turns into a
    # pyarrow predicate pushed into the panel read, and on this panel file
    # (written by a newer parquet-cpp-arrow than the pinned pyarrow 19 can
    # parse -- see harness/io.py's module docstring) a PUSHED "in" filter
    # aborts the process natively instead of raising the catchable OSError
    # the unfiltered read hits, so `read_parquet`'s polars fallback never
    # gets a chance to run. Restricting by `days` first (post-read, same as
    # every other caller here) then filtering in Python, exactly like
    # `run.py`'s own `detail_episodes = [ep for ep in episodes if ...]`,
    # avoids the pushdown path entirely.
    spot_path = manifest.get("spot_path", paths.SPOT)
    # The spot line's currency is a property of the panel, not of the plot, and
    # on `SPOT_LONDON` it is BTC/USDT -- ~$50 clear of the oracle by
    # construction. Saying so on the axis is the difference between a reader
    # seeing the USDT premium and a reader seeing a broken model.
    spot_label = ("venue spot (BTC/USDT, UNCORRECTED)"
                  if os.path.abspath(spot_path) == os.path.abspath(
                      paths.SPOT_LONDON)
                  else "venue spot (BTC/USD)")
    warmup_s = float(manifest.get("warmup_s", 0.0))
    warmup = warmup_s > 0.0
    candidate_eps = load_episodes(spot_path=spot_path,
                                  days=manifest.get("rtds_covered_days"),
                                  rtds_path=paths.RTDS_BTC,
                                  warmup=warmup, warmup_s=warmup_s or 900.0)
    detail_eps = {ep.market_id: ep for ep in candidate_eps
                 if ep.market_id in chosen}
    for i, mkt in enumerate(chosen):
        ep = detail_eps.get(mkt)
        if ep is None:
            print(f"WARNING: could not load episode for chosen market {mkt}")
            continue
        ok = plot_market_detail(
            detail["ticks"], detail["ledger"], ep,
            os.path.join(out_dir, f"market_detail_{i}_{mkt}.png"),
            spot_label=spot_label)
        if not ok:
            print(f"WARNING: no ticks for chosen market {mkt}")

    combined_ledger = headline["ledger"]
    is_maker = combined_ledger["liquidity"] == int(Liquidity.MAKER)
    plot_markout(combined_ledger[is_maker], combined_ledger[~is_maker],
                os.path.join(out_dir, "markout_distribution.png"))

    plot_by_tte_bucket(combined_ledger, os.path.join(out_dir, "pnl_fills_by_tte.png"))

    # winner_up per market, from the same spot sample the runs used. No RTDS
    # and no warm-up here on purpose: this only needs `winner_up`, which comes
    # from the strikes, and loading either would cost minutes for nothing.
    episodes = load_episodes(spot_path=spot_path, days=manifest["spot_days"])
    winner_up_by_market = {ep.market_id: (1.0 if ep.winner_up else 0.0)
                           for ep in episodes if ep.winner_up is not None}
    plot_calibration(combined_ledger, winner_up_by_market,
                    os.path.join(out_dir, "calibration.png"))

    print("done")


if __name__ == "__main__":
    main()
