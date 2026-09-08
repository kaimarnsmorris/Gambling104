"""Merge the captures into the 100 ms panel, filter markets, and gate it.

One row per (market, 100 ms bucket) IN WHICH A CAPTURE ACTUALLY OBSERVED THE
BOOK. Nothing is carried forward, so a market's rows are not a fixed 3,000 --
a bucket nobody saw is absent, and `t_ms` says which.

⚠ BUCKET LABELLING IS A LOOKAHEAD TRAP. A row labelled `t_ms` holds an
observation drawn from the interval [t_ms, t_ms+100). It was therefore NOT
knowable at t_ms; the earliest a strategy could act on it is t_ms+100. The
first observation in the bucket is the one kept, so the row is as close to its
own label as the capture allows, but the rule still stands: shift by one
bucket before treating a row as a decision input.

Two filters, in this order:
  completeness  a market is dropped unless a capture saw its first 10 s AND
                its last 10 s AND at least 90 % of its 3,000 buckets. This is
                the "cut off half way" filter; every dropped market and its
                reason go to `results/dropped_markets.tsv`.
  strike        a market with no strike from any source is dropped, so the
                panel and `strikes_5m.parquet` cover exactly the same markets.
"""
import os
import sys

import numpy as np
import pandas as pd

import common

#: Two captures of one book agree to a cent or they are not the same book --
#: but only when the book is STILL. Both captures are 10 Hz samplers, so while
#: the book moves they legitimately catch it one tick apart: measured, the
#: median difference when they disagree is exactly 1.0 c, and the rate scales
#: with movement (5.6 % static, 46.5 % when the book moved 3+ ticks). So a
#: tick of disagreement is physics, and the gate is set on the thing that
#: is NOT: a gap over 5 c means one capture has the wrong book.
AGREE_C = 0.01
BIG_C = 0.05
MIN_AGREE = 0.85
MAX_BIG = 0.02

#: A book contradicted by BOTH its neighbours by this much is not a book. The
#: 5 m binary legitimately sits at 0 or 1 near expiry -- 94 % of rows in the
#: last 10 s are at a bound -- so being extreme is normal and is NOT the test.
#: Being extreme for a single 100 ms bucket while the buckets either side are
#: 20 c away is the roll artefact: the previous market's resolved book
#: surviving the attribution trim. Measured at 88 rows, 85 of them a market's
#: first bucket.
SPIKE_C = 0.20


def load(gamma, off):
    """Every capture's quotes, on ONE vantage, bucketed, ids reduced to codes.

    The id is a 77-character CLOB token; at ~48 M rows the strings alone are
    several GB, so they become codes on the way in and come back at the end.

    `ts` is corrected to polydata's vantage before bucketing -- see
    `s03b_vantage.py`. Without it the same book state carries two different
    times and the buckets do not line up.
    """
    codes = {m: i for i, m in enumerate(gamma.market_id)}
    open_of = gamma.set_index("market_id").open_ts

    frames = []
    for src in common.PRIORITY:
        p = common.QUOTES[src]
        if not os.path.exists(p):
            print(f"  {src}: no quotes file, skipped", flush=True)
            continue
        d = pd.read_parquet(p)
        d = d[d.market_id.isin(codes)]
        ts = d.ts.values.copy()
        if src in off:
            day = (ts // 86400).astype("int64")
            shift = pd.Series(day).map(off[src]).values / 1000.0
            miss = np.isnan(shift)
            if miss.any():
                raise SystemExit(f"{src}: {int(miss.sum()):,} quotes on days "
                                 "with no vantage offset -- run s03b first")
            ts = ts - shift
        o = d.market_id.map(open_of).values
        t_ms = np.floor((ts - o) * 1000.0 / common.BUCKET_MS)
        keep = (t_ms >= 0) & (t_ms < common.N_BUCKET)
        if src != "archive":
            keep &= (t_ms * common.BUCKET_MS) >= common.TDB_TRIM_MS
        f = pd.DataFrame({
            "mkt": d.market_id.map(codes).values[keep].astype("int32"),
            "t_ms": (t_ms[keep] * common.BUCKET_MS).astype("int32"),
            "ts": ts[keep],
            "bid": d.bid.values[keep].astype("float32"),
            "ask": d.ask.values[keep].astype("float32"),
            "src": src,
        })
        # One row per (market, bucket, capture): the FIRST observation in the
        # bucket, i.e. the one closest to the label the row will carry.
        f = f.sort_values(["mkt", "t_ms", "ts"], kind="stable")
        f = f.drop_duplicates(["mkt", "t_ms"], keep="first")
        frames.append(f)
        print(f"  {src}: {len(f):,} bucketed quotes", flush=True)
    return frames, codes


def gate_agreement(frames):
    """Do two independent hosts see the same book? And is either inverted?

    An UP/DOWN token mix-up would NOT look like disagreement in a mid -- it
    looks like `bid ≈ 1 - ask`. So both are measured, and the inverted match
    rate must be the low one.
    """
    by = {f.src.iloc[0]: f.set_index(["mkt", "t_ms"]) for f in frames}
    if "archive" not in by:
        return pd.DataFrame()
    a = by["archive"]
    rows = []
    for src, o in by.items():
        if src == "archive":
            continue
        j = a[["bid", "ask"]].join(o[["bid", "ask"]], how="inner",
                                   lsuffix="_a", rsuffix="_o")
        if not len(j):
            continue
        direct = ((j.bid_a - j.bid_o).abs() <= AGREE_C) & \
                 ((j.ask_a - j.ask_o).abs() <= AGREE_C)
        inverted = ((j.bid_a - (1 - j.ask_o)).abs() <= AGREE_C) & \
                   ((j.ask_a - (1 - j.bid_o)).abs() <= AGREE_C)
        big = (((j.bid_a - j.bid_o).abs() > BIG_C)
               | ((j.ask_a - j.ask_o).abs() > BIG_C))
        rows.append({"src": src, "shared_buckets": len(j),
                     "agree_direct": round(float(direct.mean()), 4),
                     "agree_inverted": round(float(inverted.mean()), 4),
                     "disagree_over_5c": round(float(big.mean()), 4)})
    return pd.DataFrame(rows)


def gate_attribution(frames):
    """Disagreement by position in the window.

    `ticks` carries no market id, so a trading.db quote is attributed to the
    window containing it. If the v1 quoter lingered on the old market after a
    roll, that shows up as a spike of disagreement in the first buckets --
    which is exactly what this measures, rather than assuming either way.
    """
    by = {f.src.iloc[0]: f.set_index(["mkt", "t_ms"]) for f in frames}
    if "archive" not in by:
        return pd.DataFrame()
    a = by["archive"]
    rows = []
    for src, o in by.items():
        if src == "archive":
            continue
        j = a[["bid"]].join(o[["bid"]], how="inner", lsuffix="_a", rsuffix="_o")
        if not len(j):
            continue
        t = j.index.get_level_values("t_ms").values
        bad = (j.bid_a - j.bid_o).abs() > AGREE_C
        for lo, hi, lab in [(0, 1000, "0-1s"), (1000, 5000, "1-5s"),
                            (5000, 290000, "5-290s"),
                            (290000, 300000, "290-300s")]:
            m = (t >= lo) & (t < hi)
            if m.any():
                rows.append({"src": src, "window": lab, "n": int(m.sum()),
                             "disagree": round(float(bad.values[m].mean()), 4)})
    return pd.DataFrame(rows)


def assemble(frames):
    """Best capture wins each bucket; corroboration counted per capture."""
    all_ = pd.concat(frames, ignore_index=True)
    all_["rank"] = all_.src.map({s: i for i, s in enumerate(common.PRIORITY)})
    all_ = all_.sort_values(["mkt", "t_ms", "rank"], kind="stable")

    best = all_.drop_duplicates(["mkt", "t_ms"], keep="first")
    key = ["mkt", "t_ms"]
    j = all_.merge(best[key + ["bid", "ask"]], on=key, suffixes=("", "_w"))
    agree = ((j.bid - j.bid_w).abs() <= AGREE_C) & \
            ((j.ask - j.ask_w).abs() <= AGREE_C)
    n_src = j[agree].groupby(key).src.nunique().rename("n_src")
    g = j.groupby(key)
    spread = ((g.bid.max() - g.bid.min()) * 100).where(g.size() > 1)
    return best.set_index(key).join(n_src).join(spread.rename("src_spread_c"))


def drop_spikes(panel):
    """Remove single-bucket books both neighbours contradict."""
    p = panel.reset_index()
    g = p.groupby("mkt").bid
    prv, nxt = g.shift(1), g.shift(-1)
    # A first or last bucket has one neighbour; one contradiction is enough
    # there, because there is nothing else to corroborate it.
    bad_prev = prv.isna() | (prv - p.bid).abs().gt(SPIKE_C)
    bad_next = nxt.isna() | (nxt - p.bid).abs().gt(SPIKE_C)
    spike = bad_prev & bad_next & ~(prv.isna() & nxt.isna())
    n = int(spike.sum())
    print(f"  dropping {n:,} single-bucket spikes "
          f"({n/max(len(p),1):.5%})", flush=True)
    return panel[~spike.values]


def select(panel, strikes_ids, codes):
    """Apply the completeness and strike filters, and say what went and why."""
    inv = {v: k for k, v in codes.items()}
    t = panel.index.get_level_values("t_ms").values
    mk = panel.index.get_level_values("mkt").values
    df = pd.DataFrame({"mkt": mk, "t": t})
    st = df.groupby("mkt").agg(
        n=("t", "size"),
        has_open=("t", lambda s: bool((s < common.END_S * 1000).any())),
        has_close=("t", lambda s: bool((s >= (common.H - common.END_S) * 1000).any())),
    )
    st["coverage"] = st.n / common.N_BUCKET
    st["market_id"] = [inv[i] for i in st.index]
    st["has_strike"] = st.market_id.isin(strikes_ids)
    st["keep"] = (st.has_open & st.has_close
                  & (st.coverage >= common.MIN_COVERAGE) & st.has_strike)
    st["reason"] = np.where(
        st.keep, "",
        np.where(~st.has_strike, "no_strike",
                 np.where(~st.has_open, "no_open_10s",
                          np.where(~st.has_close, "no_close_10s",
                                   "coverage_below_90pct"))))
    return st


def main():
    gamma = pd.read_parquet(os.path.join(common.DATA, "gamma_5m.parquet"))
    strikes = pd.read_parquet(common.STRIKES)
    vp = os.path.join(common.DATA, "vantage_offsets.parquet")
    if not os.path.exists(vp):
        raise SystemExit("run s03b_vantage.py first -- the captures are on "
                         "different vantages and their buckets will not align")
    v = pd.read_parquet(vp)
    off = {s: g.set_index("day").offset_ms for s, g in v.groupby("src")}
    frames, codes = load(gamma, off)

    agree = gate_agreement(frames)
    attr = gate_attribution(frames)
    panel = drop_spikes(assemble(frames))
    st = select(panel, set(strikes.market_id), codes)

    print("\n-- gate 1: cross-capture agreement, and token orientation")
    print(agree.to_string(index=False))
    print("\n-- gate 2: trading.db attribution by position in the window")
    print(attr.to_string(index=False))

    keep = set(st.index[st.keep])
    out = panel[panel.index.get_level_values("mkt").isin(keep)].reset_index()
    inv = {v: k for k, v in codes.items()}
    open_of = gamma.set_index("market_id").open_ts

    out["market_id"] = [inv[i] for i in out.mkt.values]
    out["open_ts"] = out.market_id.map(open_of).astype("int64")
    out["mid"] = ((out.bid + out.ask) / 2).astype("float32")
    # Integer milliseconds, not float seconds: the captures are 10 Hz samplers
    # with millisecond stamps, so ms is the real resolution and an int says so
    # without inviting a float-precision question.
    out["recv_ms"] = np.round(out.ts.values * 1000.0).astype("int64")
    out["n_src"] = out.n_src.astype("int32")
    out = out[["market_id", "open_ts", "t_ms", "recv_ms", "bid", "ask", "mid",
               "src", "n_src", "src_spread_c"]]
    out = out.sort_values(["open_ts", "t_ms"], kind="stable").reset_index(drop=True)

    # -- gates that must hold on what is actually written
    dup = int(out.duplicated(["market_id", "t_ms"]).sum())
    bad_t = int(((out.t_ms < 0) | (out.t_ms >= common.H * 1000)
                 | (out.t_ms % common.BUCKET_MS != 0)).sum())
    era = int((out.open_ts < common.ERA_START).sum())
    nostrike = int((~out.market_id.isin(set(strikes.market_id))).sum())

    print(f"\n-- gate 3: duplicate (market, bucket) {dup}; bad t_ms {bad_t}")
    print(f"-- gate 4: rows before the 60s-TWAP cutover {era}; "
          f"rows with no strike {nostrike}")

    fail = []
    if not agree.empty:
        if (agree.agree_direct < MIN_AGREE).any():
            fail.append("cross-capture agreement below floor")
        if (agree.disagree_over_5c > MAX_BIG).any():
            fail.append(f"a capture disagrees by over {BIG_C*100:.0f} c on more "
                        f"than {MAX_BIG:.0%} of shared buckets -- that is a "
                        "wrong book, not sampling jitter")
        if (agree.agree_inverted > agree.agree_direct).any():
            fail.append("a capture matches INVERTED better than direct -- "
                        "UP/DOWN token orientation is wrong")
    for n, lab in ((dup, "duplicate (market, bucket) rows"),
                   (bad_t, "t_ms off the 100 ms grid or out of range"),
                   (era, "rows before the cutover"),
                   (nostrike, "rows with no strike")):
        if n:
            fail.append(f"{n:,} {lab}")
    if fail:
        raise SystemExit("REFUSING TO WRITE: " + "; ".join(fail))

    os.makedirs(common.RESULTS, exist_ok=True)
    agree.to_csv(os.path.join(common.RESULTS, "gate_agreement.tsv"),
                 sep="\t", index=False)
    attr.to_csv(os.path.join(common.RESULTS, "gate_attribution.tsv"),
                sep="\t", index=False)
    st[~st.keep][["market_id", "n", "coverage", "reason"]].to_csv(
        os.path.join(common.RESULTS, "dropped_markets.tsv"), sep="\t",
        index=False)
    out.to_parquet(common.PANEL, index=False)

    kept = st.keep.sum()
    summary = pd.DataFrame([{
        "markets_listed": len(gamma),
        "markets_kept": int(kept),
        "markets_dropped": int((~st.keep).sum()),
        "rows": len(out),
        "median_rows_per_market": int(out.groupby("market_id").size().median()),
        "median_coverage": round(float(st.coverage[st.keep].median()), 4),
        "buckets_2plus_captures": int((out.n_src >= 2).sum()),
        "first_open": common.iso(out.open_ts.min()),
        "last_open": common.iso(out.open_ts.max()),
    }])
    summary.to_csv(os.path.join(common.RESULTS, "summary.tsv"), sep="\t",
                   index=False)
    print("\n" + summary.T.to_string(header=False))
    print(f"\nwrote {common.PANEL}", flush=True)
    print(st[~st.keep].reason.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
