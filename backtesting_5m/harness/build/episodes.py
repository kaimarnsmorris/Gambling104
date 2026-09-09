"""Panel rows -> Episodes.

Settlement is not stored in this dataset. settle(N) is the strike of the
market opening at open_ts + 300, which is why open_ts is carried on the panel.
Verified exact on all 7,330 testable back-to-back pairs; winner_up is
settle >= strike, with ties going Up.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from harness import paths
from harness.core.episode import DEFAULT_WARMUP_S, build_episode
from harness.io import read_parquet
from harness.streams import RESERVED, resolve
from harness.streams.reader import grid_stream

# Converts an epoch-second window bound into a registered stream's own time
# unit. `grid_stream` requires an absolute epoch time column, but that column
# need not be nanoseconds -- assuming ns here would make a millisecond-stamped
# stream's window bounds three orders of magnitude too large and silently
# select nothing.
_UNITS_PER_S = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}


def load_chainlink(rtds_path=None):
    """Chainlink BTC/USD ticks as (receipt_ms, px), sorted by receipt.

    Keyed on `px_first_recv_ns` -- when the oracle price FIRST reached this
    vantage -- not on `oracle_ms`, which is the oracle's own stamp and runs
    ~1.5 s ahead of arrival. Using `oracle_ms` would hand a decision a price
    it had not yet been told, which is lookahead of exactly the size the
    basis model is trying to measure.
    """
    rtds = read_parquet(rtds_path or paths.RTDS_BTC,
                        columns=["px", "px_first_recv_ns"])
    rtds = rtds.dropna(subset=["px", "px_first_recv_ns"])
    recv_ms = rtds["px_first_recv_ns"].to_numpy(dtype="float64") / 1e6
    order = np.argsort(recv_ms, kind="stable")
    return recv_ms[order], rtds["px"].to_numpy(dtype="float64")[order]


def chainlink_window(recv_ms, px, open_ts):
    """The oracle ticks that arrived inside one market's 300 s window.

    Returned on the panel's own bucket convention (`t_ms` = the START of the
    100 ms bucket the observation fell in), so `build_episode` can grid and
    decision-shift it with the same code path as the book and the spot.
    """
    lo = float(open_ts) * 1000.0
    hi = lo + paths.H * 1000.0
    a, b = np.searchsorted(recv_ms, (lo, hi))
    if b <= a:
        return None
    t_ms = (((recv_ms[a:b] - lo) // paths.BUCKET_MS)
            * paths.BUCKET_MS).astype("int64")
    return pd.DataFrame({"t_ms": t_ms, "px": px[a:b]})


def chainlink_warmup(recv_ms, px, open_ts, warmup_s):
    """The oracle ticks that arrived in the `warmup_s` seconds BEFORE the open.

    Labelled `t_ms` from the START of the warm-up region, which is the same
    convention `chainlink_window` uses for the in-window region, so the frame
    goes through `build_episode`'s ordinary grid-and-shift path unchanged.
    The interval is half open on the right: an observation at exactly the open
    belongs to the market, not to its history.
    """
    if warmup_s <= 0:
        return None
    hi = float(open_ts) * 1000.0
    lo = hi - float(warmup_s) * 1000.0
    a, b = np.searchsorted(recv_ms, (lo, hi))
    if b <= a:
        return None
    t_ms = (((recv_ms[a:b] - lo) // paths.BUCKET_MS)
            * paths.BUCKET_MS).astype("int64")
    return pd.DataFrame({"t_ms": t_ms, "px": px[a:b]})


def spot_warmup(spot_by_open, open_ts, warmup_s, columns):
    """The spot rows covering the `warmup_s` seconds before `open_ts`.

    Assembled from the PRIOR markets' own rows -- the spot panel is stored
    per market, with `t_ms` measured from that market's open -- and relabelled
    onto a single warm-up grid whose origin is `open_ts - warmup_s`. Rows that
    fall outside the region are dropped here rather than by `_grid`, so the
    frame handed to `build_episode` means exactly what it says.

    Returns None when no prior market contributes a single row, which
    `build_episode` renders as an all-NaN region of the right length.
    """
    if warmup_s <= 0 or not spot_by_open:
        return None
    base = int(open_ts) - int(warmup_s)
    span = int(warmup_s) * 1000

    frames = []
    # Markets are back to back on a 300 s grid, so the region is covered by
    # the handful of opens at base, base+300, ... strictly below open_ts. A
    # market that started before `base` still contributes its tail, hence the
    # extra step backwards.
    first = base - (base % paths.H) - paths.H
    for prior in range(first, int(open_ts), paths.H):
        rows = spot_by_open.get(prior)
        if rows is None or not len(rows):
            continue
        t_ms = rows["t_ms"].to_numpy() + (prior - base) * 1000
        keep = (t_ms >= 0) & (t_ms < span)
        if not keep.any():
            continue
        frame = rows.loc[keep, columns[1:]].copy()
        frame.insert(0, "t_ms", t_ms[keep])
        frames.append(frame)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def settlement_map(strikes):
    """open_ts -> the NEXT market's strike, i.e. this market's settlement."""
    s = strikes.sort_values("open_ts")
    nxt = dict(zip(s["open_ts"] - paths.H, s["strike"]))
    return {int(ts): float(nxt[ts]) for ts in s["open_ts"] if ts in nxt}


def load_episodes(panel_path=None, strikes_path=None, spot_path=None,
                  fair_path=None, days=None, markets=None, max_markets=None,
                  fair_is_causal=False, rtds_path=None,
                  warmup=False, warmup_s=DEFAULT_WARMUP_S, streams=()):
    """Build Episodes from the panel, the strikes and (optionally) spot/fair.

    `fair_is_causal` is forwarded to `build_episode`: leave it False unless the
    fair export's timestamp contract has been confirmed in writing to be
    decision aligned. See harness/core/episode.py for why the default shifts.

    `rtds_path` attaches the Chainlink oracle line (`ep.chainlink`) that a
    basis-learning fair model needs. It is OPTIONAL and off by default,
    because the capture at `paths.RTDS_BTC` stops at 2026-08-21 01:59 UTC and
    so covers only the first two of this harness's six evaluation days --
    episodes outside it get an all-NaN `chainlink` and an infinite age, and a
    block that requires the oracle must decide for itself what to do about
    that rather than have this loader quietly invent a value.

    `warmup` turns on the pre-open history region described in
    harness/core/episode.py; `warmup_s` is its length, defaulting to
    DEFAULT_WARMUP_S = 900 s (three prior markets). Warm-up is OFF by default
    so that a caller who does not ask for it gets exactly the episodes this
    loader produced before the feature existed -- the in-window arrays are
    identical either way, and with `warmup=False` the warm-up arrays are
    empty rather than merely NaN.

    Warm-up is drawn from the WHOLE spot panel and the whole oracle capture,
    not from the `days`/`markets` selection. A day filter says which markets
    to trade, not what history existed before them, and an estimator that
    restarts on every day boundary is the bug this feature exists to fix.

    THE BOUNDARY. `has_warmup` is set per episode by comparing the region's
    start against the earliest market the spot panel carries (or, with no
    spot, the earliest market in this panel read, taken before the day
    filter). It says the region is COMPLETE. Episodes at the start of the
    sample get False -- and may still carry part of a region, where it
    straddles the first market -- but never a short array and never a padded
    one. Everything later gets True, including episodes whose
    `warmup_chainlink` is all NaN because the oracle capture had already
    ended; that is a feed outage, and a block that cannot tell the two apart
    would treat the start of the sample as an outage and the end of the
    oracle as history.

    Measured on 2026-08-17..18 off `paths.SPOT_ORACLE_WINDOW`: 508 of 562
    episodes get a complete region. Of the 54 that do not, 51 are markets
    that open before the venue capture starts at 04:19 UTC on 08-17 and have
    no spot of their own either; the remaining 3 are the first three markets
    after it.
    """
    panel_path = panel_path or paths.PANEL
    strikes_path = strikes_path or paths.STRIKES

    strikes = read_parquet(strikes_path)
    settle_by_open = settlement_map(strikes)

    filters = []
    if markets:
        filters.append(("market_id", "in", list(markets)))
    panel = read_parquet(panel_path,
                         filters=filters or None,
                         columns=["market_id", "open_ts", "t_ms",
                                  "bid", "ask", "mid", "n_src"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")
    # Taken BEFORE the day filter, so a run restricted to one day does not
    # decide that its first three markets have no history.
    panel_start = int(panel["open_ts"].min()) if len(panel) else None
    if days:
        panel = panel[panel["day"].isin(set(days))]

    warmup_s = float(warmup_s) if warmup else 0.0

    spot = None
    spot_cols = None
    spot_by_open = None
    if spot_path:
        spot = read_parquet(spot_path)
        spot_cols = ["t_ms", "spot"]
        if "spot_usdt" in spot.columns:
            spot_cols.append("spot_usdt")
        # Group ONCE. The previous per-market lookup below did a full boolean
        # scan of the spot frame for every market -- O(markets x rows) over a
        # 4.6 M-row frame. Group order is the frame's own row order, so the
        # frames handed to `build_episode` are the same rows in the same
        # order the boolean mask used to produce.
        spot_by_open = {int(k): v for k, v in
                        spot[spot_cols + ["open_ts"]].groupby(
                            "open_ts", sort=False)}
    cl_recv = cl_px = None
    if rtds_path:
        cl_recv, cl_px = load_chainlink(rtds_path)
    fair = None
    fair_by_market = None
    if fair_path:
        fair = read_parquet(fair_path)
        # Group once, mirroring the spot fix above -- including `sort=False`
        # for the same reason.
        fair_by_market = {k: v for k, v in
                          fair.groupby("market_id", sort=False)}

    # The earliest open for which any history could exist at all. Anything
    # whose warm-up region starts before this has no prior data, as opposed
    # to prior data that happens to be missing.
    if spot is not None and len(spot):
        history_start = int(spot["open_ts"].min())
    else:
        history_start = panel_start

    strike_by_id = dict(zip(strikes["market_id"], strikes["strike"]))

    stream_frames = {}
    for name in streams:
        reg = resolve(name)
        if reg.pre_gridded:
            # Already keyed on (open_ts, t_ms) and already on the decision
            # grid. Routing it through grid_stream would treat t_ms as an
            # absolute epoch and drop every row. Skip: load_episodes reads
            # these panels by its existing path.
            continue
        stream_frames[name] = (reg, read_parquet(reg.path))

    episodes = []
    for (market_id, open_ts), obs in panel.groupby(["market_id", "open_ts"],
                                                   sort=True):
        if market_id not in strike_by_id:
            continue
        ep_spot = None
        if spot_by_open is not None:
            rows = spot_by_open.get(int(open_ts))
            ep_spot = (rows[spot_cols] if rows is not None
                      else spot.iloc[0:0][spot_cols])
        ep_cl = None
        if cl_recv is not None:
            ep_cl = chainlink_window(cl_recv, cl_px, int(open_ts))

        wu_spot = wu_cl = None
        has_warmup = False
        if warmup_s > 0:
            has_warmup = (history_start is not None
                          and int(open_ts) - warmup_s >= history_start)
            if spot_by_open is not None:
                wu_spot = spot_warmup(spot_by_open, int(open_ts), warmup_s,
                                      spot_cols)
            if cl_recv is not None:
                wu_cl = chainlink_warmup(cl_recv, cl_px, int(open_ts),
                                         warmup_s)
        ep_s = None
        if fair is not None:
            rows = fair_by_market.get(market_id)
            ep_s = np.full(paths.N_BUCKET, np.nan)
            if rows is not None:
                k = (rows["t_ms"].to_numpy() // paths.BUCKET_MS).astype(
                    "int64")
                keep = (k >= 0) & (k < paths.N_BUCKET)
                ep_s[k[keep]] = rows["s"].to_numpy()[keep]

        ep = build_episode(
            market_id=market_id, open_ts=int(open_ts),
            day=obs["day"].iloc[0],
            strike=strike_by_id[market_id],
            settle=settle_by_open.get(int(open_ts)),
            obs=obs, spot=ep_spot, s=ep_s,
            fair_is_causal=fair_is_causal, chainlink=ep_cl,
            warmup_spot=wu_spot, warmup_chainlink=wu_cl,
            warmup_s=warmup_s, has_warmup=has_warmup)

        if stream_frames:
            window = {}
            for name, (reg, df) in stream_frames.items():
                # Bounds are converted using the STREAM'S OWN time unit --
                # not assumed to be nanoseconds -- so a millisecond-stamped
                # stream does not silently select nothing.
                to_unit = _UNITS_PER_S[reg.time_unit]
                lo = int(open_ts) * to_unit
                hi = (int(open_ts) + paths.H) * to_unit
                sub = df[(df[reg.time_col] >= lo) & (df[reg.time_col] < hi)]
                window[name] = grid_stream(
                    sub, int(open_ts), reg.values or
                    tuple(c for c in sub.columns if c not in RESERVED),
                    time_col=reg.time_col, time_unit=reg.time_unit,
                    causal=reg.causal)
            ep = replace(ep, streams=window)

        episodes.append(ep)

        if max_markets is not None and len(episodes) >= max_markets:
            break

    return episodes
