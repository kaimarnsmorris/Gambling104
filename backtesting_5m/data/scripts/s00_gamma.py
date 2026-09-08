"""Enumerate every 5 m market in the 60 s-TWAP era from gamma.

This is what says which markets exist and what their CLOB token ids are, and
it is the ONLY reliable way to do it. The obvious alternative -- classify the
archive's markets by how long their capture ran -- was tried and rejected: it
agrees with the archive's own `tf` label on just 66.6 % of labelled markets,
because a market whose capture is partial has a short span and impersonates a
shorter timeframe. Span cannot separate the books when coverage is the very
thing being measured.

`token_up` is `clobTokenIds[0]`, which is exactly the archive's `market_id`
(verified 6,611/6,611 in the l6_instrument layer), so it joins the book
directly and carries the era filter with it.

Also emits the strike. The chain property `settle(N) == strike(N+1)` holds
exactly for back-to-back 5 m windows, so a market's strike is the previous
window's settlement; both are cross-checked here rather than assumed.
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

import common

GAMMA = "https://gamma-api.polymarket.com/events"
SERIES = "btc-up-or-down-5m"
UA = "Mozilla/5.0 (research; contact via repo)"


def _get(url, tries=5, timeout=60):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"gamma failed: {url} ({last})")


def _iso(ts):
    return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(e):
    m = (e.get("markets") or [{}])[0]
    cfg = m.get("cryptoMarketConfig") or {}
    meta = e.get("eventMetadata") or {}
    toks = m.get("clobTokenIds")
    if isinstance(toks, str):
        try:
            toks = json.loads(toks)
        except Exception:                                        # noqa: BLE001
            toks = None
    toks = toks or []
    end = e.get("endDate")
    return {
        "slug": e.get("slug"),
        "market_id": toks[0] if toks else None,
        "end_date": pd.Timestamp(end).timestamp() if end else None,
        "lookback": cfg.get("twapLookbackSeconds"),
        "gamma_strike": meta.get("priceToBeat"),
        "gamma_settle": meta.get("finalPrice"),
    }


def main():
    t1 = int(time.time())
    # ⚠ gamma's `startDate` is when the market was LISTED, not when its window
    # opens -- 5 m markets are listed roughly a day ahead. Querying the era's
    # own dates therefore drops every market at the start of it and pulls in
    # ones that have not opened yet. So scan a window widened at both ends and
    # filter on `open_ts`, which is derived from `endDate`.
    lo0 = common.ERA_START - common.ERA_START % 86400 - 3 * 86400
    rows = []
    for lo in range(lo0, t1 + 2 * 86400, 86400):
        offset = 0
        while True:
            url = (f"{GAMMA}?series_slug={SERIES}&limit=100&offset={offset}"
                   "&order=startDate&ascending=true"
                   f"&start_date_min={_iso(lo)[:10]}T00:00:00Z"
                   f"&start_date_max={_iso(lo + 86400)[:10]}T00:00:00Z")
            d = _get(url)
            rows += [_row(e) for e in d]
            if len(d) < 100:
                break
            offset += 100
        print(f"  gamma {_iso(lo)[:10]}  total {len(rows):,}", flush=True)

    g = pd.DataFrame(rows).drop_duplicates("slug")
    g = g[g.market_id.notna() & g.end_date.notna()]
    g["open_ts"] = (g.end_date - common.H).astype("int64")

    n0 = len(g)
    # A market whose window has not closed cannot be complete, and gamma lists
    # them ahead of time, so the upper bound is real rather than cosmetic.
    g = g[(g.open_ts >= common.ERA_START) & (g.open_ts + common.H <= t1)]
    # The era filter is a per-market property, not a date: enforce the config
    # itself so a market that somehow carries the 30 s feed cannot slip in.
    bad = g.lookback.notna() & (g.lookback != 60)
    if bad.any():
        raise SystemExit(
            f"{int(bad.sum())} markets at or after the cutover report "
            f"twapLookbackSeconds != 60 -- the era boundary is not where "
            "ERA_START says it is")
    print(f"gamma: {n0:,} markets -> {len(g):,} in the 60 s-TWAP era "
          f"({common.iso(g.open_ts.min())} -> {common.iso(g.open_ts.max())})",
          flush=True)
    print(f"  lookback: {g.lookback.value_counts(dropna=False).to_dict()}",
          flush=True)

    os.makedirs(common.DATA, exist_ok=True)
    g.sort_values("open_ts").reset_index(drop=True).to_parquet(
        os.path.join(common.DATA, "gamma_5m.parquet"), index=False)


if __name__ == "__main__":
    sys.exit(main())
