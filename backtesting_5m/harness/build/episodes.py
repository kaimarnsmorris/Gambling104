"""Panel rows -> Episodes.

Settlement is not stored in this dataset. settle(N) is the strike of the
market opening at open_ts + 300, which is why open_ts is carried on the panel.
Verified exact on all 7,330 testable back-to-back pairs; winner_up is
settle >= strike, with ties going Up.
"""
import numpy as np
import pandas as pd

from harness import paths
from harness.core.episode import build_episode


def settlement_map(strikes):
    """open_ts -> the NEXT market's strike, i.e. this market's settlement."""
    s = strikes.sort_values("open_ts")
    nxt = dict(zip(s["open_ts"] - paths.H, s["strike"]))
    return {int(ts): float(nxt[ts]) for ts in s["open_ts"] if ts in nxt}


def load_episodes(panel_path=None, strikes_path=None, spot_path=None,
                  fair_path=None, days=None, markets=None, max_markets=None):
    panel_path = panel_path or paths.PANEL
    strikes_path = strikes_path or paths.STRIKES

    strikes = pd.read_parquet(strikes_path)
    settle_by_open = settlement_map(strikes)

    filters = []
    if markets:
        filters.append(("market_id", "in", list(markets)))
    panel = pd.read_parquet(panel_path,
                            filters=filters or None,
                            columns=["market_id", "open_ts", "t_ms",
                                     "bid", "ask", "mid", "n_src"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")
    if days:
        panel = panel[panel["day"].isin(set(days))]

    spot = None
    if spot_path:
        spot = pd.read_parquet(spot_path)
    fair = None
    if fair_path:
        fair = pd.read_parquet(fair_path)

    strike_by_id = dict(zip(strikes["market_id"], strikes["strike"]))

    episodes = []
    for (market_id, open_ts), obs in panel.groupby(["market_id", "open_ts"],
                                                   sort=True):
        if market_id not in strike_by_id:
            continue
        ep_spot = None
        if spot is not None:
            ep_spot = spot[spot["open_ts"] == open_ts][["t_ms", "spot"]]
        ep_s = None
        if fair is not None:
            rows = fair[fair["market_id"] == market_id]
            ep_s = np.full(paths.N_BUCKET, np.nan)
            k = (rows["t_ms"].to_numpy() // paths.BUCKET_MS).astype("int64")
            keep = (k >= 0) & (k < paths.N_BUCKET)
            ep_s[k[keep]] = rows["s"].to_numpy()[keep]

        episodes.append(build_episode(
            market_id=market_id, open_ts=int(open_ts),
            day=obs["day"].iloc[0],
            strike=strike_by_id[market_id],
            settle=settle_by_open.get(int(open_ts)),
            obs=obs, spot=ep_spot, s=ep_s))

        if max_markets is not None and len(episodes) >= max_markets:
            break

    return episodes
