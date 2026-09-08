"""One market, as the strategy is allowed to see it.

THE CAUSALITY RULE, enforced here and nowhere else:

    Decision index i carries exactly the information available at
    t_ms = i * 100, which is every observation from buckets k <= i - 1.

The panel labels a row by the START of the 100 ms bucket its observation was
drawn from, so that row was not knowable until the bucket closed. Everything
downstream indexes these arrays freely and therefore cannot reach forward,
because no index contains a future value.

That guarantee covers EVERY column, `s` included. The fair export is assumed
to label its rows the way the panel does -- a value at `t_ms` derived from
[t_ms, t_ms+100) -- so `s` is shifted onto the decision grid exactly like the
book and the spot. The asymmetry decides it: shifting an already-causal export
costs 100 ms of information, while failing to shift a panel-labelled one puts
a future observation in the alpha column and voids every result built on it.
A caller who has confirmed IN WRITING that the export is already decision
aligned may pass `fair_is_causal=True` to opt out of the shift.

Carrying the last quote is not the same as inventing one. A live trader knows
the last book they saw; what they do not know is whether it is still valid.
That is what `book_age_ms` is for, and why the engine refuses to trade against
a book older than `max_book_age_ms`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from harness.paths import BUCKET_MS, N_BUCKET


@dataclass(frozen=True)
class Episode:
    market_id: str
    open_ts: int
    day: str
    strike: float
    settle: float | None
    winner_up: bool | None

    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    book_age_ms: np.ndarray
    has_book: np.ndarray
    n_src: np.ndarray

    s: np.ndarray
    sigma: np.ndarray
    spot: np.ndarray
    spot_age_ms: np.ndarray
    has_spot: np.ndarray

    def __len__(self) -> int:
        return N_BUCKET

    def tte_s(self, i: int) -> float:
        return 300.0 - i * (BUCKET_MS / 1000.0)


def shift_to_decision_grid(values, present):
    """Carry observations forward onto the decision grid, one bucket late.

    Returns (carried, age_ms). `carried[i]` is the freshest observation from a
    bucket k <= i-1; `age_ms[i]` is how stale it was by the time index i could
    act on it, measured from the close of its own bucket. A bucket observed at
    k is age 0 at index k+1.
    """
    values = np.asarray(values, dtype="float64")
    present = np.asarray(present, dtype=bool)
    n = values.shape[0]

    src = np.where(present, np.arange(n), -1)
    src = np.maximum.accumulate(src)

    # shift by one bucket: index i may only see buckets <= i-1
    shifted = np.full(n, -1, dtype="int64")
    shifted[1:] = src[:-1]

    carried = np.full(n, np.nan)
    seen = shifted >= 0
    carried[seen] = values[shifted[seen]]

    age = np.full(n, np.inf)
    idx = np.arange(n)
    age[seen] = (idx[seen] - shifted[seen] - 1) * BUCKET_MS
    return carried, age


def _grid(obs: pd.DataFrame, column: str):
    values = np.full(N_BUCKET, np.nan)
    present = np.zeros(N_BUCKET, dtype=bool)
    if obs is not None and len(obs):
        k = (obs["t_ms"].to_numpy() // BUCKET_MS).astype("int64")
        keep = (k >= 0) & (k < N_BUCKET)
        values[k[keep]] = obs[column].to_numpy()[keep]
        present[k[keep]] = True
    return values, present


def build_episode(market_id, open_ts, day, strike, settle,
                  obs, spot=None, s=None, fair_is_causal=False):
    """Assemble one Episode. `obs` needs t_ms, bid, ask, mid, n_src.

    `s` is a full-length array indexed by bucket, as the fair export labels it.
    By DEFAULT it is shifted onto the decision grid like every other column,
    because the export is assumed to use the panel's convention (a value at
    t_ms drawn from [t_ms, t_ms+100), hence not knowable at t_ms).

    `fair_is_causal=True` asserts the opposite -- that the export is already
    decision aligned -- and uses `s` unshifted. Only set it once the export's
    timestamp contract has been confirmed in writing by whoever produces it.
    Setting it on a panel-labelled export manufactures lookahead in the alpha
    column, which no downstream gate can detect.
    """
    raw_bid, present = _grid(obs, "bid")
    raw_ask, _ = _grid(obs, "ask")
    raw_mid, _ = _grid(obs, "mid")
    raw_src, _ = _grid(obs, "n_src")

    bid, age = shift_to_decision_grid(raw_bid, present)
    ask, _ = shift_to_decision_grid(raw_ask, present)
    mid, _ = shift_to_decision_grid(raw_mid, present)
    n_src, _ = shift_to_decision_grid(raw_src, present)
    has_book = np.isfinite(bid) & np.isfinite(ask)

    if spot is not None and len(spot):
        raw_spot, spot_present = _grid(spot, "spot")
        spot_arr, spot_age = shift_to_decision_grid(raw_spot, spot_present)
    else:
        spot_arr = np.full(N_BUCKET, np.nan)
        spot_age = np.full(N_BUCKET, np.inf)
    has_spot = np.isfinite(spot_arr)

    if s is None:
        s_arr = np.full(N_BUCKET, np.nan)
    else:
        raw_s = np.asarray(s, dtype="float64")
        if fair_is_causal:
            s_arr = raw_s
        else:
            s_arr, _ = shift_to_decision_grid(raw_s, np.isfinite(raw_s))

    winner = None if settle is None else bool(settle >= strike)
    return Episode(
        market_id=market_id, open_ts=int(open_ts), day=day,
        strike=float(strike), settle=settle, winner_up=winner,
        bid=bid, ask=ask, mid=mid, book_age_ms=age, has_book=has_book,
        n_src=n_src,
        s=s_arr, sigma=np.full(N_BUCKET, np.nan),
        spot=spot_arr, spot_age_ms=spot_age, has_spot=has_spot,
    )
