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

THE WARM-UP REGION, and why it is a separate set of arrays.

An Episode covers exactly one market: 3,000 buckets, 300 seconds. A signal
block's `precompute` therefore used to see no history at all before the open,
so every stateful estimator restarted from nothing every 300 s. Two blocks
were measurably hurt by that: the vol block's EWMA of realised variance had
no burn-in and was still warming when the market was nearly over, and the
fair block's basis term `B_t = ewm(M - C)` had to run a 60 s halflife -- far
too short for a peg-timescale quantity -- purely so it could converge inside
one window.

The fix here is deliberately NOT carried state. Episodes stay independent:
that is what makes the run parallelisable and what makes per-market PnL an
i.i.d. sample for the day-blocked bootstrap. Instead each Episode carries its
OWN copy of the `warmup_s` seconds immediately before its open, on the same
100 ms grid, as `warmup_*` arrays.

Three properties hold, and the tests enforce all three:

  * The causality treatment is identical. Warm-up observations go through the
    same `shift_to_decision_grid`, on their own grid, so warm-up index j
    carries only warm-up buckets k <= j-1. Everything in the region was
    knowable strictly before the open; nothing in it is a licence to peek.
  * The in-window arrays are untouched. The warm-up region is shifted
    SEPARATELY, not concatenated and shifted as one, precisely so that
    `bid[0]`, `spot[0]`, `s[0]` and every other in-window value are bit for
    bit what they were before warm-up existed. The cost is the single bucket
    at the seam: the last 100 ms before the open is observed but never
    carried anywhere, because carrying it into in-window index 0 would change
    what the trading loop sees.
  * The trading loop never indexes warm-up. It exists for `precompute`.

`has_warmup` is a boundary flag, not a data-quality flag. It is True when the
WHOLE warm-up region lies inside the extent of the sample the loader read,
and False at the start of the sample, where part or all of the region falls
before any data exists. A False episode may still carry some observations --
the region can straddle the start -- so the flag says "this history is
complete", not "this history is empty".

That is what separates the two ways an array can be NaN. `has_warmup` False
means the sample does not reach back this far. `has_warmup` True with an
all-NaN array means the region is real and THAT FEED was down through it,
which is the normal state of `warmup_chainlink` past the end of the oracle
capture. A block that conflates them reads the start of the sample as an
outage and the end of the oracle as history.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from harness.paths import BUCKET_MS, N_BUCKET

#: How much history a signal block gets before the open, when warm-up is
#: requested. 900 s is three prior markets, and it is chosen against the
#: halflives that actually run on it, not for roundness:
#:
#:   * The fair block's basis term is `B_t = ewm(M - C)`, the venue-to-oracle
#:     gap. That gap moves on the USDT peg's timescale -- hours -- while the
#:     transient it has to reject (oracle publication plus ~1.5 s of transit)
#:     lives on seconds. Separating those wants a halflife of minutes. At a
#:     180 s halflife, 900 s is five halflives: the seed's weight is down to
#:     2^-5 = 3 %, so B reaches the open governed by the data and not by its
#:     first observation. The 60 s halflife its author was forced into, and
#:     the ~$16 seeding error he measured, both come off the table.
#:   * The vol block's EWMA of realised variance runs on tens of seconds. Even
#:     a 60 s halflife gets fifteen halflives of burn-in here, so it opens
#:     warm rather than warming up through the window it is being scored on.
#:
#: The cost is linear in the length: three extra markets of gridding and
#: 5 x 3,000 float64 per episode. Going further buys the basis little (a
#: sixth halflife moves the seed weight from 3 % to 1.6 %) and costs the same
#: again, so 900 is where the curve flattens.
DEFAULT_WARMUP_S = 900.0


def _nans(n=N_BUCKET):
    return np.full(n, np.nan)


def _infs(n=N_BUCKET):
    return np.full(n, np.inf)


def _empty():
    return np.zeros(0)


class StreamView:
    """One registered stream's decision-aligned arrays.

    Values are reached as attributes; `age_ms` and `has` always exist. A
    mistyped column raises immediately and names the real ones, so it fails at
    the first tick rather than surfacing as NaN deep in a sweep.
    """
    __slots__ = ("_name", "_cols")

    def __init__(self, name, cols):
        self._name = name
        self._cols = cols

    def __getattr__(self, item):
        try:
            return self._cols[item]
        except KeyError:
            raise AttributeError(
                f"stream {self._name!r} has no column {item!r}; "
                f"it has {tuple(sorted(self._cols))}") from None

    def __repr__(self):
        return f"StreamView({self._name!r}, {tuple(sorted(self._cols))})"


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

    #: The RAW venue mid, BTC/**USDT**, before any currency correction, and
    #: the Chainlink BTC/USD oracle price on the same decision grid.
    #:
    #: `spot` above is already USD -- the capture's `usdt_basis` subtracted
    #: row by row. These two exist for a fair model that would rather learn
    #: the basis from the settlement oracle itself than trust the capture's
    #: quoted USDT/USD rate. A model uses `spot`, or it uses this pair; using
    #: both double-counts the correction.
    #:
    #: `chainlink` is gridded by RECEIPT, not by `oracle_ms`: the feed is
    #: stamped with the oracle's own clock but lands ~1.5 s later (median
    #: 1.53 s, p25 1.31, p75 1.76 over the RTDS capture), and a decision at
    #: index i can only use what had arrived by then.
    #:
    #: Both default to all-NaN, so an Episode built without them behaves as
    #: it did before they existed.
    spot_usdt: np.ndarray = field(default_factory=lambda: _nans())
    chainlink: np.ndarray = field(default_factory=lambda: _nans())
    chainlink_age_ms: np.ndarray = field(default_factory=lambda: _infs())

    #: PRE-OPEN HISTORY. These cover the `warmup_s` seconds ENDING at the
    #: open, on the same 100 ms grid and under the same causality shift as
    #: the in-window arrays -- warm-up index j carries warm-up buckets
    #: k <= j-1. They are for `precompute` only. The trading loop must never
    #: index them, and `len(ep)` deliberately still reports 3,000 so that a
    #: loop written against `range(len(ep))` cannot reach into them.
    #:
    #: Length is `warmup_n`, which is 0 when no warm-up was requested. When
    #: warm-up WAS requested the arrays are always full length: an episode
    #: too early in the sample to have prior data gets NaN, never a truncated
    #: or padded array. `has_warmup` says whether the region is COMPLETE, and
    #: is what separates the start of the sample from a feed outage -- see
    #: the module docstring.
    warmup_s: float = 0.0
    has_warmup: bool = False
    warmup_spot: np.ndarray = field(default_factory=_empty)
    warmup_spot_age_ms: np.ndarray = field(default_factory=_empty)
    warmup_has_spot: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    warmup_spot_usdt: np.ndarray = field(default_factory=_empty)
    warmup_chainlink: np.ndarray = field(default_factory=_empty)
    warmup_chainlink_age_ms: np.ndarray = field(default_factory=_empty)

    #: Decision time of each index, milliseconds since the market open. On the
    #: 100 ms grid this is exactly i*100. It exists so blocks and the engine ask
    #: "what time is index i" rather than assuming buckets -- which is what
    #: makes event replay additive rather than a rewrite.
    t_ms: np.ndarray = field(
        default_factory=lambda: np.arange(N_BUCKET, dtype="int64") * BUCKET_MS)

    #: Registered streams: {name: {col: array, "age_ms": array, "has": array}}
    streams: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return N_BUCKET

    @property
    def warmup_n(self) -> int:
        """Buckets of pre-open history. 0 when no warm-up was requested."""
        return int(self.warmup_spot.shape[0])

    def warmup_tte_s(self, j: int) -> float:
        """Seconds from warm-up index `j` to THIS market's open.

        Positive and decreasing: the first warm-up index is `warmup_s` away,
        the last is one bucket away. Lets a block weight its history by age
        without having to know the grid's arithmetic.
        """
        return (self.warmup_n - j) * (BUCKET_MS / 1000.0)

    def tte_s(self, i: int) -> float:
        return 300.0 - self.t_ms[i] / 1000.0

    def stream(self, name):
        """The decision-aligned arrays of a registered stream."""
        if name not in self.streams:
            raise KeyError(
                f"{name!r} is not on this episode; it has "
                f"{tuple(sorted(self.streams))}")
        return StreamView(name, self.streams[name])


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


def _grid(obs: pd.DataFrame, column: str, n: int = N_BUCKET):
    """Scatter a `t_ms`-labelled frame onto an `n`-bucket grid.

    `n` is a parameter only so the warm-up region can reuse this exact code
    path. Warm-up frames label `t_ms` from the START of their own region, so
    the arithmetic below is unchanged.
    """
    values = np.full(n, np.nan)
    present = np.zeros(n, dtype=bool)
    if obs is not None and len(obs):
        k = (obs["t_ms"].to_numpy() // BUCKET_MS).astype("int64")
        keep = (k >= 0) & (k < n)
        values[k[keep]] = obs[column].to_numpy()[keep]
        present[k[keep]] = True
    return values, present


def _warmup_series(frame, column, n, present=None):
    """One warm-up column, gridded and decision-shifted like an in-window one.

    Returns (carried, age_ms), both length `n`. `present` lets a caller share
    one presence mask across columns of the same frame, exactly as the
    in-window spot does for `spot` and `spot_usdt`.
    """
    if n <= 0:
        return _empty(), _empty()
    if frame is None or not len(frame) or column not in frame:
        return _nans(n), _infs(n)
    raw, own_present = _grid(frame, column, n)
    return shift_to_decision_grid(
        raw, own_present if present is None else present)


def build_episode(market_id, open_ts, day, strike, settle,
                  obs, spot=None, s=None, fair_is_causal=False,
                  chainlink=None, warmup_spot=None, warmup_chainlink=None,
                  warmup_s=0.0, has_warmup=False):
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

    WARM-UP. `warmup_s` seconds of pre-open history, or 0.0 for none, which
    is the default and leaves this function byte-for-byte what it was.
    `warmup_spot` and `warmup_chainlink` are frames shaped exactly like
    `spot` and `chainlink`, except that `t_ms` counts from the START of the
    warm-up region -- i.e. from `open_ts - warmup_s` -- so they run through
    the same `_grid` and the same `shift_to_decision_grid` as everything
    else. Passing `warmup_s` without frames yields all-NaN warm-up arrays of
    the right length, which is the correct answer for an episode whose feeds
    were down before the open.

    `has_warmup` is the loader's assertion that the whole region lies inside
    the sample's extent. This function does not and cannot infer it: an
    all-NaN warm-up frame looks identical whether the data ran out or the
    feed did.
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
        if "spot_usdt" in spot:
            raw_usdt, _ = _grid(spot, "spot_usdt")
            usdt_arr, _ = shift_to_decision_grid(raw_usdt, spot_present)
        else:
            usdt_arr = _nans()
    else:
        spot_arr = np.full(N_BUCKET, np.nan)
        spot_age = np.full(N_BUCKET, np.inf)
        usdt_arr = _nans()
    has_spot = np.isfinite(spot_arr)

    # Gridded on `present` from its own frame, so a bucket with no oracle
    # update carries the last one forward with an honest age, exactly as the
    # book and the spot do.
    if chainlink is not None and len(chainlink):
        raw_cl, cl_present = _grid(chainlink, "px")
        cl_arr, cl_age = shift_to_decision_grid(raw_cl, cl_present)
    else:
        cl_arr, cl_age = _nans(), _infs()

    # --- the warm-up region -------------------------------------------
    # Shifted SEPARATELY from the in-window arrays, never concatenated with
    # them. Concatenating would let the last pre-open bucket carry into
    # in-window index 0 and change what the trading loop sees, which is the
    # one thing this feature must not do.
    warmup_n = int(round(float(warmup_s) * 1000.0 / BUCKET_MS))
    if warmup_n <= 0:
        warmup_n = 0
        wu_spot = wu_spot_age = wu_usdt = _empty()
        wu_cl = wu_cl_age = _empty()
        wu_has_spot = np.zeros(0, dtype=bool)
    else:
        wu_present = None
        if warmup_spot is not None and len(warmup_spot):
            _, wu_present = _grid(warmup_spot, "spot", warmup_n)
        wu_spot, wu_spot_age = _warmup_series(
            warmup_spot, "spot", warmup_n, wu_present)
        wu_usdt, _ = _warmup_series(
            warmup_spot, "spot_usdt", warmup_n, wu_present)
        wu_cl, wu_cl_age = _warmup_series(
            warmup_chainlink, "px", warmup_n)
        wu_has_spot = np.isfinite(wu_spot)

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
        spot_usdt=usdt_arr, chainlink=cl_arr, chainlink_age_ms=cl_age,
        warmup_s=float(warmup_s) if warmup_n else 0.0,
        has_warmup=bool(has_warmup) and warmup_n > 0,
        warmup_spot=wu_spot, warmup_spot_age_ms=wu_spot_age,
        warmup_has_spot=wu_has_spot, warmup_spot_usdt=wu_usdt,
        warmup_chainlink=wu_cl, warmup_chainlink_age_ms=wu_cl_age,
    )
