"""s -- the expected settling TWAP, in USD.

The market does not settle on a price, it settles on a 60 s Chainlink TWAP.
So the quantity to forecast is an AVERAGE over [T-w, T], and this block is its
conditional expectation, E[A_T | F_t], which has two regimes:

    tau >= w   E[A] = S_t
               the whole average is still in the future and S is a martingale,
               so today's spot IS the forecast

    tau <  w   E[A] = (1 - tau/w) * TWAP_realised + (tau/w) * S_t
               the elapsed part of the window is no longer a forecast, it is a
               measurement, so it enters at full weight

An EWM lagged to "impersonate" the TWAP would be a biased estimator of the
first regime, which is 80 % of the window. The lag is not the answer; the
explicit average is.

WHAT S_t IS, AND WHAT IS SMOOTHED. There is NO smoother on the venue mid. A
price is a martingale: an EWM of it is a strictly lagged estimate of a
quantity whose best estimate is its current value, so it buys nothing and
pays for it in lag. An earlier version of this block ran a 2 s denoise here
on the grounds that "the bookTicker tick is not itself the Chainlink feed".
That premise is true and the instrument was wrong. The venue and the oracle
differ by a BASIS, not by noise, and a basis is the thing to smooth:

    S_t = M_t - B_t          M = raw venue mid, BTC/USDT
    B_t = ewm(M - C)         C = Chainlink BTC/USD oracle price

M moves at the speed of the market and is passed through untouched. B moves
at the speed of the USDT peg, which is hours, and is the only smoothed
quantity in this file. The fast thing stays fast; the slow thing gets the
averaging.

WHY THE BASIS IS LEARNED AND NOT READ OFF THE CAPTURE. The venue capture
quotes a `usdt_basis`, and `ep.spot` is already M minus that quote. This
block deliberately does NOT use it. That number explains ~87 % of the level
and leaves a residual that is itself neither zero nor constant -- +4.50 mean
over 2026-08-19..21, drifting +3.29 / +5.66 / +4.77 by day, by more than its
own standard error. Learning the whole basis against the settlement oracle
absorbs the quote and its residual together, and is answerable to the only
series that actually settles these markets. So the inputs here are `spot_usdt`
and `chainlink`, never `spot`.

THE HALFLIFE IS LONG ON PURPOSE. `M - C` is not the basis alone. It also
carries a transient: the oracle publishes at 1 Hz and reaches this vantage a
further ~1.5 s later, so every venue move shows up in the difference before
the oracle has caught up. That spike is oracle staleness, not basis. A short
halflife would absorb genuine price moves into B and drag S_t back toward a
stale oracle -- reintroducing, through the back door, exactly the lag that
removing the venue smoother was meant to eliminate. The two components live
on cleanly separated timescales (peg drift in hours, staleness in seconds),
so a long halflife separates them almost perfectly.

BASIS_HALFLIFE_S IS 180 s, AND THE WARM-UP IS WHAT PAYS FOR IT. It used to
be 60 s, and this docstring called that "the weakest number in this file":
`precompute` saw ONE market, so B restarted every 300 s and a peg-timescale
halflife could never converge inside a window. 60 s bought five halflives per
episode, and the price was the seed -- B started at a single (M - C)
observation whose sampling error is the raw gap's own sd, ~$16, decaying away
over the first minute, i.e. exactly while the market was opening.

`Episode` now carries a pre-open warm-up region (see
`harness/core/episode.py`), and this block burns the basis in across it
before index 0. With 900 s of warm-up a 180 s halflife is again five
halflives, so the seed's weight at the open is 2^-5 = 3 % rather than 100 %,
and the halflife is free to sit on the peg's timescale instead of on the
episode's. The two changes are one change: raising the halflife WITHOUT the
warm-up pays the convergence cost and buys nothing, and warming up a 60 s
filter burns in a number that was only ever short because it had to be.

The burn-in uses whatever history the region carries, including a partial one
(`ep.has_warmup` False, at the very start of the sample). An EWM seeded from
its own first observation degrades gracefully: less history is less burn-in,
not a wrong answer. A NaN bucket updates nothing, so an outage inside the
region costs exactly the observations it removed and no more.

BOTH SIDES OF THE GAP MUST BE FRESH, and this is the defect the warm-up
exposed. `M - C` is only a basis observation if M and C were observed at
about the same instant; measured a minute apart it is a price MOVE wearing a
basis's units. The oracle side was always guarded (`oracle_age == 0`), the
venue side never was -- the block sampled whatever mid the panel was carrying
forward. In-window that was nearly harmless, because the mid is fresh in
98.9 % of buckets and the exceptions are rare. Across the warm-up region it
is not: 50 of the 1,389 market slots on `SPOT_ORACLE_WINDOW` carry no venue
spot at all, and each one blanks the last 300 s of the warm-up of the three
markets that follow it. There the mid is carried for up to 295 s while the
oracle keeps printing, so every gap in that stretch is minus the BTC move
since the feed died -- measured on 2026-08-19 that dragged B to +136 where
the true basis was +44.

`MAX_MID_AGE_MS` closes it. The spot age distribution is bimodal -- fresh
inside 200 ms, or an outage of seconds to minutes; the fraction over 200 ms
(0.25 % in-window, 1.17 % in warm-up) and the fraction over 5 s agree to
three decimals -- so the threshold selects outages and nothing else, and any
value between them gives the same answer. 1 s is one oracle publication
interval: further apart than that and the two prices are not contemporaneous
in the only sense that matters here.

NO ORACLE, NO FAIR. Where `ep.chainlink` is all NaN the basis is unknowable
and this block returns NaN rather than falling back to the uncorrected mid --
which would be wrong by ~$43, about 0.17 of a 300 s sigma, roughly 7 c of
probability bias toward UP, some four times the taker fee. A missing forecast
is a market not traded; a silent $43 error is a manufactured edge. Note that
`harness.paths.RTDS_BTC` stops at 2026-08-21 01:59 UTC, which is why the
panel to run this on is `harness.paths.SPOT_ORACLE_WINDOW` (2026-08-17..21),
built over the overlap of the three feeds rather than across it.

TWO OPPOSITE READINGS OF THE SAME ARRAY, both deliberate:

  * S_t updates only where `spot_age_ms == 0`. The panel carries the last spot
    forward, and a repeat is not news; letting it update would drag the level
    at 10 Hz because nothing happened.
  * TWAP_realised averages EVERY bucket, carried values included. A flat mean
    over a uniform 100 ms grid IS the time-weighted average, and time passes
    whether or not a quote arrives. Skipping carried buckets here would
    silently reweight the average toward the busy moments.

`vol.py` carries its own copy of the window length, and its own scan. Blocks
are frozen into a run one file per slot, so a block that imports its neighbour
is a block whose frozen copy no longer means what it meant.
"""
import math

import numpy as np

from harness.paths import BUCKET_MS

TWAP_WINDOW_S = 60.0            # Chainlink's lookback, per the 2026-08-14 era
BASIS_HALFLIFE_S = 180.0        # on M - C only; never on the mid itself
MAX_MID_AGE_MS = 1000.0         # a gap needs a CONTEMPORANEOUS mid, not any mid
BUCKET_S = BUCKET_MS / 1000.0


def _scan(state, mid, age, oracle, oracle_age, offset,
          out_level=None, out_basis=None):
    """Walk one contiguous block of buckets, advancing (M_t, B_t).

    `offset` is the block's first bucket measured from the market open, so
    the warm-up region passes -warmup_n and the market itself passes 0. The
    EWM's alpha is drawn from that absolute clock, which is what lets the
    basis cross the open without noticing that it did.
    """
    level, basis, prev_basis = state
    for j in range(mid.shape[0]):
        px = float(mid[j])
        usable = np.isfinite(px) and px > 0.0

        if usable and age[j] == 0.0:
            level = px          # no smoother: the price passes straight through

        # Sample the gap only on a FRESH oracle print, AGAINST A FRESH MID.
        # Re-sampling a carried oracle value would weight one 1 Hz
        # observation ten times and pull B toward whatever the venue happened
        # to do while the oracle was silent; differencing a carried MID
        # against a live oracle is the same error with the feeds swapped, and
        # it is the one a blanked warm-up region makes minutes wide.
        cl = float(oracle[j])
        if (usable and age[j] <= MAX_MID_AGE_MS and oracle_age[j] == 0.0
                and np.isfinite(cl) and cl > 0.0):
            gap = px - cl
            k = offset + j
            if prev_basis is None or not np.isfinite(basis):
                basis = gap
            else:
                dt = (k - prev_basis) * BUCKET_S
                alpha = 1.0 - math.exp(
                    -dt * math.log(2.0) / BASIS_HALFLIFE_S)
                basis = alpha * gap + (1.0 - alpha) * basis
            prev_basis = k

        if out_level is not None:
            out_level[j] = level
            out_basis[j] = basis

    return level, basis, prev_basis


def basis_path(ep):
    """(M_t, B_t) at every in-window decision index, burnt in on the warm-up.

    Exposed rather than inlined so the warm-up can be MEASURED -- B_t at the
    open with and without a pre-open region, and the size of the step in B
    across a market boundary -- without a diagnostic having to reimplement
    the recursion and thereby measure a different filter than the one that
    priced the run.
    """
    state = (float("nan"), float("nan"), None)

    n_wu = ep.warmup_n
    if n_wu:
        state = _scan(
            state,
            np.asarray(ep.warmup_spot_usdt, dtype="float64"),
            np.asarray(ep.warmup_spot_age_ms, dtype="float64"),
            np.asarray(ep.warmup_chainlink, dtype="float64"),
            np.asarray(ep.warmup_chainlink_age_ms, dtype="float64"),
            -n_wu)

    n = len(ep)
    level = np.full(n, np.nan)
    basis = np.full(n, np.nan)
    _scan(state,
          np.asarray(ep.spot_usdt, dtype="float64"),
          np.asarray(ep.spot_age_ms, dtype="float64"),
          np.asarray(ep.chainlink, dtype="float64"),
          np.asarray(ep.chainlink_age_ms, dtype="float64"),
          0, level, basis)
    return level, basis


def precompute(ep):
    """E[settling TWAP] at each decision index. NaN before the first spot."""
    mid = np.asarray(ep.spot_usdt, dtype="float64")     # raw BTC/USDT
    age = np.asarray(ep.spot_age_ms, dtype="float64")
    level, basis = basis_path(ep)

    out = np.full(len(ep), np.nan)
    window_sum = 0.0
    window_n = 0

    for i in range(len(ep)):
        fair = level[i] - basis[i]   # NaN until both a mid and an oracle exist

        tau = ep.tte_s(i)
        if tau > TWAP_WINDOW_S:
            out[i] = fair
            continue

        px = float(mid[i])
        if np.isfinite(px) and px > 0.0 and np.isfinite(basis[i]):
            window_sum += px - basis[i]
            window_n += 1

        if window_n == 0 or not np.isfinite(fair):
            out[i] = fair
            continue

        frac = max(0.0, tau) / TWAP_WINDOW_S
        out[i] = (1.0 - frac) * (window_sum / window_n) + frac * fair

    return out
