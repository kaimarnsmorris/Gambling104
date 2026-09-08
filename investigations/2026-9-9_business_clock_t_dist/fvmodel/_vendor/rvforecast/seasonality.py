"""Step 2 - seasonality: a deterministic multiplicative factor s(t) on the variance rate.

    log E[r_t^2] = c + f_NY(tau_NY) + g_UTC(tau_UTC) + sum_j delta_j K_j(t) + log sigma~_t^2

f_NY is a weekly Fourier series in New-York local time-of-week (DST aware), g_UTC a daily
Fourier series in UTC time-of-day, and K_j are narrow Gaussian spike kernels at scheduled
events. The two clocks are fitted *jointly*: they are only separated by the fact that they
drift an hour against each other at the March/November DST boundaries, so a sequential fit
(NY first, then UTC on the residual) smears both. Ablation 1 in the report quantifies that.

The smooth parts are stored as evaluated tables (10080 NY minutes-of-week, 1440 UTC
minutes-of-day) and interpolated linearly to 1s; spikes are evaluated analytically, so the
effective resolution is 1 second where it matters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from . import tzclock as T

SEC_WEEK = 604800
SEC_DAY = 86400
MIN_WEEK = 10080
MIN_DAY = 1440

# E[log chi^2_1] = psi(1/2) + log 2 = -1.2703628...
LOG_CHI2_1_BIAS = -1.2703628454614782


# --------------------------------------------------------------------------- bases
def fourier(x: np.ndarray, period: float, order: int) -> np.ndarray:
    """[cos(2pi j x/P), sin(2pi j x/P)] for j = 1..order -> (n, 2*order)."""
    w = 2.0 * np.pi * np.asarray(x, dtype=np.float64)[:, None] / period
    j = np.arange(1, order + 1, dtype=np.float64)[None, :]
    a = w * j
    return np.concatenate([np.cos(a), np.sin(a)], axis=1)


@dataclass
class BasisSpec:
    ny_order: int = 96
    utc_order: int = 36
    weekend_order: int = 6
    lam_ny: float = 1e-3
    lam_utc: float = 1e-3
    lam_weekend: float = 1e-3

    @property
    def n_ny(self) -> int:
        return 2 * self.ny_order

    @property
    def n_utc(self) -> int:
        return 2 * self.utc_order

    @property
    def n_we(self) -> int:
        return 2 * self.weekend_order + 1

    @property
    def n_par(self) -> int:
        return 1 + self.n_ny + self.n_utc + self.n_we


def design(ny_tow_s: np.ndarray, utc_tod_s: np.ndarray, spec: BasisSpec) -> np.ndarray:
    """Stacked design matrix [1 | NY weekly Fourier | UTC daily Fourier | weekend block]."""
    n = ny_tow_s.size
    blocks = [np.ones((n, 1))]
    blocks.append(fourier(ny_tow_s, SEC_WEEK, spec.ny_order))
    blocks.append(fourier(utc_tod_s, SEC_DAY, spec.utc_order))
    # weekend block: indicator (NY Sat/Sun) times a low-order daily Fourier, so the
    # weekend can have its own level and shape without forcing high weekly orders.
    we = (ny_tow_s >= 5 * SEC_DAY).astype(np.float64)
    tod_ny = ny_tow_s % SEC_DAY
    blocks.append(we[:, None])
    blocks.append(we[:, None] * fourier(tod_ny, SEC_DAY, spec.weekend_order))
    return np.concatenate(blocks, axis=1)


def penalty(spec: BasisSpec) -> np.ndarray:
    """Diagonal ridge penalty; grows like j^2 so high harmonics are shrunk hardest."""
    p = np.zeros(spec.n_par)
    o = 1
    j = np.arange(1, spec.ny_order + 1, dtype=np.float64)
    p[o:o + spec.n_ny] = spec.lam_ny * np.concatenate([j ** 2, j ** 2])
    o += spec.n_ny
    j = np.arange(1, spec.utc_order + 1, dtype=np.float64)
    p[o:o + spec.n_utc] = spec.lam_utc * np.concatenate([j ** 2, j ** 2])
    o += spec.n_utc
    p[o] = spec.lam_weekend
    j = np.arange(1, spec.weekend_order + 1, dtype=np.float64)
    p[o + 1:o + spec.n_we] = spec.lam_weekend * np.concatenate([j ** 2, j ** 2])
    return p


def ridge_solve(X: np.ndarray, y: np.ndarray, w: np.ndarray, pen: np.ndarray):
    Xw = X * w[:, None]
    A = X.T @ Xw
    A.flat[:: A.shape[0] + 1] += pen * w.sum() / max(X.shape[0], 1)
    b = Xw.T @ y
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------- cell stats
def trimmed_mean(a: np.ndarray, frac: float = 0.1) -> float:
    if a.size == 0:
        return np.nan
    a = np.sort(a)
    k = int(np.floor(a.size * frac))
    b = a[k:a.size - k] if a.size - 2 * k > 0 else a
    return float(b.mean())


@dataclass
class CellStats:
    """log-variance-rate statistics binned by (NY minute-of-week, UTC minute-of-day)."""
    ny_min: np.ndarray
    utc_min: np.ndarray
    count: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    trimmed: np.ndarray
    month: np.ndarray | None = None  # per-(cell, month) when built for CV


def build_cells(ts_s: np.ndarray, y: np.ndarray, month: np.ndarray | None = None,
                estimators=("mean", "median", "trimmed")) -> CellStats:
    """Bin y by (NY minute-of-week, UTC minute-of-day) [, month]."""
    nym = (T.ny_tow(ts_s) // 60).astype(np.int64)
    utm = (T.utc_tod(ts_s) // 60).astype(np.int64)
    if month is None:
        key = nym * MIN_DAY + utm
    else:
        key = (nym * MIN_DAY + utm) * 16 + month
    order = np.argsort(key, kind="stable")
    key_s, y_s = key[order], y[order]
    edges = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [key_s.size]])
    uk = key_s[starts]
    cnt = (ends - starts).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(y_s)])
    mean = (csum[ends] - csum[starts]) / cnt
    med = np.empty_like(mean)
    trm = np.empty_like(mean)
    if "median" in estimators or "trimmed" in estimators:
        for i, (a, b) in enumerate(zip(starts, ends)):
            seg = np.sort(y_s[a:b])
            med[i] = seg[seg.size // 2] if seg.size % 2 else 0.5 * (
                seg[seg.size // 2 - 1] + seg[seg.size // 2])
            k = int(np.floor(seg.size * 0.10))
            trm[i] = seg[k:seg.size - k].mean() if seg.size - 2 * k > 0 else seg.mean()
    else:
        med[:] = np.nan
        trm[:] = np.nan
    if month is None:
        nyi, uti, mo = uk // MIN_DAY, uk % MIN_DAY, None
    else:
        mo = (uk % 16).astype(np.int64)
        k2 = uk // 16
        nyi, uti = k2 // MIN_DAY, k2 % MIN_DAY
    return CellStats(nyi, uti, cnt, mean, med, trm, mo)


# ------------------------------------------------------------------------- spikes
@dataclass
class Spike:
    """A signed two-lobe Gaussian event kernel on log variance.

    A scheduled release is not a bump, it is a *step*. Liquidity providers pull quotes
    ahead of a known print, trading thins out and 1-second realised variance falls
    **below** baseline; the number lands; variance jumps and then decays over minutes.
    Measured model-free against each occurrence's own +/-20-30 minute reference band,
    eight of the nine candidate events on this sample show that lull, with held-out
    t-statistics between -3.6 and -8.5.

    So the kernel carries an independent amplitude on each side of the peak:

        K(d) = log_mult_pre  * exp(-0.5 (d/rise_s)^2)    for d < 0
             = log_mult      * exp(-0.5 (d/width_s)^2)   for d >= 0

    with d measured from `center_s`. `log_mult_pre` is normally negative (the lull) and
    `log_mult` positive (the release). The deliberate discontinuity at d = 0 is the
    release itself. Setting `log_mult_pre = log_mult` and `rise_s = width_s` recovers a
    plain symmetric Gaussian, and `log_mult_pre = 0` recovers the earlier one-sided
    kernel, so older parameter files keep working.
    """
    name: str
    clock: str            # "NY" or "UTC"
    weekday_mask: list    # 7 bools, Mon..Sun in that clock
    local_time_s: int     # seconds since local midnight of the nominal event
    center_s: float       # fitted offset of the peak from the nominal time
    width_s: float        # Gaussian sigma AFTER the peak (the decay)
    log_mult: float       # log-variance multiplier just AFTER the event
    tstat: float = 0.0    # held-out t on the post lobe
    keep: bool = True
    rise_s: float = -1.0  # Gaussian sigma BEFORE the peak; <0 means "same as width_s"
    log_mult_pre: float = 0.0   # log-variance multiplier just BEFORE the event
    tstat_pre: float = 0.0      # held-out t on the pre lobe

    def __post_init__(self):
        # Backward compatibility, made explicit rather than implicit. A kernel written
        # before the two-lobe change has rise_s < 0 and no log_mult_pre, and meant a
        # plain symmetric Gaussian - so mirror the amplitude. A kernel from the current
        # fitter always carries rise_s > 0, and there log_mult_pre == 0 genuinely means
        # "this lobe did not clear its significance bar", so it must stay zero.
        if self.rise_s is not None and self.rise_s < 0 and self.log_mult_pre == 0.0:
            self.log_mult_pre = self.log_mult


def _spike_shape(delta: np.ndarray, center: float, width: float,
                 rise: float = -1.0):
    """(gaussian shape, before-mask) for a two-piece kernel centred at `center`."""
    d = np.asarray(delta, dtype=np.float64) - center
    w_up = max(rise, 1e-6) if rise is not None and rise > 0 else max(width, 1e-6)
    z = d / np.where(d < 0.0, w_up, max(width, 1e-6))
    return np.exp(-0.5 * z * z), d < 0.0


def _spike_kernel(delta: np.ndarray, center: float, width: float,
                  rise: float = -1.0) -> np.ndarray:
    """Unsigned shape only - kept for the symmetric/one-sided legacy path."""
    return _spike_shape(delta, center, width, rise)[0]


# ------------------------------------------------------------------ dated events
SEC_HOUR = 3600
SEC_5MIN = 300
SEC_MIN = 60


@dataclass
class DatedSpike:
    """A signed two-lobe kernel attached to an explicit list of UTC timestamps.

    The weekly/daily kernels in `Spike` recur on a clock. FOMC decisions, CPI and
    payroll prints and the monthly option expiry do not: they sit on particular dates.
    The kernel shape is the same signed two-lobe Gaussian (a lull before a scheduled
    release, a jump after it), but the occurrence list is a table of epoch seconds and
    the evaluation is a nearest-occurrence lookup.

    `occurrences` holds every date in the fitted calendar; `n_train` records how many of
    them fell inside the fitting window, which is the number the amplitude was actually
    estimated from.
    """
    name: str
    occurrences: list                 # UTC epoch seconds, ascending
    center_s: float = 0.0
    width_s: float = 300.0            # sigma AFTER the peak
    rise_s: float = 120.0             # sigma BEFORE the peak
    log_mult: float = 0.0             # amplitude after the peak
    log_mult_pre: float = 0.0         # amplitude before the peak (normally negative)
    tstat: float = 0.0
    tstat_pre: float = 0.0
    keep: bool = True
    n_train: int = 0
    note: str = ""

    def as_array(self) -> np.ndarray:
        return np.asarray(self.occurrences, dtype=np.int64)


def _nearest_delta(ts_s: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Signed seconds from the nearest occurrence (positive = after the event)."""
    if occ.size == 0:
        return np.full(ts_s.shape, 1e18)
    pos = np.searchsorted(occ, ts_s)
    lo = occ[np.clip(pos - 1, 0, occ.size - 1)]
    hi = occ[np.clip(pos, 0, occ.size - 1)]
    dlo = (ts_s - lo).astype(np.float64)
    dhi = (ts_s - hi).astype(np.float64)
    return np.where(np.abs(dlo) <= np.abs(dhi), dlo, dhi)


# --------------------------------------------------------------------- main class
@dataclass
class Seasonality:
    """Evaluated seasonal factor. `s(ts)` returns the multiplier on the variance rate."""
    ny_tow_log: np.ndarray                 # (10080,) minute-of-week, NY
    utc_tod_log: np.ndarray                # (1440,)  minute-of-day, UTC
    spikes: list = field(default_factory=list)
    const: float = 0.0                     # normalisation: mean_t s(t) == 1
    meta: dict = field(default_factory=dict)
    # --- v2 layers, all optional so a v1 params.json still loads unchanged
    min_log: np.ndarray = field(          # (60,) second-of-minute, mean zero
        default_factory=lambda: np.zeros(0))
    fm_log: np.ndarray = field(           # (300,) second-within-5-minutes, mean zero
        default_factory=lambda: np.zeros(0))
    moh_log: np.ndarray = field(          # (3600,) second-of-hour, UTC, mean zero
        default_factory=lambda: np.zeros(0))
    dated: list = field(default_factory=list)   # DatedSpike list (macro calendar)

    # ---- evaluation -------------------------------------------------------
    def _interp(self, table: np.ndarray, x_s: np.ndarray, period_s: int) -> np.ndarray:
        """Linear interpolation of a minute table whose values sit at minute centres."""
        n = table.size
        u = (np.asarray(x_s, dtype=np.float64) - 30.0) / 60.0
        i0 = np.floor(u).astype(np.int64)
        frac = u - i0
        i0 = np.mod(i0, n)
        i1 = np.mod(i0 + 1, n)
        return table[i0] * (1.0 - frac) + table[i1] * frac

    def log_smooth(self, ts_s: np.ndarray) -> np.ndarray:
        ts_s = np.asarray(ts_s, dtype=np.int64)
        out = (self._interp(self.ny_tow_log, T.ny_tow(ts_s), SEC_WEEK)
               + self._interp(self.utc_tod_log, T.utc_tod(ts_s), SEC_DAY))
        # Sub-hourly layers are stored already evaluated at 1s, so there is nothing to
        # interpolate: candle closes and expiry boundaries are sharp by nature and a
        # minute table would smear exactly the feature they exist to hold.
        if self.min_log.size:
            out = out + self.min_log[np.mod(ts_s, SEC_MIN)]
        if self.fm_log.size:
            out = out + self.fm_log[np.mod(ts_s, SEC_5MIN)]
        if self.moh_log.size:
            out = out + self.moh_log[np.mod(ts_s, SEC_HOUR)]
        return out

    def log_dated(self, ts_s: np.ndarray) -> np.ndarray:
        """Macro-calendar contribution: signed two-lobe kernels on explicit dates."""
        ts_s = np.asarray(ts_s, dtype=np.int64)
        out = np.zeros(ts_s.shape, dtype=np.float64)
        for sp in self.dated:
            if not sp.keep:
                continue
            occ = sp.as_array()
            if occ.size == 0:
                continue
            d = _nearest_delta(ts_s, occ)
            g, before = _spike_shape(d, sp.center_s, sp.width_s, sp.rise_s)
            out += np.where(before, sp.log_mult_pre, sp.log_mult) * g
        return out

    def log_spike(self, ts_s: np.ndarray) -> np.ndarray:
        ts_s = np.asarray(ts_s, dtype=np.int64)
        out = np.zeros(ts_s.shape, dtype=np.float64)
        if not self.spikes:
            return out
        ny = T.ny_tow(ts_s)
        ny_tod, ny_dow = ny % SEC_DAY, ny // SEC_DAY
        utc = T.utc_tod(ts_s)
        utc_dow = ((ts_s // SEC_DAY) + 3) % 7   # epoch day 0 = Thursday
        for sp in self.spikes:
            if not sp.keep:
                continue
            tod, dow = (ny_tod, ny_dow) if sp.clock == "NY" else (utc, utc_dow)
            d = tod.astype(np.float64) - sp.local_time_s
            d = np.mod(d + SEC_DAY / 2, SEC_DAY) - SEC_DAY / 2
            m = np.asarray(sp.weekday_mask, dtype=bool)[dow]
            g, before = _spike_shape(d, sp.center_s, sp.width_s,
                                     getattr(sp, "rise_s", -1.0))
            amp = np.where(before, getattr(sp, "log_mult_pre", 0.0), sp.log_mult)
            out += np.where(m, amp * g, 0.0)
        return out

    def log_s(self, ts_s: np.ndarray) -> np.ndarray:
        out = self.log_smooth(ts_s) + self.log_spike(ts_s) + self.const
        if self.dated:
            out = out + self.log_dated(ts_s)
        return out

    def s(self, ts_s: np.ndarray) -> np.ndarray:
        return np.exp(self.log_s(ts_s))

    # ---- normalisation ----------------------------------------------------
    def normalise(self, ts_s: np.ndarray) -> None:
        """Set the constant so that mean over the given timestamps of s == 1."""
        self.const = 0.0
        m = np.mean(np.exp(self.log_s(ts_s)))
        self.const = -float(np.log(m))

    def normalise_range(self, t0: int, t1: int, chunk: int = 4_000_000) -> None:
        """Normalise over EVERY second in [t0, t1).

        Sampling the mean at 1-minute resolution would under-resolve the narrow event
        spikes and leave `s` a percent or so away from mean 1, which shows up directly
        as a level error in the business clock. A second-by-second mean is exact and
        costs a couple of seconds.
        """
        self.const = 0.0
        tot, cnt = 0.0, 0
        for a in range(int(t0), int(t1), chunk):
            tt = np.arange(a, min(a + chunk, int(t1)), dtype=np.int64)
            tot += float(np.sum(np.exp(self.log_s(tt))))
            cnt += tt.size
        self.const = -float(np.log(tot / max(cnt, 1)))

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "ny_tow_log": self.ny_tow_log.tolist(),
            "utc_tod_log": self.utc_tod_log.tolist(),
            "spikes": [asdict(s) for s in self.spikes],
            "const": self.const,
            "meta": self.meta,
            "min_log": np.asarray(self.min_log).tolist(),
            "fm_log": np.asarray(self.fm_log).tolist(),
            "moh_log": np.asarray(self.moh_log).tolist(),
            "dated": [asdict(s) for s in self.dated],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Seasonality":
        return cls(
            ny_tow_log=np.asarray(d["ny_tow_log"], dtype=np.float64),
            utc_tod_log=np.asarray(d["utc_tod_log"], dtype=np.float64),
            spikes=[Spike(**s) for s in d["spikes"]],
            const=float(d["const"]),
            meta=d.get("meta", {}),
            min_log=np.asarray(d.get("min_log", []), dtype=np.float64),
            fm_log=np.asarray(d.get("fm_log", []), dtype=np.float64),
            moh_log=np.asarray(d.get("moh_log", []), dtype=np.float64),
            dated=[DatedSpike(**s) for s in d.get("dated", [])],
        )

    def copy(self) -> "Seasonality":
        """Deep-enough copy: the kernels are re-instantiated so a caller can drop or
        re-amplitude one layer without mutating the source."""
        return Seasonality(
            ny_tow_log=np.array(self.ny_tow_log, copy=True),
            utc_tod_log=np.array(self.utc_tod_log, copy=True),
            spikes=[Spike(**asdict(s)) for s in self.spikes],
            const=self.const, meta=dict(self.meta),
            min_log=np.array(self.min_log, copy=True),
            fm_log=np.array(self.fm_log, copy=True),
            moh_log=np.array(self.moh_log, copy=True),
            dated=[DatedSpike(**asdict(s)) for s in self.dated])

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path) -> "Seasonality":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ------------------------------------------------------------------- smooth fitting
def _tables_from_coef(coef: np.ndarray, spec: BasisSpec):
    """Evaluate the fitted smooth parts onto the (10080,) and (1440,) minute grids.

    The UTC table is centred to mean zero so the level lives entirely in the NY table;
    the overall level is then absorbed by `Seasonality.normalise`.
    """
    ny_grid = np.arange(MIN_WEEK) * 60.0 + 30.0
    utc_grid = np.arange(MIN_DAY) * 60.0 + 30.0

    o = 1
    c_ny = coef[o:o + spec.n_ny]; o += spec.n_ny
    c_utc = coef[o:o + spec.n_utc]; o += spec.n_utc
    c_we = coef[o:o + spec.n_we]

    ny = coef[0] + fourier(ny_grid, SEC_WEEK, spec.ny_order) @ c_ny
    we = (ny_grid >= 5 * SEC_DAY).astype(np.float64)
    ny = ny + we * (c_we[0] + fourier(ny_grid % SEC_DAY, SEC_DAY, spec.weekend_order)
                    @ c_we[1:])
    utc = fourier(utc_grid, SEC_DAY, spec.utc_order) @ c_utc
    m = utc.mean()
    return ny + m, utc - m


def fit_smooth(cells: CellStats, spec: BasisSpec, stat: str = "mean",
               mode: str = "joint"):
    """Weighted ridge fit of the smooth two-clock model on binned cell statistics.

    mode: "joint"       - both clocks in one least-squares (the brief's recommendation)
          "sequential"  - NY first, then UTC on the NY residual (the old approach)
          "ny_only" / "utc_only" / "none"
    """
    y = {"mean": cells.mean, "median": cells.median, "trimmed": cells.trimmed}[stat]
    w = cells.count.astype(np.float64)
    ok = np.isfinite(y)
    ny_s = cells.ny_min.astype(np.float64) * 60.0 + 30.0
    utc_s = cells.utc_min.astype(np.float64) * 60.0 + 30.0

    if mode == "none":
        c = np.zeros(spec.n_par)
        c[0] = np.average(y[ok], weights=w[ok])
        return c, spec

    if mode == "joint":
        X = design(ny_s, utc_s, spec)
        coef = ridge_solve(X[ok], y[ok], w[ok], penalty(spec))
        return coef, spec

    if mode == "ny_only":
        s2 = BasisSpec(spec.ny_order, 0, spec.weekend_order, spec.lam_ny, spec.lam_utc,
                       spec.lam_weekend)
        X = design(ny_s, utc_s, s2)
        coef = ridge_solve(X[ok], y[ok], w[ok], penalty(s2))
        return coef, s2

    if mode == "utc_only":
        s2 = BasisSpec(0, spec.utc_order, 0, spec.lam_ny, spec.lam_utc, spec.lam_weekend)
        X = design(ny_s, utc_s, s2)
        coef = ridge_solve(X[ok], y[ok], w[ok], penalty(s2))
        return coef, s2

    if mode == "sequential":
        s_ny = BasisSpec(spec.ny_order, 0, spec.weekend_order, spec.lam_ny, spec.lam_utc,
                         spec.lam_weekend)
        X1 = design(ny_s, utc_s, s_ny)
        c1 = ridge_solve(X1[ok], y[ok], w[ok], penalty(s_ny))
        resid = y - X1 @ c1
        s_utc = BasisSpec(0, spec.utc_order, 0, spec.lam_ny, spec.lam_utc, spec.lam_weekend)
        X2 = design(ny_s, utc_s, s_utc)
        c2 = ridge_solve(X2[ok], resid[ok], w[ok], penalty(s_utc))
        # splice into a full-spec coefficient vector
        coef = np.zeros(spec.n_par)
        coef[0] = c1[0] + c2[0]
        coef[1:1 + spec.n_ny] = c1[1:1 + spec.n_ny]
        coef[1 + spec.n_ny:1 + spec.n_ny + spec.n_utc] = c2[1:1 + s_utc.n_utc]
        coef[1 + spec.n_ny + spec.n_utc:] = c1[1 + s_ny.n_ny:]
        return coef, spec

    raise ValueError(mode)


def seasonality_from_coef(coef: np.ndarray, spec: BasisSpec, spikes=None,
                          meta=None) -> Seasonality:
    ny, utc = _tables_from_coef(coef, spec)
    return Seasonality(ny_tow_log=ny, utc_tod_log=utc, spikes=list(spikes or []),
                       const=0.0, meta=meta or {})
