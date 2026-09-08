"""The order-book location term.

The preliminary note's C4 section established two things about top-of-book imbalance
`I = ln(ask size / bid size)` on this instrument, and both change how it should be
parameterised:

* the alpha decays in **business time**, not clock time - matched on clock seconds the
  beta curves differ across clock-speed terciles by 23%, matched on business time by 7%;
* its saturated size is a fixed number of **basis points** (about -0.18 bp per unit of
  I), not a fixed fraction of risk, so it is worth relatively more when the market is
  quiet.

So the model is `beta(dT)` in basis points on the business-time axis the volatility
clock already computes, plus the quote-age interaction C4 found: a quote that has just
moved predicts the next second, a quote that has sat unchanged for ten seconds predicts
the next ten and beyond.

    E[ r over (t, t+h) ] = ( beta0(dT_h) + beta1(dT_h) * log(age / age_ref) ) * I_t

and the settlement's location shift is the weighted sum of the per-second increments,

    m_Y = I_t * sum_k W_k [ beta(dT_k) - beta(dT_{k-1}) ]

which is bounded because beta saturates: past the fitted range the increments are zero.

The fit rests on five book captures totalling 107.8 hours in one market regime, and
**they do not overlap the Chainlink history at all** - they end 2026-04-02, Chainlink
starts 2026-04-13. Nothing here has been measured against a Chainlink settlement. The
interface is built so an event-driven bookTicker feed running alongside the Chainlink
recorder is a drop-in: `AlphaModel.m_Y` only ever sees `I`, an age, and the weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BP = 1e4


@dataclass
class AlphaModel:
    """beta(dT) in basis points per unit of imbalance, with the quote-age interaction."""
    dT_knots: np.ndarray            # ascending business-time knots (business days)
    beta0_bp: np.ndarray            # cumulative expected move, bp, at age = age_ref
    beta1_bp: np.ndarray            # coefficient on log(age / age_ref)
    age_ref: float = 2.2            # seconds since the L1 price last moved
    i_sd: float = 1.0               # sd of I, for reporting a "one-sigma" shift
    meta: dict = field(default_factory=dict)

    # ---- the curve -------------------------------------------------------
    def beta(self, dT, log_age_ratio=0.0) -> np.ndarray:
        """Cumulative expected move in bp per unit of I, at business age `dT`.

        Flat outside the fitted range: below it there has been no time for anything to
        happen, above it the alpha has saturated. Both are what the data show and both
        keep `m_Y` bounded.
        """
        dT = np.asarray(dT, dtype=np.float64)
        b0 = np.interp(dT, self.dT_knots, self.beta0_bp,
                       left=self.beta0_bp[0], right=self.beta0_bp[-1])
        if np.all(self.beta1_bp == 0.0):
            return b0
        b1 = np.interp(dT, self.dT_knots, self.beta1_bp,
                       left=self.beta1_bp[0], right=self.beta1_bp[-1])
        return b0 + b1 * np.asarray(log_age_ratio, dtype=np.float64)

    def log_age_ratio(self, px_age_s) -> np.ndarray:
        a = np.maximum(np.asarray(px_age_s, dtype=np.float64), 0.05)
        return np.log(a / self.age_ref)

    # ---- the settlement shift -------------------------------------------
    def m_Y(self, I, W: np.ndarray, dT_cum: np.ndarray, px_age_s=None) -> np.ndarray:
        """Location shift of the settlement, as a fraction (not bp).

        `dT_cum[..., k]` is the business time from the quote origin to the end of second
        k; `W` are the settlement weights over the same seconds. Returns `I * sum_k W_k
        d_beta_k / 1e4`, shaped like the leading axes of `dT_cum`.
        """
        if W.size == 0:
            return np.zeros(np.shape(I))
        d = np.atleast_2d(np.asarray(dT_cum, dtype=np.float64))
        lar = 0.0 if px_age_s is None else self.log_age_ratio(px_age_s)
        lar = np.reshape(lar, (-1, 1)) if np.ndim(lar) else lar
        b = self.beta(d, lar)
        db = np.diff(np.concatenate([np.zeros((b.shape[0], 1)), b], axis=1), axis=1)
        out = (np.asarray(I, dtype=np.float64).reshape(-1)
               * (db[:, :W.size] @ W[:db.shape[1]]) / BP)
        return out if np.ndim(I) else float(out[0])

    # ---- io --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"dT_knots": self.dT_knots.tolist(), "beta0_bp": self.beta0_bp.tolist(),
                "beta1_bp": self.beta1_bp.tolist(), "age_ref": self.age_ref,
                "i_sd": self.i_sd, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "AlphaModel":
        return cls(np.asarray(d["dT_knots"], dtype=np.float64),
                   np.asarray(d["beta0_bp"], dtype=np.float64),
                   np.asarray(d["beta1_bp"], dtype=np.float64),
                   float(d.get("age_ref", 2.2)), float(d.get("i_sd", 1.0)),
                   d.get("meta", {}))

    @classmethod
    def zero(cls) -> "AlphaModel":
        """The no-book-feed case: every location shift is exactly zero."""
        return cls(np.array([1e-9, 1.0]), np.zeros(2), np.zeros(2), 2.2, 1.0,
                   {"note": "no book feed"})


# ================================================================== fitting helpers
def bin_edges(dT: np.ndarray, n_bins: int = 14) -> np.ndarray:
    lo = float(np.quantile(dT, 0.002))
    hi = float(np.quantile(dT, 0.998))
    return np.geomspace(max(lo, 1e-10), hi, n_bins + 1)


def _wls(X, y, w=None):
    """Normal-equation solve. Small and fixed-size, so this is far cheaper than lstsq."""
    if w is None:
        A = X.T @ X
        b = X.T @ y
    else:
        Xw = X * w[:, None]
        A = Xw.T @ X
        b = Xw.T @ y
    A[np.diag_indices_from(A)] += 1e-12
    return np.linalg.solve(A, b)


def fit_beta_dT(dT: np.ndarray, I: np.ndarray, R_bp: np.ndarray,
                log_age: np.ndarray, edges: np.ndarray,
                block: np.ndarray = None, n_boot: int = 60,
                seed: int = 0, with_age: bool = True, min_n: int = 500) -> dict:
    """beta0 and beta1 per business-time bin, with a block-bootstrap standard error.

    Every origin contributes one row per horizon and the horizons overlap, so the rows
    are dependent both across h at one origin and across nearby origins. A Newey-West
    correction handles one of those and not the other. Resampling contiguous blocks of
    origins handles both, assuming the blocks are long enough to be near-independent
    (they are half an hour).

    The rows are sorted into their bins once so that each bin is a contiguous slice;
    without that, every (bin, bootstrap) pair costs a full-length boolean mask and the
    fit does not finish.
    """
    rng = np.random.default_rng(seed)
    j = np.clip(np.searchsorted(edges, dT, side="right") - 1, 0, edges.size - 2)
    order = np.argsort(j, kind="stable")
    j = j[order]
    cols = [I[order]] + ([I[order] * log_age[order]] if with_age else [])
    X = np.column_stack([np.ones(j.size)] + cols)
    y = R_bp[order]
    blk = None if block is None else block[order]
    starts = np.searchsorted(j, np.arange(edges.size))
    nb = edges.size - 1

    b0 = np.zeros(nb)
    b1 = np.zeros(nb)
    se0 = np.full(nb, np.nan)
    se1 = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=np.int64)
    for k in range(nb):
        sl = slice(starts[k], starts[k + 1])
        cnt[k] = sl.stop - sl.start
        if cnt[k] < min_n:
            continue
        c = _wls(X[sl], y[sl])
        b0[k] = c[1]
        if with_age:
            b1[k] = c[2]
        if blk is None or n_boot <= 0:
            continue
        ub, inv = np.unique(blk[sl], return_inverse=True)
        B = np.empty((n_boot, 2))
        for s_ in range(n_boot):
            cnts = np.bincount(rng.integers(0, ub.size, ub.size), minlength=ub.size)
            w = cnts[inv].astype(np.float64)
            cc = _wls(X[sl], y[sl], w)
            B[s_] = (cc[1], cc[2] if with_age else 0.0)
        se0[k] = float(np.std(B[:, 0]))
        se1[k] = float(np.std(B[:, 1]))

    centres = np.sqrt(edges[:-1] * edges[1:])
    return {"dT": centres, "beta0_bp": b0, "beta1_bp": b1, "se0": se0, "se1": se1,
            "n": cnt, "edges": edges}


def cumulate(fit: dict) -> tuple:
    """Turn per-bin *cumulative* beta estimates into a monotone saturating curve.

    The regression already estimates the cumulative move to business age dT, so the
    curve is the estimates themselves; this only enforces the shape the note measured
    (monotone toward the saturation level) so that the per-second increments the
    settlement sum uses cannot change sign on estimation noise.
    """
    ok = fit["n"] >= 500
    d = fit["dT"][ok]
    b0 = fit["beta0_bp"][ok]
    b1 = fit["beta1_bp"][ok]
    b0 = np.minimum.accumulate(b0) if b0[0] > b0[-1] else np.maximum.accumulate(b0)
    return d, b0, b1
