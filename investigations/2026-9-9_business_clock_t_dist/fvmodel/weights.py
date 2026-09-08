"""One weight builder for every settlement type this model prices.

A settlement is a weighted average of future prices. Everything the estimator does with
it - the variance, the order-book location term, the standardised residual, the tail -
depends on the settlement only through the vector `W` of loadings on the future
one-second returns, plus a deterministic carry. So there is exactly one function that
turns a settlement type into that vector, and three thin kind-specific branches inside
it.

The algebra
-----------
Let `x_u = log X(u)` be the log of the settlement's input price series and
`r_k = x_{t+k} - x_{t+k-1}` the future one-second returns at the quote origin `t`.

**A single print** at `t + K` loads `W_k = 1` for `k = 1..K`.

**A TWAP** of the `L` prices ending at `T` loads `c_k = min(L, n-k+1) / L`, i.e. one for
every second before the window opens and a linear ramp inside it.

**A Chainlink print** is not the price, it is an EWMA of it. With `lam = exp(-1/tau)`,

    F(u+K) = lam^K F(u) + (1 - lam^K) x_u + sum_{k=1..K} (1 - lam^(K-k+1)) dx_{u+k}

so the print stamped `s` loads `1 - lam^(K-k+1)` on `r_k` with `K = s - delta - t`, and
carries `lam^K` of the amount by which the filter currently trails the price. `delta`
enters only through `K`; the quote origin's own filter value `F(t)` is known, so it is
the natural reference point and the carry is a *known* mean shift, not noise.

**A Chainlink 60-second TWAP** is the average of sixty of those, one per second of the
window, which is what `chainlink_twap60` composes. The composition matters only inside
the last few seconds: at `tau = 1.6 s`, `lam^K` is 2e-3 by `K = 10`, so past about ten
seconds from expiry the composed weights and the raw triangular ramp are the same vector
to within a tenth of a per cent.

Normalisation
-------------
`W` is the loading of the **unknown part's own average**, not of the whole settlement,
because that is the quantity the variance, the tail and `y*` are all expressed in. The
share of the settlement still unknown is `omega`; the loading of the full settlement
average is `omega * W`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("perp_single", "perp_twap", "chainlink_twap60")


@dataclass
class Settlement:
    """The complete description of what is left to be determined."""
    kind: str
    n: int                        # seconds from the quote origin to expiry
    W: np.ndarray                 # (n,) loading of the unknown average on r_1..r_n
    omega: float                  # share of the settlement weight still unknown
    n_components: int             # prints (or seconds) the settlement averages over
    n_known: int                  # of those, how many are already determined
    known_idx: np.ndarray         # their offsets j (component stamped at T - j)
    unknown_idx: np.ndarray       # the rest
    carry: float = 0.0            # coefficient on (F_t - x_t); Chainlink kinds only
    L: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def n_unknown(self) -> int:
        return self.n_components - self.n_known

    @property
    def W_raw(self) -> np.ndarray:
        """Loading of the **full** settlement average, before omega-normalisation.

        `W` is normalised by the structurally unknown share; a caller that knows which
        prints have actually been received re-normalises this by its own omega.
        """
        return self.W * self.omega

    def describe(self) -> str:
        return ("%s: %d of %d components known (%.1f%% of the settlement weight), "
                "%d future seconds load the answer"
                % (self.kind, self.n_known, self.n_components,
                   100.0 * (1.0 - self.omega), self.n))


def _empty(kind: str, n_components: int, L: int) -> Settlement:
    """Expiry has passed (or the whole settlement is already fixed)."""
    return Settlement(kind, 0, np.zeros(0), 0.0, n_components, n_components,
                      np.arange(n_components), np.zeros(0, dtype=np.int64), 0.0, L)


def settlement_weights(kind: str, n_remaining: int, params: dict = None,
                       L: int = None) -> Settlement:
    """`W_k` over the remaining seconds k = 1..n, and the known/unknown split.

    `params` needs `tau_s` and `delta_s` for the Chainlink kinds and is ignored
    otherwise. `L` overrides the TWAP window length (default 60 for `perp_twap`).
    """
    n = int(n_remaining)
    if kind == "perp_single":
        if n <= 0:
            return _empty(kind, 1, 1)
        return Settlement(kind, n, np.ones(n), 1.0, 1, 0,
                          np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64),
                          0.0, 1)

    if kind == "perp_twap":
        Lw = int(L if L is not None else 60)
        if Lw < 1:
            raise ValueError("perp_twap needs L >= 1")
        if n <= 0:
            return _empty(kind, Lw, Lw)
        k = np.arange(1, n + 1, dtype=np.float64)
        c = np.minimum(Lw, n - k + 1.0) / Lw           # loading of the FULL average
        n_unknown = min(n, Lw)
        omega = n_unknown / Lw
        return Settlement(kind, n, c / omega, omega, Lw, Lw - n_unknown,
                          np.arange(n_unknown, Lw, dtype=np.int64),
                          np.arange(0, n_unknown, dtype=np.int64), 0.0, Lw)

    if kind == "chainlink_twap60":
        Lw = int(L if L is not None else 60)
        p = params or {}
        tau = float(p.get("tau_s", 1.6))
        delta = float(p.get("delta_s", 0.2))
        lam = float(np.exp(-1.0 / tau)) if tau > 0 else 0.0
        if n <= 0:
            return _empty(kind, Lw, Lw)
        # component j is the print stamped at T - j; it reads X at T - j - delta, which
        # is K_j seconds after the quote origin
        j = np.arange(Lw, dtype=np.float64)
        K = n - j - delta
        fut = K > 0.0
        n_unknown = int(fut.sum())
        if n_unknown == 0:
            return _empty(kind, Lw, Lw)
        Kf = K[fut]                                          # (m,)
        k = np.arange(1, n + 1, dtype=np.float64)            # (n,)
        if lam > 0.0:
            # 1 - lam^(K - k + 1), clipped at zero past the end of each print's reach
            Wc = np.maximum(1.0 - lam ** (Kf[:, None] - k[None, :] + 1.0), 0.0)
            carry = float(np.sum(lam ** Kf))
        else:                                                # tau -> 0: the raw ramp
            Wc = (Kf[:, None] - k[None, :] + 1.0 > 0.0).astype(np.float64)
            carry = 0.0
        W_full = Wc.sum(axis=0) / Lw                         # loading of the FULL avg
        omega = n_unknown / Lw
        return Settlement(kind, n, W_full / omega, omega, Lw, Lw - n_unknown,
                          np.arange(n_unknown, Lw, dtype=np.int64),
                          np.arange(0, n_unknown, dtype=np.int64),
                          carry / Lw / omega, Lw,
                          meta={"tau_s": tau, "delta_s": delta, "lambda": lam})

    raise ValueError("unknown settlement kind %r; expected one of %s" % (kind, KINDS))


def uncomposed_ramp(n_remaining: int, L: int = 60) -> np.ndarray:
    """The triangular TWAP weights a model that forgot the EWMA composition would use."""
    return settlement_weights("perp_twap", n_remaining, L=L).W


def composition_gap(n_remaining: int, params: dict, L: int = 60) -> dict:
    """How far the EWMA-composed weights sit from the raw ramp, at one horizon."""
    a = settlement_weights("chainlink_twap60", n_remaining, params, L=L).W
    b = uncomposed_ramp(n_remaining, L=L)
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    den = max(float(np.abs(b).sum()), 1e-300)
    return {"n": int(n_remaining), "max_abs": float(np.max(np.abs(a - b))),
            "l1_rel": float(np.abs(a - b).sum() / den),
            "sum_composed": float(a.sum()), "sum_ramp": float(b.sum())}
