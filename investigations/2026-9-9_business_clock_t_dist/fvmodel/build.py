"""Assemble the fitted pieces into one `FairValueModel`, and score a set of quotes.

Everything the estimator ships is a file on disk written by one of the fitting scripts;
this is the single place that reads them all and puts the object together, so a script,
a test and the autotrader all get exactly the same model.
"""
from __future__ import annotations

import json

import numpy as np

from .alpha import AlphaModel
from .base import TABLES_DIR, load_params
from .chainlink import EpsModel, FilterParams
from .fairvalue import FairValueModel
from .tails import SettlementTail
from .variance import QLIKE_FLOOR, mz, qlike

KINDS = ("chainlink_twap60", "perp_twap", "perp_single")


def build_model(eps_fit: str = "train", tails: dict = None,
                basis_hl_s: float = None,
                input_mode: str = "blend") -> FairValueModel:
    """The shipped model: v2.1 plus the filter, the residual, the kernels and the alpha.

    `input_mode="perp_only"` returns the same object built on the perp-only print model
    - its own filter constants, its own basis drift and its own residual - which is what
    the "perp-only input" ablation needs. Anything less would score the blend's residual
    against a filter it was not fitted for.
    """
    from rvforecast.streaming import Model

    params = load_params()
    pm = json.load(open(TABLES_DIR / "print_model.json"))
    if input_mode == "perp_only":
        po = pm["perp_only"]
        fp = FilterParams.from_dict(dict(po["filter"], w_spot=0.0))
        fp.basis_hl_s = float(po["half_life_s"])
        fp.lag_s = int(po["lag_s"])
        eps = EpsModel.from_dict(pm["eps"]["perp_only"])
        eps.meta = dict(eps.meta or {})
        eps.meta["basis_drift"] = po["drift_rows"]
    else:
        fp = FilterParams.from_dict(pm["filter"]["shipped"])
        fp.basis_hl_s = float(basis_hl_s if basis_hl_s is not None
                              else pm["basis"]["half_life_s"])
        fp.lag_s = int(pm["basis"]["lag_s"])
        eps = EpsModel.from_dict(pm["eps"][eps_fit])
        eps.meta = dict(eps.meta or {})
        eps.meta["basis_drift"] = pm["basis"]["drift_rows"]

    rho = {}
    ivr = 1.0
    cuts = (0.75, 1.25)
    if (TABLES_DIR / "kernels.json").exists():
        kj = json.load(open(TABLES_DIR / "kernels.json"))
        rho = {k: np.asarray(v, dtype=np.float64) for k, v in kj["rho"].items()}
        rho["shipped"] = np.asarray(kj["rho_shipped"], dtype=np.float64)
        ivr = float(kj["input_var_ratio"])
        cuts = tuple(kj["act_cuts"])
    else:
        rho = {"all": np.asarray(params["acf_kernel"]["last"], dtype=np.float64)}

    alpha = AlphaModel.zero()
    if (TABLES_DIR / "alpha.json").exists():
        alpha = AlphaModel.from_dict(json.load(open(TABLES_DIR / "alpha.json"))["alpha"])

    tl = {}
    if tails:
        tl = {k: (v if isinstance(v, SettlementTail) else SettlementTail.from_dict(v))
              for k, v in tails.items()}
    elif (TABLES_DIR / "tails.json").exists():
        tl = {k: SettlementTail.from_dict(v)
              for k, v in json.load(open(TABLES_DIR / "tails.json")).items()}

    return FairValueModel(Model(params), fp, eps, alpha, rho, cuts, tl, ivr, 0.0)


# ------------------------------------------------------- the basis-drift variance
def _drift_curve(model: FairValueModel):
    rows = (model.eps.meta or {}).get("basis_drift")
    if not rows:
        return None, None
    return (np.array([r[0] for r in rows], dtype=np.float64),
            np.array([r[1] for r in rows], dtype=np.float64))


def basis_drift_var(model: FairValueModel, a) -> np.ndarray:
    """`V(a) = Var(b_{t+a} - b_t)`, interpolated in log a from the measured curve."""
    ax, vx = _drift_curve(model)
    if ax is None:
        return np.zeros(np.shape(a))
    return np.interp(np.log(np.maximum(np.asarray(a, dtype=np.float64), 1.0)),
                     np.log(ax), vx, left=vx[0], right=vx[-1])


def basis_drift_window_var(model: FairValueModel, ages: np.ndarray,
                           weights: np.ndarray = None) -> float:
    """Var of the weighted mean of the basis over a settlement window.

    The settlement model treats the slow basis as constant across the window. It is not,
    and its own movement is a source of settlement uncertainty the return variance knows
    nothing about. Evaluating `V(a_eff)` at the mean age would be wrong in both
    directions - the components' drifts are strongly correlated with each other - so the
    full covariance is used. For any process with stationary increments,

        Cov(b_{t+a} - b_t, b_{t+a'} - b_t) = [ V(a) + V(a') - V(|a - a'|) ] / 2

    which needs nothing but the measured `V` and no assumption that it is a random walk.
    """
    ax, _ = _drift_curve(model)
    if ax is None or np.size(ages) == 0:
        return 0.0
    a = np.asarray(ages, dtype=np.float64)
    w = (np.full(a.size, 1.0 / a.size) if weights is None
         else np.asarray(weights, dtype=np.float64))
    Va = basis_drift_var(model, a)
    C = 0.5 * (Va[:, None] + Va[None, :] - basis_drift_var(model,
                                                           np.abs(a[:, None] - a[None, :])))
    return float(max(w @ C @ w, 0.0))


# ================================================================== the scorecard
STRIKE_GRID = (-1.0, -0.5, 0.5, 1.0)


def grid_strikes(ref: dict, grid=STRIKE_GRID) -> np.ndarray:
    """Strike prices at fixed multiples of the reference model's own sd.

    `y*` is affine in the strike - `y*(K) = y*(K0) + (K - K0) / (p_ref * omega)` - so
    the strike at which the reference model's standardised distance is exactly `c` is
    `K0 + (c*sd - y*(K0)) * p_ref * omega`. Building the grid in **price** space is what
    lets two models with different variances be scored on the same question; a grid
    placed at each model's own sd asks the wider model an easier one, and the wider
    model then appears to win.
    """
    sd = np.sqrt(np.maximum(ref["var_y"], 1e-300))
    scale = ref["p_ref"] * ref["omega"]
    return np.stack([ref["strike"] + (c * sd - ref["y_star"]) * scale for c in grid])


def _y_star_at(rows: dict, K: np.ndarray) -> np.ndarray:
    """This arm's `y*` for an arbitrary strike, from the affine relation above."""
    return rows["y_star"] + (K - rows["strike"]) / (rows["p_ref"] * rows["omega"])


def score(rows: dict, tail, informative: float = 0.25, grid=STRIKE_GRID,
          ref: dict = None) -> dict:
    """Binary and variance metrics for one cell of quotes.

    Two binary scores, because they answer different questions.

    `log_loss` is on the **real market**: the strike is the settlement reference price
    at the market's open, which is what a Polymarket up/down window actually is. Late in
    such a market most quotes are decided, so the score is small and dominated by how
    often the model is confidently right.

    `log_loss_grid` is on a **strike grid** placed at fixed multiples of a reference
    model's predicted standard deviation, which is v2's convention and the only way to
    have informative quotes at every remaining time. `ref` is that reference - pass the
    full model's rows when scoring an ablation, so both arms are priced at the same
    strikes and the comparison means something.
    """
    # An ablation can survive a slightly different set of markets - the
    # market-observable baseline quotes from two seconds earlier, so a handful of
    # origins fall outside the window - and the common strike grid needs the two arms
    # aligned market by market. Both are sorted by expiry, so intersecting on it is
    # enough, and doing it here means every metric is computed on the same set.
    if ref is not None and not (ref["T"].size == rows["T"].size
                                and np.array_equal(ref["T"], rows["T"])):
        common = np.intersect1d(ref["T"], rows["T"])
        ref = {k: v[np.searchsorted(ref["T"], common)] for k, v in ref.items()}
        rows = {k: v[np.searchsorted(rows["T"], common)] for k, v in rows.items()}

    y = rows["y_star"]
    v = rows["var_y"]
    z = rows["z"]
    up = rows["up"].astype(np.float64)
    p = np.clip(tail.prob_up(y, v, z), 1e-9, 1 - 1e-9)
    sel = np.abs(y) / np.sqrt(np.maximum(v, 1e-300)) > informative
    out = {"n_markets": int(y.size), "n_informative": int(sel.sum())}
    if sel.sum() > 30:
        out["log_loss"] = float(-np.mean(up[sel] * np.log(p[sel])
                                         + (1 - up[sel]) * np.log(1 - p[sel])))
        out["brier"] = float(np.mean((p[sel] - up[sel]) ** 2))
    else:
        out["log_loss"] = out["brier"] = float("nan")
    out["log_loss_all"] = float(-np.mean(up * np.log(p) + (1 - up) * np.log(1 - p)))
    out["brier_all"] = float(np.mean((p - up) ** 2))
    r2 = rows["resid"] ** 2
    # A single squared return is an atomic proxy: on the perp at one and two seconds it
    # is *exactly* zero about forty per cent of the time (tick discreteness), and QLIKE
    # diverges there for any finite forecast. The metric is taken over the realisations
    # it is defined on and the excluded share is reported next to it, rather than
    # quietly clamped to a floor that would put the number wherever the clamp is.
    nz = r2 > 0
    out["frac_zero_resid"] = float(1.0 - nz.mean())
    out["qlike"] = qlike(r2[nz], v[nz]) if nz.sum() > 30 else float("nan")
    out["qlike_excess"] = out["qlike"] - QLIKE_FLOOR
    out["qlike_n"] = int(nz.sum())
    sl, ic, r2m = mz(r2, v)
    out["mz_slope"], out["mz_intercept"], out["mz_r2"] = sl, ic, r2m
    # two calibration statistics, because they disagree and the disagreement is the
    # finding: the ratio of means is a level check, the mean of the ratios is the
    # multiplicative scale a likelihood would choose. See scripts/15_inherited.py.
    out["level"] = float(np.mean(r2) / max(np.mean(v), 1e-300))
    out["level_mean_ratio"] = float(np.mean(r2 / np.maximum(v, 1e-300)))
    out["sd_resid_bp"] = float(np.std(rows["resid"]) * 1e4)
    out["sd_pred_bp"] = float(np.sqrt(np.mean(v)) * 1e4)
    out["mean_p"] = float(np.mean(p))
    out["freq_up"] = float(np.mean(up))

    Ks = grid_strikes(ref if ref is not None else rows, grid)
    ll, br = [], []
    for K in Ks:
        yc = _y_star_at(rows, K)
        pc = np.clip(tail.prob_up(yc, v, z), 1e-9, 1 - 1e-9)
        uc = (rows["settle"] > K).astype(np.float64)
        ll.append(-np.mean(uc * np.log(pc) + (1 - uc) * np.log(1 - pc)))
        br.append(np.mean((pc - uc) ** 2))
    out["log_loss_grid"] = float(np.mean(ll))
    out["brier_grid"] = float(np.mean(br))
    return out


def reliability(rows: dict, tail, bins: int = 10, grid=STRIKE_GRID,
                ref: dict = None) -> list:
    """Reliability over the strike grid, which is where the probabilities actually
    spread out; the at-open strike puts almost every late quote at zero or one."""
    v, z = rows["var_y"], rows["z"]
    Ks = grid_strikes(ref if ref is not None else rows, grid)
    p = np.concatenate([np.clip(tail.prob_up(_y_star_at(rows, K), v, z), 1e-9, 1 - 1e-9)
                        for K in Ks])
    up = np.concatenate([(rows["settle"] > K).astype(np.float64) for K in Ks])
    edges = np.linspace(0, 1, bins + 1)
    j = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    out = []
    for k in range(bins):
        m = j == k
        if m.sum() < 20:
            continue
        out.append([float(0.5 * (edges[k] + edges[k + 1])), float(p[m].mean()),
                    float(up[m].mean()), int(m.sum())])
    return out
