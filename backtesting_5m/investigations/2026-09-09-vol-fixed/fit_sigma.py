"""Measure the model's sigma against realised movement, then fit sigma(tau).

TWO JOBS, in this order, and the first one is allowed to end the investigation:

  1. DIAGNOSE. For a grid of time-to-expiry buckets, compare `vol_baseline.py`'s
     sigma_T against the realised dispersion of

         r = ln(A_T / s_t)

     where A_T is the settling value (the next market's strike, which IS the
     settling Chainlink 60 s TWAP) and s_t is `fair.py`'s E[A_T | F_t] -- the
     SAME level `f.standardise` divides by. Comparing against raw spot would
     measure a different model than the one being scored.

  2. FIT. If sigma really is too small, fit sigma(tau) to the realised
     dispersion ON THE FIT DAYS ONLY and write it to `sigma_fit.json`, which
     `vol.py` reads. The block stays declarative; the number is reproducible
     by re-running this file.

THE SCALE ESTIMATOR is the mean-absolute one,

     sigma_hat(tau) = sqrt(pi/2) * mean |r|

not the sample sd. Both are reported and both are plotted, but 5-minute BTC
returns are fat tailed, and a handful of jumps move the sd by more than they
move anything the model is scored on. `link.py`'s QQ correction exists
precisely to carry the tails, so the scale handed to it should be a scale, not
a tail statistic.

THE FITTED OBJECT IS A TABLE, not a formula. The whole point of the exercise
is that the analytic term structure in `vol_baseline.py` is suspected of being
wrong, so replacing it with a different analytic term structure would beg the
question. A log-log linear interpolation over the empirical bucket table
asserts nothing about shape. Two parametric fits ARE reported, because their
coefficients are the readable summary of what the table says:

  * a free power law, sigma ~ tau^b. b = 0.5 is plain diffusion; the baseline's
    tau_eff implies b -> 1.5 as tau -> 0.
  * the baseline's own tau_eff form with the TWAP window w set free, which
    answers "is 60 s the wrong window, or is the whole form wrong?"

FLAT, the level-vs-term-structure control, is the same weighted fit with the
exponent PINNED at 0.5. Same data, same estimator, one constraint -- so the
gap between `vol.py` and `vol_flat.py` is term structure and nothing else.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

from harness.build.episodes import load_episodes           # noqa: E402
from harness.core import provenance                        # noqa: E402
from harness.io import read_parquet                        # noqa: E402
from harness import paths                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "runs", "cache")

#: WHICH SPOT PANEL, AND WHICH CALENDAR SPLIT. Each entry carries its own
#: days, because the panels do not cover the same ones.
#:
#: `oracle_window` is the panel to use and the default:
#: `harness.paths.SPOT_ORACLE_WINDOW`, 2026-08-17..21, built over the overlap
#: of the three feeds this fit needs. It matters because `fair.py` learns its
#: basis from the Chainlink oracle and returns NaN without one, and the RTDS
#: capture stops at 2026-08-21 01:59 UTC. On the old `usd` panel (08-19..24)
#: that left the FIT half two days of oracle and the TEST half none at all,
#: so the split was not a split.
#:
#: THE SPLIT IS FIT = 08-17..18, TEST = 08-19..21, and it is chosen so both
#: halves have oracle coverage. Scorable markets -- spot, oracle and a
#: settlement -- run 236 / 275 / 283 / 284 / 24 across the five days, so this
#: cut is 511 fit against 591 test. 08-21 contributes only its first two
#: hours because that is where the oracle capture ends; it is put in the test
#: half, where a short day dilutes rather than distorts.
#:
#: `usd` and `usdt` are kept runnable only so the before/after in REPORT.md
#: reproduces. `usdt` is now `harness.paths.SPOT_LEGACY_USDT` explicitly:
#: `paths.SPOT` was repointed at the corrected build, and leaving this entry
#: reading `SPOT` would have quietly refitted the "uncorrected" control on the
#: corrected panel.
PANELS = {
    "oracle_window": ("SPOT_ORACLE_WINDOW",
                      ("2026-08-17", "2026-08-18"),
                      ("2026-08-19", "2026-08-20", "2026-08-21"),
                      "sigma_fit.json", "sigma_obs_oracle_window.parquet",
                      "sigma_vs_realised.png"),
    "usd": ("SPOT_USD",
            ("2026-08-19", "2026-08-20", "2026-08-21"),
            ("2026-08-22", "2026-08-23", "2026-08-24"),
            "sigma_fit_usd_0819.json", "sigma_obs_usd.parquet",
            "sigma_vs_realised_usd_0819.png"),
    "usdt": ("SPOT_LEGACY_USDT",
             ("2026-08-19", "2026-08-20", "2026-08-21"),
             ("2026-08-22", "2026-08-23", "2026-08-24"),
             "sigma_fit_uncorrected.json",
             "sigma_obs_uncorrected.parquet",
             "sigma_vs_realised_uncorrected.png"),
}
PANEL = "oracle_window"         # set by main() from --panel

#: Pre-open history handed to the blocks, matching the runners. `fair.py`'s
#: basis halflife is 180 s and only converges because of it, so a fit run
#: without warm-up would be fitting sigma to a different `fair` than the one
#: the backtest prices with.
WARMUP_S = 900.0

#: Decision indices the term structure is measured at. 5 s apart over most of
#: the window, then dense over the last 5 s, because that is where `tau_eff`
#: does its most aggressive shrinking and where the table would otherwise be
#: extrapolating rather than reading.
GRID = tuple(range(0, 2951, 50)) + (2960, 2970, 2980, 2990, 2995)


def _days():
    _, fit_days, test_days, _, _, _ = PANELS[PANEL]
    return fit_days, test_days


def _paths():
    spot_attr, _, _, fit_name, obs_name, plot_name = PANELS[PANEL]
    return (getattr(paths, spot_attr), os.path.join(HERE, fit_name),
            os.path.join(CACHE_DIR, obs_name), os.path.join(HERE, plot_name))


def observations(days=None, rebuild=False):
    """One row per (market, decision index): model sigma and realised r."""
    spot_path, _, obs_parquet, _ = _paths()
    if days is None:
        fit_days, test_days = _days()
        days = fit_days + test_days
    if os.path.exists(obs_parquet) and not rebuild:
        return read_parquet(obs_parquet)

    res = provenance.resolve_slots(HERE)
    fair = provenance.load_slot(res["fair"], "fair")
    base = provenance.load_slot(os.path.join(HERE, "vol_baseline.py"), "vol")

    # `rtds_path` is not optional here: `fair.py` learns its basis from the
    # oracle and returns NaN without one, so omitting it would silently
    # produce an empty fit. `warmup` matches the runners for the same reason
    # -- fit the sigma of the model that actually trades.
    eps = load_episodes(spot_path=spot_path, days=days,
                        rtds_path=paths.RTDS_BTC,
                        warmup=True, warmup_s=WARMUP_S)
    eps = [e for e in eps if e.has_spot.any() and e.settle is not None]
    print(f"episodes with spot and a settlement: {len(eps)}", flush=True)

    rows = []
    for ep in eps:
        s = fair.precompute(ep)
        sig = base.precompute(ep)
        settle = float(ep.settle)
        for i in GRID:
            if not (np.isfinite(s[i]) and s[i] > 0.0):
                continue
            rows.append((ep.market_id, ep.day, ep.tte_s(i), float(s[i]),
                         float(sig[i]), settle,
                         math.log(settle / float(s[i]))))

    obs = pd.DataFrame(rows, columns=["market_id", "day", "tte", "s",
                                      "sigma_model", "settle", "r"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    obs.to_parquet(obs_parquet, index=False)
    print(f"wrote {len(obs)} observations -> {obs_parquet}", flush=True)
    return obs


def term_structure(obs):
    """Per-tau realised scale, sd, and the mean model sigma on the same rows."""
    g = obs.groupby("tte", sort=True)
    tab = pd.DataFrame({
        "n": g["r"].size(),
        "abs_scale": g["r"].apply(lambda x: math.sqrt(math.pi / 2) * x.abs().mean()),
        "sd": g["r"].std(ddof=1),
        "sigma_model": g["sigma_model"].mean(),
        "sigma_model_n": g["sigma_model"].apply(lambda x: int(np.isfinite(x).sum())),
    }).reset_index()
    tab["ratio"] = tab["sigma_model"] / tab["abs_scale"]
    return tab


def _wls(x, y, w):
    """Weighted least squares slope and intercept of y on x."""
    w = np.asarray(w, dtype="float64")
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    b = (w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum()
    return b, my - b * mx


def power_law(tab):
    """Free power law sigma = exp(a) * tau^b, weighted by bucket count."""
    m = tab[(tab["abs_scale"] > 0) & (tab["tte"] > 0)]
    b, a = _wls(np.log(m["tte"]), np.log(m["abs_scale"]), m["n"])
    return {"a": float(a), "b": float(b), "k": float(math.exp(a))}


def flat_law(tab):
    """The same fit with the exponent PINNED at 0.5: level, no term structure."""
    m = tab[(tab["abs_scale"] > 0) & (tab["tte"] > 0)]
    w = m["n"].to_numpy(dtype="float64")
    resid = np.log(m["abs_scale"].to_numpy()) - 0.5 * np.log(m["tte"].to_numpy())
    a = float((w * resid).sum() / w.sum())
    return {"a": a, "b": 0.5, "k": float(math.exp(a))}


def free_window(tab):
    """The baseline's own tau_eff form with the TWAP window w set free.

    sigma = k * sqrt(tau_eff(tau; w)). Answers whether the SHAPE is right and
    only the window is wrong -- the baseline asserts w = 60 s.
    """
    m = tab[(tab["abs_scale"] > 0) & (tab["tte"] > 0)]
    tau = m["tte"].to_numpy(dtype="float64")
    y = np.log(m["abs_scale"].to_numpy())
    w = m["n"].to_numpy(dtype="float64")

    def tau_eff(t, win):
        return np.where(t >= win, np.maximum(t - 2.0 * win / 3.0, 1e-9),
                        t ** 3 / (3.0 * win * win))

    best = None
    for win in np.concatenate([np.linspace(0.5, 60.0, 240),
                               np.linspace(60.0, 600.0, 109)]):
        pred = 0.5 * np.log(tau_eff(tau, win))
        a = float((w * (y - pred)).sum() / w.sum())
        sse = float((w * (y - pred - a) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(win), a)
    return {"window_s": best[1], "k": float(math.exp(best[2])), "sse": best[0]}


def main():
    global PANEL
    for arg in sys.argv[1:]:
        if arg.startswith("--panel="):
            PANEL = arg.split("=", 1)[1]
    if PANEL not in PANELS:
        raise SystemExit(f"--panel must be one of {sorted(PANELS)}")
    spot_path, fit_json, _, plot_path = _paths()
    FIT_DAYS, TEST_DAYS = _days()
    print(f"panel: {PANEL}  ({os.path.basename(spot_path)})")
    print(f"split: fit {list(FIT_DAYS)}  ->  test {list(TEST_DAYS)}")

    rebuild = "--rebuild" in sys.argv
    obs = observations(rebuild=rebuild)
    obs = obs[np.isfinite(obs["r"])]

    fit_obs = obs[obs["day"].isin(FIT_DAYS)]
    test_obs = obs[obs["day"].isin(TEST_DAYS)]
    fit_tab = term_structure(fit_obs)
    test_tab = term_structure(test_obs)

    print(f"fit rows {len(fit_obs)} over {fit_obs.market_id.nunique()} markets; "
          f"test rows {len(test_obs)} over {test_obs.market_id.nunique()} markets")

    print("\n-- model sigma vs realised scale, FIT days --")
    print("  tte     n   model_sigma  realised_scale  realised_sd  model/realised")
    for _, r in fit_tab.iterrows():
        print(f"{r.tte:6.1f} {int(r.n):6d}   {r.sigma_model:10.6f}  "
              f"{r.abs_scale:13.6f}  {r.sd:11.6f}  {r.ratio:13.3f}")

    pl = power_law(fit_tab)
    fl = flat_law(fit_tab)
    fw = free_window(fit_tab)
    print(f"\nfree power law   : sigma = {pl['k']:.6g} * tau^{pl['b']:.4f}")
    print(f"pinned (flat)    : sigma = {fl['k']:.6g} * tau^0.5")
    print(f"free tau_eff win : w = {fw['window_s']:.2f} s "
          f"(baseline asserts 60.0), k = {fw['k']:.6g}")

    # Does the baseline's EWMA carry any conditional information at all? If
    # dropping it for an unconditional table costs nothing, say so out loud.
    cond = fit_obs[np.isfinite(fit_obs["sigma_model"])].copy()
    cond["rel"] = cond["r"].abs() / cond.groupby("tte")["r"].transform(
        lambda x: x.abs().mean())
    cond["rel_sig"] = (cond["sigma_model"]
                       / cond.groupby("tte")["sigma_model"].transform("mean"))
    x = np.log(cond["rel_sig"].clip(lower=1e-12).to_numpy())
    y = np.log(cond["rel"].clip(lower=1e-12).to_numpy())
    keep = np.isfinite(x) & np.isfinite(y)
    rho = float(np.corrcoef(x[keep], y[keep])[0, 1])
    print(f"EWMA conditional information: corr(log rel sigma, log rel |r|) "
          f"= {rho:.4f} on {int(keep.sum())} fit rows")

    # The bias the vol block cannot fix, stated in the same units.
    bias = fit_obs.groupby("tte")["r"].mean()
    frac_neg = fit_obs.groupby("tte")["r"].apply(lambda v: float((v < 0).mean()))
    print(f"forecast BIAS mean ln(A/s): {bias.min():+.6f} .. {bias.max():+.6f} "
          f"across tau; fraction negative {frac_neg.min():.3f} .. "
          f"{frac_neg.max():.3f}")

    # Is the calendar split finally clean? On the uncorrected panel the
    # short-tau bias was 3.6x larger on the fit half than the test half, which
    # made any fit/test comparison a comparison of two data defects.
    near = fit_obs[fit_obs["tte"] <= 5.0]["r"]
    near_t = test_obs[test_obs["tte"] <= 5.0]["r"]
    bias_fit, bias_test = float(near.mean()), float(near_t.mean())
    ratio = bias_fit / bias_test if bias_test else float("nan")
    se_fit = float(near.std(ddof=1) / math.sqrt(len(near)))
    se_test = float(near_t.std(ddof=1) / math.sqrt(len(near_t)))
    print(f"short-tau (tte<=5s) bias  fit {bias_fit*1e4:+.3f} bp "
          f"(se {se_fit*1e4:.3f})  test {bias_test*1e4:+.3f} bp "
          f"(se {se_test*1e4:.3f})  ratio {ratio:.3f}")

    # THE VOLATILITY REGIME, which on this window dominates the split. Realised
    # movement rises monotonically across 2026-08-17..21, so ANY calendar cut
    # puts the quiet days in the fit half and the loud ones in the test half,
    # and an UNCONDITIONAL sigma table carries none of that across. This is
    # not a defect in the fit; it is the largest single caveat on it, and it
    # is printed and recorded so no reader has to rediscover it.
    def _scale(df, tau):
        v = df[np.isclose(df["tte"], tau)]["r"].to_numpy()
        return (math.sqrt(math.pi / 2) * np.abs(v).mean()) if len(v) else float("nan")

    regime = {"by_day_scale_300s_bp": {}, }
    print("\nrealised scale by day (bp)   300 s      30 s       5 s")
    for day in sorted(obs["day"].unique()):
        d = obs[obs["day"] == day]
        s300, s30, s5 = _scale(d, 300.0), _scale(d, 30.0), _scale(d, 5.0)
        regime["by_day_scale_300s_bp"][str(day)] = float(s300 * 1e4)
        print(f"  {day}  n={d.market_id.nunique():4d}   "
              f"{s300*1e4:8.3f}  {s30*1e4:8.3f}  {s5*1e4:8.3f}")
    f300, t300 = _scale(fit_obs, 300.0), _scale(test_obs, 300.0)
    regime.update({"fit_scale_300s_bp": float(f300 * 1e4),
                   "test_scale_300s_bp": float(t300 * 1e4),
                   "test_over_fit_300s": float(t300 / f300)})
    print(f"  FIT half {f300*1e4:.3f} bp vs TEST half {t300*1e4:.3f} bp at "
          f"300 s -- the test half is {t300/f300:.2f}x more volatile, so an "
          f"unconditional\n  table fitted here is that much too small there.")

    fit = {
        "generated_by": "fit_sigma.py",
        "panel": PANEL,
        "spot_path": spot_path,
        "warmup_s": WARMUP_S,
        "basis_halflife_s": float(
            provenance.load_slot(provenance.resolve_slots(HERE)["fair"],
                                 "fair").BASIS_HALFLIFE_S),
        "bias_short_tau_fit": bias_fit,
        "bias_short_tau_test": bias_test,
        "bias_short_tau_se_fit": se_fit,
        "bias_short_tau_se_test": se_test,
        "bias_fit_over_test": ratio,
        "vol_regime": regime,
        "fit_days": list(FIT_DAYS),
        "test_days": list(TEST_DAYS),
        "estimator": "sqrt(pi/2) * mean|ln(settle / fair_s)| per tte bucket",
        "n_fit_rows": int(len(fit_obs)),
        "n_fit_markets": int(fit_obs.market_id.nunique()),
        "table": {"tte": [float(t) for t in fit_tab["tte"]],
                  "sigma": [float(v) for v in fit_tab["abs_scale"]],
                  "n": [int(v) for v in fit_tab["n"]]},
        "power_law": pl,
        "flat": fl,
        "free_tau_eff_window": fw,
        "ewma_conditional_corr": rho,
        "bias_mean_log_by_tte": {str(float(t)): float(v)
                                 for t, v in bias.items()},
        "bias_frac_negative_by_tte": {str(float(t)): float(v)
                                      for t, v in frac_neg.items()},
    }
    with open(fit_json, "w") as fh:
        json.dump(fit, fh, indent=2)
    print(f"\nwrote {fit_json}")

    plot_sigma(fit_tab, test_tab, pl, fl, plot_path)
    print(f"wrote {os.path.basename(plot_path)}")


def plot_sigma(fit_tab, test_tab, pl, fl, path):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    tau = np.asarray(fit_tab["tte"], dtype="float64")
    ax[0].plot(tau, fit_tab["abs_scale"], "o-", ms=3.5, color="#1b6ca8",
               label="realised scale, fit days")
    ax[0].plot(test_tab["tte"], test_tab["abs_scale"], "s--", ms=3.0,
               color="#7fb2d6", label="realised scale, test days")
    ax[0].plot(tau, fit_tab["sd"], ":", color="#0b3c5d", lw=1.2,
               label="realised sd, fit days")
    ax[0].plot(tau, fit_tab["sigma_model"], "o-", ms=3.5, color="#c0392b",
               label="model sigma (vol_baseline.py)")
    ax[0].plot(tau, pl["k"] * tau ** pl["b"], "-", color="#27853f", lw=1.4,
               label=f"calibrated power law, tau^{pl['b']:.2f}")
    ax[0].plot(tau, fl["k"] * tau ** 0.5, "-", color="#e08a1e", lw=1.4,
               label="flat control, tau^0.50")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("time to expiry tau (s)")
    ax[0].set_ylabel("sigma_T  (sd of ln(A_T / s_t))")
    ax[0].set_title(f"Model sigma against realised movement ({PANEL} panel)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7.5, loc="upper left")

    ax[1].axhline(1.0, color="k", lw=1.0)
    ax[1].plot(tau, fit_tab["ratio"], "o-", ms=3.5, color="#c0392b",
               label="fit days")
    ax[1].plot(test_tab["tte"], test_tab["ratio"], "s--", ms=3.0,
               color="#e59a93", label="test days")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("time to expiry tau (s)")
    ax[1].set_ylabel("model sigma / realised scale")
    ax[1].set_title("Ratio: below 1 means the model is overconfident")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
