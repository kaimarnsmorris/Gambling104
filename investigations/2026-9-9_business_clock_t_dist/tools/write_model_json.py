"""Generate model.json with the current table hashes. Re-run after changing a table,
and bump `version` and CHANGELOG.md when you do.

    python tools/write_model_json.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "print_model": ("scripts/10_print_model.py", "report_final.html section 4",
                    "filter constants, basis tracker, eps model, reconstruction"),
    "kernels": ("scripts/11_kernels.py", "report_final.html section 6",
                "activity-conditioned return autocorrelation, input variance ratio"),
    "alpha": ("scripts/12_alpha.py", "report_final.html section 10",
              "beta(dT) in bp with the quote-age interaction"),
    "tails": ("scripts/13_eval.py", "report_final.html section 11",
              "settlement tail (nu, mu, sigma) by log business time, per kind"),
    "xi_cap": ("scripts/22_fit_xi_cap.py", "report_final.html section 11.2",
               "short-business-time forward-curve cap; OFF by default"),
    "filter_by_w": ("tools/extract_filter_by_w.py", "spec ruling R11",
                    "best (tau, delta, p_late) per blend weight, coarse grid"),
}
DEFAULT_OVERRIDES = {
    "version": "ov-1.0.0",
    "source": "consolidation brief 2026-09-09",
    "kappa_vol": 0.0, "kappa_vol_short": 0.0,
    "shrink_w": 0.0, "shrink_decay_s": 60.0,
    "rho_kernel": "conditional",
    "eps_scale": 1.0, "eps_condition": True,
    "basis_tracker": "main", "information_set": "full",
    "reconstruct_in_transit": True,
    "kappa_tail": 0.0, "tail_scale": 1.0, "tail_family": "t",
    "alpha_scale": 1.0, "alpha_scale_age": 1.0, "alpha_cap_sd": None,
    "w_spot": None, "xi_cap_c": None, "temperature": 1.0,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    params = ROOT / "tables" / "params_v2_1_0.json"
    cfg = {
        "version": "fv-1.0.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/write_model_json.py",
        "source": ("btc_volatility_clock_chainlink, report_final.html; "
                   "vendored 2026-09-09, not refitted here"),
        "window": {"backtest_t0": 1786665600, "backtest_t1": 1788220799,
                   "select_t1": 1787702399, "holdout_t0": 1787702400,
                   "burn_days": 14,
                   "note": ("2026-08-14 .. 2026-08-31. The panel runs to 09-08 but "
                            "the perp/spot 1 s inputs end 08-31, so 09-01 onward is "
                            "unpriceable. See spec section 1.1.")},
        "base": {"params": "tables/params_v2_1_0.json", "params_version": "2.1.0",
                 "params_sha256": sha(params)},
        "tables": {
            name: {"file": "tables/%s.json" % name,
                   "sha256": sha(ROOT / "tables" / ("%s.json" % name)),
                   "fitted_by": by, "report": rep, "contents": what}
            for name, (by, rep, what) in TABLES.items()},
        "defaults": {"eps_fit": "train", "input_mode": "blend"},
        "overrides": DEFAULT_OVERRIDES,
        "data": {
            "note": ("Large 1 s inputs are not vendored. Override the root with the "
                     "FV_SOURCE_ROOT environment variable."),
            "source_root": (r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
                            r"\btc_volatility_clock_chainlink"),
            "chainlink_1s": "data/chainlink/chainlink_1s.parquet",
            "perp_1s": "../btc_volatility_clock/output/data/btcusdt_1s.parquet",
            "spot_1s_glob": "cache/spot_1s_*.parquet"},
    }
    (ROOT / "model.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote model.json version %s with %d tables"
          % (cfg["version"], len(cfg["tables"])))


if __name__ == "__main__":
    main()
