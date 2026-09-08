"""Build the fair export one variant reads: strike-free, one row per market second.

`s` is the settlement level at which the model is indifferent, in USD, which is the
harness's `E[A]`; `sigma` is the settlement standard deviation in the same units.
Because y* is affine in the strike, those two plus the tail parameters are the whole
model - the strike work happens in `f.py` and `link.py`, which is exactly the split
the harness's slot contract asks for.

    python -m export.build_export --variant baseline

Every market gets exactly `len(T_S_RANGE)` == 301 rows, priceable or not. `evaluate`
filters its expiries through a `keep` mask (`it > 0`, `iT < win.n`, `iO >= 0`) BEFORE
`ok` is computed, so `emit_all=True` still returns a variable-length row set - the
returned rows are a SUBSET of the markets passed in, and which markets survive
differs with `t_s` (because `it = iT - n` moves with it). To keep every market/second
cell present, each `t_s`'s returned rows are scattered into full-length arrays
(indexed by the complete market list) via `np.searchsorted`, with the unpriced
positions left as NaN / `ok=False` rather than dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from fvmodel import config                                  # noqa: E402
from fvmodel.base import load_params                         # noqa: E402
from fvmodel.engine import Window, evaluate                  # noqa: E402
from fvmodel.overrides import quoted_prob                    # noqa: E402
from export.cache import register_bank                       # noqa: E402

MARKET_LEN = 300           # the 5 m window
TWAP_LEN = 60              # the Chainlink settlement average
BUCKET_MS = 100
# t_s = -1 fills bucket 0; t_s = 299 fills bucket 2999. See usable_t_ms.
T_S_RANGE = list(range(-1, MARKET_LEN))
FAIR_DIR = INV.parents[1] / "backtesting_5m" / "data" / "fair"

# columns that come out of `evaluate` and are scattered as float64, NaN where the
# market/second cell was not priceable (or was outside `evaluate`'s `keep` mask)
_FLOAT_COLS = ("s", "sigma", "nu", "mu", "sigma_t", "p_model", "p_quoted", "omega",
               "n_known", "n_transit", "m_Y", "eps_bar", "carry", "var_eps",
               "var_basis", "z_business", "p_ref")


def usable_t_ms(t_s: int) -> int:
    """The first 100 ms bucket at which a row carrying information through the
    instant `open_ts + t_s` may be used.

    The harness's convention (its spec section 1.4) is that a decision at bucket i
    may only use information from strictly before it. The model's state at instant
    `t` is the price at `t`, so a value becomes usable one 100 ms bucket later and
    holds until the next second's value supersedes it. `t_s = -1` is the pre-open row
    that serves bucket 0.
    """
    return max(0, int(t_s) * 1000 + 100)


def bucket_owner() -> np.ndarray:
    """For each of the 3,000 buckets, the `t_s` whose value it must use.

    The inverse of `usable_t_ms`, materialised once. Every block reads the export
    through this, so the causality shift is defined in exactly one place.
    """
    i = np.arange(MARKET_LEN * 1000 // BUCKET_MS, dtype=np.int64)
    return (i * BUCKET_MS - 100) // 1000


_WINDOWS: dict = {}

# `evaluate`'s forward_block looks ahead from the quote second all the way to the
# expiry second plus one (curve.py's dT_at indexes bv.B[i + u + 1]), so a market
# whose expiry lands on the window's very last second is one row short - market_grid
# itself avoids this by stopping two seconds before win.ts[-1], but the strikes-file
# expiries used here are not filtered that way. A few seconds of tail margin on the
# Window (which is not itself a market boundary - no market's `T` sits inside it)
# gives every real expiry the headroom `evaluate` needs.
_TAIL_MARGIN_S = 5


def _window(model, t0: int, t1: int, burn: int):
    """One Window per (span, filter constants), shared across variant runs.

    Building a Window sweeps the seasonality, the activity factor and the whole print
    model over a month of seconds. Only `w_spot` and the basis half-lives change any
    of that, so every other variant reuses the same object and pays for it once.
    """
    fp = model.fp
    key = (t0, t1, burn, fp.w_spot, fp.tau_s, fp.delta_s, fp.basis_hl_s,
           fp.basis_hl_alt_s, fp.lag_s)
    if key not in _WINDOWS:
        _WINDOWS.clear()                   # one at a time; a Window is large
        _WINDOWS[key] = Window(load_params(), fp, t0 - burn, t1 + _TAIL_MARGIN_S,
                               warm=0, chainlink=True,
                               log=lambda *a: print("   ", *a, flush=True))
    return _WINDOWS[key]


def _strikes(t0: int, t1: int) -> pl.DataFrame:
    from harness_paths import STRIKES

    return (pl.read_parquet(STRIKES)
            .filter((pl.col("open_ts") >= t0) & (pl.col("open_ts") + MARKET_LEN <= t1))
            .sort("open_ts"))


def _scatter_float(n: int, pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    out[pos] = vals
    return out


def _scatter_bool(n: int, pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[pos] = vals
    return out


def build(variant: str, t0: int, t1: int, out_path: Path | None = None) -> Path:
    loaded = config.load(variant)
    model, ov = loaded.model, loaded.overrides
    print("variant %s: %s" % (variant, ov.label()), flush=True)

    mk = _strikes(t0, t1)
    T = (mk["open_ts"].to_numpy() + MARKET_LEN).astype(np.int64)
    ids = mk["market_id"].to_numpy()
    opens = mk["open_ts"].to_numpy().astype(np.int64)
    strikes_venue = mk["strike"].to_numpy().astype(np.float64)
    n_mk = T.size

    burn = loaded.cfg["window"]["burn_days"] * 86400
    win = _window(model, t0, t1, burn)
    iT = win.i(T)
    idx = np.unique(np.concatenate([iT - (MARKET_LEN - t) for t in T_S_RANGE]))
    idx = idx[(idx > 0) & (idx < win.n)]
    register_bank(win, idx, loaded.cfg["base"]["params_sha256"])

    frames = []
    for t_s in T_S_RANGE:
        n = MARKET_LEN - t_s                # seconds remaining at the info instant
        cell = evaluate(win, model, "chainlink_twap60", MARKET_LEN, n,
                        expiries=T, book=None, emit_all=True)
        r = cell.rows
        # r["T"] is a SUBSET of T (evaluate's `keep` mask runs before `ok`), so the
        # returned rows are scattered back onto the full market list by position -
        # markets `evaluate` dropped for this t_s get NaN / ok=False, not a hole.
        pos = np.searchsorted(T, r["T"])
        s = r["strike"] - r["y_star"] * r["p_ref"] * r["omega"]
        sigma = np.sqrt(r["var_y"]) * r["p_ref"] * r["omega"]
        # `p_model`/`p_quoted` must answer "what does the model say about the venue's
        # own strike" (the spec: "at the market's real strike, for the scorecard"),
        # not about `evaluate`'s internal ATM linearisation anchor (`lvl[iO]`) - the
        # two disagree by real dollars (median ~$3.57 venue vs instantaneous level;
        # Ruling 15). `s` is exactly strike-free (test_export.py), so the venue's own
        # y* is recovered by the same affine relationship `evaluate` used internally,
        # just evaluated at K = strikes_venue instead of at `r["strike"]`. The tail
        # object (not the t.cdf((z+mu)/sigma_t) form `f`/`link` use) does the
        # recentring, so this stays an independent computation of the same quantity -
        # the harness blocks and this export path must not share one formula, or a
        # test that block output equals this column would be circular.
        K = strikes_venue[pos]
        y_star_venue = (K - s) / (r["p_ref"] * r["omega"])
        p_model = model.tail_for("chainlink_twap60").prob_up(
            y_star_venue, r["var_y"], r["z"])
        p_quoted = np.asarray(quoted_prob(model.ov, p_model), dtype=np.float64)
        values = {
            "s": s, "sigma": sigma, "nu": r["nu"], "mu": r["mu"],
            "sigma_t": r["sigma_t"], "p_model": p_model,
            "p_quoted": p_quoted, "omega": r["omega"],
            "n_known": r["n_known"], "n_transit": r["n_transit"],
            "m_Y": r["m_Y"], "eps_bar": r["eps_bar"], "carry": r["carry"],
            "var_eps": r["var_eps"], "var_basis": r["var_basis"],
            "z_business": r["z"], "p_ref": r["p_ref"],
        }
        cols = {name: _scatter_float(n_mk, pos, np.asarray(vals, dtype=np.float64))
                for name, vals in values.items()}
        cols["ok"] = _scatter_bool(n_mk, pos, r["ok"])
        cols["market_id"] = ids
        cols["open_ts"] = opens
        cols["t_s"] = np.full(n_mk, t_s, dtype=np.int32)
        frames.append(pl.DataFrame(cols))
    df = pl.concat(frames).sort(["open_ts", "t_s"])

    out = Path(out_path) if out_path else FAIR_DIR / ("%s.parquet" % variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({"provenance": loaded.provenance,
                                "window": {"t0": t0, "t1": t1},
                                "rows": len(df),
                                "markets": int(df["market_id"].n_unique()),
                                "ok_fraction": float(df["ok"].mean())}, indent=2,
                               default=str) + "\n")
    print("wrote %s: %d rows, %d markets, ok %.4f"
          % (out, len(df), df["market_id"].n_unique(), df["ok"].mean()))
    return out


def main() -> None:
    cfg = config.read_config()["window"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--t0", type=int, default=cfg["backtest_t0"])
    ap.add_argument("--t1", type=int, default=cfg["backtest_t1"])
    a = ap.parse_args()
    build(a.variant, a.t0, a.t1)


if __name__ == "__main__":
    main()
