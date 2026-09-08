"""Freeze the pre-consolidation model's outputs, so the consolidation can be proved
bit-identical. Runs against the SOURCE repo, never against the vendored copy.

    python tools/capture_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
           r"\btc_volatility_clock_chainlink")
VC = SRC.parent / "btc_volatility_clock"
for p in (str(SRC), str(VC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fv.base import CL_T0, load_params            # noqa: E402
from fv.build import build_model                  # noqa: E402
from fv.engine import Window, market_at, state_at  # noqa: E402
from fv.fairvalue import Switches, fair_value     # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "fv_golden.npz"
WIN_T0 = CL_T0 + 20 * 86400
WIN_T1 = WIN_T0 + 3 * 86400
# (kind, market length L, seconds remaining n) - spans the composed-weight regime
# (n < 60), the crossover, and the long horizons where the head integral dominates
CASES = [("chainlink_twap60", 300, n) for n in (300, 120, 60, 30, 10, 5, 2, 1)]
CASES += [("chainlink_twap60", 900, n) for n in (900, 300, 60, 5)]
CASES += [("perp_twap", 900, n) for n in (900, 120, 30)]
CASES += [("perp_single", 900, n) for n in (900, 300, 10, 1)]
BOOK = {"imbalance": 0.8, "px_age_s": 1.4}


def main() -> None:
    model = build_model()
    win = Window(load_params(), model.fp, WIN_T0, WIN_T1, warm=0, chainlink=True,
                 log=lambda *a: None)
    # eight quote origins spread across the window, all past the register burn-in
    its = np.linspace(win.n - 200_000, win.n - 5_000, 8).astype(np.int64)
    win.prepare(np.concatenate([its, its - 2]))

    cols = {k: [] for k in ("kind", "n", "L", "it", "p_up", "var_y", "y_star", "m_Y",
                            "eps_bar", "var_eps", "var_basis", "carry", "omega",
                            "known_value", "nu", "mu", "sigma_t", "z", "p_ref")}
    for it in its:
        state = state_at(win, int(it), model)
        state.book = dict(BOOK)
        t = int(win.ts[it])
        for kind, L, n in CASES:
            mk = market_at(win, kind, t + n, n, L, 60)
            if not np.isfinite(mk.strike):
                continue
            # market_at() in SRC hardcodes book_snapshot=None; fair_value() reads
            # market.book_snapshot (not state.book, which is dead w.r.t. m_Y), so the
            # alpha path never fires unless we attach it here.
            mk.book_snapshot = dict(BOOK)
            fv = fair_value(state, mk, model, Switches(), t_now=t)
            cols["kind"].append(kind)
            cols["n"].append(n)
            cols["L"].append(L)
            cols["it"].append(int(it))
            for k in ("p_up", "var_y", "y_star", "m_Y", "eps_bar", "var_eps",
                      "var_basis", "carry", "omega", "known_value", "z", "p_ref"):
                cols[k].append(float(getattr(fv, k)))
            cols["nu"].append(float(fv.tail[0]))
            cols["mu"].append(float(fv.tail[1]))
            cols["sigma_t"].append(float(fv.tail[2]))

    assert len(cols["p_up"]) >= 100, "too few golden rows: %d" % len(cols["p_up"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, window_t0=WIN_T0, window_t1=WIN_T1,
             kind=np.array(cols.pop("kind")),
             **{k: np.asarray(v) for k, v in cols.items()})
    print("wrote %s: %d rows" % (OUT, len(cols["p_up"])))


if __name__ == "__main__":
    main()
