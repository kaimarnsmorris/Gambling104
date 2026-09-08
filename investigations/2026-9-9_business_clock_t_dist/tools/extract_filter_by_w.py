"""Reduce the filter grid-search surface to one best row per blend weight.

`fit_filter` searches (w, tau, delta, p_late) and `print_model.json` keeps only the
winner, but the `w_spot` override needs the constants at every w. The full surface
survives in the source repo's cache/print_model.npz as `surface_train` (17,238 rows
of w, tau, delta, p_late, rmse). This takes the argmin per w and writes the 13-row
table, which is small enough to commit and read.

    python tools/extract_filter_by_w.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get(
    "FV_SOURCE_ROOT",
    r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
    r"\btc_volatility_clock_chainlink"))
OUT = Path(__file__).resolve().parents[1] / "tables" / "filter_by_w.json"


def main() -> None:
    s = np.load(SRC / "cache" / "print_model.npz")["surface_train"].astype(np.float64)
    rows = []
    for w in np.unique(s[:, 0]):
        r = s[s[:, 0] == w]
        b = r[int(np.argmin(r[:, 4]))]
        rows.append([round(float(b[0]), 4), float(b[1]), float(b[2]),
                     float(b[3]), float(b[4])])
    rows.sort(key=lambda x: x[0])
    OUT.write_text(json.dumps({
        "source": "cache/print_model.npz:surface_train (coarse grid), argmin rmse per w",
        "cols": ["w_spot", "tau_s", "delta_s", "p_stamp_late", "rmse_bp"],
        "note": ("This is the COARSE grid. At the shipped w = 0.6 it gives "
                 "tau = 0.7872, not the fine-refined shipped tau = 0.8447, so "
                 "w_spot: 0.6 is NOT the baseline model. See spec ruling R11."),
        "rows": rows}, indent=2) + "\n")
    print("wrote %s: %d rows, w from %.2f to %.2f"
          % (OUT, len(rows), rows[0][0], rows[-1][0]))


if __name__ == "__main__":
    main()
