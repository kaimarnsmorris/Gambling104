"""The consolidation must not change a single number.

Task 1 froze the pre-consolidation model's answers on a fixed set of quotes. This
prices the same quotes through the vendored, consolidated model and requires exact
equality - not approximate. Anything else means the consolidation changed the model,
which is the one thing it is not allowed to do.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

GOLDEN = "tests/golden/fv_golden.npz"
FIELDS = ("p_up", "var_y", "y_star", "m_Y", "eps_bar", "var_eps", "var_basis",
          "carry", "omega", "known_value", "z", "p_ref")


@pytest.fixture(scope="module")
def replay():
    from pathlib import Path

    from fvmodel.base import ROOT, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value

    g = np.load(Path(ROOT) / GOLDEN)
    model = build_model()
    win = Window(load_params(), model.fp, int(g["window_t0"]), int(g["window_t1"]),
                 warm=0, chainlink=True, log=lambda *a: None)
    its = np.unique(g["it"])
    win.prepare(np.concatenate([its, its - 2]))

    got = {k: np.empty(g["p_up"].size) for k in FIELDS}
    got.update({k: np.empty(g["p_up"].size) for k in ("nu", "mu", "sigma_t")})
    states = {}
    for r in range(g["p_up"].size):
        it = int(g["it"][r])
        if it not in states:
            st = state_at(win, it, model)
            st.book = {"imbalance": 0.8, "px_age_s": 1.4}
            states[it] = st
        st = states[it]
        t = int(win.ts[it])
        n = int(g["n"][r])
        kind = str(g["kind"][r])
        L = int(g["L"][r])
        mk = market_at(win, kind, t + n, n, L, 60)
        mk.book_snapshot = {"imbalance": 0.8, "px_age_s": 1.4}
        fv = fair_value(st, mk, model, t_now=t)
        for k in FIELDS:
            got[k][r] = float(getattr(fv, k))
        got["nu"][r], got["mu"][r], got["sigma_t"][r] = (float(x) for x in fv.tail)
    return g, got


@pytest.mark.parametrize("field", FIELDS + ("nu", "mu", "sigma_t"))
def test_bit_identical_to_pre_consolidation(replay, field):
    g, got = replay
    want = g[field]
    bad = ~(got[field] == want)
    assert not bad.any(), (
        "%s differs on %d of %d rows; worst |delta| = %.3g at row %d "
        "(golden %.17g, got %.17g)"
        % (field, bad.sum(), want.size, np.max(np.abs(got[field] - want)),
           int(np.argmax(np.abs(got[field] - want))),
           want[np.argmax(np.abs(got[field] - want))],
           got[field][np.argmax(np.abs(got[field] - want))]))
