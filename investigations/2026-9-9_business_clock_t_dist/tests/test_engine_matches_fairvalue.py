"""The batch evaluation and the production call must be the same model.

Two implementations of the same arithmetic drift the moment either is touched, and
the drift is invisible: the export would keep producing plausible numbers for a model
the harness does not run. So the same quotes are priced both ways, under EVERY
override, and the answers have to agree to floating point.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# one representative value per override key, chosen to actually bite
OVERRIDE_CASES = [
    {},
    {"kappa_vol": 0.15},
    {"kappa_vol_short": -0.22},
    {"shrink_w": 0.5, "shrink_decay_s": 60.0},
    {"rho_kernel": "unconditional"},
    {"rho_kernel": "off"},
    {"eps_scale": 2.0},
    {"eps_scale": 0.0},
    {"eps_condition": False},
    {"basis_tracker": "alt"},
    {"basis_tracker": "off"},
    {"information_set": "prints_only"},
    {"reconstruct_in_transit": False},
    {"kappa_tail": -0.5},
    {"tail_scale": 1.2},
    {"tail_family": "normal"},
    {"alpha_scale": 0.0},
    {"alpha_scale": 2.0},
    {"alpha_scale_age": 0.0},
    {"alpha_cap_sd": 0.5},
    {"w_spot": 0.35},
    {"xi_cap_c": 2.0},
    {"temperature": 1.5},
]
CELLS = [("chainlink_twap60", 300, 300), ("chainlink_twap60", 300, 60),
         ("chainlink_twap60", 300, 5), ("perp_single", 900, 300),
         ("perp_twap", 900, 120)]


@pytest.fixture(scope="module")
def win_base():
    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window

    base = build_model()
    t0 = CL_T0 + 20 * 86400
    win = Window(load_params(), base.fp, t0, t0 + 3 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    return win, base


@pytest.mark.parametrize("ovkw", OVERRIDE_CASES,
                         ids=[",".join(d) or "defaults" for d in OVERRIDE_CASES])
@pytest.mark.parametrize("kind,L,n", CELLS)
def test_batch_equals_single_quote(win_base, ovkw, kind, L, n):
    from fvmodel.engine import evaluate, market_at, market_grid, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import Overrides, apply_overrides

    win, base = win_base
    ov = Overrides(**ovkw)
    # w_spot changes the blend, so the window's print-model series must be rebuilt
    if ovkw.get("w_spot") is not None:
        pytest.skip("w_spot needs its own Window; covered by test_w_spot_rebuild")
    # prints_only changes the INFORMATION SET the batch path constructs a state
    # from (it quotes off a print received LAG_S seconds ago, with the matching
    # lag on the variance window) - not the pricing arithmetic. `evaluate` can
    # build that lagged state because it holds the whole window; `fair_value` is
    # handed a state and structurally cannot re-derive an earlier one. The two
    # paths are therefore not comparable on this override by construction; see
    # test_prints_only_lags_the_quote for what it actually does.
    if ovkw.get("information_set") == "prints_only":
        pytest.skip("prints_only changes the batch path's state construction, not "
                    "the arithmetic; covered by test_prints_only_lags_the_quote")
    model = apply_overrides(base, ov)

    T = market_grid(win, L, burn_days=2)
    assert T.size > 20
    win.prepare(np.unique(np.concatenate([win.i(T) - n, win.i(T) - n - 2])))
    cell = evaluate(win, model, kind, L, n, expiries=T)
    assert len(cell) > 10

    picked = 0
    for k in range(0, len(cell), max(len(cell) // 6, 1)):
        Tk = int(cell.rows["T"][k])
        it = win.i(Tk) - n
        if it < 64:
            continue
        st = state_at(win, it, model)
        mk = market_at(win, kind, Tk, n, L, 60)
        one = fair_value(st, mk, model, t_now=int(win.ts[it]))
        for field, tol in (("var_y", 1e-8), ("y_star", 1e-7), ("omega", 0),
                           ("carry", 1e-7), ("eps_bar", 1e-8), ("var_eps", 1e-8),
                           ("p_model", 1e-8), ("p_quoted", 1e-8)):
            a = getattr(one, field)
            b = float(cell.rows[field][k])
            assert a == pytest.approx(b, rel=tol, abs=1e-14), (
                "%s under %s at kind=%s n=%d: single quote %.12g vs batch %.12g"
                % (field, ov.label(), kind, n, a, b))
        picked += 1
    assert picked >= 3


def test_w_spot_rebuild(win_base):
    """w_spot changes the input blend, so the batch path needs its own Window."""
    from fvmodel.base import load_params
    from fvmodel.engine import Window, evaluate, market_at, market_grid, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import Overrides, apply_overrides

    win0, base = win_base
    model = apply_overrides(base, Overrides(w_spot=0.35))
    win = Window(load_params(), model.fp, int(win0.ts[0]), int(win0.ts[-1]),
                 warm=0, chainlink=True, log=lambda *a: None)
    T = market_grid(win, 300, burn_days=2)
    win.prepare(np.unique(np.concatenate([win.i(T) - 60, win.i(T) - 62])))
    cell = evaluate(win, model, "chainlink_twap60", 300, 60, expiries=T)
    k = len(cell) // 2
    Tk = int(cell.rows["T"][k])
    it = win.i(Tk) - 60
    one = fair_value(state_at(win, it, model),
                     market_at(win, "chainlink_twap60", Tk, 60, 300, 60),
                     model, t_now=int(win.ts[it]))
    assert one.var_y == pytest.approx(float(cell.rows["var_y"][k]), rel=1e-8)
    assert one.y_star == pytest.approx(float(cell.rows["y_star"][k]), rel=1e-7)


def test_prints_only_lags_the_quote(win_base):
    """`information_set="prints_only"` is the honest no-exchange-feed counterparty:
    it quotes off the print it received LAG_S seconds ago, with the matching lag on
    the variance window, and it never carries a deterministic drift term because it
    has nothing beyond the received prints to extrapolate from. This is the lag edge
    the shipped `market_observable` variant exists to measure (spec ruling 12), so a
    restored behaviour needs a test pinning it or it will be deleted again the next
    time someone notices the scalar path can't reproduce it."""
    from fvmodel.engine import evaluate, market_grid
    from fvmodel.overrides import Overrides, apply_overrides

    win, base = win_base
    kind, L, n = "chainlink_twap60", 300, 60
    T = market_grid(win, L, burn_days=2)
    win.prepare(np.unique(np.concatenate([win.i(T) - n, win.i(T) - n - 2,
                                          win.i(T) - n - 2 - 2])))

    default = evaluate(win, apply_overrides(base, Overrides()), kind, L, n, expiries=T)
    lagged = evaluate(win, apply_overrides(base, Overrides(information_set="prints_only")),
                      kind, L, n, expiries=T)
    common = np.intersect1d(default.rows["T"], lagged.rows["T"])
    assert common.size > 10
    i0 = np.searchsorted(default.rows["T"], common)
    i1 = np.searchsorted(lagged.rows["T"], common)

    assert not np.allclose(default.rows["y_star"][i0], lagged.rows["y_star"][i1])
    assert not np.allclose(default.rows["var_y"][i0], lagged.rows["var_y"][i1])
    assert np.all(lagged.rows["carry"][i1] == 0.0)


def test_basis_tracker_alt_switches_the_residual_too(win_base):
    """`d_lvl = b + eps` by construction, so selecting `basis_tracker="alt"` for the
    level must select the matching residual `eps_alt`, or the two trackers' gap gets
    double-counted (spec ruling 13). This must fail before that fix: with only the
    level switched, `eps_bar` is unchanged from the default run because both paths
    would still be reading the main tracker's residual."""
    from fvmodel.engine import evaluate, market_grid
    from fvmodel.overrides import Overrides, apply_overrides

    win, base = win_base
    kind, L, n = "chainlink_twap60", 300, 60
    T = market_grid(win, L, burn_days=2)
    win.prepare(np.unique(np.concatenate([win.i(T) - n, win.i(T) - n - 2])))

    default = evaluate(win, apply_overrides(base, Overrides()), kind, L, n, expiries=T)
    alt = evaluate(win, apply_overrides(base, Overrides(basis_tracker="alt")),
                   kind, L, n, expiries=T)
    common = np.intersect1d(default.rows["T"], alt.rows["T"])
    assert common.size > 10
    i0 = np.searchsorted(default.rows["T"], common)
    i1 = np.searchsorted(alt.rows["T"], common)

    assert not np.allclose(default.rows["eps_bar"][i0], alt.rows["eps_bar"][i1])


def test_emit_all_keeps_unpriceable_rows(win_base):
    """Task 7's export builder needs every market second, priceable or not, so
    `emit_all=True` must keep the rows the `ok` mask would otherwise drop."""
    from fvmodel.engine import evaluate, market_grid

    win, model = win_base
    kind, L, n = "chainlink_twap60", 300, 5
    T = market_grid(win, L, burn_days=2, step=1)
    win.prepare(np.unique(np.concatenate([win.i(T) - n, win.i(T) - n - 2])))

    dropped = evaluate(win, model, kind, L, n, expiries=T)
    kept = evaluate(win, model, kind, L, n, expiries=T, emit_all=True)

    assert "ok" in kept.rows
    assert len(kept) > len(dropped)
    assert kept.rows["ok"].any()
    assert not kept.rows["ok"].all()


def test_no_lookahead(win_base):
    """Perturbing data strictly after the quote must not move the quote."""
    from fvmodel.base import load_params
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value

    win, model = win_base
    it = win.n - 4000
    win.prepare(np.array([it, it - 2]))
    st = state_at(win, it, model)
    mk = market_at(win, "chainlink_twap60", int(win.ts[it]) + 300, 300, 900, 60)
    ref = fair_value(st, mk, model, t_now=int(win.ts[it]))

    win2 = Window(load_params(), model.fp, int(win.ts[0]), int(win.ts[-1]), warm=0,
                  chainlink=True, log=lambda *a: None)
    for name in ("perp", "cl_step"):
        a = getattr(win2, name)
        a[it + 1:] = a[it + 1:] * 1.05
    win2.prepare(np.array([it, it - 2]))
    got = fair_value(state_at(win2, it, model), mk, model, t_now=int(win.ts[it]))
    for field in ("var_y", "carry", "eps_bar", "var_eps", "omega"):
        assert getattr(ref, field) == pytest.approx(getattr(got, field), abs=1e-15), (
            "%s moved when data after the quote time changed" % field)
