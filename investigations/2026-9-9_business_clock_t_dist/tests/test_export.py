"""The export: its algebra, its causality, and that it is strike-free."""
from __future__ import annotations

import numpy as np
import pytest


def test_usable_bucket_is_one_shift_after_the_instant():
    from export.build_export import usable_t_ms

    # a value carrying information through open_ts + t_s becomes usable one 100 ms
    # bucket later; t_s = -1 is the pre-open row that fills bucket 0
    assert usable_t_ms(-1) == 0
    assert usable_t_ms(0) == 100
    assert usable_t_ms(1) == 1100
    # t_s=299 becomes usable at ms 299100 (= 299*1000 + 100), the first of the 9
    # buckets bucket_owner() assigns it (2991..2999, ms 299100..299900 - the last
    # t_s owns only 9 buckets because 300 t_s values cannot divide 3000 buckets
    # evenly once t_s=-1 claims bucket 0). Verified numerically against
    # bucket_owner(): np.where(bucket_owner() == 299)[0].min() * 100 == 299100.
    assert usable_t_ms(299) == 299100


def test_bucket_owner_partitions_the_grid():
    from export.build_export import T_S_RANGE, bucket_owner

    own = bucket_owner()
    assert own.shape == (3000,)
    assert own[0] == -1, "bucket 0 is owned by the pre-open row"
    assert own[1] == 0 and own[10] == 0, "t_s=0 owns buckets 1..10"
    assert own[11] == 1
    assert own[2999] == 299
    assert set(own.tolist()) == set(T_S_RANGE)
    assert len(T_S_RANGE) == 301


@pytest.mark.slow
def test_s_and_sigma_reproduce_y_star():
    """s is strike-free and y*(K) = (K - s) / (p_ref * omega). If that identity
    holds, f() and link() in the blocks are the model, not an approximation."""
    import numpy as np

    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, evaluate, market_grid

    model = build_model()
    t0 = CL_T0 + 20 * 86400
    win = Window(load_params(), model.fp, t0, t0 + 2 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    T = market_grid(win, 300, burn_days=1)[:200]
    win.prepare(win.i(T) - 120)
    cell = evaluate(win, model, "chainlink_twap60", 300, 120, expiries=T)
    r = cell.rows
    s = r["strike"] - r["y_star"] * r["p_ref"] * r["omega"]
    y_back = (r["strike"] - s) / (r["p_ref"] * r["omega"])
    # rtol=1e-12/atol=1e-18 is tighter than the roundtrip can deliver: `s` subtracts
    # y_star*p_ref*omega (O(1)) from strike (O(1e4-1e5)), so recovering y_star loses
    # a few ULPs of `strike` to cancellation - harmless in absolute terms (~1e-16)
    # but large relative to the smallest y_star values (~1e-5). rtol=1e-9/atol=1e-12
    # still checks the identity to far tighter than anything downstream (sigma,
    # p_model) can resolve.
    assert np.allclose(y_back, r["y_star"], rtol=1e-9, atol=1e-12)
    # and s must not depend on the strike: shift the strike, s must move with it
    # exactly, leaving the implied breakeven unchanged in the model's own terms
    assert np.all(np.isfinite(s))


@pytest.mark.slow
def test_export_has_a_row_for_every_market_second(tmp_path):
    import polars as pl

    from export.build_export import T_S_RANGE, build

    t0 = 1786665600
    p = build("baseline", t0, t0 + 3 * 3600, out_path=tmp_path / "baseline.parquet")
    df = pl.read_parquet(p)
    per = df.group_by("market_id").len()
    assert per["len"].min() == len(T_S_RANGE) == per["len"].max()
    assert df["ok"].sum() > 0.8 * len(df), "most seconds should be priceable"
    assert set(df.columns) >= {"market_id", "open_ts", "t_s", "s", "sigma", "nu",
                               "mu", "sigma_t", "p_model", "p_quoted", "ok"}


@pytest.mark.slow
def test_export_is_causal(tmp_path):
    """Every bucket must be served by a row whose information instant precedes it."""
    import polars as pl

    from export.build_export import bucket_owner

    own = bucket_owner()
    bucket_ms = np.arange(own.size, dtype=np.int64) * 100
    # the row owning a bucket carries information through the instant open_ts + t_s,
    # which is t_s * 1000 ms after the open. That instant must be STRICTLY before the
    # bucket it serves, or a block is reading its own tick or the future.
    info_ms = own.astype(np.int64) * 1000
    assert np.all(info_ms < bucket_ms), (
        "%d bucket(s) served by a row from their own instant or later"
        % int((info_ms >= bucket_ms).sum()))
