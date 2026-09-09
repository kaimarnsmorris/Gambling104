"""Pins stream gridding.

A registered stream must inherit the SAME causality treatment as the built-in
columns -- it does not get to re-earn the lookahead guarantee, it gets it by
routing through the same shift.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams.reader import grid_stream

OPEN = 1_786_665_600


def _rows(offsets_s, values):
    return pd.DataFrame({
        "recv_ns": [(OPEN + o) * 1_000_000_000 for o in offsets_s],
        "px": values,
    })


def test_an_observation_is_not_visible_at_its_own_bucket():
    out = grid_stream(_rows([0.0], [7.0]), OPEN, ("px",))
    assert np.isnan(out["px"][0]), "bucket 0 leaked into decision index 0"
    assert out["px"][1] == 7.0


def test_age_is_zero_when_first_usable_then_grows():
    out = grid_stream(_rows([0.7], [7.0]), OPEN, ("px",))
    assert out["px"][8] == 7.0 and out["age_ms"][8] == 0.0
    assert out["px"][9] == 7.0 and out["age_ms"][9] == 100.0


def test_has_is_false_before_the_first_observation():
    out = grid_stream(_rows([0.4], [1.0]), OPEN, ("px",))
    assert not out["has"][:5].any()
    assert out["has"][5:].all()


def test_first_observation_in_a_bucket_wins():
    df = _rows([0.00, 0.05, 0.10], [1.0, 999.0, 2.0])
    out = grid_stream(df, OPEN, ("px",))
    assert out["px"][1] == 1.0
    assert out["px"][2] == 2.0


def test_observations_outside_the_window_are_dropped():
    df = _rows([-1.0, 0.5, 301.0], [1.0, 2.0, 3.0])
    out = grid_stream(df, OPEN, ("px",))
    assert np.isfinite(out["px"]).sum() > 0
    assert out["px"][-1] == 2.0


def test_a_non_causal_stream_is_not_shifted():
    """Ground-truth series plotted after the fact are not decision inputs."""
    out = grid_stream(_rows([0.0], [7.0]), OPEN, ("px",), causal=False)
    assert out["px"][0] == 7.0


def test_millisecond_time_units_are_honoured():
    df = pd.DataFrame({"recv_ns": [(OPEN * 1000) + 700], "px": [3.0]})
    out = grid_stream(df, OPEN, ("px",), time_unit="ms")
    assert out["px"][8] == 3.0


def test_bucket_boundary_survives_float_error():
    """0.1 s after open must land in bucket 100, not fall back to 0."""
    out = grid_stream(_rows([0.1, 0.0996], [1.0, 2.0]), OPEN, ("px",))
    assert out["px"][2] == 1.0, "exact 0.1 s must not fall a bucket early"
    assert out["px"][1] == 2.0, "99.6 ms must not be rounded up a bucket"


def test_the_earliest_observation_in_a_bucket_wins_even_when_the_frame_is_unsorted():
    """Ties must break by TIME, not by row order.

    write_stream sorts by day, never by recv_ns within a day, so an unsorted
    frame is a real input -- and getting this wrong puts the wrong value on
    the grid with no error at all.
    """
    df = _rows([0.05, 0.00, 0.10], [999.0, 1.0, 2.0])   # deliberately unsorted
    out = grid_stream(df, OPEN, ("px",))
    assert out["px"][1] == 1.0, "the 0.00 s observation is the first in bucket 0"
    assert out["px"][2] == 2.0


def test_has_does_not_depend_on_which_value_column_comes_first():
    """`has` is about whether an observation arrived, not whether one
    particular column happened to be non-null."""
    df = pd.DataFrame({
        "recv_ns": [(OPEN + 0.0) * 1_000_000_000],
        "a": [float("nan")],
        "b": [5.0],
    })
    ab = grid_stream(df, OPEN, ("a", "b"))
    ba = grid_stream(df, OPEN, ("b", "a"))
    assert list(ab["has"]) == list(ba["has"])
    assert ab["has"][1], "an observation did arrive, whatever column a holds"
