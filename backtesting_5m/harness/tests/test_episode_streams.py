"""Pins the stream accessor and the time base.

Core fields stay statically typed; registered streams come through an explicit
accessor. A typo must fail at the first tick, not surface as NaN deep in a run.
"""
import numpy as np
import pytest

from harness.paths import N_BUCKET


def test_t_ms_is_the_decision_time_of_each_index(flat_episode):
    assert flat_episode.t_ms.shape == (N_BUCKET,)
    assert flat_episode.t_ms[0] == 0
    assert flat_episode.t_ms[1] == 100
    assert flat_episode.t_ms[-1] == (N_BUCKET - 1) * 100


def test_tte_is_consistent_with_the_time_base(flat_episode):
    i = 1500
    assert flat_episode.tte_s(i) == pytest.approx(
        300.0 - flat_episode.t_ms[i] / 1000.0)


def test_a_registered_stream_is_reachable_by_name(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={
        "chainlink": {"px": np.full(N_BUCKET, 5.0),
                      "age_ms": np.zeros(N_BUCKET),
                      "has": np.ones(N_BUCKET, dtype=bool)}})
    cl = ep.stream("chainlink")
    assert cl.px[10] == 5.0
    assert cl.age_ms[10] == 0.0
    assert cl.has[10]


def test_an_unregistered_stream_raises_and_lists_what_is_there(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={"chainlink": {"px": np.zeros(1),
                                                      "age_ms": np.zeros(1),
                                                      "has": np.zeros(1, bool)}})
    with pytest.raises(KeyError, match="chainlink"):
        ep.stream("chainlnk")


def test_a_mistyped_column_raises_listing_the_real_ones(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={"cl": {"px": np.zeros(1),
                                               "age_ms": np.zeros(1),
                                               "has": np.zeros(1, bool)}})
    with pytest.raises(AttributeError, match="px"):
        ep.stream("cl").pxx


def test_episodes_without_streams_still_work(flat_episode):
    assert flat_episode.streams == {}
    with pytest.raises(KeyError):
        flat_episode.stream("anything")
