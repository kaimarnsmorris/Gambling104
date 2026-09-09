"""Pins one receipt per value column.

`rtds_btc` stamps px, twap30 and twap60 each with its own arrival, and they
disagree -- twap60's receipt runs later than px's on 70.8 % of 568,261 rows,
median +46.6 ms, past a whole bucket on 27.5 %. Bucketing all three at the
price's receipt put a not-yet-arrived twap60 on 4.6 % of one real window's
buckets, and twap60 is the settlement variable. These tests assert on the
VALUE on the grid, because the shape was always right.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams import (Stream, StreamInvalid, TimeKind, clear_registry,
                             register, resolve)
from harness.streams.reader import grid_stream
from harness.streams.spec import check_receipt_map

OPEN = 1_786_665_600

#: px arrives in buckets 0 and 2; the twap it shares a row with arrives a
#: bucket LATER each time. Bucket the twap on px's receipt and index 1 holds a
#: twap the host had not been told yet.
MAP = (("twap60", "twap60_recv_ns"),)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _rows():
    def ns(offset):
        return int((OPEN + offset) * 1_000_000_000)
    return pd.DataFrame({
        "px_recv_ns": [ns(0.05), ns(0.25)],
        "twap60_recv_ns": [ns(0.15), ns(0.35)],
        "px": [1.0, 2.0],
        "twap60": [10.0, 20.0],
    })


def _grid(**kw):
    return grid_stream(_rows(), OPEN, ("px", "twap60"),
                       time_col="px_recv_ns", **kw)


def test_a_value_is_absent_until_its_own_receipt_says_it_arrived():
    out = _grid(value_time_cols=MAP)
    assert np.isnan(out["twap60"][1]), (
        "twap60 arrived in bucket 1, so index 1 must not see it -- this is "
        "the settlement variable, not a nice-to-have")
    assert out["twap60"][2] == 10.0
    assert out["px"][1] == 1.0, "px still rides on its own receipt"


def test_the_shared_receipt_is_what_delivered_the_value_early():
    """The defect itself, pinned so the fix cannot be quietly undone."""
    wrong = _grid()
    assert wrong["twap60"][1] == 10.0
    right = _grid(value_time_cols=MAP)
    assert np.isnan(right["twap60"][1])


def test_each_receipt_carries_its_own_column_forward():
    out = _grid(value_time_cols=MAP)
    assert out["twap60"][3] == 10.0, "still the freshest twap at index 3"
    assert out["twap60"][4] == 20.0
    assert out["px"][3] == 2.0


def test_age_and_has_describe_the_stream_level_receipt():
    """Documented explicitly: they cover the stream's own time column, not a
    column bucketed on a receipt of its own."""
    out = _grid(value_time_cols=MAP)
    assert out["has"][1] and out["age_ms"][1] == 0.0    # px arrived in b0
    assert np.isnan(out["twap60"][1])                   # twap60 had not
    assert list(out["has"]) == list(_grid()["has"])


def test_an_unmapped_column_still_rides_the_stream_level_receipt():
    out = grid_stream(_rows(), OPEN, ("px", "twap60"),
                      time_col="px_recv_ns",
                      value_time_cols=(("nothing_here", "px_recv_ns"),))
    assert out["twap60"][1] == 10.0


def test_a_non_causal_stream_keeps_per_column_receipts():
    out = _grid(value_time_cols=MAP, causal=False)
    assert out["twap60"][1] == 10.0 and np.isnan(out["twap60"][0])
    assert out["px"][0] == 1.0


# -- the general case fails loudly rather than silently ---------------------

def test_a_value_column_with_an_ignored_receipt_is_refused():
    with pytest.raises(StreamInvalid) as exc:
        check_receipt_map("rtds", ("px", "px_first_recv_ns", "twap60",
                                   "twap60_first_recv_ns"),
                          ("px", "twap60"), "px_first_recv_ns")
    assert "twap60" in str(exc.value)
    assert "twap60_first_recv_ns" in str(exc.value)


def test_declaring_the_receipt_satisfies_the_check():
    check_receipt_map("rtds", ("px", "px_first_recv_ns", "twap60",
                               "twap60_first_recv_ns"),
                      ("px", "twap60"), "px_first_recv_ns",
                      (("twap60", "twap60_first_recv_ns"),))


def test_a_column_governed_by_its_own_receipt_needs_no_declaration():
    check_receipt_map("one_field", ("px", "px_first_recv_ns"), ("px",),
                      "px_first_recv_ns")


def test_registering_a_per_field_receipt_file_without_a_map_raises(tmp_path):
    """The whole point of the standard: loud at registration, not a silent
    4.6 % lookahead in the alpha column."""
    p = tmp_path / "rtds.parquet"
    pd.DataFrame({"px": [1.0], "px_first_recv_ns": [1],
                  "twap60": [2.0], "twap60_first_recv_ns": [2]}).to_parquet(p)
    with pytest.raises(StreamInvalid, match="twap60_first_recv_ns"):
        register("rtds", str(p), adapter=Stream(
            name="rtds", time_col="px_first_recv_ns",
            time_kind=TimeKind.RECEIPT, causal=True))


def test_the_same_file_registers_once_every_receipt_is_named(tmp_path):
    p = tmp_path / "rtds.parquet"
    pd.DataFrame({"px": [1.0], "px_first_recv_ns": [1],
                  "twap60": [2.0], "twap60_first_recv_ns": [2]}).to_parquet(p)
    register("rtds", str(p), adapter=Stream(
        name="rtds", time_col="px_first_recv_ns", time_kind=TimeKind.RECEIPT,
        causal=True,
        value_time_cols=(("px", "px_first_recv_ns"),
                         ("twap60", "twap60_first_recv_ns"))))
    assert resolve("rtds").values == ("px", "twap60")


# -- the declaration stays hashable ----------------------------------------

def test_the_map_is_pairs_not_a_dict_so_the_stream_stays_hashable():
    s = Stream(name="s", time_kind=TimeKind.RECEIPT, causal=True,
               value_time_cols=[["a", "a_recv_ns"]])
    assert s.value_time_cols == (("a", "a_recv_ns"),)
    assert hash(s) == hash(Stream(name="s", time_kind=TimeKind.RECEIPT,
                                  causal=True,
                                  value_time_cols=(("a", "a_recv_ns"),)))


def test_a_dict_shaped_map_is_refused_with_the_reason():
    with pytest.raises(StreamInvalid, match="unhashable"):
        Stream(name="s", time_kind=TimeKind.RECEIPT, causal=True,
               value_time_cols={"a": "a_recv_ns"})


def test_one_value_column_has_exactly_one_receipt():
    with pytest.raises(StreamInvalid, match="more than once"):
        Stream(name="s", time_kind=TimeKind.RECEIPT, causal=True,
               value_time_cols=(("a", "a_recv_ns"), ("a", "a_last_recv_ns")))


# -- end to end, through the registry and the loader ------------------------

def _panel(tmp_path):
    rows = [{"market_id": "m0", "open_ts": OPEN, "t_ms": t, "bid": 0.49,
             "ask": 0.51, "mid": 0.50, "n_src": 2} for t in (0, 100, 200)]
    p = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p)
    s = tmp_path / "strikes.parquet"
    pd.DataFrame({"market_id": ["m0"], "open_ts": [OPEN],
                  "strike": [100.0]}).to_parquet(s)
    return str(p), str(s)


def test_the_loader_grids_each_column_on_the_receipt_it_declared(tmp_path):
    from harness.build.episodes import load_episodes

    panel, strikes = _panel(tmp_path)
    f = tmp_path / "rtds.parquet"
    _rows().to_parquet(f)
    register("chainlink", str(f), adapter=Stream(
        name="chainlink", time_col="px_recv_ns", time_kind=TimeKind.RECEIPT,
        time_unit="ns", causal=True, value_time_cols=(
            ("px", "px_recv_ns"), ("twap60", "twap60_recv_ns"))))

    ep = load_episodes(panel_path=panel, strikes_path=strikes,
                       streams=("chainlink",))[0]
    cl = ep.stream("chainlink")
    assert cl.px[1] == 1.0
    assert np.isnan(cl.twap60[1]), "the twap had not arrived by index 1"
    assert cl.twap60[2] == 10.0


def test_only_the_declared_values_reach_the_grid(tmp_path):
    """Bookkeeping is not a value. A receipt column gridded as a float64 is
    lossy past ~256 ns at epoch-nanosecond magnitudes and says nothing
    `age_ms` does not; a copy count is a property of the recorder."""
    from harness.build.episodes import load_episodes

    panel, strikes = _panel(tmp_path)
    f = tmp_path / "rtds.parquet"
    df = _rows()
    df["twap60_n_copies"] = [1, 1]
    df.to_parquet(f)
    register("chainlink", str(f), adapter=Stream(
        name="chainlink", time_col="px_recv_ns", time_kind=TimeKind.RECEIPT,
        time_unit="ns", causal=True, value_time_cols=(
            ("px", "px_recv_ns"), ("twap60", "twap60_recv_ns"))))

    ep = load_episodes(panel_path=panel, strikes_path=strikes,
                       streams=("chainlink",))[0]
    assert set(ep.streams["chainlink"]) == {"px", "twap60", "age_ms", "has"}


def test_the_catalog_chainlink_entry_names_a_receipt_per_value():
    from harness.streams import catalog

    catalog.install()
    got = resolve("chainlink")
    assert got.values == ("px", "twap30", "twap60")
    assert dict(got.value_time_cols) == {
        "px": "px_first_recv_ns",
        "twap30": "twap30_first_recv_ns",
        "twap60": "twap60_first_recv_ns"}


def test_a_mapping_that_points_at_another_fields_receipt_raises():
    """A presence test would pass this and reinstate the whole defect.

    Mapping `twap60` to the PRICE's receipt satisfies "is it declared?" while
    putting twap60 on the grid before it arrived -- the original bug, now
    written down explicitly instead of inherited from a default.
    """
    with pytest.raises(StreamInvalid) as e:
        check_receipt_map("rtds", ("px", "px_first_recv_ns", "twap60",
                                   "twap60_first_recv_ns"),
                          ("px", "twap60"), "px_first_recv_ns",
                          (("twap60", "px_first_recv_ns"),))
    assert "twap60_first_recv_ns" in str(e.value)


def test_a_partial_mapping_that_would_drop_a_field_raises():
    """Declaring pairs narrows the stream's values to the mapped names.

    So mapping twap60 alone does not mis-align px -- it removes px from the
    grid entirely. That is quieter than the bug it replaces and surfaces as a
    KeyError somewhere else, so it has to fail here.
    """
    with pytest.raises(StreamInvalid) as e:
        check_receipt_map("rtds", ("px", "px_first_recv_ns", "twap60",
                                   "twap60_first_recv_ns"),
                          ("twap60",), "px_first_recv_ns",
                          (("twap60", "twap60_first_recv_ns"),))
    assert "'px'" in str(e.value)
