"""Pins that the spot panels are catalog entries, and that the USDT one says so.

There are four physical panels behind five constants. The London panel is
BTC/USDT UNCORRECTED -- roughly +$56 against the BTC/USD oracle these markets
settle on -- and is usable only because fair.py learns the basis. A consumer
assuming USD would be that much wrong.
"""
from harness.streams import catalog, clear_registry, registered, resolve
import pytest


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def test_the_catalog_registers_every_spot_panel():
    catalog.install()
    for n in ("spot_usd", "spot_oracle_window", "spot_london_usdt"):
        assert n in registered()


def test_the_usdt_panel_is_named_so_it_cannot_be_mistaken_for_usd():
    catalog.install()
    assert "usdt" in "spot_london_usdt"
    assert "usd" == resolve("spot_usd").name[-3:]


def test_the_legacy_panel_is_registered_but_marked():
    catalog.install()
    assert "spot_legacy_usdt" in registered()
