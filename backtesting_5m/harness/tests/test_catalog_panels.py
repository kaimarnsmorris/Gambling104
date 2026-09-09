"""Pins that the spot panels are catalog entries, and that the USDT one says so.

There are four physical panels behind five constants. The London panel is
BTC/USDT UNCORRECTED -- +$43.17 (sd 16.36) against the BTC/USD oracle these
markets settle on -- and is usable only because fair.py learns the basis. A
consumer assuming USD would be that much wrong.

Every assertion here resolves through the registry to a PATH. The previous
version compared a literal to itself ("usdt" in "spot_london_usdt") and
re-read the alias it had just registered, so it would have passed with every
panel pointed at the wrong file.
"""
import inspect
import os

import pytest

from harness import paths
from harness.build import spot_5m_100ms
from harness.streams import catalog, clear_registry, registered, resolve


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
    """The name must track the FILE, not just read plausibly. Pointing the
    usdt alias at the usd panel is the mistake worth catching, and it is
    invisible to any assertion made about the alias alone."""
    catalog.install()
    usdt, usd = resolve("spot_london_usdt"), resolve("spot_usd")
    assert usdt.path == paths.SPOT_LONDON_USDT
    assert usd.path == paths.SPOT_USD
    assert usdt.path != usd.path
    assert "usdt" in os.path.basename(usdt.path)
    assert "usdt" not in os.path.basename(usd.path)


def test_the_legacy_panel_is_registered_but_marked():
    """Registration alone was all this asserted. The marking is the point:
    the entry exists only so a historical run folder can be reproduced."""
    catalog.install()
    got = resolve("spot_legacy_usdt")
    assert got.path == paths.SPOT_LEGACY_USDT
    assert got.path != resolve("spot_usd").path

    entry = _catalog_note("spot_legacy_usdt")
    assert "Do not build on it" in entry
    assert "43.17" in entry


def test_the_panel_currency_figure_matches_the_build_that_measured_it():
    """The catalog is now the single authority on panel currency -- paths.py's
    statement was deleted -- and it carried +$56, a number nothing in the repo
    derives. The build measures +$43.17 (sd 16.36)."""
    note = _catalog_note("spot_london_usdt")
    assert "43.17" in note and "16.36" in note
    assert "$56" not in inspect.getsource(catalog)
    assert "+43.17" in inspect.getsource(spot_5m_100ms)


def _catalog_note(name):
    """The contiguous `#:` comment block attached to one catalog entry.

    The marking is documentation, so the test that it exists has to read the
    documentation -- and read it AT the entry, so moving the warning off this
    panel fails rather than passing on some other entry's words.
    """
    lines = inspect.getsource(catalog).splitlines()
    at = next(i for i, ln in enumerate(lines) if f'_install("{name}"' in ln)
    block = []
    while at > 0 and lines[at - 1].strip().startswith("#:"):
        at -= 1
        block.insert(0, lines[at].strip()[2:].strip())
    return " ".join(block)
