"""Pins the quote algebra.

    z_bid   = f(s - e_s - q*rpl_s)
    eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p

e_* are symmetric half-spreads: they WIDEN the pair.
rpl_* are retreat per lot: they PUSH the pair, and must not widen it.
"""
import pytest

from harness.blocks.defaults.f import standardise
from harness.blocks.defaults.link import link
from harness.blocks.defaults.quote import quotes
from harness.core.config import QuoteParams

STRIKE, SIGMA = 100_000.0, 50.0


def q_at(s=100_000.0, q=0.0, **kw):
    return quotes(s, q, STRIKE, SIGMA, QuoteParams(**kw), standardise, link)


def test_zero_edge_and_zero_inventory_collapse_to_fair():
    bid, ask = q_at()
    assert bid == pytest.approx(0.50, abs=0.011)
    assert ask == pytest.approx(0.50, abs=0.011)


def test_probability_edge_widens_symmetrically():
    bid, ask = q_at(e_p=0.05)
    mid = 0.5 * (bid + ask)
    assert mid == pytest.approx(0.50, abs=0.011)
    assert ask - bid > 0.09


def test_retreat_pushes_without_widening():
    """A long position must move both quotes down by the same amount."""
    b0, a0 = q_at(e_p=0.05, q=0.0)
    b1, a1 = q_at(e_p=0.05, q=10.0, rpl_p=0.002)
    assert b1 < b0 and a1 < a0
    assert (a1 - b1) == pytest.approx(a0 - b0, abs=1e-9)


def test_a_short_position_retreats_upwards():
    b0, a0 = q_at(e_p=0.05, q=0.0)
    b1, a1 = q_at(e_p=0.05, q=-10.0, rpl_p=0.002)
    assert b1 > b0 and a1 > a0


def test_edge_in_level_space_moves_the_pair_apart():
    narrow = q_at(e_s=0.0)
    wide = q_at(e_s=100.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_edge_in_latent_space_moves_the_pair_apart():
    narrow = q_at(e_z=0.0)
    wide = q_at(e_z=0.5)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_quotes_snap_to_the_cent_grid_conservatively():
    """Bids round DOWN and asks round UP -- never quote better than intended."""
    bid, ask = q_at(e_p=0.037)
    assert bid == pytest.approx(round(bid, 2), abs=1e-9)
    assert ask == pytest.approx(round(ask, 2), abs=1e-9)
    assert bid <= 0.463 + 1e-9 and ask >= 0.537 - 1e-9


def test_quotes_stay_inside_the_bounds():
    bid, ask = q_at(s=100_500.0, e_p=0.9)
    assert 0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0


def test_inventory_is_clamped_to_max_pos():
    unclamped = q_at(q=1000.0, rpl_p=0.001, max_pos=10.0)
    clamped = q_at(q=10.0, rpl_p=0.001, max_pos=10.0)
    assert unclamped == clamped
