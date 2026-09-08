"""Pins the Polymarket fee model to values verified on-chain.

fee = BASE_FEE_RATE * p * (1-p) * shares, peaking at 1.75 c/share at p=0.50.
Takers pay it net of the Silver rebate; makers are PAID a fifth of it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness.blocks.defaults.fees import FeeSchedule, Liquidity  # noqa: E402


def test_taker_fee_peaks_at_the_money():
    f = FeeSchedule()
    at_mid = f.charge(Liquidity.TAKER, shares=100.0, p=0.50)
    off_mid = f.charge(Liquidity.TAKER, shares=100.0, p=0.10)
    assert at_mid > off_mid > 0.0


def test_taker_fee_at_mid_is_1_75_cents_per_share_net_of_silver_rebate():
    """0.07 * 0.25 = 1.75 c/share gross, x (1 - 0.0833) at Silver."""
    f = FeeSchedule()
    usd = f.charge(Liquidity.TAKER, shares=100.0, p=0.50)
    assert usd == pytest.approx(100.0 * 0.0175 * (1.0 - 0.0833), rel=1e-12)


def test_maker_is_paid_a_fifth_of_the_base_fee():
    f = FeeSchedule()
    usd = f.charge(Liquidity.MAKER, shares=100.0, p=0.50)
    assert usd == pytest.approx(-100.0 * 0.0175 * 0.20, rel=1e-12)
    assert usd < 0.0


def test_fee_vanishes_at_the_bounds():
    f = FeeSchedule()
    assert f.charge(Liquidity.TAKER, 100.0, 0.0) == pytest.approx(0.0)
    assert f.charge(Liquidity.TAKER, 100.0, 1.0) == pytest.approx(0.0)


def test_fee_is_symmetric_about_a_half():
    f = FeeSchedule()
    assert f.charge(Liquidity.TAKER, 100.0, 0.3) == pytest.approx(
        f.charge(Liquidity.TAKER, 100.0, 0.7))


def test_the_realised_average_is_not_the_constant():
    """1.263 c/share is an average over a fill distribution, never a rate.

    Guards against anyone re-hard-coding it: no single p reproduces it as
    the *gross* per-share fee except off-mid, so the model must stay a curve.
    """
    f = FeeSchedule()
    per_share_at_mid = f.charge(Liquidity.TAKER, 1.0, 0.5) * 100.0
    assert per_share_at_mid > 1.263
