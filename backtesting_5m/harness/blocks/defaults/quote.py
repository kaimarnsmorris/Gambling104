"""The quote construction.

    z_bid   = f(s - e_s - q*rpl_s)
    z_ask   = f(s + e_s - q*rpl_s)
    eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p
    eff_ask = link(z_ask + e_z - q*rpl_z) + e_p - q*rpl_p

Edge widens; retreat pushes. Bids snap DOWN to the cent grid and asks snap UP,
so rounding never quotes better than intended.
"""
import math


def _snap_down(p, tick):
    return math.floor(p / tick + 1e-9) * tick


def _snap_up(p, tick):
    return math.ceil(p / tick - 1e-9) * tick


def quotes(s_i, q, strike, sigma_i, params, standardise, link):
    """Return (eff_bid, eff_ask) on the tick grid, clipped to [0, 1]."""
    if params.max_pos > 0.0:
        q = max(-params.max_pos, min(params.max_pos, q))

    lean_s = q * params.rpl_s
    lean_z = q * params.rpl_z
    lean_p = q * params.rpl_p

    z_bid = standardise(s_i - params.e_s - lean_s, strike, sigma_i)
    z_ask = standardise(s_i + params.e_s - lean_s, strike, sigma_i)

    eff_bid = link(z_bid - params.e_z - lean_z) - params.e_p - lean_p
    eff_ask = link(z_ask + params.e_z - lean_z) + params.e_p - lean_p

    eff_bid = min(1.0, max(0.0, _snap_down(eff_bid, params.tick)))
    eff_ask = min(1.0, max(0.0, _snap_up(eff_ask, params.tick)))
    return eff_bid, eff_ask
