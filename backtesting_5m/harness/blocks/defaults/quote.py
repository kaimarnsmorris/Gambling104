"""The quote construction.

    z_bid   = f(s - e_s - q*rpl_s)
    z_ask   = f(s + e_s - q*rpl_s)
    eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p
    eff_ask = link(z_ask + e_z - q*rpl_z) + e_p - q*rpl_p

Edge widens; retreat pushes.

These are THEORETICAL prices, deliberately left off the cent grid. Snapping is
an order-placement concern and lives in `execution.py`, which applies the maker
fee adjustment first and snaps once, afterwards. Snapping here as well made the
adjustment a no-op by construction: the whole rebate is sub-tick, so adding it
to an already-on-grid number and re-snapping in the same direction can never
move the price. Rounding conservatively is still the rule -- it is just applied
once, to the number actually sent to the venue.
"""


def quotes(s_i, q, strike, sigma_i, params, standardise, link):
    """Return the unsnapped theoretical (eff_bid, eff_ask), clipped to [0, 1]."""
    if params.max_pos > 0.0:
        q = max(-params.max_pos, min(params.max_pos, q))

    lean_s = q * params.rpl_s
    lean_z = q * params.rpl_z
    lean_p = q * params.rpl_p

    z_bid = standardise(s_i - params.e_s - lean_s, strike, sigma_i)
    z_ask = standardise(s_i + params.e_s - lean_s, strike, sigma_i)

    eff_bid = link(z_bid - params.e_z - lean_z) - params.e_p - lean_p
    eff_ask = link(z_ask + params.e_z - lean_z) + params.e_p - lean_p

    eff_bid = min(1.0, max(0.0, eff_bid))
    eff_ask = min(1.0, max(0.0, eff_ask))
    return eff_bid, eff_ask
