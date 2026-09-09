"""s -> z. Log moneyness, standardised by the total-horizon vol.

    z = (ln(S / K) - 1/2 sigma_T^2) / sigma_T

This is d2: P(S_T > K) with S driftless under the pricing measure. The
half-sigma-squared term is small over 300 s but it is not zero, and it is what
makes an exactly at-the-money market pay slightly under 0.5.

`sigma` here is sigma_T, the TOTAL vol to expiry, not an annualised rate. The
harness hands this function no time argument -- the vol block folds tte in.
"""
import math


def standardise(level: float, strike: float, sigma: float) -> float:
    """How many sigma the expected settlement sits above the strike."""
    if not (sigma > 0.0) or not math.isfinite(sigma):
        return 0.0
    if not (level > 0.0) or not (strike > 0.0):
        return 0.0
    return (math.log(level / strike) - 0.5 * sigma * sigma) / sigma
