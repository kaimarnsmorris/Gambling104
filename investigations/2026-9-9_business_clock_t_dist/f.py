"""f block: standardise a settlement level against the strike.

    z = (level - strike) / sigma

so z is the signed moneyness in settlement standard deviations and increases with
the level, which is the direction `link` expects: a higher fair value means UP is
more likely.
"""
import numpy as np


def standardise(level, strike, sigma):
    sg = np.asarray(sigma, dtype=np.float64)
    return np.where(sg > 0, (np.asarray(level, dtype=np.float64)
                             - np.asarray(strike, dtype=np.float64))
                    / np.where(sg > 0, sg, 1.0), np.nan)
