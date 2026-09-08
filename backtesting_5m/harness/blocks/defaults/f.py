"""s -> z. The moneyness standardisation of the settlement chain."""


def standardise(level: float, strike: float, sigma: float) -> float:
    """How many sigma the expected settlement sits above the strike."""
    if sigma <= 0.0:
        return 0.0
    return (level - strike) / sigma
