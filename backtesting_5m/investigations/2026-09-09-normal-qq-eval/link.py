"""z -> p. The normal link, with a QQ correction on the standardised distance.

    p = Phi(QQ(z))

A plain Phi(z) prices the tails of a 5 m BTC window too thin. QQ is the
quantile-quantile repair: a quintic fitted from the empirical quantiles back
onto the normal ones, so `qq` maps a Gaussian z to the z the data actually
behaved like, and Phi then reads off the probability.

The polynomial is ZEROED -- no constant term -- so an at-the-money z survives
untouched and the link cannot smuggle in a directional bias. Its coefficients
are ordered high power first, matching the source table.
"""
import math

#          z^5              z^4               z^3              z^2      z^1
QQ = (1.188460485423e-02, -1.177159929325e-03,
      4.886489346730e-02,  4.291009504391e-03, 9.196954091043e-01)

Z_CLAMP = 40.0      # |z| beyond this is saturated anyway; z**5 is not


def qq(z: float) -> float:
    """The zeroed quintic, by Horner. f(0) = 0 by construction."""
    a5, a4, a3, a2, a1 = QQ
    return z * (a1 + z * (a2 + z * (a3 + z * (a4 + z * a5))))


def phi(x: float) -> float:
    """The standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def link(z: float) -> float:
    """P(up) given the standardised distance from the strike."""
    if math.isnan(z):
        return float("nan")
    return phi(qq(max(-Z_CLAMP, min(Z_CLAMP, z))))
