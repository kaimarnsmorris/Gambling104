"""Fair value for perp-settled and Chainlink-settled BTC up/down markets.

    from fv import build_model, fair_value, FVState, Market

    model = build_model()
    fv = fair_value(state, Market("chainlink_twap60", expiry, strike,
                                  received_prints, book_snapshot), model)
    fv.p_up, fv.var_y, fv.y_star, fv.n_known, fv.m_Y, fv.eps_bar, fv.tail

The volatility model underneath is `btc_volatility_clock` v2.1 and is not refitted here.
Everything in this package sits between that model and the settlement.
"""
from .alpha import AlphaModel  # noqa: F401
from .build import build_model, reliability, score  # noqa: F401
from .chainlink import EpsModel, FilterParams, PrintFilter  # noqa: F401
from .fairvalue import (FairValue, FairValueModel, FVState, Market,  # noqa: F401
                        Switches, fair_value)
from .tails import SettlementTail  # noqa: F401
from .weights import Settlement, settlement_weights  # noqa: F401

__all__ = ["AlphaModel", "EpsModel", "FairValue", "FairValueModel", "FVState",
           "FilterParams", "Market", "PrintFilter", "Settlement", "SettlementTail",
           "Switches", "build_model", "fair_value", "reliability", "score",
           "settlement_weights"]
