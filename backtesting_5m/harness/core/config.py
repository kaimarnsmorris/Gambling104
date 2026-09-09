"""Every knob a run turns. Dataclasses only -- no behaviour."""
from dataclasses import dataclass, field

from harness.blocks.defaults.fees import FeeSchedule
from harness.core.latency import LatencyModel
from harness.paths import TICK


@dataclass(frozen=True)
class QuoteParams:
    """Edge and retreat, expressible at three layers.

    e_*   symmetric half-spread: widens the pair.
    rpl_* retreat per lot: pushes the pair, does not widen it.

    Three layers because the pass-through is not constant -- 0.79 c/$ at the
    money at tau=300 s against 14.5 c/$ at tau=30 s. A single-layer edge is
    mis-specified across the window by construction.
    """
    e_s: float = 0.0        # USD of BTC
    e_z: float = 0.0        # sigma
    e_p: float = 0.0        # probability
    rpl_s: float = 0.0
    rpl_z: float = 0.0
    rpl_p: float = 0.0
    max_pos: float = 0.0    # 0 disables the clamp
    tick: float = TICK
    shares: float = 1.0     # lot size


@dataclass(frozen=True)
class ExecConfig:
    """Execution knobs. There is no mode: the policy is unified.

    Making and taking are simultaneous, not alternatives -- we rest on both
    sides and cross whenever the book is through the fee-adjusted
    threshold. Which is why `fees` is a policy input here and not only a
    PnL input: the thresholds themselves are fee-adjusted.
    """

    latency: LatencyModel = field(default_factory=LatencyModel)
    #: None means "use whatever the resolved `fees` block supplies", which is
    #: how an investigation overrides the schedule by dropping in a fees.py.
    #: A schedule set here is a run parameter and wins over the block. `run`
    #: resolves it and stamps it back onto the config it replays with, so
    #: the policy prices against the same schedule the ledger charges.
    fees: FeeSchedule | None = None
    max_book_age_ms: float = 1000.0
    requote_every: int = 10             # decision indices between requotes
    min_tte_s: float = 0.0
    max_tte_s: float = 300.0
    fill_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    t0: int | None = None
    t1: int | None = None
    days: tuple = ()
    markets: tuple = ()
    max_markets: int | None = None
    require_spot: bool = False
    #: Exclude any market for which a named stream has no usable observation
    #: in the window. EXCLUDED MEANS ABSENT -- not present-with-NaN, and never
    #: scored as zero. `require_spot` is the older special case and is kept
    #: working; prefer require=("spot",).
    require: tuple = ()


@dataclass(frozen=True)
class Output:
    emit_ticks: bool = False
    tick_markets: tuple = ()
    seeds: tuple = (0,)
    plots: bool = True
