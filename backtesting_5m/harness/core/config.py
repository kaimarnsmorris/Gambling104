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
    #: Apply the venue's $1.00/day per-stream minimum maker-rebate payout.
    #: Measured with perfect separation over 131 earn-days: smallest paid
    #: $1.0359, largest skipped $0.7209. Off by default because it changes the
    #: headline; when on, the caveats say so.
    apply_daily_minimum: bool = False


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

    #: Drop markets whose L1 never moved AND which never had more than one
    #: quote source: the venue-maintenance signature. ON BY DEFAULT, because
    #: including them does not degrade a result, it manufactures one.
    #:
    #: On 2026-08-19 the venue sat in maintenance through 04:05-05:35 and
    #: 09:05-10:45, publishing bid 0.50 / ask 0.51 unchanged for all 2,940
    #: buckets of 37 consecutive markets while BTC moved normally. A model
    #: quoting against a frozen book prints money that was never available:
    #: those 36 markets returned +$44.94 EACH against -$0.48 on the other
    #: 1,066, and single-handedly turned a -0.48/market strategy into a
    #: +1.01/market one.
    #:
    #: Across the whole panel (7,111 markets, 26 days) the signature appears
    #: on 8 days and 253 markets, and it separates perfectly: every frozen
    #: market is single-source, no two-source market has fewer than 11
    #: distinct mids, and the 32 single-source markets with genuinely moving
    #: books (up to 144 distinct mids) are kept. The conjunction is what
    #: makes it safe -- either test alone would either miss markets or drop
    #: real ones.
    drop_frozen_book: bool = True

    #: A market whose mid takes this many distinct values or fewer over the
    #: whole 300 s is frozen. Two-source markets bottom out at 11, so 10
    #: cannot reach one even before the single-source condition applies.
    frozen_book_max_distinct_mid: int = 10


@dataclass(frozen=True)
class Output:
    emit_ticks: bool = False
    tick_markets: tuple = ()
    seeds: tuple = (0,)
