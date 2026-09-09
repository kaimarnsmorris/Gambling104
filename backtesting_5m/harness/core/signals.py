"""Precomputed `s` and `sigma`, reused across arms that cannot change them.

WHY THIS EXISTS. `fair.precompute(ep)` and `vol.precompute(ep)` take the
episode and nothing else -- no quote parameter, no execution knob, no seed.
Profiled over one day (284 markets), they are **59.5 ms per episode against
37.3 ms for the whole feed loop**: 61 % of an arm's cost is signal the arm
cannot influence. `run()` used to call them inside the seed loop, so a
3-seed run computed the same two arrays three times, and a 100-point grid
computed them three hundred times.

WHAT THE KEY HAS TO COVER. The cache is keyed on the CONTENT of the block
files that produced the arrays, not on the episode alone. A grid that varies
only `QuoteParams` reuses them safely; a grid that swaps `fair.py` or
`vol.py` -- or an investigation that edits one between runs -- gets a
different key and recomputes. Keying on the episode alone would hand the
next model the previous model's signal and report it as a result, which is
the one failure this harness exists to prevent.

Only `fair` and `vol` enter the key. They are the two SignalBlocks: the
slots declared vectorised, stateless and precomputed per episode. Every
other slot runs inside the loop and cannot reach these arrays.

SWEEPING A MODEL PARAMETER. A grid over vol scale is as ordinary as a grid
over half-spread, and requiring an edit to `vol.py` per point would make it
impossible. `signal_params` carries those parameters, KEYED BY SLOT:

    signal_params={"vol": {"scale": 1.6}}

Keyed by slot rather than flat, because a parameter belongs to one block.
A flat dict has to be broadcast to both, and then a `scale` meant for `vol`
reaches `fair.precompute` as an unexpected keyword and the run dies -- which
is exactly what the first version of this did. Naming the slot also makes
the config hash say which block was swept.

An unknown slot name raises rather than being ignored: a typo that silently
swept nothing would report the baseline under the swept arm's label.
"""
import hashlib

#: The slots whose output this module caches. Both are SignalBlocks --
#: `precompute(ep) -> array(N_BUCKET)` -- and nothing else in the block set
#: can change `s` or `sigma`.
SIGNAL_SLOTS = ("fair", "vol")


def block_signature(resolved):
    """A digest of the block files that produce `s` and `sigma`.

    Content, not path: two investigations pointing at the same shared model
    should share a cache entry, and one that edits its local `vol.py` should
    not keep the arrays the old one made. Reads the files each call -- they
    are a few kilobytes, and a stale signature is the failure mode worth
    paying to avoid.

    A slot that cannot be read falls back to its path plus a marker, so an
    unreadable block degrades to "never shares a cache entry" rather than
    colliding with a readable one.
    """
    h = hashlib.sha256()
    for slot in SIGNAL_SLOTS:
        path = resolved.get(slot)
        h.update(f"{slot}\0".encode())
        if path is None:
            h.update(b"<unresolved>\0")
            continue
        try:
            with open(path, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).hexdigest().encode())
        except OSError:
            h.update(f"<unreadable:{path}>".encode())
        h.update(b"\0")
    return h.hexdigest()


def normalise_params(signal_params):
    """`{slot: {name: value}}` as a sorted tuple of pairs -- hashable.

    A dict is the natural thing to write and an unusable cache key; sorting
    at both levels means `{"vol": {"a": 1, "b": 2}}` and
    `{"vol": {"b": 2, "a": 1}}` are one cache entry rather than two.
    """
    if not signal_params:
        return ()
    items = (signal_params.items() if hasattr(signal_params, "items")
             else signal_params)
    out = []
    for slot, params in items:
        if slot not in SIGNAL_SLOTS:
            raise ValueError(
                f"signal_params slot {slot!r} is not one of {SIGNAL_SLOTS}. "
                f"Parameters are keyed by the block they belong to, e.g. "
                f'signal_params={{"vol": {{"scale": 1.6}}}}. A slot nobody '
                f"reads would sweep nothing and report the baseline under "
                f"the swept arm's name.")
        inner = (params.items() if hasattr(params, "items") else params)
        out.append((slot, tuple(sorted((str(k), v) for k, v in inner))))
    return tuple(sorted(out))


def precompute_signals(episodes, modules, signature, cache=None,
                       signal_params=()):
    """`{market_id: (s, sigma)}` for `episodes`, filling `cache` in place.

    `cache` is any dict-like keyed by `(signature, params, market_id)`; pass
    the same one across a grid and every arm after the first pays nothing.
    Pass None for a one-shot run.

    `signal_params` is `{slot: {name: value}}`: each slot's parameters reach
    only THAT block's `precompute`, as keyword arguments, and are part of the
    key. A slot with no entry is called `precompute(ep)` exactly as before,
    so a block that never opted in cannot be broken by a sweep of the other
    one.

    Episodes are keyed by `market_id`, which is unique per market and already
    the key every artefact uses.
    """
    if cache is None:
        cache = {}
    params = normalise_params(signal_params)
    per_slot = {slot: dict(pairs) for slot, pairs in params}
    out = {}
    for ep in episodes:
        key = (signature, params, ep.market_id)
        if key not in cache:
            cache[key] = tuple(
                modules[slot].precompute(ep, **per_slot[slot])
                if per_slot.get(slot) else modules[slot].precompute(ep)
                for slot in SIGNAL_SLOTS)
        out[ep.market_id] = cache[key]
    return out
