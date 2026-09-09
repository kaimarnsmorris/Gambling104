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
impossible. `signal_params` is passed through to `precompute` as keyword
arguments and forms part of the cache key, so a block that wants to be swept
declares the parameter and every value gets its own entry. A block that
takes `(ep)` alone is unaffected: an empty `signal_params` calls it exactly
as before.
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
    """`signal_params` as a sorted tuple of pairs -- hashable, order-free.

    A dict is the natural thing to write and an unusable cache key; sorting
    means `{"scale": 2, "floor": 1}` and `{"floor": 1, "scale": 2}` are one
    entry rather than two.
    """
    if not signal_params:
        return ()
    items = (signal_params.items() if hasattr(signal_params, "items")
             else signal_params)
    return tuple(sorted((str(k), v) for k, v in items))


def precompute_signals(episodes, modules, signature, cache=None,
                       signal_params=()):
    """`{market_id: (s, sigma)}` for `episodes`, filling `cache` in place.

    `cache` is any dict-like keyed by `(signature, params, market_id)`; pass
    the same one across a grid and every arm after the first pays nothing.
    Pass None for a one-shot run.

    `signal_params` reaches `precompute` as keyword arguments and is part of
    the key, which is what lets a grid sweep a model parameter without
    editing the block. Empty means the block is called `precompute(ep)`, so
    a block that never opted in cannot be broken by this.

    Episodes are keyed by `market_id`, which is unique per market and already
    the key every artefact uses.
    """
    if cache is None:
        cache = {}
    params = normalise_params(signal_params)
    kwargs = dict(params)
    out = {}
    for ep in episodes:
        key = (signature, params, ep.market_id)
        if key not in cache:
            cache[key] = (modules["fair"].precompute(ep, **kwargs)
                          if kwargs else modules["fair"].precompute(ep),
                          modules["vol"].precompute(ep, **kwargs)
                          if kwargs else modules["vol"].precompute(ep))
        out[ep.market_id] = cache[key]
    return out
