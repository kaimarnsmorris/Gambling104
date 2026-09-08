"""vol block: the settlement standard deviation, in the same USD units as `s`."""
from _fvexport import for_episode


def precompute(ep):
    return for_episode(ep)["sigma"]
