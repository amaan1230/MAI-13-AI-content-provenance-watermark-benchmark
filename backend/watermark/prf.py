"""Library-version-independent pseudo-random function for green-list membership.

The reference KGW implementation seeds `torch.randperm` with `key * prev_token`.
That ties detection to torch's RNG implementation, which is not guaranteed stable
across versions/platforms. We instead use a keyed splitmix64 mix: membership of
token `t` in the green list for context `c` is

    green(c, t)  <=>  U(mix(seed(c) ^ mix(t))) < gamma,   U in [0, 1)

Under H0 (text written without knowledge of the key) each scored token is green
with probability gamma, which is the assumption behind the one-proportion z-test.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
_TWO53 = float(2**53)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised over uint64 arrays (wrap-around arithmetic)."""
    with np.errstate(over="ignore"):
        z = (x + _C1) & _MASK64
        z = ((z ^ (z >> np.uint64(30))) * _C2) & _MASK64
        z = ((z ^ (z >> np.uint64(27))) * _C3) & _MASK64
        return z ^ (z >> np.uint64(31))


def key_to_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def context_seed(key_int: int, context: Sequence[int]) -> np.uint64:
    """Fold the key and the preceding `h` token ids into a 64-bit seed (h may be 0)."""
    s = _mix(np.array([key_int], dtype=np.uint64))
    for tok in context:
        s = _mix(s ^ _mix(np.array([int(tok) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)))
    return s[0]


def uniform_scores(seed: np.uint64, token_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.uint64)
    h = _mix(np.uint64(seed) ^ _mix(ids))
    return (h >> np.uint64(11)).astype(np.float64) / _TWO53


def green_mask(seed: np.uint64, vocab_size: int, gamma: float) -> np.ndarray:
    return uniform_scores(seed, np.arange(vocab_size, dtype=np.uint64)) < gamma


def is_green(seed: np.uint64, token_id: int, gamma: float) -> bool:
    return bool(uniform_scores(seed, np.array([token_id]))[0] < gamma)
