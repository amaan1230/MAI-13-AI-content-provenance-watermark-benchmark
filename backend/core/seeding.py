"""Deterministic seeding helpers."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def derive_seed(base_seed: int, *parts: object) -> int:
    """Stable per-item seed derived from a base seed and identifying parts."""
    h = hashlib.sha256(":".join([str(base_seed), *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)
