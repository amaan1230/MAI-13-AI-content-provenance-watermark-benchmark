"""Content hashing used for identity, integrity checks, and deduplication."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(text: str) -> str:
    """Exact-integrity hash over the raw UTF-8 bytes (no normalisation).

    Any change, including whitespace or invisible characters, changes the hash.
    """
    return sha256_hex(text.encode("utf-8"))


def normalized_text(text: str) -> str:
    """Normalisation used for duplicate detection only (not integrity)."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return " ".join(text.lower().split())


def normalized_hash(text: str) -> str:
    return sha256_hex(normalized_text(text).encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))
