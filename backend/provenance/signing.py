"""HMAC-SHA256 record signing.

Threat model / assumptions (see docs/SECURITY.md):
- The signing key is a server-side secret (PROVENANCE_SIGNING_KEY). Anyone holding it
  can forge signatures, so a valid signature proves only that *this server* registered
  the (content hash, metadata) tuple, and that it has not been altered since.
- It is NOT a proof of authorship and NOT publicly verifiable (symmetric key). A
  public-key scheme (e.g. Ed25519 / C2PA manifests) would be needed for third-party
  verification; that is out of scope for the text-modality MVP.
"""

from __future__ import annotations

import hashlib
import hmac

from backend.core.hashing import canonical_json


def sign_record(key: str, record: dict) -> str:
    return hmac.new(key.encode("utf-8"), canonical_json(record).encode("utf-8"), hashlib.sha256).hexdigest()


def verify_record(key: str, record: dict, signature: str) -> bool:
    return hmac.compare_digest(sign_record(key, record), signature)


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
