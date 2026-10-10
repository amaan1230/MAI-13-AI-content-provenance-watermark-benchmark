"""Append-only, hash-chained audit log."""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from backend.core.hashing import canonical_json, sha256_hex
from backend.core.logging import request_id_var
from backend.models import AuditEvent
from backend.models.base import utcnow

_PG_LOCK_KEY = 0x4D414931  # arbitrary constant: "MAI1"
_process_lock = threading.RLock()


def _release(session: Session, *_: Any) -> None:
    if session.info.pop("audit_lock_held", False):
        _process_lock.release()


def _serialize_appends(session: Session) -> None:
    """Hold an append lock until the surrounding transaction ends.

    Two concurrent transactions must not both read the same chain head, or the chain
    forks. PostgreSQL: transaction-scoped advisory lock (works across processes).
    SQLite (single process): a process lock released on commit/rollback.
    """
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _PG_LOCK_KEY})
        return
    if session.info.get("audit_lock_held"):
        return
    _process_lock.acquire()
    session.info["audit_lock_held"] = True
    if not session.info.get("audit_listeners"):
        event.listen(session, "after_commit", _release)
        event.listen(session, "after_rollback", _release)
        session.info["audit_listeners"] = True


def _event_hash(prev_hash: str | None, payload: dict[str, Any]) -> str:
    return sha256_hex(f"{prev_hash or ''}|{canonical_json(payload)}".encode())


def record_event(
    session: Session,
    *,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    _serialize_appends(session)
    last = session.execute(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(1)).scalar_one_or_none()
    prev_hash = last.event_hash if last else None
    ts = utcnow()
    payload = {
        "ts": _iso(ts),
        "actor": actor,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "details": details or {},
    }
    event = AuditEvent(
        ts=ts,
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        request_id=request_id_var.get(),
        details=details or {},
        prev_hash=prev_hash,
        event_hash=_event_hash(prev_hash, payload),
    )
    session.add(event)
    session.flush()
    return event


def verify_chain(session: Session) -> dict[str, Any]:
    """Recompute the hash chain. Returns the first broken event if any."""
    prev: str | None = None
    n = 0
    for ev in session.execute(select(AuditEvent).order_by(AuditEvent.id)).scalars():
        payload = {
            "ts": _iso(ev.ts),
            "actor": ev.actor,
            "action": ev.action,
            "entity_type": ev.entity_type,
            "entity_id": ev.entity_id,
            "details": ev.details or {},
        }
        if ev.prev_hash != prev or _event_hash(prev, payload) != ev.event_hash:
            return {"valid": False, "events_checked": n, "broken_at_event_id": ev.id}
        prev = ev.event_hash
        n += 1
    return {"valid": True, "events_checked": n, "broken_at_event_id": None}


def _iso(ts) -> str:
    # SQLite drops tzinfo on read; events are always written in UTC.
    from datetime import UTC

    # PostgreSQL returns timestamptz in the session time zone; hash over UTC always.
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat()
