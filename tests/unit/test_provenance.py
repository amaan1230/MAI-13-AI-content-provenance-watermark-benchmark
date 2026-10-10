from __future__ import annotations

import pytest
from sqlalchemy import update

from backend.core.config import reset_settings_cache
from backend.core.errors import NotFound, ValidationFailed
from backend.models import AuditEvent, ContentItem
from backend.provenance.audit import verify_chain
from backend.provenance.registry import (
    NewContent,
    ancestors,
    descendants,
    provenance_record,
    register_content,
    verify_integrity,
)


def _reg(s, text, parent=None, **kw):
    return register_content(
        s, NewContent(text=text, source_type=kw.pop("source_type", "unknown"), parent_id=parent, **kw)
    )


def test_lineage_chain(db_session):
    root = _reg(db_session, "root text")
    child = _reg(db_session, "child text", root.id, source_type="transformed")
    grand = _reg(db_session, "grandchild text", child.id, source_type="transformed")
    sibling = _reg(db_session, "sibling", root.id)
    assert [a.id for a in ancestors(db_session, grand.id)] == [child.id, root.id]
    assert {d.id for d in descendants(db_session, root.id)} == {child.id, grand.id, sibling.id}
    assert ancestors(db_session, root.id) == []


def test_parent_must_exist(db_session):
    with pytest.raises(NotFound):
        _reg(db_session, "x", "00000000-0000-0000-0000-000000000000")


def test_label_requires_justification(db_session):
    with pytest.raises(ValidationFailed, match="label_source"):
        _reg(db_session, "x", label="human")
    item = _reg(db_session, "x", label="human", label_source="test fixture")
    assert item.label == "human"


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_rejected(db_session, bad):
    with pytest.raises(ValidationFailed):
        _reg(db_session, bad)


def test_text_length_limit(db_session, monkeypatch):
    monkeypatch.setenv("MAX_TEXT_CHARS", "10")
    reset_settings_cache()
    with pytest.raises(ValidationFailed):
        _reg(db_session, "x" * 11)


def test_integrity_verification(db_session):
    item = _reg(db_session, "Exact text.")
    v = verify_integrity(db_session, item.id, "Exact text.")
    assert v["verified"] and v["candidate_matches_hash"]
    v = verify_integrity(db_session, item.id, "exact   TEXT.")
    assert not v["verified"] and v["candidate_normalized_match"]
    db_session.execute(update(ContentItem).where(ContentItem.id == item.id).values(text="tampered"))
    db_session.expire_all()
    assert verify_integrity(db_session, item.id)["stored_text_matches_hash"] is False


def test_signature_roundtrip(db_session, monkeypatch):
    monkeypatch.setenv("PROVENANCE_SIGNING_KEY", "sign-key")
    reset_settings_cache()
    item = _reg(db_session, "signed")
    assert item.signature
    assert verify_integrity(db_session, item.id)["signature_valid"] is True
    item.label = "ai"  # tamper a signed field without re-signing
    item.label_source = "x"
    db_session.flush()
    assert verify_integrity(db_session, item.id)["signature_valid"] is False


def test_no_raw_text_mode(db_session, monkeypatch):
    monkeypatch.setenv("STORE_RAW_TEXT", "false")
    reset_settings_cache()
    item = _reg(db_session, "private text")
    assert item.text is None and item.content_hash
    assert verify_integrity(db_session, item.id, "private text")["verified"]


def test_audit_chain_and_tamper_detection(db_session):
    for i in range(5):
        _reg(db_session, f"t{i}")
    db_session.flush()
    r = verify_chain(db_session)
    assert r["valid"] and r["events_checked"] == 5
    db_session.execute(update(AuditEvent).where(AuditEvent.id == 3).values(actor="mallory"))
    db_session.expire_all()
    r = verify_chain(db_session)
    assert not r["valid"] and r["broken_at_event_id"] == 3


def test_provenance_record_shape(db_session):
    root = _reg(db_session, "root")
    rec = provenance_record(db_session, root.id)
    assert rec["content"].id == root.id
    assert rec["integrity"]["stored_text_matches_hash"] is True
