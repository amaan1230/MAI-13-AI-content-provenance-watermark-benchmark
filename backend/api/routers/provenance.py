from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import Principal, get_db, viewer
from backend.api.schemas import ProvenanceOut, VerifyIn
from backend.provenance.registry import provenance_record, verify_integrity

router = APIRouter(prefix="/api/v1/provenance", tags=["provenance"])


@router.get("/{content_id}", response_model=ProvenanceOut)
def get_provenance(content_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    """Content record with its full lineage (ancestors and descendants), watermark runs,
    transformations, detection history and integrity checks."""
    return provenance_record(db, content_id)


@router.post("/{content_id}/verify")
def verify(content_id: str, body: VerifyIn, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    """Check whether a candidate text is byte-identical (or normalisation-equivalent) to the registered content."""
    return verify_integrity(db, content_id, body.text)
