"""FastAPI dependencies: DB sessions and API-key authentication with role-based access."""

from __future__ import annotations

import hmac
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.config import Role, get_settings
from backend.db.session import get_sessionmaker

ROLE_RANK = {"viewer": 0, "researcher": 1, "admin": 2}


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role


def get_db() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_principal(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal:
    s = get_settings()
    if not s.auth_enabled:
        return Principal(name="local", role="admin")
    if not x_api_key:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header", headers={"WWW-Authenticate": "ApiKey"}
        )
    for i, (key, role) in enumerate(s.api_key_roles().items()):
        if hmac.compare_digest(key.encode(), x_api_key.encode()):
            return Principal(name=f"key-{i}:{role}", role=role)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key", headers={"WWW-Authenticate": "ApiKey"})


def require(role: Role):
    def _dep(p: Principal = Depends(get_principal)) -> Principal:
        if ROLE_RANK[p.role] < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role '{role}' (you are '{p.role}')")
        return p

    return _dep


viewer = require("viewer")
researcher = require("researcher")
admin = require("admin")
