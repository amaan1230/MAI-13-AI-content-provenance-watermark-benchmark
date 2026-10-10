"""Programmatic Alembic helpers (used by app startup, CLI and tests)."""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from backend.core.config import PROJECT_ROOT, get_settings


def alembic_config(url: str | None = None) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "backend" / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", (url or get_settings().database_url).replace("%", "%%"))
    cfg.attributes["configure_logger"] = False
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    get_settings().ensure_dirs()
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


if __name__ == "__main__":
    upgrade()
    print("Database migrated to head.")
