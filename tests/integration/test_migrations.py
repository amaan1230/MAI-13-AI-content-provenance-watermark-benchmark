from __future__ import annotations

import os

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from backend.db.migrate import downgrade, upgrade
from backend.models import Base

TABLES = {
    "content_items",
    "dataset_versions",
    "model_configs",
    "watermark_runs",
    "transformations",
    "detection_results",
    "experiment_runs",
    "experiment_artifacts",
    "audit_events",
}


def _check_roundtrip(url: str):
    upgrade(url)
    eng = create_engine(url)
    assert TABLES <= set(inspect(eng).get_table_names())
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    assert diff == [], f"models and migrations differ: {diff}"
    downgrade(url)
    assert not (TABLES & set(inspect(eng).get_table_names()))
    upgrade(url)
    assert TABLES <= set(inspect(eng).get_table_names())
    eng.dispose()


def test_sqlite_migration_roundtrip(settings_env):
    _check_roundtrip(f"sqlite:///{(settings_env / 'mig.db').as_posix()}")


@pytest.mark.postgres
@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_postgres_migration_roundtrip(settings_env):
    _check_roundtrip(os.environ["TEST_POSTGRES_URL"])
