"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-10 14:58:48.186973

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    op.create_table('audit_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor', sa.String(length=100), nullable=False),
    sa.Column('action', sa.String(length=100), nullable=False),
    sa.Column('entity_type', sa.String(length=50), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('request_id', sa.String(length=64), nullable=True),
    sa.Column('details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('prev_hash', sa.String(length=64), nullable=True),
    sa.Column('event_hash', sa.String(length=64), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_events')),
    sa.UniqueConstraint('event_hash', name=op.f('uq_audit_events_event_hash'))
    )
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        batch_op.create_index('ix_audit_events_entity', ['entity_type', 'entity_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_audit_events_ts'), ['ts'], unique=False)

    op.create_table('dataset_versions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('version', sa.String(length=50), nullable=False),
    sa.Column('source', sa.String(length=500), nullable=True),
    sa.Column('license', sa.String(length=200), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('manifest_uri', sa.String(length=1000), nullable=True),
    sa.Column('manifest_sha256', sa.String(length=64), nullable=True),
    sa.Column('n_items', sa.Integer(), nullable=False),
    sa.Column('split_seed', sa.Integer(), nullable=True),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_versions')),
    sa.UniqueConstraint('name', 'version', name=op.f('uq_dataset_versions_name'))
    )
    op.create_table('model_configs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('model_name', sa.String(length=300), nullable=False),
    sa.Column('revision', sa.String(length=100), nullable=True),
    sa.Column('role', sa.String(length=50), nullable=False),
    sa.Column('params', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_configs')),
    sa.UniqueConstraint('config_hash', name=op.f('uq_model_configs_config_hash'))
    )
    op.create_table('experiment_runs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('experiment_type', sa.String(length=50), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('stage', sa.String(length=100), nullable=True),
    sa.Column('progress', sa.Float(), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('seed', sa.Integer(), nullable=False),
    sa.Column('dataset_version_id', sa.String(length=36), nullable=True),
    sa.Column('rerun_of_id', sa.String(length=36), nullable=True),
    sa.Column('software_version', sa.String(length=50), nullable=False),
    sa.Column('git_commit', sa.String(length=40), nullable=True),
    sa.Column('environment', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
    sa.Column('metrics_summary', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('error_details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('created_by', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')", name=op.f('ck_experiment_runs_status')),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_experiment_runs_dataset_version_id_dataset_versions'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['rerun_of_id'], ['experiment_runs.id'], name=op.f('fk_experiment_runs_rerun_of_id_experiment_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_experiment_runs'))
    )
    with op.batch_alter_table('experiment_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_experiment_runs_config_hash'), ['config_hash'], unique=False)
        batch_op.create_index(batch_op.f('ix_experiment_runs_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_experiment_runs_experiment_type'), ['experiment_type'], unique=False)
        batch_op.create_index(batch_op.f('ix_experiment_runs_status'), ['status'], unique=False)

    op.create_table('content_items',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('normalized_hash', sa.String(length=64), nullable=False),
    sa.Column('text', sa.Text(), nullable=True),
    sa.Column('char_len', sa.Integer(), nullable=False),
    sa.Column('modality', sa.String(length=20), nullable=False),
    sa.Column('language', sa.String(length=16), nullable=True),
    sa.Column('source_type', sa.String(length=20), nullable=False),
    sa.Column('label', sa.String(length=10), nullable=False),
    sa.Column('label_source', sa.String(length=500), nullable=True),
    sa.Column('dataset_version_id', sa.String(length=36), nullable=True),
    sa.Column('split', sa.String(length=10), nullable=True),
    sa.Column('group_id', sa.String(length=100), nullable=True),
    sa.Column('external_id', sa.String(length=200), nullable=True),
    sa.Column('parent_id', sa.String(length=36), nullable=True),
    sa.Column('model_config_id', sa.String(length=36), nullable=True),
    sa.Column('experiment_run_id', sa.String(length=36), nullable=True),
    sa.Column('generation_config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('signature', sa.String(length=128), nullable=True),
    sa.Column('created_by', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("label IN ('human', 'ai', 'unknown')", name=op.f('ck_content_items_label')),
    sa.CheckConstraint("source_type IN ('human', 'ai', 'ai_watermarked', 'transformed', 'unknown')", name=op.f('ck_content_items_source_type')),
    sa.CheckConstraint("split IS NULL OR split IN ('train', 'val', 'test')", name=op.f('ck_content_items_split')),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_content_items_dataset_version_id_dataset_versions'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['experiment_run_id'], ['experiment_runs.id'], name=op.f('fk_content_items_experiment_run_id_experiment_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['model_config_id'], ['model_configs.id'], name=op.f('fk_content_items_model_config_id_model_configs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['parent_id'], ['content_items.id'], name=op.f('fk_content_items_parent_id_content_items'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_content_items'))
    )
    with op.batch_alter_table('content_items', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_content_items_content_hash'), ['content_hash'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_created_at'), ['created_at'], unique=False)
        batch_op.create_index('ix_content_items_dataset_split', ['dataset_version_id', 'split'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_dataset_version_id'), ['dataset_version_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_experiment_run_id'), ['experiment_run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_group_id'), ['group_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_model_config_id'), ['model_config_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_normalized_hash'), ['normalized_hash'], unique=False)
        batch_op.create_index(batch_op.f('ix_content_items_parent_id'), ['parent_id'], unique=False)

    op.create_table('experiment_artifacts',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('experiment_run_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('kind', sa.String(length=50), nullable=False),
    sa.Column('uri', sa.String(length=1000), nullable=False),
    sa.Column('storage_key', sa.String(length=500), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['experiment_run_id'], ['experiment_runs.id'], name=op.f('fk_experiment_artifacts_experiment_run_id_experiment_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_experiment_artifacts')),
    sa.UniqueConstraint('experiment_run_id', 'name', name=op.f('uq_experiment_artifacts_experiment_run_id'))
    )
    with op.batch_alter_table('experiment_artifacts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_experiment_artifacts_experiment_run_id'), ['experiment_run_id'], unique=False)

    op.create_table('detection_results',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('content_id', sa.String(length=36), nullable=False),
    sa.Column('detector_type', sa.String(length=20), nullable=False),
    sa.Column('detector_name', sa.String(length=100), nullable=False),
    sa.Column('detector_version', sa.String(length=20), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('threshold', sa.Float(), nullable=True),
    sa.Column('decision', sa.String(length=20), nullable=False),
    sa.Column('p_value', sa.Float(), nullable=True),
    sa.Column('n_tokens_scored', sa.Integer(), nullable=True),
    sa.Column('latency_ms', sa.Float(), nullable=True),
    sa.Column('experiment_run_id', sa.String(length=36), nullable=True),
    sa.Column('details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("decision IN ('positive', 'negative', 'insufficient', 'error')", name=op.f('ck_detection_results_decision')),
    sa.CheckConstraint("detector_type IN ('watermark', 'ai_classifier', 'provenance')", name=op.f('ck_detection_results_detector_type')),
    sa.ForeignKeyConstraint(['content_id'], ['content_items.id'], name=op.f('fk_detection_results_content_id_content_items'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['experiment_run_id'], ['experiment_runs.id'], name=op.f('fk_detection_results_experiment_run_id_experiment_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_detection_results'))
    )
    with op.batch_alter_table('detection_results', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_detection_results_content_id'), ['content_id'], unique=False)
        batch_op.create_index('ix_detection_results_exp_detector', ['experiment_run_id', 'detector_name'], unique=False)

    op.create_table('transformations',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('parent_content_id', sa.String(length=36), nullable=False),
    sa.Column('child_content_id', sa.String(length=36), nullable=True),
    sa.Column('attack_type', sa.String(length=50), nullable=False),
    sa.Column('attack_name', sa.String(length=100), nullable=False),
    sa.Column('attack_version', sa.String(length=20), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('seed', sa.BigInteger(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('latency_ms', sa.Float(), nullable=True),
    sa.Column('semantic_similarity', sa.Float(), nullable=True),
    sa.Column('experiment_run_id', sa.String(length=36), nullable=True),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("status IN ('succeeded', 'failed', 'skipped')", name=op.f('ck_transformations_status')),
    sa.ForeignKeyConstraint(['child_content_id'], ['content_items.id'], name=op.f('fk_transformations_child_content_id_content_items'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['experiment_run_id'], ['experiment_runs.id'], name=op.f('fk_transformations_experiment_run_id_experiment_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_content_id'], ['content_items.id'], name=op.f('fk_transformations_parent_content_id_content_items'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_transformations'))
    )
    with op.batch_alter_table('transformations', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_transformations_attack_type'), ['attack_type'], unique=False)
        batch_op.create_index(batch_op.f('ix_transformations_child_content_id'), ['child_content_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_transformations_experiment_run_id'), ['experiment_run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_transformations_parent_content_id'), ['parent_content_id'], unique=False)

    op.create_table('watermark_runs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('method', sa.String(length=50), nullable=False),
    sa.Column('method_version', sa.String(length=20), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('key_fingerprint', sa.String(length=16), nullable=False),
    sa.Column('model_config_id', sa.String(length=36), nullable=True),
    sa.Column('input_content_id', sa.String(length=36), nullable=True),
    sa.Column('output_content_id', sa.String(length=36), nullable=True),
    sa.Column('experiment_run_id', sa.String(length=36), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('n_tokens', sa.Integer(), nullable=True),
    sa.Column('latency_ms', sa.Float(), nullable=True),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['experiment_run_id'], ['experiment_runs.id'], name=op.f('fk_watermark_runs_experiment_run_id_experiment_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['input_content_id'], ['content_items.id'], name=op.f('fk_watermark_runs_input_content_id_content_items'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['model_config_id'], ['model_configs.id'], name=op.f('fk_watermark_runs_model_config_id_model_configs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['output_content_id'], ['content_items.id'], name=op.f('fk_watermark_runs_output_content_id_content_items'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_watermark_runs'))
    )
    with op.batch_alter_table('watermark_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_watermark_runs_config_hash'), ['config_hash'], unique=False)
        batch_op.create_index(batch_op.f('ix_watermark_runs_experiment_run_id'), ['experiment_run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_watermark_runs_method'), ['method'], unique=False)
        batch_op.create_index(batch_op.f('ix_watermark_runs_output_content_id'), ['output_content_id'], unique=False)




def downgrade() -> None:

    with op.batch_alter_table('watermark_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_watermark_runs_output_content_id'))
        batch_op.drop_index(batch_op.f('ix_watermark_runs_method'))
        batch_op.drop_index(batch_op.f('ix_watermark_runs_experiment_run_id'))
        batch_op.drop_index(batch_op.f('ix_watermark_runs_config_hash'))

    op.drop_table('watermark_runs')
    with op.batch_alter_table('transformations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_transformations_parent_content_id'))
        batch_op.drop_index(batch_op.f('ix_transformations_experiment_run_id'))
        batch_op.drop_index(batch_op.f('ix_transformations_child_content_id'))
        batch_op.drop_index(batch_op.f('ix_transformations_attack_type'))

    op.drop_table('transformations')
    with op.batch_alter_table('detection_results', schema=None) as batch_op:
        batch_op.drop_index('ix_detection_results_exp_detector')
        batch_op.drop_index(batch_op.f('ix_detection_results_content_id'))

    op.drop_table('detection_results')
    with op.batch_alter_table('experiment_artifacts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_experiment_artifacts_experiment_run_id'))

    op.drop_table('experiment_artifacts')
    with op.batch_alter_table('content_items', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_content_items_parent_id'))
        batch_op.drop_index(batch_op.f('ix_content_items_normalized_hash'))
        batch_op.drop_index(batch_op.f('ix_content_items_model_config_id'))
        batch_op.drop_index(batch_op.f('ix_content_items_group_id'))
        batch_op.drop_index(batch_op.f('ix_content_items_experiment_run_id'))
        batch_op.drop_index(batch_op.f('ix_content_items_dataset_version_id'))
        batch_op.drop_index('ix_content_items_dataset_split')
        batch_op.drop_index(batch_op.f('ix_content_items_created_at'))
        batch_op.drop_index(batch_op.f('ix_content_items_content_hash'))

    op.drop_table('content_items')
    with op.batch_alter_table('experiment_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_experiment_runs_status'))
        batch_op.drop_index(batch_op.f('ix_experiment_runs_experiment_type'))
        batch_op.drop_index(batch_op.f('ix_experiment_runs_created_at'))
        batch_op.drop_index(batch_op.f('ix_experiment_runs_config_hash'))

    op.drop_table('experiment_runs')
    op.drop_table('model_configs')
    op.drop_table('dataset_versions')
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_audit_events_ts'))
        batch_op.drop_index('ix_audit_events_entity')

    op.drop_table('audit_events')

