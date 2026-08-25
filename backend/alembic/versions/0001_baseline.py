"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-10 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


model_format_enum = sa.Enum("HF", "ONNX", "GGUF", "TRT", name="modelformat")


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("cuda_version", sa.String(), nullable=True),
        sa.Column("driver_version", sa.String(), nullable=True),
        sa.Column("hardware", sa.String(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("family", sa.String(), nullable=True),
        sa.Column("architecture", sa.String(), nullable=True),
        sa.Column("parameters", sa.String(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=True),
        sa.Column("tokenizer_type", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(op.f("ix_models_name"), "models", ["name"], unique=True)
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_id", sa.Integer(), nullable=True),
        sa.Column("version_tag", sa.String(), nullable=True),
        sa.Column("format", model_format_enum, nullable=True),
        sa.Column("sha256", sa.String(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
    )
    op.create_table(
        "benchmarks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("runtime", sa.String(), nullable=True),
        sa.Column("batch_size", sa.Integer(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("p99_latency_ms", sa.Float(), nullable=True),
        sa.Column("throughput_tps", sa.Float(), nullable=True),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("memory_usage_gb", sa.Float(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
    )
    op.create_table(
        "evaluations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("benchmark_name", sa.String(), nullable=True),
        sa.Column("score", sa.String(), nullable=True),
        sa.Column("delta_from_base", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
    )
    op.create_table(
        "quantizations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("method", sa.String(), nullable=True),
        sa.Column("precision", sa.String(), nullable=True),
        sa.Column("mse_error", sa.Float(), nullable=True),
        sa.Column("perplexity_delta", sa.Float(), nullable=True),
        sa.Column("memory_reduction_pct", sa.Float(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
    )


def downgrade() -> None:
    op.drop_table("quantizations")
    op.drop_table("evaluations")
    op.drop_table("benchmarks")
    op.drop_table("model_versions")
    op.drop_index(op.f("ix_models_name"), table_name="models")
    op.drop_table("models")
    op.drop_table("experiments")
    model_format_enum.drop(op.get_bind(), checkfirst=False)
