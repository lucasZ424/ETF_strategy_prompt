"""Initial schema - 6 core tables for DB migration Phases 1-7.

Revision ID: 001
Revises: None
Create Date: 2026-03-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---- instrument_master ----
    op.create_table(
        "instrument_master",
        sa.Column("instrument_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("market", sa.String(10), nullable=False),
        sa.Column("asset_type", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100)),
        sa.Column("yfinance_ticker", sa.String(30)),
        sa.Column("is_core_training", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_cross_market", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_macro_proxy", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("history_policy_years", sa.Integer, nullable=False, server_default="10"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ---- daily_bars ----
    op.create_table(
        "daily_bars",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column(
            "instrument_id",
            sa.Integer,
            sa.ForeignKey("instrument_master.instrument_id"),
            nullable=False,
        ),
        sa.Column("open", sa.Float),
        sa.Column("high", sa.Float),
        sa.Column("low", sa.Float),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("adj_close", sa.Float),
        sa.Column("volume", sa.Float),
        sa.Column("data_source", sa.String(20), server_default="yfinance"),
        sa.Column("loaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("symbol", "trade_date"),
    )
    op.create_index("idx_daily_bars_date", "daily_bars", ["trade_date"])
    op.create_index("idx_daily_bars_instrument", "daily_bars", ["instrument_id", "trade_date"])

    # ---- data_fetch_log ----
    op.create_table(
        "data_fetch_log",
        sa.Column("fetch_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("fetch_start_date", sa.Date, nullable=False),
        sa.Column("fetch_end_date", sa.Date, nullable=False),
        sa.Column("rows_fetched", sa.Integer, server_default="0"),
        sa.Column("rows_inserted", sa.Integer, server_default="0"),
        sa.Column("rows_updated", sa.Integer, server_default="0"),
        sa.Column("status", sa.String(20), server_default="success"),
        sa.Column("error_message", sa.Text),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ---- prediction_snapshots ----
    op.create_table(
        "prediction_snapshots",
        sa.Column("prediction_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(50)),
        sa.Column("model_name", sa.String(50), server_default="dashboard"),
        sa.Column("model_version", sa.String(50)),
        sa.Column("asof_date", sa.Date, nullable=False),
        sa.Column("prediction_time", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("current_close", sa.Float, nullable=False),
        sa.Column("y_close_1d_pred", sa.Float),
        sa.Column("y_close_3d_pred", sa.Float),
        sa.Column("y_close_5d_pred", sa.Float),
        sa.Column("feature_manifest_version", sa.String(50)),
        sa.Column("data_freshness_date", sa.Date),
        sa.UniqueConstraint("model_version", "asof_date", "symbol", name="uq_pred_version_date_symbol"),
    )
    op.create_index("idx_pred_asof", "prediction_snapshots", ["asof_date", "symbol"])

    # ---- model_runs (Phase 6+) ----
    op.create_table(
        "model_runs",
        sa.Column("run_id", sa.String(50), primary_key=True),
        sa.Column("model_name", sa.String(50), nullable=False),
        sa.Column("model_version", sa.String(50)),
        sa.Column("config_hash", sa.String(64)),
        sa.Column("train_start_date", sa.Date),
        sa.Column("train_end_date", sa.Date),
        sa.Column("val_start_date", sa.Date),
        sa.Column("val_end_date", sa.Date),
        sa.Column("test_start_date", sa.Date),
        sa.Column("test_end_date", sa.Date),
        sa.Column("metrics_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ---- dashboard_metrics_history (Phase 7+) ----
    op.create_table(
        "dashboard_metrics_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(50), sa.ForeignKey("model_runs.run_id")),
        sa.Column("model_version", sa.String(50)),
        sa.Column("horizon_name", sa.String(20), nullable=False),
        sa.Column("mse", sa.Float),
        sa.Column("rmse", sa.Float),
        sa.Column("mae", sa.Float),
        sa.Column("mape", sa.Float),
        sa.Column("r2", sa.Float),
        sa.Column("n_rows", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("dashboard_metrics_history")
    op.drop_table("model_runs")
    op.drop_index("idx_pred_asof", table_name="prediction_snapshots")
    op.drop_table("prediction_snapshots")
    op.drop_table("data_fetch_log")
    op.drop_index("idx_daily_bars_instrument", table_name="daily_bars")
    op.drop_index("idx_daily_bars_date", table_name="daily_bars")
    op.drop_table("daily_bars")
    op.drop_table("instrument_master")
