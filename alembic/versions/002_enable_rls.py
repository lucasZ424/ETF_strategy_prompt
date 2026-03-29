"""Enable Row Level Security on all tables.

Supabase exposes the public schema via PostgREST.  Without RLS, every
table is world-readable through the Supabase REST API.  Enabling RLS
with no permissive policies for anon/authenticated locks down API
access while the ``postgres`` superuser (used by Grafana and the sync
script) bypasses RLS automatically.

Revision ID: 002
Revises: 001
"""

from alembic import op
from sqlalchemy import text

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

TABLES = [
    "instrument_master",
    "daily_bars",
    "data_fetch_log",
    "prediction_snapshots",
    "model_runs",
    "dashboard_metrics_history",
    "alembic_version",
]

# Supabase-specific roles — only exist on Supabase, not vanilla PostgreSQL.
SUPABASE_ROLES = ["anon", "authenticated"]


def _role_exists(conn, role: str) -> bool:
    result = conn.execute(
        text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}
    )
    return result.scalar() is not None


def upgrade() -> None:
    conn = op.get_bind()

    # Check which Supabase roles exist on this server
    revoke_roles = [r for r in SUPABASE_ROLES if _role_exists(conn, r)]

    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

        if revoke_roles:
            roles_csv = ", ".join(revoke_roles)
            op.execute(f"REVOKE ALL ON {table} FROM {roles_csv}")


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
