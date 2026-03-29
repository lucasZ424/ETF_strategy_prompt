"""Database engine and session management.

Provides a cached SQLAlchemy engine and session factory for the ETF strategy
database.  The DB URL is resolved from the ``DatabaseConfig`` — either a
direct URL string or an environment-variable override.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.data.models_db import Base

logger = logging.getLogger(__name__)


def resolve_db_url(url: str = "", url_env: str = "") -> str:
    """Return the effective database URL.

    Priority: environment variable ``url_env`` > explicit ``url``.
    Raises ``ValueError`` if neither is set.
    """
    if url_env:
        env_val = os.environ.get(url_env, "")
        if env_val:
            return env_val
    if url:
        return url
    raise ValueError(
        "No database URL configured. Set [database].url in the TOML config "
        "or provide an environment variable via [database].url_env."
    )


@lru_cache(maxsize=1)
def get_engine(db_url: str, echo: bool = False) -> Engine:
    """Create (and cache) a SQLAlchemy engine for *db_url*.

    Parameters
    ----------
    db_url : str
        PostgreSQL connection URL, e.g.
        ``postgresql://user:pass@localhost:5432/etf_strategy``
    echo : bool
        If True, SQLAlchemy logs all SQL statements (useful for debugging).
    """
    logger.info("Creating SQLAlchemy engine for %s", db_url.split("@")[-1])
    engine = create_engine(
        db_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to *engine*."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables that do not yet exist.

    This is a convenience for development and first-time setup.  For
    production, prefer Alembic migrations.
    """
    Base.metadata.create_all(engine)
    logger.info("Database tables ensured via create_all.")
