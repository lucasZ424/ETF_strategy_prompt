"""Backend-agnostic data access dispatcher.

``DataBackend`` wraps both the file-based loaders and the DB repository
behind a single interface.  Callers (pipeline, scripts) interact with
``DataBackend`` — the backend selection is driven by the TOML config's
``[database].backend`` value: ``"file"`` | ``"db"`` | ``"hybrid"``.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backend modes."""

    FILE = "file"
    DB = "db"
    HYBRID = "hybrid"  # DB first, fall back to file on error


class DataBackend:
    """Unified entry-point for loading raw market data.

    Public methods mirror the signatures that ``pipeline.py`` and scripts
    already expect, returning DataFrames with identical schemas regardless
    of the active backend.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration (includes ``database`` sub-config and
        universe lists).
    project_root : Path
        Absolute path to the project root directory.
    """

    def __init__(self, config: PipelineConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.backend = StorageBackend(config.database.backend)

        self._repo = None
        if self.backend in (StorageBackend.DB, StorageBackend.HYBRID):
            from src.data.db import get_engine, get_session_factory, resolve_db_url
            from src.data.repository import BarRepository

            db_url = resolve_db_url(config.database.url, config.database.url_env)
            engine = get_engine(db_url)
            self._repo = BarRepository(get_session_factory(engine))

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load_china_etfs(self) -> pd.DataFrame:
        """Load pooled China ETF bars.

        Returns ``[date, open, high, low, close, adj_close, volume, symbol]``.
        """
        if self.backend == StorageBackend.FILE:
            return self._file_load_china()
        if self.backend == StorageBackend.DB:
            return self._repo.load_china_etf_bars(
                self.config.universe_core, self.config.universe_optional
            )
        # HYBRID
        try:
            return self._repo.load_china_etf_bars(
                self.config.universe_core, self.config.universe_optional
            )
        except Exception:
            logger.warning("DB load failed for China ETFs, falling back to file.")
            return self._file_load_china()

    def load_cross_market_etfs(self) -> pd.DataFrame:
        """Load cross-market ETF bars (SPY, QQQ, IEUR)."""
        if self.backend == StorageBackend.FILE:
            return self._file_load_cross_market()
        if self.backend == StorageBackend.DB:
            return self._repo.load_cross_market_bars(self.config.cross_market)
        # HYBRID
        try:
            return self._repo.load_cross_market_bars(self.config.cross_market)
        except Exception:
            logger.warning("DB load failed for cross-market, falling back to file.")
            return self._file_load_cross_market()

    def load_macro_series(self, symbol: str) -> pd.DataFrame:
        """Load a single macro proxy series (VIX, TNX, DXY).

        Returns ``[date, close]`` matching ``cross_market._load_and_clean_series()``
        output schema.
        """
        if self.backend == StorageBackend.FILE:
            return self._file_load_macro(symbol)
        if self.backend == StorageBackend.DB:
            return self._repo.load_macro_series(symbol)
        # HYBRID
        try:
            return self._repo.load_macro_series(symbol)
        except Exception:
            logger.warning("DB load failed for %s, falling back to file.", symbol)
            return self._file_load_macro(symbol)

    def load_unseen_etfs(self, symbols: List[str] | None = None) -> pd.DataFrame:
        """Load unseen / out-of-sample ETF bars."""
        if self.backend == StorageBackend.FILE:
            return self._file_load_unseen(symbols)
        if self.backend == StorageBackend.DB:
            return self._repo.load_unseen_bars(symbols)
        # HYBRID
        try:
            return self._repo.load_unseen_bars(symbols)
        except Exception:
            logger.warning("DB load failed for unseen ETFs, falling back to file.")
            return self._file_load_unseen(symbols)

    # ------------------------------------------------------------------
    # File-backend helpers (delegate to existing loader.py)
    # ------------------------------------------------------------------

    def _file_load_china(self) -> pd.DataFrame:
        from src.data.loader import load_china_etfs

        raw_dir = self.project_root / self.config.raw_dir
        return load_china_etfs(
            raw_dir, self.config.universe_core, self.config.universe_optional
        )

    def _file_load_cross_market(self) -> pd.DataFrame:
        from src.data.loader import load_cross_market_etfs

        raw_dir = self.project_root / self.config.raw_dir
        return load_cross_market_etfs(raw_dir, self.config.cross_market)

    def _file_load_macro(self, symbol: str) -> pd.DataFrame:
        from src.data.cross_market import _load_and_clean_series

        raw_dir = self.project_root / self.config.raw_dir
        csv_path = raw_dir / "cross_market" / f"{symbol}.csv"
        return _load_and_clean_series(csv_path)

    def _file_load_unseen(self, symbols: List[str] | None) -> pd.DataFrame:
        from src.data.loader import load_unseen_etfs

        data_dir = self.project_root / "data"
        return load_unseen_etfs(data_dir, symbols)
