"""Read and validate TOML configuration for the pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DashboardTargetConfig:
    """Dashboard target: multi-horizon raw close-price forecast."""

    horizons: List[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass(frozen=True)
class BarrierConfig:
    """Triple-barrier ternary gate label construction."""

    horizon: int = 10
    upper_multiplier: float = 1.0
    lower_multiplier: float = 1.0
    vol_lookback: int = 20
    target_scaling: str = "daily"  # {"daily", "sqrt_horizon"}


@dataclass(frozen=True)
class GateConfig:
    """Gate-specific feature parameters."""

    cost_threshold: float = 0.0016  # round-trip cost in log-return terms
    gap_threshold: float = 0.005   # abs gap threshold for gap_freq_10


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection and backend selection."""

    backend: str = "file"      # "file" | "db" | "hybrid"
    url: str = ""              # PostgreSQL connection URL
    url_env: str = ""          # env var name for URL (overrides url)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Two-stage feature selection: Pearson filter + XGBoost importance."""

    correlation_threshold: float = 0.90
    importance_top_k: int | None = None  # None = keep all post-correlation
    protected_prefixes: List[str] = field(
        default_factory=lambda: [
            "spy_", "qqq_", "ieur_", "vix_", "us10y_", "dxy_",
            "rel10_", "zscore_ret10d_cross", "rank_volume_cross",
        ]
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the data processing pipeline."""

    timezone: str = "Asia/Shanghai"
    seed: int = 42
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    universe_core: List[str] = field(
        default_factory=lambda: [
            "510050.SS", "510300.SS", "510500.SS", "510880.SS", "588000.SS",
            "159915.SZ", "513130.SS", "512690.SS", "512170.SS", "512480.SS",
            "516160.SS", "518880.SS",
        ]
    )
    universe_optional: List[str] = field(default_factory=list)
    unseen_etfs: List[str] = field(
        default_factory=lambda: ["512880.SS", "159919.SZ"]
    )
    # dashboard_etfs removed — non-core ETFs are now discovered dynamically
    # and stored in instrument_master (see scripts/db_discover_etfs.py)
    cross_market: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IEUR"])
    global_risk: List[str] = field(default_factory=lambda: ["VIX", "TNX", "DXY"])
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    dashboard_target: DashboardTargetConfig = field(default_factory=DashboardTargetConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model training and evaluation."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    optuna_trials: int = 100
    optuna_pruning: bool = True
    optuna_pruner_startup_trials: int = 10
    optuna_pruner_warmup_steps: int = 50
    early_stopping_patience: int = 10
    model_dir: str = "models"
    output_dir: str = "outputs"
    seed: int = 42


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtest simulation."""

    initial_cash: float = 1_000_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    max_gross_leverage: float = 1.0
    risk_free_rate: float = 0.02


def load_config(path: Path) -> PipelineConfig:
    """Parse a TOML file and return a PipelineConfig."""

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    raw = tomllib.loads(raw_bytes.decode("utf-8"))

    proj = raw.get("project", {})
    data = raw.get("data", {})
    feat = raw.get("features", {})

    # Barrier config
    b = raw.get("barrier", {})
    barrier = BarrierConfig(
        horizon=b.get("horizon", 10),
        upper_multiplier=b.get("upper_multiplier", 1.0),
        lower_multiplier=b.get("lower_multiplier", 1.0),
        vol_lookback=b.get("vol_lookback", 20),
        target_scaling=b.get("target_scaling", "daily"),
    )

    # Gate config
    gc = raw.get("gate", {})
    gate = GateConfig(
        cost_threshold=gc.get("cost_threshold", 0.0016),
        gap_threshold=gc.get("gap_threshold", 0.005),
    )

    # Dashboard target
    dt = raw.get("target", {}).get("dashboard", {})
    dashboard_target = DashboardTargetConfig(
        horizons=dt.get("horizons", [1, 5, 10]),
    )

    # Feature selection
    fs = feat.get("selection", {})
    _default_protected = [
        "spy_", "qqq_", "ieur_", "vix_", "us10y_", "dxy_",
        "rel10_", "zscore_ret10d_cross", "rank_volume_cross",
    ]
    feature_selection = FeatureSelectionConfig(
        correlation_threshold=fs.get("correlation_threshold", 0.90),
        importance_top_k=fs.get("importance_top_k", None),
        protected_prefixes=fs.get("protected_prefixes", _default_protected),
    )

    # Database config
    db = raw.get("database", {})
    database = DatabaseConfig(
        backend=db.get("backend", "file"),
        url=db.get("url", ""),
        url_env=db.get("url_env", ""),
    )

    return PipelineConfig(
        timezone=proj.get("timezone", "Asia/Shanghai"),
        seed=proj.get("seed", 42),
        raw_dir=data.get("raw_dir", "data/raw"),
        interim_dir=data.get("interim_dir", "data/interim"),
        processed_dir=data.get("processed_dir", "data/processed"),
        universe_core=data.get(
            "universe_core", data.get("universe", ["510300.SS", "510500.SS"])
        ),
        universe_optional=data.get("universe_optional", []),
        unseen_etfs=data.get("unseen_etfs", ["512880.SS", "159919.SZ"]),
        # dashboard_etfs removed — discovered dynamically via db_discover_etfs
        cross_market=data.get(
            "cross_market_features", data.get("cross_market", ["SPY", "QQQ", "IEUR"])
        ),
        global_risk=data.get("global_risk", ["VIX", "TNX", "DXY"]),
        lookback_windows=feat.get("lookback_windows", [5, 10, 20]),
        barrier=barrier,
        gate=gate,
        dashboard_target=dashboard_target,
        feature_selection=feature_selection,
        database=database,
    )


def load_model_config(path: Path) -> ModelConfig:
    """Parse [model] section from TOML and return ModelConfig."""

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    raw = tomllib.loads(raw_bytes.decode("utf-8"))

    m = raw.get("model", {})
    seed = m.get("seed", raw.get("project", {}).get("seed", 42))

    return ModelConfig(
        train_ratio=m.get("train_ratio", 0.70),
        val_ratio=m.get("val_ratio", 0.15),
        test_ratio=m.get("test_ratio", 0.15),
        optuna_trials=m.get("optuna_trials", 100),
        optuna_pruning=m.get("optuna_pruning", True),
        optuna_pruner_startup_trials=m.get("optuna_pruner_startup_trials", 10),
        optuna_pruner_warmup_steps=m.get("optuna_pruner_warmup_steps", 50),
        early_stopping_patience=m.get("early_stopping_patience", 10),
        model_dir=m.get("model_dir", "models"),
        output_dir=m.get("output_dir", "outputs"),
        seed=seed,
    )


def load_backtest_config(path: Path) -> BacktestConfig:
    """Parse [backtest] section from TOML and return BacktestConfig."""

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    raw = tomllib.loads(raw_bytes.decode("utf-8"))

    b = raw.get("backtest", {})
    return BacktestConfig(
        initial_cash=b.get("initial_cash", 1_000_000.0),
        fee_bps=b.get("fee_bps", 5.0),
        slippage_bps=b.get("slippage_bps", 3.0),
        max_gross_leverage=b.get("max_gross_leverage", 1.0),
        risk_free_rate=b.get("risk_free_rate", 0.02),
    )
