"""Read and validate TOML configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    """Flat configuration for the data processing pipeline."""

    timezone: str = "Asia/Shanghai"
    seed: int = 42
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    universe_core: List[str] = field(
        default_factory=lambda: ["510050.SS", "510300.SS", "510500.SS", "159915.SZ"]
    )
    universe_optional: List[str] = field(default_factory=lambda: ["588000.SS"])
    cross_market: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IEUR"])
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    target_definition: str = "open_to_prev_close_log_return"
    target_horizon: int = 1
    top_k_features: int | None = None  # GRA-GRG screening; None = keep all


def load_config(path: Path) -> PipelineConfig:
    """Parse a TOML file and return a PipelineConfig."""

    raw_bytes = path.read_bytes()
    # Strip UTF-8 BOM if present (common on Windows)
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    raw = tomllib.loads(raw_bytes.decode("utf-8"))

    proj = raw.get("project", {})
    data = raw.get("data", {})
    feat = raw.get("features", {})
    tgt = raw.get("target", {})

    return PipelineConfig(
        timezone=proj.get("timezone", "Asia/Shanghai"),
        seed=proj.get("seed", 42),
        raw_dir=data.get("raw_dir", "data/raw"),
        interim_dir=data.get("interim_dir", "data/interim"),
        processed_dir=data.get("processed_dir", "data/processed"),
        universe_core=data.get(
            "universe_core", data.get("universe", ["510300.SS", "510500.SS"])
        ),
        universe_optional=data.get("universe_optional", ["588000.SS"]),
        cross_market=data.get(
            "cross_market_features", data.get("cross_market", ["SPY", "QQQ", "IEUR"])
        ),
        lookback_windows=feat.get("lookback_windows", [5, 10, 20]),
        target_definition=tgt.get("definition", "open_to_prev_close_log_return"),
        target_horizon=tgt.get("horizon", 1),
        top_k_features=feat.get("top_k_features", None),
    )
