"""CLI entry point: python scripts/train_xgboost.py [--config path/to/config.toml]

Placeholder — model training logic is pending redesign.
The previous ranker + gate dual-model approach was removed because the
engineered features (vol-adjusted 5d forward returns) had poor correlation
with the converted y_rank target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost for ETF strategy (pending redesign)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    print("Model training is pending redesign.")
    print("Feature pipeline is intact — run scripts/run_pipeline.py first.")


if __name__ == "__main__":
    main()
