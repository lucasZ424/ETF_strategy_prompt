"""CLI entry point: python scripts/run_pipeline.py [--config path/to/config.toml]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETF data processing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Construct backend when DB or hybrid mode is active.
    backend = None
    if config.database.backend != "file":
        from src.data.backend import DataBackend
        backend = DataBackend(config, PROJECT_ROOT)

    output_paths = run_pipeline(config, PROJECT_ROOT, backend=backend)

    print("\nPipeline complete. Outputs:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
