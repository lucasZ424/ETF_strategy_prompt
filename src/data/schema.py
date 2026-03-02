"""Schema validation helpers for open-universe ETF inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd


# Keep this list synchronized with training/inference feature builders.
REQUIRED_BASE_COLUMNS: Sequence[str] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True)
class ValidationResult:
    """Result container for schema and history checks."""

    is_valid: bool
    errors: List[str]


def validate_open_universe_input(
    frame: pd.DataFrame,
    required_feature_columns: Iterable[str],
    min_history_rows: int = 60,
    max_missing_feature_ratio: float = 0.05,
) -> ValidationResult:
    """Validates inference input for one ETF request.

    Checks:
    - Required base columns exist.
    - Required feature columns exist.
    - Data has at least ``min_history_rows`` rows.
    - Date column is monotonic increasing.
    - Missing ratio in required features is below threshold.
    """

    errors: List[str] = []
    required_cols = list(REQUIRED_BASE_COLUMNS) + list(required_feature_columns)

    missing_cols = [c for c in required_cols if c not in frame.columns]
    if missing_cols:
        errors.append(f"missing_columns={missing_cols}")

    if frame.shape[0] < min_history_rows:
        errors.append(
            f"insufficient_history_rows={frame.shape[0]}<required_{min_history_rows}"
        )

    if "date" in frame.columns:
        date_series = pd.to_datetime(frame["date"], errors="coerce")
        if date_series.isna().any():
            errors.append("invalid_date_values")
        elif not date_series.is_monotonic_increasing:
            errors.append("date_not_monotonic_increasing")

    feature_cols = [c for c in required_feature_columns if c in frame.columns]
    if feature_cols:
        missing_ratio = float(frame[feature_cols].isna().mean().max())
        if missing_ratio > max_missing_feature_ratio:
            errors.append(
                "feature_missing_ratio_above_threshold="
                f"{missing_ratio:.4f}>{max_missing_feature_ratio:.4f}"
            )

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)
