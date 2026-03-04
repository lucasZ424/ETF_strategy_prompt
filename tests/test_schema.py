"""Tests for open-universe schema validation."""

import pandas as pd
# import pytest

from src.data.schema import validate_open_universe_input


def _make_frame(n_rows: int = 80) -> pd.DataFrame:
    """Create a minimal valid DataFrame for testing."""
    dates = pd.bdate_range("2024-01-01", periods=n_rows)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SS",
            "open": range(n_rows),
            "high": range(n_rows),
            "low": range(n_rows),
            "close": range(n_rows),
            "volume": range(n_rows),
            "feat_a": range(n_rows),
        }
    )


def test_valid_input_passes():
    result = validate_open_universe_input(_make_frame(), required_feature_columns=["feat_a"])
    assert result.is_valid
    assert result.errors == []


def test_insufficient_history():
    result = validate_open_universe_input(
        _make_frame(n_rows=10), required_feature_columns=["feat_a"], min_history_rows=60
    )
    assert not result.is_valid
    assert any("insufficient_history" in e for e in result.errors)


def test_missing_columns():
    frame = _make_frame().drop(columns=["volume"])
    result = validate_open_universe_input(frame, required_feature_columns=["feat_a"])
    assert not result.is_valid
    assert any("missing_columns" in e for e in result.errors)
