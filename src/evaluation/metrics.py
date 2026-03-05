"""Regression evaluation metrics for return prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for standard regression evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "directional_accuracy": self.directional_accuracy,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.8f}  RMSE={self.rmse:.6f}  MAE={self.mae:.6f}  "
            f"R2={self.r2:.4f}  DirAcc={self.directional_accuracy:.4f}  N={self.n_samples}"
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute regression metrics including directional accuracy."""
    n = len(y_true)
    residuals = y_true - y_pred

    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    sign_match = np.sign(y_pred) == np.sign(y_true)
    dir_acc = float(np.mean(sign_match))

    return RegressionMetrics(
        mse=mse, rmse=rmse, mae=mae, r2=r2,
        directional_accuracy=dir_acc, n_samples=n,
    )
