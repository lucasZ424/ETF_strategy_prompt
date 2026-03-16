"""Evaluation metrics: ranking quality + classification measures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import warnings 
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k (sorted descending by predicted score)."""
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0
    gains = relevance / np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(gains))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """NDCG@k for a single query (one date's ETF slate)."""
    order = np.argsort(y_score)[::-1]
    sorted_true = y_true[order]
    # Shift to non-negative relevance (min=0)
    shifted = sorted_true - sorted_true.min()
    dcg = _dcg_at_k(shifted, k)
    ideal_order = np.argsort(shifted)[::-1]
    idcg = _dcg_at_k(shifted[ideal_order], k)
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class RankerMetrics:
    ndcg_at_3: float
    ndcg_at_5: float
    spearman_rho: float
    spearman_pval: float
    n_dates: int
    n_rows: int

    def to_dict(self) -> Dict:
        return {
            "ndcg_at_3": self.ndcg_at_3,
            "ndcg_at_5": self.ndcg_at_5,
            "spearman_rho": self.spearman_rho,
            "spearman_pval": self.spearman_pval,
            "n_dates": self.n_dates,
            "n_rows": self.n_rows,
        }

    def __str__(self) -> str:
        return (
            f"NDCG@3={self.ndcg_at_3:.4f}  NDCG@5={self.ndcg_at_5:.4f}  "
            f"Spearman={self.spearman_rho:.4f} (p={self.spearman_pval:.3f})  "
            f"N_dates={self.n_dates}  N_rows={self.n_rows}"
        )


def compute_ranker_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    y_col: str = "target",
    date_col: str = "date",
) -> RankerMetrics:
    """Compute per-date NDCG@k and pooled Spearman rho.

    Parameters
    ----------
    df : DataFrame with columns [date, <y_col>, ...]
    scores : predicted ranking scores (same row order as df)
    y_col : name of the ground-truth target column
    """
    df = df.copy()
    df["_score"] = scores

    ndcg3_list: List[float] = []
    ndcg5_list: List[float] = []

    for _, g in df.groupby(date_col):
        if len(g) < 2:
            continue
        y = g[y_col].values
        s = g["_score"].values
        ndcg3_list.append(ndcg_at_k(y, s, k=3))
        ndcg5_list.append(ndcg_at_k(y, s, k=5))

    rho, pval = spearmanr(df[y_col].values, df["_score"].values)

    return RankerMetrics(
        ndcg_at_3=float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
        ndcg_at_5=float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
        spearman_rho=float(rho),
        spearman_pval=float(pval),
        n_dates=len(ndcg3_list),
        n_rows=len(df),
    )


# ---------------------------------------------------------------------------
# Regression metrics (alpha model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    n_rows: int

    def to_dict(self) -> Dict:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "n_rows": self.n_rows,
        }

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.6f}  RMSE={self.rmse:.6f}  MAE={self.mae:.6f}  "
            f"R2={self.r2:.4f}  N={self.n_rows}"
        )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Compute standard regression metrics for the alpha model."""
    mse = float(mean_squared_error(y_true, y_pred))
    return RegressionMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        n_rows=len(y_true),
    )


# ---------------------------------------------------------------------------
# Classification metrics (gate / regime models)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    confusion: List[List[int]]
    class_labels: List[str]
    n_rows: int

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_per_class": self.f1_per_class,
            "confusion_matrix": self.confusion,
            "class_labels": self.class_labels,
            "n_rows": self.n_rows,
        }

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.4f}  F1_macro={self.f1_macro:.4f}  "
            f"N={self.n_rows}"
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str] | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics for gate or regime models."""
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if class_labels is None:
        class_labels = [str(c) for c in unique_classes]

    acc = float(accuracy_score(y_true, y_pred))
    f1_mac = float(f1_score(y_true, y_pred, average="macro"))
    f1_per = f1_score(y_true, y_pred, average=None, labels=unique_classes)
    f1_dict = {label: float(v) for label, v in zip(class_labels, f1_per)}
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes).tolist()

    return ClassificationMetrics(
        accuracy=acc,
        f1_macro=f1_mac,
        f1_per_class=f1_dict,
        confusion=cm,
        class_labels=class_labels,
        n_rows=len(y_true),
    )


# ---------------------------------------------------------------------------
# Dashboard metrics (per-horizon price prediction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardMetrics:
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    n_rows: int

    def to_dict(self) -> Dict:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "n_rows": self.n_rows,
        }

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.4f}  RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  "
            f"MAPE={self.mape:.4f}%  R2={self.r2:.6f}  N={self.n_rows}"
        )


def compute_dashboard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> DashboardMetrics:
    """Compute regression + MAPE metrics for dashboard price predictions."""
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    # MAPE: mean absolute percentage error (exclude zeros to avoid inf)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return DashboardMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=mae,
        mape=mape,
        r2=float(r2_score(y_true, y_pred)),
        n_rows=len(y_true),
    )
