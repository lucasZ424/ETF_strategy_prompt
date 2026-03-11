"""Evaluation metrics: ranking quality measures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
