"""XGBoost training with Optuna hyperparameter optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner

from src.config import ModelConfig

logger = logging.getLogger(__name__)

_META_COLS = {"date", "symbol", "target"}


@dataclass
class TrainResult:
    """Output of the training process."""

    model: xgb.XGBRegressor
    best_params: Dict[str, Any]
    feature_names: List[str]
    val_rmse: float
    eval_history: Dict[str, List[float]]


def _get_feature_cols(df) -> List[str]:
    """Return sorted list of feature column names."""
    return sorted([c for c in df.columns if c not in _META_COLS])


def safe_feature_array(df, feature_cols: List[str]) -> np.ndarray:
    """Extract feature matrix, replacing inf/NaN with 0.0."""
    raw = df[feature_cols].values.astype(float)
    return np.nan_to_num(np.where(np.isinf(raw), np.nan, raw), nan=0.0)


def _make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    patience: int,
):
    """Create Optuna objective closure for XGBoost regression."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        model = xgb.XGBRegressor(
            **params,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            early_stopping_rounds=patience,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

        trial.report(rmse, step=model.best_iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return rmse

    return objective


def train_xgboost(
    train_df,
    val_df,
    config: ModelConfig,
) -> TrainResult:
    """Run Optuna HP search and train final XGBoost model."""
    feature_cols = _get_feature_cols(train_df)
    _X_train_raw = train_df[feature_cols].values.astype(float)
    X_train = np.nan_to_num(np.where(np.isinf(_X_train_raw), np.nan, _X_train_raw), nan=0.0)
    y_train = train_df["target"].values
    _X_val_raw = val_df[feature_cols].values.astype(float)
    X_val = np.nan_to_num(np.where(np.isinf(_X_val_raw), np.nan, _X_val_raw), nan=0.0)
    y_val = val_df["target"].values

    logger.info(
        "Features: %d, Train rows: %d, Val rows: %d",
        len(feature_cols), len(X_train), len(X_val),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=50),
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )

    objective = _make_objective(
        X_train, y_train, X_val, y_val, config.seed, config.early_stopping_patience
    )
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info("Best Optuna params: %s", best_params)
    logger.info("Best val RMSE: %.6f", study.best_value)

    # Retrain final model with best params, recording eval history for plotting
    final_model = xgb.XGBRegressor(
        **best_params,
        random_state=config.seed,
        tree_method="hist",
        verbosity=0,
        early_stopping_rounds=config.early_stopping_patience,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    raw_history = final_model.evals_result()
    eval_history = {
        "train_rmse": raw_history["validation_0"]["rmse"],
        "val_rmse": raw_history["validation_1"]["rmse"],
    }

    logger.info(
        "Final model: best_iteration=%d, best_score=%.6f (patience=%d)",
        final_model.best_iteration, final_model.best_score,
        config.early_stopping_patience,
    )

    return TrainResult(
        model=final_model,
        best_params=best_params,
        feature_names=feature_cols,
        val_rmse=study.best_value,
        eval_history=eval_history,
    )


def save_model(result: TrainResult, model_dir: Path, name: str = "xgboost_v1") -> Path:
    """Save model artifact and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_path = model_dir / f"{name}.json"
    result.model.save_model(str(xgb_path))
    logger.info("Saved XGBoost native model: %s", xgb_path)

    bundle_path = model_dir / f"{name}.joblib"
    joblib.dump(
        {
            "model": result.model,
            "feature_names": result.feature_names,
            "best_params": result.best_params,
            "val_rmse": result.val_rmse,
            "eval_history": result.eval_history,
        },
        bundle_path,
    )
    logger.info("Saved joblib bundle: %s", bundle_path)

    return bundle_path
