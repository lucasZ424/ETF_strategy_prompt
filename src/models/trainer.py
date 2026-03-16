"""XGBoost model training for alpha, gate, and regime models.

Three coordinated models:
  1. Alpha regressor  — XGBRegressor, target: y_alpha (10d forward log return)
  2. Gate classifier  — XGBClassifier, target: barrier_label {-1, 0, +1}
  3. Regime classifier — XGBClassifier, target: regime_label {0, 1, 2}

Each model uses Optuna for hyperparameter search with:
  - Median-pruner based trial pruning
  - Validation-loss based early stopping (patience rounds on validation_1)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
)

from src.config import ModelConfig

logger = logging.getLogger(__name__)

_META_COLS = {"date", "symbol"}

# Suppress Optuna info spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Utility helpers (preserved from v0)
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return sorted list of feature columns (everything except meta cols)."""
    return sorted([c for c in df.columns if c not in _META_COLS])


def safe_X(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Extract feature matrix with NaN/inf handling."""
    raw = df[feature_cols].values.astype(float)
    return np.nan_to_num(np.where(np.isinf(raw), np.nan, raw), nan=0.0)


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Container for a trained model and its metadata."""

    model: Any  # xgb.XGBRegressor or xgb.XGBClassifier
    feature_cols: List[str]
    best_params: Dict[str, Any]
    best_iteration: int
    val_metric_name: str
    val_metric_value: float
    eval_history: Dict[str, List[float]]


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------

def _suggest_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Common XGBoost hyperparameter search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


class _TrialPruningCallback(xgb.callback.TrainingCallback):
    """Report XGBoost iteration metrics to Optuna and prune bad trials early."""

    def __init__(self, trial: optuna.Trial, observation_key: str) -> None:
        data_name, metric_name = observation_key.split("-", 1)
        self._trial = trial
        self._data_name = data_name
        self._metric_name = metric_name

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Any]) -> bool:
        metric_series = evals_log.get(self._data_name, {}).get(self._metric_name, [])
        if not metric_series:
            return False

        value = float(metric_series[-1])
        self._trial.report(value, step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at iter={epoch}, {self._data_name}-{self._metric_name}={value:.6f}",
            )
        return False


def _create_optuna_study(config: ModelConfig) -> optuna.Study:
    """Create a seeded Optuna study with the standard MedianPruner."""
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner: optuna.pruners.BasePruner
    if config.optuna_pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.optuna_pruner_startup_trials,
            n_warmup_steps=config.optuna_pruner_warmup_steps,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)


def _build_fit_callbacks(
    config: ModelConfig,
    metric_name: str,
    trial: Optional[optuna.Trial] = None,
    monitor_data_name: str = "validation_1",
) -> List[Any]:
    """Build callbacks for validation-loss early stopping and optional Optuna pruning."""
    callbacks: List[Any] = [
        xgb.callback.EarlyStopping(
            rounds=config.early_stopping_patience,
            metric_name=metric_name,
            data_name=monitor_data_name,
            maximize=False,
            save_best=True,
        ),
    ]
    if trial is not None and config.optuna_pruning:
        callbacks.append(_TrialPruningCallback(trial, f"{monitor_data_name}-{metric_name}"))
    return callbacks


# ---------------------------------------------------------------------------
# Alpha regressor
# ---------------------------------------------------------------------------

def train_alpha_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    config: ModelConfig,
) -> TrainResult:
    """Train XGBoost regressor for alpha prediction with Optuna HPO."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial)
        params["n_estimators"] = 1000  # rely on early stopping
        model = xgb.XGBRegressor(
            **params,
            eval_metric="rmse",
            callbacks=_build_fit_callbacks(config, metric_name="rmse", trial=trial),
            random_state=config.seed,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        return float(mean_squared_error(y_val, preds))

    study = _create_optuna_study(config)
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["n_estimators"] = 1000

    # Retrain with best params, capture eval history
    final_model = xgb.XGBRegressor(
        **best_params,
        eval_metric="rmse",
        callbacks=_build_fit_callbacks(config, metric_name="rmse"),
        random_state=config.seed,
        verbosity=0,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    eval_history = {
        "train_rmse": final_model.evals_result()["validation_0"]["rmse"],
        "val_rmse": final_model.evals_result()["validation_1"]["rmse"],
    }

    val_preds = final_model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, val_preds))

    logger.info(
        "Alpha regressor: best_iteration=%d, val_MSE=%.6f, params=%s",
        final_model.best_iteration, val_mse, best_params,
    )

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        best_params=best_params,
        best_iteration=final_model.best_iteration,
        val_metric_name="val_mse",
        val_metric_value=val_mse,
        eval_history=eval_history,
    )


# ---------------------------------------------------------------------------
# Gate classifier (ternary: -1, 0, +1)
# ---------------------------------------------------------------------------

def _remap_gate_labels(y: np.ndarray) -> np.ndarray:
    """Remap {-1, 0, +1} to {0, 1, 2} for XGBoost multi:softprob."""
    return (y + 1).astype(int)


def _unmap_gate_labels(y: np.ndarray) -> np.ndarray:
    """Reverse remap: {0, 1, 2} back to {-1, 0, +1}."""
    return (y - 1).astype(int)


def train_gate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    config: ModelConfig,
) -> TrainResult:
    """Train XGBoost classifier for triple-barrier gate labels."""

    # Remap labels for XGBoost
    y_train_mapped = _remap_gate_labels(y_train)
    y_val_mapped = _remap_gate_labels(y_val)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial)
        params["n_estimators"] = 1000
        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            callbacks=_build_fit_callbacks(config, metric_name="mlogloss", trial=trial),
            random_state=config.seed,
            verbosity=0,
        )
        model.fit(
            X_train, y_train_mapped,
            eval_set=[(X_train, y_train_mapped), (X_val, y_val_mapped)],
            verbose=False,
        )
        probs = model.predict_proba(X_val)
        return float(log_loss(y_val_mapped, probs))

    study = _create_optuna_study(config)
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["n_estimators"] = 1000

    final_model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        callbacks=_build_fit_callbacks(config, metric_name="mlogloss"),
        random_state=config.seed,
        verbosity=0,
    )
    final_model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_train, y_train_mapped), (X_val, y_val_mapped)],
        verbose=False,
    )

    eval_history = {
        "train_mlogloss": final_model.evals_result()["validation_0"]["mlogloss"],
        "val_mlogloss": final_model.evals_result()["validation_1"]["mlogloss"],
    }

    val_probs = final_model.predict_proba(X_val)
    val_logloss = float(log_loss(y_val_mapped, val_probs))
    val_preds = final_model.predict(X_val)
    val_acc = float(accuracy_score(y_val_mapped, val_preds))
    val_f1 = float(f1_score(y_val_mapped, val_preds, average="macro"))

    logger.info(
        "Gate classifier: best_iteration=%d, val_logloss=%.4f, "
        "val_acc=%.4f, val_f1_macro=%.4f",
        final_model.best_iteration, val_logloss, val_acc, val_f1,
    )

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        best_params=best_params,
        best_iteration=final_model.best_iteration,
        val_metric_name="val_logloss",
        val_metric_value=val_logloss,
        eval_history=eval_history,
    )


# ---------------------------------------------------------------------------
# Regime classifier (ternary: 0=defensive, 1=balanced, 2=aggressive)
# ---------------------------------------------------------------------------

def train_regime_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    config: ModelConfig,
) -> TrainResult:
    """Train XGBoost classifier for regime labels."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial)
        params["n_estimators"] = 1000
        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            callbacks=_build_fit_callbacks(config, metric_name="mlogloss", trial=trial),
            random_state=config.seed,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        probs = model.predict_proba(X_val)
        return float(log_loss(y_val, probs))

    study = _create_optuna_study(config)
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["n_estimators"] = 1000

    final_model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        callbacks=_build_fit_callbacks(config, metric_name="mlogloss"),
        random_state=config.seed,
        verbosity=0,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    eval_history = {
        "train_mlogloss": final_model.evals_result()["validation_0"]["mlogloss"],
        "val_mlogloss": final_model.evals_result()["validation_1"]["mlogloss"],
    }

    val_probs = final_model.predict_proba(X_val)
    val_logloss = float(log_loss(y_val, val_probs))
    val_preds = final_model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds, average="macro"))

    logger.info(
        "Regime classifier: best_iteration=%d, val_logloss=%.4f, "
        "val_acc=%.4f, val_f1_macro=%.4f",
        final_model.best_iteration, val_logloss, val_acc, val_f1,
    )

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        best_params=best_params,
        best_iteration=final_model.best_iteration,
        val_metric_name="val_logloss",
        val_metric_value=val_logloss,
        eval_history=eval_history,
    )


# ---------------------------------------------------------------------------
# Dashboard regressor (multi-output: 1d / 3d / 5d raw close price)
# ---------------------------------------------------------------------------

@dataclass
class DashboardTrainResult:
    """Container for per-horizon dashboard models."""

    models: Dict[str, Any]           # horizon_name → fitted XGBRegressor
    feature_cols: List[str]
    best_params: Dict[str, Dict[str, Any]]   # horizon_name → params
    best_iterations: Dict[str, int]
    val_metrics: Dict[str, float]    # horizon_name → val_mse
    eval_histories: Dict[str, Dict[str, List[float]]]


def train_dashboard_regressor(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    feature_cols: List[str],
    config: ModelConfig,
) -> DashboardTrainResult:
    """Train one XGBoost regressor per horizon for dashboard close-price prediction.

    Parameters
    ----------
    y_train / y_val : dict mapping horizon name (e.g. "y_close_1d") to target array.
    """
    models = {}
    best_params_all = {}
    best_iters = {}
    val_metrics = {}
    eval_histories = {}

    for horizon_name in sorted(y_train.keys()):
        yt = y_train[horizon_name]
        yv = y_val[horizon_name]

        logger.info("--- Dashboard horizon: %s ---", horizon_name)

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_xgb_params(trial)
            params["n_estimators"] = 1000
            model = xgb.XGBRegressor(
                **params,
                eval_metric="rmse",
                callbacks=_build_fit_callbacks(config, metric_name="rmse", trial=trial),
                random_state=config.seed,
                verbosity=0,
            )
            model.fit(
                X_train, yt,
                eval_set=[(X_train, yt), (X_val, yv)],
                verbose=False,
            )
            preds = model.predict(X_val)
            return float(mean_squared_error(yv, preds))

        study = _create_optuna_study(config)
        study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)

        bp = study.best_params
        bp["n_estimators"] = 1000

        final_model = xgb.XGBRegressor(
            **bp,
            eval_metric="rmse",
            callbacks=_build_fit_callbacks(config, metric_name="rmse"),
            random_state=config.seed,
            verbosity=0,
        )
        final_model.fit(
            X_train, yt,
            eval_set=[(X_train, yt), (X_val, yv)],
            verbose=False,
        )

        val_preds = final_model.predict(X_val)
        val_mse = float(mean_squared_error(yv, val_preds))

        models[horizon_name] = final_model
        best_params_all[horizon_name] = bp
        best_iters[horizon_name] = final_model.best_iteration
        val_metrics[horizon_name] = val_mse
        eval_histories[horizon_name] = {
            "train_rmse": final_model.evals_result()["validation_0"]["rmse"],
            "val_rmse": final_model.evals_result()["validation_1"]["rmse"],
        }

        logger.info(
            "Dashboard %s: best_iter=%d, val_MSE=%.4f",
            horizon_name, final_model.best_iteration, val_mse,
        )

    return DashboardTrainResult(
        models=models,
        feature_cols=feature_cols,
        best_params=best_params_all,
        best_iterations=best_iters,
        val_metrics=val_metrics,
        eval_histories=eval_histories,
    )


def save_dashboard_bundle(
    result: DashboardTrainResult,
    model_dir: Path,
) -> None:
    """Save all dashboard horizon models as a single bundle."""
    model_dir.mkdir(parents=True, exist_ok=True)

    for horizon_name, model in result.models.items():
        model_path = model_dir / f"dashboard_{horizon_name}.joblib"
        joblib.dump(model, model_path)

    manifest_path = model_dir / "dashboard_feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(result.feature_cols, f, indent=2)

    meta = {
        "model_name": "dashboard",
        "horizons": list(result.models.keys()),
        "best_params": result.best_params,
        "best_iterations": result.best_iterations,
        "val_metrics": result.val_metrics,
        "n_features": len(result.feature_cols),
    }
    meta_path = model_dir / "dashboard_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    for horizon_name, hist in result.eval_histories.items():
        hist_path = model_dir / f"dashboard_{horizon_name}_eval_history.json"
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)

    logger.info("Saved dashboard bundle: %d horizons → %s", len(result.models), model_dir)


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def save_model_bundle(
    result: TrainResult,
    model_name: str,
    model_dir: Path,
) -> None:
    """Save model artifact, feature manifest, and training metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    # Model artifact
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(result.model, model_path)

    # Feature manifest
    manifest_path = model_dir / f"{model_name}_feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(result.feature_cols, f, indent=2)

    # Training metadata
    meta = {
        "model_name": model_name,
        "best_params": result.best_params,
        "best_iteration": result.best_iteration,
        "val_metric_name": result.val_metric_name,
        "val_metric_value": result.val_metric_value,
        "n_features": len(result.feature_cols),
    }
    meta_path = model_dir / f"{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Eval history
    history_path = model_dir / f"{model_name}_eval_history.json"
    with open(history_path, "w") as f:
        json.dump(result.eval_history, f, indent=2)

    logger.info("Saved model bundle: %s → %s", model_name, model_dir)


def load_model(model_dir: Path, model_name: str) -> Any:
    """Load a saved model artifact."""
    return joblib.load(model_dir / f"{model_name}.joblib")


def load_feature_manifest(model_dir: Path, model_name: str) -> List[str]:
    """Load the feature manifest for a saved model."""
    with open(model_dir / f"{model_name}_feature_manifest.json") as f:
        return json.load(f)
