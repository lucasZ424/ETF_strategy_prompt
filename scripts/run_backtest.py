"""CLI entry point: python scripts/run_backtest.py [--config path] [--output-dir path]

Loads test predictions from the latest (or specified) model output directory,
runs a backtest simulation, computes performance metrics, and generates plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import SimpleBacktestEngine, load_market_bars  # noqa: E402
from src.backtest.interface import Forecast, Signal  # noqa: E402
from src.backtest.strategies import EqualWeightSignalStrategy  # noqa: E402
from src.config import load_backtest_config, load_config, load_model_config  # noqa: E402
from src.evaluation.backtest_metrics import compute_backtest_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SIGNAL_MAP = {"long": Signal.LONG, "short": Signal.SHORT, "flat": Signal.FLAT}


def _build_version_tag(config_path: Path) -> str:
    """Build the full version tag like 'xgboost_30ft_t70_v15_esp10' from config."""
    model_cfg = load_model_config(config_path)
    pipe_cfg = load_config(config_path)

    t = int(model_cfg.train_ratio * 100)
    v = int(model_cfg.val_ratio * 100)
    esp = model_cfg.early_stopping_patience

    # Determine expected feature count from top_k or fall back to suffix match
    if pipe_cfg.top_k_features is not None:
        return f"xgboost_{pipe_cfg.top_k_features}ft_t{t}_v{v}_esp{esp}"
    return f"t{t}_v{v}_esp{esp}"  # partial suffix when top_k is None


def _find_matching_output_dir(base: Path, config_path: Path) -> Path:
    """Find the output directory matching the current config's version tag."""
    tag = _build_version_tag(config_path)
    dirs = [d for d in base.iterdir() if d.is_dir() and tag in d.name]
    if not dirs:
        all_dirs = [d.name for d in base.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No output directory matching '{tag}' under {base}. "
            f"Available: {all_dirs}"
        )
    chosen = max(dirs, key=lambda d: d.stat().st_mtime)
    if len(dirs) > 1:
        logger.info("Multiple matches for '%s': %s — using %s", tag, [d.name for d in dirs], chosen.name)
    return chosen


def _load_predictions(output_dir: Path):
    """Load test_predictions.csv and build forecasts + signal lookup."""
    csv_path = output_dir / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No test_predictions.csv in {output_dir}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    logger.info("Loaded %d predictions from %s", len(df), csv_path)

    forecasts = []
    signal_lookup = {}

    for _, row in df.iterrows():
        dt = row["date"].to_pydatetime()
        d = dt.date()
        sym = row["symbol"]

        forecasts.append(Forecast(
            timestamp=dt,
            symbol=sym,
            predicted_return=float(row["predicted_return"]),
        ))
        signal_lookup[(d, sym)] = _SIGNAL_MAP.get(row["signal"], Signal.FLAT)

    return forecasts, signal_lookup, df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curve(equity_curve, metrics, plots_dir: Path):
    dates = [d for d, _ in equity_curve]
    values = [v for _, v in equity_curve]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, linewidth=1.2, color="steelblue")
    ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5, label="Initial Cash")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()

    text = (
        f"CAGR: {metrics.cagr_pct:.2f}%\n"
        f"Sharpe: {metrics.sharpe_ratio:.2f}\n"
        f"Max DD: {metrics.max_drawdown_pct:.2f}%\n"
        f"Total Return: {metrics.total_return_pct:.2f}%"
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes, verticalalignment="top",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(plots_dir / "equity_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved equity_curve.png")


def plot_drawdown(equity_curve, plots_dir: Path):
    values = np.array([v for _, v in equity_curve])
    dates = [d for d, _ in equity_curve]
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, drawdown, color="salmon", alpha=0.6)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plots_dir / "drawdown.png", dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown.png")


def plot_returns_distribution(returns, plots_dir: Path):
    ret = np.array(returns) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ret, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=np.mean(ret), color="green", linestyle="-", alpha=0.7, label=f"Mean: {np.mean(ret):.3f}%")
    ax.set_title("Daily Returns Distribution")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "returns_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved returns_distribution.png")


def plot_monthly_returns(equity_curve, plots_dir: Path):
    """Monthly return heatmap-style bar chart."""
    df = pd.DataFrame(equity_curve, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    monthly = df["value"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100

    if len(monthly_ret) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if r > 0 else "red" for r in monthly_ret.values]
    ax.bar(range(len(monthly_ret)), monthly_ret.values, color=colors, alpha=0.7)
    labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Monthly Returns (%)")
    ax.set_ylabel("Return %")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "monthly_returns.png", dpi=150)
    plt.close(fig)
    logger.info("Saved monthly_returns.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run backtest on test set predictions.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml"))
    parser.add_argument("--output-dir", default=None, help="Versioned output dir (auto-detect if omitted)")
    args = parser.parse_args()

    config_path = Path(args.config)
    pipe_cfg = load_config(config_path)
    bt_cfg = load_backtest_config(config_path)

    # Locate output directory
    base_output = PROJECT_ROOT / "outputs"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _find_matching_output_dir(base_output, config_path)
    logger.info("Using output directory: %s", output_dir)

    # Load predictions and signals
    forecasts, signal_lookup, pred_df = _load_predictions(output_dir)
    symbols = pred_df["symbol"].unique().tolist()
    date_min = pred_df["date"].min()
    date_max = pred_df["date"].max()

    # Load market bars for the test period
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    bars = load_market_bars(raw_dir, symbols, start_date=date_min, end_date=date_max)

    # Build strategy and run engine
    strategy = EqualWeightSignalStrategy(
        signal_lookup=signal_lookup,
        max_gross_leverage=bt_cfg.max_gross_leverage,
    )
    engine = SimpleBacktestEngine()
    result = engine.run(
        bars=bars,
        forecasts=forecasts,
        strategy=strategy,
        initial_cash=bt_cfg.initial_cash,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
    )

    # Compute metrics
    metrics = compute_backtest_metrics(
        equity_curve=result.equity_curve,
        returns=result.returns,
        metadata=result.metadata,
        risk_free_rate=bt_cfg.risk_free_rate,
    )

    # Print summary
    logger.info("=== Backtest Results ===")
    logger.info("  Initial Cash:    %.2f", metrics.initial_cash)
    logger.info("  Final Value:     %.2f", metrics.final_value)
    logger.info("  Total Return:    %.2f%%", metrics.total_return_pct)
    logger.info("  CAGR:            %.2f%%", metrics.cagr_pct)
    logger.info("  Sharpe Ratio:    %.2f", metrics.sharpe_ratio)
    logger.info("  Max Drawdown:    %.2f%%", metrics.max_drawdown_pct)
    logger.info("  Max DD Duration: %d days", metrics.max_drawdown_duration_days)
    logger.info("  Win Rate:        %.2f%%", metrics.win_rate_pct)
    logger.info("  Total Trades:    %d", metrics.total_trades)
    logger.info("  Total Fees:      %.2f", metrics.total_fees)
    logger.info("  Total Slippage:  %.2f", metrics.total_slippage)

    # Save report
    report = {
        "metrics": metrics.to_dict(),
        "assumptions": {
            "initial_cash": bt_cfg.initial_cash,
            "fee_bps": bt_cfg.fee_bps,
            "slippage_bps": bt_cfg.slippage_bps,
            "max_gross_leverage": bt_cfg.max_gross_leverage,
            "risk_free_rate": bt_cfg.risk_free_rate,
            "execution_model": "Open price + slippage",
            "mark_to_market": "Close price",
            "rebalance": "Daily",
            "position_sizing": "Equal weight across active signals",
            "holding_period": "1 day",
            "test_period": f"{date_min.date()} to {date_max.date()}",
            "symbols": symbols,
        },
    }
    report_path = output_dir / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved backtest report: %s", report_path)

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_equity_curve(result.equity_curve, metrics, plots_dir)
    plot_drawdown(result.equity_curve, plots_dir)
    plot_returns_distribution(result.returns, plots_dir)
    plot_monthly_returns(result.equity_curve, plots_dir)

    logger.info("All backtest outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
