"""CLI entry point: python scripts/run_backtest.py [--config path] [--unseen SYM1 SYM2 ...]

Runs backtest on core test-set predictions, optionally tests unseen ETFs,
produces comparison table and visualizations.

Usage:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --unseen 512880.SS 159919.SZ
  python scripts/run_backtest.py --output-dir outputs/xgboost_30ft_t70_v15_esp10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import yfinance as yf  # noqa: E402
from src.backtest.engine import SimpleBacktestEngine, load_market_bars  # noqa: E402
from src.backtest.interface import Forecast, MarketBar, Signal  # noqa: E402
from src.backtest.strategies import EqualWeightSignalStrategy  # noqa: E402
from src.config import (  # noqa: E402
    BacktestConfig,
    load_backtest_config,
    load_config,
    load_model_config,
)
from src.data.cleaner import clean_china_etfs, clean_cross_market  # noqa: E402
from src.data.cross_market import align_cross_market_to_china  # noqa: E402
from src.data.loader import load_cross_market_etfs, load_single_csv  # noqa: E402
from src.evaluation.backtest_metrics import compute_backtest_metrics  # noqa: E402
from src.evaluation.signal import SignalConfig, predict_to_signal  # noqa: E402
from src.features.builder import build_trade_features  # noqa: E402
from src.models.trainer import safe_X  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SIGNAL_MAP = {"long": Signal.LONG, "short": Signal.SHORT, "flat": Signal.FLAT}


# ---------------------------------------------------------------------------
# Version tag / output directory discovery
# ---------------------------------------------------------------------------

def _build_version_tag(config_path: Path) -> str:
    model_cfg = load_model_config(config_path)
    t = int(model_cfg.train_ratio * 100)
    v = int(model_cfg.val_ratio * 100)
    esp = model_cfg.early_stopping_patience
    return f"t{t}_v{v}_esp{esp}"


def _find_matching_output_dir(base: Path, config_path: Path) -> Path:
    tag = _build_version_tag(config_path)
    dirs = [d for d in base.iterdir() if d.is_dir() and tag in d.name]
    if not dirs:
        all_dirs = [d.name for d in base.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No output directory matching '{tag}' under {base}. Available: {all_dirs}"
        )
    chosen = max(dirs, key=lambda d: d.stat().st_mtime)
    if len(dirs) > 1:
        logger.info("Multiple matches for '%s': %s — using %s", tag, [d.name for d in dirs], chosen.name)
    return chosen


# ---------------------------------------------------------------------------
# Core backtest on test-set predictions
# ---------------------------------------------------------------------------

def _load_predictions(output_dir: Path):
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
        forecasts.append(Forecast(timestamp=dt, symbol=sym, predicted_return=float(row["predicted_return"])))
        signal_lookup[(d, sym)] = _SIGNAL_MAP.get(row["signal"], Signal.FLAT)

    return forecasts, signal_lookup, df


def run_core_backtest(output_dir, pipe_cfg, bt_cfg):
    """Run backtest on core ETF test-set predictions."""
    forecasts, signal_lookup, pred_df = _load_predictions(output_dir)
    symbols = pred_df["symbol"].unique().tolist()
    date_min, date_max = pred_df["date"].min(), pred_df["date"].max()

    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    bars = load_market_bars(raw_dir, symbols, start_date=date_min, end_date=date_max)

    strategy = EqualWeightSignalStrategy(signal_lookup, max_gross_leverage=bt_cfg.max_gross_leverage)
    result = SimpleBacktestEngine().run(bars, forecasts, strategy, bt_cfg)

    metrics = compute_backtest_metrics(
        result.equity_curve, result.returns, result.metadata, bt_cfg.risk_free_rate)

    return {
        "label": "core_model",
        "symbols": symbols,
        "test_period": f"{date_min.date()} to {date_max.date()}",
        "metrics": metrics,
        "equity_curve": result.equity_curve,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Unseen ETF support
# ---------------------------------------------------------------------------

def _fetch_etf(symbol: str, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{symbol}.csv"
    if csv_path.exists():
        logger.info("Using cached data for %s", symbol)
        return csv_path
    logger.info("Fetching %s from Yahoo Finance...", symbol)
    frame = yf.download(symbol, start="2015-01-01", end="2025-12-31", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame.to_csv(csv_path, index=False)
    logger.info("Saved %s: %d rows", symbol, len(frame))
    return csv_path


def _build_unseen_features(symbol, csv_path, pipe_cfg, feature_names):
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    unseen_df = load_single_csv(csv_path, symbol)
    unseen_clean = clean_china_etfs(unseen_df)
    if len(unseen_clean) < 60:
        raise ValueError(f"Insufficient history for {symbol}: {len(unseen_clean)} rows < 60 minimum")

    cross_raw = load_cross_market_etfs(raw_dir, pipe_cfg.cross_market)
    cross_clean = clean_cross_market(cross_raw)
    china_dates = unseen_clean["date"].drop_duplicates().sort_values()
    cross_aligned = align_cross_market_to_china(
        china_dates, cross_clean, pipe_cfg.cross_market, raw_dir=raw_dir)

    datasets = build_trade_features(
        unseen_clean, cross_aligned,
        feature_selection=None,  # no selection for unseen inference
        seed=pipe_cfg.seed,
        cost_threshold=pipe_cfg.gate.cost_threshold,
        gap_threshold=pipe_cfg.gate.gap_threshold,
    )
    processed = datasets.backbone

    missing = [f for f in feature_names if f not in processed.columns]
    if missing:
        logger.warning("Missing features for %s: %s", symbol, missing)
        for f in missing:
            processed[f] = 0.0
    return processed


def _load_bars_from_dir(bar_dir: Path, symbol: str, start_date, end_date):
    csv_path = bar_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    ts = pd.Timestamp(start_date)
    idx = df["Date"].searchsorted(ts)
    df = df.iloc[max(0, idx - 1):]
    df = df[df["Date"] <= pd.Timestamp(end_date)]
    return [MarketBar(timestamp=r["Date"].to_pydatetime(), symbol=symbol,
                      open=float(r["Open"]), high=float(r["High"]),
                      low=float(r["Low"]), close=float(r["Close"]),
                      volume=float(r["Volume"])) for _, r in df.iterrows()]


def run_unseen_backtest(symbol, model, feature_names, signal_config, pipe_cfg, bt_cfg):
    """Run model backtest on a single unseen ETF. Returns result dict."""
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    unseen_dir = raw_dir.parent / "unseen_etfs"
    csv_path = _fetch_etf(symbol, unseen_dir)

    try:
        feat_df = _build_unseen_features(symbol, csv_path, pipe_cfg, feature_names)
    except ValueError as e:
        return {"label": f"unseen_{symbol}", "symbol": symbol, "error": str(e)}

    dates = sorted(feat_df["date"].unique())
    n_test = max(1, int(len(dates) * 0.15))
    test_dates = dates[-n_test:]
    test_df = feat_df[feat_df["date"].isin(test_dates)].copy()

    if len(test_df) < 10:
        return {"label": f"unseen_{symbol}", "symbol": symbol, "error": "insufficient_test_data"}

    X = safe_X(test_df, feature_names)
    predictions = model.predict(X)
    test_df["predicted_return"] = predictions
    test_df["signal"] = [predict_to_signal(p, signal_config).value for p in predictions]

    forecasts, signal_lookup = [], {}
    for _, row in test_df.iterrows():
        dt = pd.Timestamp(row["date"]).to_pydatetime()
        d = dt.date()
        forecasts.append(Forecast(timestamp=dt, symbol=symbol, predicted_return=float(row["predicted_return"])))
        signal_lookup[(d, symbol)] = Signal(row["signal"])

    bars = _load_bars_from_dir(unseen_dir, symbol, test_dates[0], test_dates[-1])
    if len(bars) < 2:
        return {"label": f"unseen_{symbol}", "symbol": symbol, "error": "insufficient_bar_data"}

    strategy = EqualWeightSignalStrategy(signal_lookup, max_gross_leverage=bt_cfg.max_gross_leverage)
    result = SimpleBacktestEngine().run(bars, forecasts, strategy, bt_cfg)

    if not result.equity_curve:
        return {"label": f"unseen_{symbol}", "symbol": symbol, "error": "no_equity_curve"}

    metrics = compute_backtest_metrics(
        result.equity_curve, result.returns, result.metadata, bt_cfg.risk_free_rate)

    return {
        "label": f"unseen_{symbol}",
        "symbol": symbol,
        "strategy": "model",
        "metrics": metrics,
        "equity_curve": result.equity_curve,
    }


def run_always_long_baseline(symbol, feat_df, pipe_cfg, bt_cfg, test_dates):
    """Always-long baseline for a single ETF."""
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    unseen_dir = raw_dir.parent / "unseen_etfs"
    test_df = feat_df[feat_df["date"].isin(test_dates)]

    forecasts, signal_lookup = [], {}
    for _, row in test_df.iterrows():
        dt = pd.Timestamp(row["date"]).to_pydatetime()
        d = dt.date()
        forecasts.append(Forecast(timestamp=dt, symbol=symbol, predicted_return=0.01))
        signal_lookup[(d, symbol)] = Signal.LONG

    bars = _load_bars_from_dir(unseen_dir, symbol, test_dates[0], test_dates[-1])
    if len(bars) < 2:
        return {"label": f"baseline_long_{symbol}", "error": "insufficient_bars"}

    strategy = EqualWeightSignalStrategy(signal_lookup, max_gross_leverage=bt_cfg.max_gross_leverage)
    result = SimpleBacktestEngine().run(bars, forecasts, strategy, bt_cfg)
    if not result.equity_curve:
        return {"label": f"baseline_long_{symbol}", "error": "no_equity_curve"}

    metrics = compute_backtest_metrics(
        result.equity_curve, result.returns, result.metadata, bt_cfg.risk_free_rate)
    return {
        "label": f"baseline_long_{symbol}",
        "symbol": symbol,
        "strategy": "always_long",
        "metrics": metrics,
        "equity_curve": result.equity_curve,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curves(curves: list[dict], plots_dir: Path):
    """Overlay equity curves for all strategies on one plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for entry in curves:
        ec = entry["equity_curve"]
        dates = [d for d, _ in ec]
        values = [v / ec[0][1] for _, v in ec]  # normalize to 1.0
        ax.plot(dates, values, linewidth=1.2, label=entry["label"])

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Equity Curves (Normalized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "equity_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved equity_curves.png")


def plot_returns_distribution(returns, plots_dir: Path):
    ret = np.array(returns) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ret, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=np.mean(ret), color="green", linestyle="-", alpha=0.7, label=f"Mean: {np.mean(ret):.3f}%")
    ax.set_title("Daily Returns Distribution (Core Model)")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "returns_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved returns_distribution.png")


def plot_monthly_returns(equity_curve, plots_dir: Path):
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
    parser = argparse.ArgumentParser(description="Run backtest pipeline with optional unseen ETF test.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml"))
    parser.add_argument("--output-dir", default=None, help="Versioned output dir (auto-detect if omitted)")
    parser.add_argument("--unseen", nargs="*", default=None,
                        help="Unseen ETF symbols to test (e.g. 512880.SS 159919.SZ)")
    args = parser.parse_args()

    config_path = Path(args.config)
    pipe_cfg = load_config(config_path)
    model_cfg = load_model_config(config_path)
    bt_cfg = load_backtest_config(config_path)

    # Locate output directory
    base_output = PROJECT_ROOT / "outputs"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _find_matching_output_dir(base_output, config_path)
    logger.info("Using output directory: %s", output_dir)

    # ── 1. Core backtest ──────────────────────────────────────────────────
    core = run_core_backtest(output_dir, pipe_cfg, bt_cfg)
    m = core["metrics"]

    logger.info("=== Core Backtest Results ===")
    logger.info("  Total Return: %.2f%%  |  CAGR: %.2f%%  |  Sharpe: %.2f", m.total_return_pct, m.cagr_pct, m.sharpe_ratio)
    logger.info("  Max Drawdown: %.2f%%  |  Win Rate: %.2f%%  |  Trades: %d", m.max_drawdown_pct, m.win_rate_pct, m.total_trades)
    logger.info("  Fees: %.2f  |  Slippage: %.2f", m.total_fees, m.total_slippage)

    # Collect all equity curves for overlay plot
    all_curves = [{"label": "core_model", "equity_curve": core["equity_curve"]}]
    comparison_rows = [{
        "label": "core_model",
        "symbols": ", ".join(core["symbols"]),
        "return_%": m.total_return_pct,
        "cagr_%": m.cagr_pct,
        "sharpe": m.sharpe_ratio,
        "max_dd_%": m.max_drawdown_pct,
        "win_%": m.win_rate_pct,
        "trades": m.total_trades,
    }]

    # ── 2. Unseen ETF tests ───────────────────────────────────────────────
    unseen_results = []
    if args.unseen:
        # Load frozen model and signal config
        tag = _build_version_tag(config_path)
        model_dir = PROJECT_ROOT / model_cfg.model_dir
        # Find model file matching tag
        model_files = list(model_dir.glob(f"{tag}*.joblib"))
        if not model_files:
            model_files = list(model_dir.glob(f"*{tag}*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model .joblib matching '{tag}' in {model_dir}")
        model_path = model_files[0]

        logger.info("Loading frozen model: %s", model_path)
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_names = bundle["feature_names"]

        signal_cfg_path = output_dir / "signal_config.json"
        with open(signal_cfg_path) as f:
            sig_raw = json.load(f)
        if "criterion" in sig_raw and "criterion_name" not in sig_raw:
            sig_raw["criterion_name"] = sig_raw.pop("criterion")
        signal_config = SignalConfig(**sig_raw)
        logger.info("Signal thresholds: long > %.5f, short < %.5f",
                    signal_config.long_threshold, signal_config.short_threshold)

        for symbol in args.unseen:
            logger.info("=== Unseen ETF: %s ===", symbol)

            # Model backtest
            res = run_unseen_backtest(symbol, model, feature_names, signal_config, pipe_cfg, bt_cfg)
            unseen_results.append(res)
            if "error" not in res:
                um = res["metrics"]
                all_curves.append({"label": f"model_{symbol}", "equity_curve": res["equity_curve"]})
                comparison_rows.append({
                    "label": f"model_{symbol}",
                    "symbols": symbol,
                    "return_%": um.total_return_pct,
                    "cagr_%": um.cagr_pct,
                    "sharpe": um.sharpe_ratio,
                    "max_dd_%": um.max_drawdown_pct,
                    "win_%": um.win_rate_pct,
                    "trades": um.total_trades,
                })

                # Always-long baseline
                raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
                unseen_dir = raw_dir.parent / "unseen_etfs"
                csv_path = unseen_dir / f"{symbol}.csv"
                feat_df = _build_unseen_features(symbol, csv_path, pipe_cfg, feature_names)
                dates = sorted(feat_df["date"].unique())
                n_test = max(1, int(len(dates) * 0.15))
                test_dates = dates[-n_test:]

                long_res = run_always_long_baseline(symbol, feat_df, pipe_cfg, bt_cfg, test_dates)
                unseen_results.append(long_res)
                if "error" not in long_res:
                    lm = long_res["metrics"]
                    all_curves.append({"label": f"always_long_{symbol}", "equity_curve": long_res["equity_curve"]})
                    comparison_rows.append({
                        "label": f"always_long_{symbol}",
                        "symbols": symbol,
                        "return_%": lm.total_return_pct,
                        "cagr_%": lm.cagr_pct,
                        "sharpe": lm.sharpe_ratio,
                        "max_dd_%": lm.max_drawdown_pct,
                        "win_%": lm.win_rate_pct,
                        "trades": lm.total_trades,
                    })
            else:
                logger.warning("  %s: %s", symbol, res["error"])

    # ── 3. Save report ────────────────────────────────────────────────────
    report = {
        "metrics": core["metrics"].to_dict(),
        "assumptions": {
            "initial_cash": bt_cfg.initial_cash,
            "fee_bps": bt_cfg.fee_bps,
            "slippage_bps": bt_cfg.slippage_bps,
            "max_gross_leverage": bt_cfg.max_gross_leverage,
            "risk_free_rate": bt_cfg.risk_free_rate,
            "target": "open_to_prev_close_log_return (gap return)",
            "execution_model": "Entry at prev close + slippage, exit at open - slippage",
            "pnl_leg": "close_{T-1} -> open_T (overnight gap, aligned to model target)",
            "rebalance": "Daily, no carry between days",
            "position_sizing": "Equal weight across active signals",
            "holding_period": "Overnight (close to next open)",
            "test_period": core["test_period"],
            "symbols": core["symbols"],
        },
    }
    if args.unseen:
        report["unseen_results"] = []
        for r in unseen_results:
            entry = {"label": r.get("label"), "symbol": r.get("symbol")}
            if "error" in r:
                entry["error"] = r["error"]
            elif "metrics" in r:
                entry["metrics"] = r["metrics"].to_dict()
            report["unseen_results"].append(entry)

    report_path = output_dir / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved backtest report: %s", report_path)

    # ── 4. Comparison table ───────────────────────────────────────────────
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = output_dir / "comparison_table.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info("Saved comparison table: %s", comp_path)
    print("\n" + comp_df.to_string(index=False))

    # ── 5. Plots ──────────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_equity_curves(all_curves, plots_dir)
    plot_returns_distribution(core["result"].returns, plots_dir)
    plot_monthly_returns(core["equity_curve"], plots_dir)

    logger.info("All backtest outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
