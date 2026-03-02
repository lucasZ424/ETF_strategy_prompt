# ETF Forecasting Starter

This repository is a starter scaffold for an ETF forecasting and backtesting project.

## Scope

- Forecast target: next-step open-to-previous-close log return.
- Baselines: XGBoost and MLP first, with room for additional models.
- Evaluation: forecast metrics plus strategy metrics in backtests.

## Quick Start

1. Put downloaded market data in `data/raw/`.
2. Copy `configs/default.template.toml` to a run config (for example `configs/local.toml`) and edit values.
3. Implement data pipeline modules in `src/data/` and `src/features/`.
4. Implement model training in `src/models/`.
5. Implement execution logic using interfaces in `src/backtest/interface.py`.

## Repository Layout

- `configs/`: experiment and run configuration files
- `data/raw|interim|processed`: data lifecycle zones
- `src/data`: retrieval and cleaning
- `src/features`: feature generation
- `src/models`: model training and inference
- `src/evaluation`: forecasting metrics and reports
- `src/backtest`: trading and portfolio simulation
- `src/dashboard`: API/UI integration layer
- `tests/`: unit and integration tests
- `reports/figures`: exported charts and diagnostics

## Notes

- Keep all splits chronological for time-series validation.
- Include transaction costs and slippage in every backtest result you report.
