# ETF Forecasting Starter

This repository is a starter scaffold for an ETF forecasting and backtesting project.

## Scope

- Forecast target: next-step open-to-previous-close log return.
- Baselines: XGBoost and MLP first, with room for additional models.
- Evaluation: forecast metrics plus strategy metrics in backtests.

## Quick Start# ETF Forecasting Pilot

This repository is a pilot project for one-day ETF return forecasting and strategy-prompt generation.

## Current Objective

- Target: next-step open-to-previous-close log return.
- Core market: representative China ETFs.
- Cross-market context: SPY, QQQ, IEUR with strict no-lookahead alignment.
- Model path: XGBoost first, then MLP only after baseline pipeline stability.

## Project Status

Implemented core workflow:
1. Raw data retrieval and storage.
2. Data cleaning, alignment, and feature engineering.
3. Chronological split, XGBoost training, evaluation, and signal generation.
4. Versioned model/output artifacts and diagnostic plots.
5. Weekly retrain script scaffold.

Roadmap and phase gates are defined in [Project_Structure.txt](Project_Structure.txt).

## Quick Start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. (Optional) Fetch/update market data:

```bash
python "scripts/fetch_china&crossmarket_etfs.py"
```

3. Run feature pipeline:

```bash
python scripts/run_pipeline.py --config configs/china_open_universe_minimal.template.toml
```

4. Train model:

```bash
python scripts/train_xgboost.py --config configs/china_open_universe_minimal.template.toml
```

5. Generate diagnostics:

```bash
python scripts/plot_results.py
```

6. Run current tests:

```bash
pytest -q tests/test_schema.py
```

## Repository Layout

- `configs/`: pipeline/model configuration templates.
- `data/`: raw, interim, and processed datasets.
- `models/`: saved model artifacts.
- `outputs/`: versioned evaluation outputs and plots.
- `scripts/`: runnable pipeline/training/automation entrypoints.
- `src/data`: data loading, cleaning, schema validation, alignment.
- `src/features`: feature engineering and screening.
- `src/models`: split and training logic.
- `src/evaluation`: metrics and signal generation.
- `src/backtest`: backtest interfaces and core types.
- `src/dashboard`: backend/frontend integration layer (pilot target).
- `tests/`: unit tests.
- `reports/`: project reviews, guidelines, and checklists.

## Engineering Guardrails

- Keep all splits chronological.
- Keep cross-market features strictly lagged to avoid leakage.
- Enforce one canonical input contract across config/schema/features.
- Treat current outputs as paper-trading pilot results, not live trading advice.


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
