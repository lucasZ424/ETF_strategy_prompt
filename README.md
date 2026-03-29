# ETF Forecasting Pilot

This repository is a PostgreSQL-backed pilot for China ETF forecasting, strategy backtesting, and dashboard-oriented price prediction. Raw market bars are stored in PostgreSQL, while processed datasets, model bundles, and reports are written to local parquet/model/output directories.

## Current Status

- Incremental raw-data ingestion into PostgreSQL via `scripts/db_fetch.py`
- Instrument registry and open-universe onboarding via `instrument_master`
- Feature pipeline that reads from the DB backend and writes processed parquet datasets
- Target generation for alpha, triple-barrier gate, regime, and dashboard close-price tasks
- XGBoost strategy training and separate dashboard regression training
- Backtest scripts for both strategy outputs and dashboard models

## Default Config

The main config is [`configs/china_open_universe_minimal.template.toml`](configs/china_open_universe_minimal.template.toml).

Key defaults:

- Backend: `db`
- DB URL: set via `ETF_DB_URL` environment variable (AliCloud RDS PostgreSQL)
- Core training universe: 12 China ETFs
- Cross-market context: `SPY`, `QQQ`, `IEUR`
- Macro proxies: `VIX`, `TNX`, `DXY`
- Dashboard unseen ETFs: `512880.SS`, `159919.SZ`

If you do not want to hardcode credentials in the config, set `[database].url_env` and supply the connection string through an environment variable.

## Setup

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Set the database environment variable (AliCloud RDS PostgreSQL):

```powershell
$env:ETF_DB_URL = "postgresql://ETF_Dashboard:<password>@pgm-bp172lqhb76i9xyvdo.pg.rds.aliyuncs.com:5432/etf_strategy"
```

3. First-time DB initialization for the core universe:

```powershell
python scripts/db_seed.py --config configs/china_open_universe_minimal.template.toml
python -m scripts.db_fetch --config configs/china_open_universe_minimal.template.toml
```

5. Optional one-time discovery of all A-share ETFs for the dashboard/open universe:

```powershell
python -m scripts.db_discover_etfs --config configs/china_open_universe_minimal.template.toml
```

## Daily Workflow

Once `instrument_master` has been initialized and `ETF_DB_URL` is set, the normal workflow is:

```powershell
# 1. Fetch latest raw data into DB (incremental - only new bars)
python -m scripts.db_fetch --config configs/china_open_universe_minimal.template.toml

# 2. Run pipeline (reads raw bars from DB -> writes processed parquets)
python scripts/run_pipeline.py --config configs/china_open_universe_minimal.template.toml

# 3. Build targets (reads processed parquets -> writes target parquets)
python scripts/build_targets.py --config configs/china_open_universe_minimal.template.toml

# 4. Train models
python scripts/train_xgboost.py --config configs/china_open_universe_minimal.template.toml
python scripts/train_dashboard.py --config configs/china_open_universe_minimal.template.toml

# 5. Backtest
python scripts/run_backtest.py --config configs/china_open_universe_minimal.template.toml
python scripts/backtest_dashboard.py --config configs/china_open_universe_minimal.template.toml

# One-time: discover and onboard all A-share ETFs for dashboard coverage
python -m scripts.db_discover_etfs --config configs/china_open_universe_minimal.template.toml
```

## Main Scripts

- `scripts/db_seed.py`: initialize DB tables and seed the core, unseen, cross-market, and macro instruments.
- `scripts/db_fetch.py`: incrementally fetch new bars from yfinance for all active DB instruments.
- `scripts/db_discover_etfs.py`: discover all A-share ETFs and onboard qualifying symbols into the DB.
- `scripts/db_onboard.py`: onboard specific symbols on demand.
- `scripts/run_pipeline.py`: build processed feature datasets from raw bars.
- `scripts/build_targets.py`: build alpha, barrier, regime, and dashboard target datasets.
- `scripts/train_xgboost.py`: train the strategy models and write evaluation outputs.
- `scripts/train_dashboard.py`: train dashboard close-price regressors and save dashboard artifacts.
- `scripts/run_backtest.py`: backtest the strategy model outputs on the test set and optional unseen ETFs.
- `scripts/backtest_dashboard.py`: evaluate dashboard models on unseen ETFs.
- `scripts/predict_dashboard.py`: daily inference - predict 1d/3d/5d close prices for all eligible ETFs and write to DB.

## Outputs

Typical artifact locations:

- `data/processed/`: processed feature and target parquet/CSV files
- `models/`: saved model bundles and feature manifests
- `outputs/`: evaluation reports, prediction CSVs, plots, and backtest summaries
- `data/a_share_etf_universe.csv`: discovered full ETF universe snapshot

## Repository Layout

- `configs/`: pipeline and model configuration templates
- `scripts/`: CLI entry points for ingestion, training, and evaluation
- `src/data/`: DB access, loaders, cleaning, alignment, and pipeline logic
- `src/features/`: strategy and dashboard feature builders
- `src/targets/`: alpha, barrier, regime, and dashboard target builders
- `src/models/`: split, train, save, and load logic
- `src/evaluation/`: forecast metrics, signals, and backtest metrics
- `src/backtest/`: backtest engine, interfaces, and strategies
- `alembic/`: migration scaffolding
- `tests/`: unit and integration tests

## Notes

- Keep all splits chronological.
- Keep cross-market features lagged and China-time aligned to avoid leakage.
- `scripts/db_fetch.py` only works as expected after instruments already exist in `instrument_master`.
- For development, the DB can be created through `init_db()` paths in the scripts; for stricter schema management, use Alembic.
