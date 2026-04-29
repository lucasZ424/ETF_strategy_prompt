# ETF Forecasting Pilot

This repository is a PostgreSQL-backed pilot for China ETF forecasting, strategy backtesting, and dashboard-oriented price prediction. Raw market bars are stored in PostgreSQL, while processed datasets, model bundles, and reports are written to local parquet/model/output directories.

## Current Status

- Incremental raw-data ingestion into PostgreSQL via `scripts/db_fetch.py`
- Instrument registry and open-universe onboarding via `instrument_master`
- Feature pipeline that reads from the DB backend and writes processed parquet datasets
- Target generation for alpha, triple-barrier gate, regime, and dashboard price-ratio tasks
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
python scripts/db_fetch.py --config configs/china_open_universe_minimal.template.toml

# 2. Run pipeline (reads raw bars from DB -> writes processed parquets)
python scripts/run_pipeline.py --config configs/china_open_universe_minimal.template.toml

# 3. Build targets (reads processed parquets -> writes target parquets)
python scripts/build_targets.py --config configs/china_open_universe_minimal.template.toml

# 4. Train strategy models (alpha/gate/regime)
python scripts/train_xgboost.py --config configs/china_open_universe_minimal.template.toml

# 5. Train dashboard models (1d/3d/5d price-ratio regressors, inverse-transformed to raw price for evaluation)
python scripts/train_dashboard.py --config configs/china_open_universe_minimal.template.toml

# 6. Backtest strategy models
python scripts/run_backtest.py --config configs/china_open_universe_minimal.template.toml

# 7. Backtest dashboard models on unseen ETFs
python scripts/backtest_dashboard.py --config configs/china_open_universe_minimal.template.toml

# 8. Generate diagnostic plots (loss curves, true vs pred scatter)
python scripts/plot_results.py

# 9. Daily inference - predict price ratios, inverse-transform to raw prices, write to prediction_snapshots
python -m scripts.predict_dashboard --config configs/china_open_universe_minimal.template.toml

# 10. Launch interactive dashboard (reads from DB, no config needed)
streamlit run app/dashboard.py

# One-time: discover and onboard all A-share ETFs for dashboard coverage
python -m scripts.db_discover_etfs --config configs/china_open_universe_minimal.template.toml
```

## Interactive Dashboard

The Streamlit dashboard (`app/dashboard.py`) provides three panels:

1. **Price Explorer** — browse historical OHLCV candlestick charts for any ETF (1 filter: symbol)
2. **Price Predictions** — view predicted vs current close prices with model freshness info (2 filters: symbol + horizon)
3. **Market Overview** — pie chart showing how many ETFs are predicted increasing, decreasing, or flat

Launch locally:

```powershell
streamlit run app/dashboard.py
```

Requires `ETF_DB_URL` to be set (same as the rest of the pipeline).

## Main Scripts

- `scripts/db_seed.py`: initialize DB tables and seed the core, unseen, cross-market, and macro instruments.
- `scripts/db_fetch.py`: incrementally fetch new bars from yfinance for all active DB instruments.
- `scripts/db_discover_etfs.py`: discover all A-share ETFs and onboard qualifying symbols into the DB.
- `scripts/db_onboard.py`: onboard specific symbols on demand.
- `scripts/run_pipeline.py`: build processed feature datasets from raw bars.
- `scripts/build_targets.py`: build alpha, barrier, regime, and dashboard target datasets.
- `scripts/train_xgboost.py`: train the strategy models and write evaluation outputs.
- `scripts/train_dashboard.py`: train dashboard price-ratio regressors (y_ratio_Hd = close_{t+H}/close_t) and save dashboard artifacts.
- `scripts/run_backtest.py`: backtest the strategy model outputs on the test set and optional unseen ETFs.
- `scripts/backtest_dashboard.py`: evaluate dashboard models on unseen ETFs.
- `scripts/plot_results.py`: generate diagnostic plots (loss curves, true vs predicted scatter) from dashboard training outputs.
- `scripts/predict_dashboard.py`: daily inference - predict 1d/3d/5d price ratios, inverse-transform to raw close prices (predicted_close = current_close × ratio), and write to DB.
- `app/dashboard.py`: interactive Streamlit dashboard with price explorer, prediction viewer, and market overview.

## Outputs

Typical artifact locations:

- `data/processed/`: processed feature and target parquet/CSV files
- `models/`: saved model bundles and feature manifests
- `outputs/`: evaluation reports, prediction CSVs, plots, and backtest summaries
- `data/a_share_etf_universe.csv`: discovered full ETF universe snapshot

## Repository Layout

- `app/`: interactive Streamlit dashboard
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
- Dashboard models predict **price ratios** (`y_ratio_Hd ≈ 1.0`), not raw close prices. This makes the model scale-invariant — it works on any price level without retraining. Raw prices are reconstructed via `predicted_close = current_close × predicted_ratio` before writing to DB. The DB schema (`prediction_snapshots.y_close_*_pred`) stores the final reconstructed prices.
- `scripts/db_fetch.py` only works as expected after instruments already exist in `instrument_master`.
- For development, the DB can be created through `init_db()` paths in the scripts; for stricter schema management, use Alembic.
