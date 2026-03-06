# Pilot Robust Development Guideline

Date: 2026-03-06
Scope: Keep the current ETF toy pipeline direction, but make it robust enough for a real frontend-backend pilot.

## Adjust


Adjust (minimum required to become robust):
- Enforce one canonical target/column contract across config, schema, and features.
- Remove hardcoded symbol assumptions from feature logic.
- Make config sections fully effective (no silent ignored fields).
- Add end-to-end quality gates and reproducible run commands.
- Add minimal backend API and frontend visibility layer around the existing model pipeline.

## Target Pilot Architecture (Preserve Existing Core)

Data plane:
1. Fetch raw data -> `data/raw/...`
2. Build processed features -> `data/processed/features_v1.*`
3. Train + evaluate -> `models/` + `outputs/<version>/...`

Service plane:
1. Backend API (FastAPI) exposes:
- `GET /health`
- `GET /model/latest`
- `POST /predict` (single ETF or batch request, schema validated)
- `POST /retrain` (manual trigger for pilot only)
2. Frontend dashboard (simple):
- Latest model/version summary
- Input form or CSV upload
- Prediction table + signal + confidence text
- Recent run logs/status

Control plane:
1. Config-driven runtime (`configs/*.toml`)
2. Validation contracts (`src/data/schema.py`)
3. Test/quality gates (unit + integration)
4. Artifact traceability (version tag + report + config snapshot)

## Non-Negotiable Engineering Rules

1. Contract-first:
- Every training/inference input must pass one shared schema.
- Use a single adjusted close naming convention (`adj_close`) everywhere.

2. Reproducibility-first:
- Every training output folder must include:
- effective config
- model params
- metrics
- signal thresholds
- feature list

3. No silent config:
- If config includes unsupported keys, fail fast.
- If required keys are missing, fail fast.

4. No hidden logic branches:
- Cross-market symbols come from config only.
- Feature list used by model must be explicitly persisted and reused at inference.

## Phase Plan

## Phase 0: Stabilize Existing Pipeline (P0)

Goal: make current scripts reliably runnable by another developer.

Required outcomes:
1. Fix fetch script runtime errors (`_summarize` call mismatch, bad `sort_values` args).
2. Unify target basis and column naming (`adj_close` contract).
3. Remove hardcoded `spy/qqq/ieur` generation path in builder logic.
4. Align config parser with config templates (`[model]`, `[split]`, `[tuning]` semantics).
5. Update README to reflect real project maturity and commands.

Exit criteria:
- Fresh clone can run fetch -> pipeline -> train -> plot without manual code edits.

## Phase 1: Backend Pilot Service (P1)

Goal: wrap current model flow into a callable backend.

Required outcomes:
1. Add `src/dashboard/api.py` (or `src/service/api.py`) with FastAPI app.
2. Add request/response models for prediction endpoints.
3. Load latest model bundle and feature schema at service startup.
4. Add `/predict` path with explicit error states:
- `not_enough_history`
- `missing_columns`
- `invalid_date_order`

Exit criteria:
- Local API returns deterministic predictions from sample input.

## Phase 2: Frontend Pilot (P2)

Goal: visible frontend-backend loop for demo and development.

Required outcomes:
1. Add minimal UI (can be Streamlit or lightweight web frontend).
2. Show:
- current model version
- last retrain timestamp
- prediction outputs and signal counts
3. Add warning banner:
- pilot only
- not investment advice

Exit criteria:
- One-click local demo where user inputs data and sees outputs through API.

## Recommended Repo Conventions

1. Keep script entrypoints but define one canonical command map in README:
- `python scripts/run_pipeline.py --config ...`
- `python scripts/train_xgboost.py --config ...`
- `python -m src.dashboard.api` (or equivalent)

2. Keep artifacts organized:
- `data/raw` source snapshots
- `data/processed` deterministic features
- `models` model binaries
- `outputs/<version>` metrics and plots

## Risk Notes for This Pilot

1. Current target design can create optimism bias if price basis is inconsistent.
2. Threshold optimization by directional accuracy alone can inflate apparent performance.
3. Without transaction-cost-aware backtest, strategy quality is still incomplete.