# MLOps Assignment II — Stage-by-Stage Development Document

**Student:** Balagopal R  
**Email:** balagopal.r@shipsy.io  
**GitHub:** https://github.com/Balagopal2022BCS0058/mlops-assignment-2.

---

## Stage 1 — Data Generation

- Generated **1,000 synthetic customers** with monthly charge and tenure fields
- Generated **3,923 support tickets** with date, category, and sentiment score
- Created **churn labels** — 18.5% churn rate driven by high ticket volume, negative sentiment, and cancellation tickets
- Saved to `data/raw/customers.csv`, `data/raw/tickets.csv`, `data/raw/churn_labels.csv`
- Script: `scripts/generate_data.py`

---

## Stage 2 — Feature Engineering

- Computed **ticket frequency windows** — `ticket_freq_7d`, `ticket_freq_30d`, `ticket_freq_90d` (rolling counts, no future leakage)
- Computed **sentiment score** — mean VADER score per customer across all tickets
- Computed **ticket category counts** — billing, technical, cancellation, general
- Computed **avg time between tickets** — mean gap in days between consecutive tickets
- Computed **monthly charge delta** — customer charge minus cohort median
- Wrapped all transformations in a single **sklearn Pipeline** — scaler + classifier serialized as one artifact to prevent train/serve skew
- Stratified **70/15/15 train/val/test split**
- Script: `scripts/prepare_data.py`

---

## Stage 3 — Model Training + MLflow Experiment Tracking

- Trained two classifiers: **XGBoost** and **RandomForest**
- Logged to MLflow per run:
  - Model type and hyperparameters
  - Feature list and training sample count
  - Metrics: **F1, ROC-AUC, Precision, Recall**
  - Artifacts: **ROC curve**, **Precision-Recall curve**, **Confusion Matrix** (PNG)
  - Serialized sklearn Pipeline artifact
- Results:

| Model | F1 | ROC-AUC |
|-------|-----|---------|
| XGBoost | 0.1923 | 0.5709 |
| RandomForest | 0.2857 | 0.6212 |

- MLflow UI: `http://localhost:5001`
- Script: `scripts/train.py`

---

## Stage 4 — Model Registry + Stage Transitions

- Registered both models in **MLflow Model Registry** as `churn-classifier`
- Managed stage lifecycle:
  - `None → Staging` — newly registered model
  - `Staging → Production` — after evaluation passes
  - `Production → Archived` — when a newer version is promoted
- Current state: **v3 in Production**, v2 in Staging, v1 Archived
- Promotion CLI: `python scripts/promote_model.py <staging|production|archive> <version>`

---

## Stage 5 — Inference API (Replaces Rule Engine)

- Built **FastAPI** app that loads the Production model from MLflow at startup
- Endpoints:
  - `POST /predict` — accepts customer ticket features, returns churn probability + label
  - `GET /health` — returns model name, version, and stage
  - `GET /model-info` — returns full feature list and model metadata
  - `GET /metrics` — Prometheus metrics endpoint
  - `GET /docs` — Swagger UI
- All request inputs validated with **Pydantic v2** — rejects invalid ranges at the API boundary
- Example response:
```json
{
  "customer_id": "C0001",
  "churn_probability": 0.1658,
  "churn_label": false,
  "model_version": "3"
}
```
- API running at: `http://localhost:8000/docs`

---

## Stage 6 — Drift Detection + Automated Retraining

- Used **Evidently AI** to compare training distribution vs production data
- Detected **feature drift** across 2 of 10 columns (drift score: 0.20 > threshold: 0.15)
- Generated HTML report at `reports/drift_report.html`
- Retraining trigger logic:
  - Reads drift score from `reports/drift_summary.json`
  - If score exceeds threshold → runs full retrain → registers new version
  - Auto-promotes new version to **Staging**
  - **Production promotion stays manual** (prevents silent regressions)
- Script: `scripts/generate_drift_report.py`

---

## Stage 7 — Production Monitoring (Prometheus)

- Instrumented FastAPI with **Prometheus client**
- Metrics exposed at `/metrics`:
  - `churn_api_requests_total` — request count by endpoint
  - `churn_api_request_latency_seconds` — latency histogram (p50, p95, p99)
  - `churn_feature_drift_score` — latest Evidently drift score
  - `churn_model_version` — currently deployed version
- Docker Compose includes **Prometheus** scraping the API every 15s

---

## Stage 8 — CI/CD/CT (GitHub Actions)

- **CI** (`.github/workflows/ci.yml`):
  - Triggers on every push and pull request
  - Runs: `ruff` lint → `mypy` type check → `pytest` (16 tests, 42% coverage)
- **CD** (`.github/workflows/cd.yml`):
  - Triggers on merge to `main`
  - Builds Docker image → pushes to GitHub Container Registry (GHCR)
- **CT** (`.github/workflows/ct.yml`):
  - Triggers on weekly cron schedule (Monday 02:00 UTC) + manual dispatch
  - Runs: generate data → prepare features → drift report → retrain if triggered
  - Uploads drift report as workflow artifact

---

## Stage 9 — Tests (16 passing)

- **Unit tests** — feature engineering, pipeline reproducibility, pydantic schema validation
- **Integration tests** — FastAPI endpoints with stub model (no MLflow required)
- Key assertions:
  - Rolling window features never include future data (no leakage)
  - Sentiment score always in `[-1.0, 1.0]`
  - Negative frequency values rejected at API boundary
  - Pipeline output is deterministic across identical inputs

---

## Project Structure

```
mlops-churn/
├── .github/workflows/     → CI, CD, CT pipelines
├── data/raw/              → DVC-tracked raw data
├── src/churn/
│   ├── features/          → Feature engineering + sklearn Pipeline
│   ├── data/              → Loaders, splitter
│   ├── models/            → Train, evaluate, registry
│   ├── serving/           → FastAPI app
│   ├── monitoring/        → Evidently drift, Prometheus metrics
│   └── retraining/        → Drift-triggered retraining logic
├── tests/                 → 16 unit + integration tests
├── scripts/               → CLI entrypoints for each stage
└── docker/                → Dockerfile + docker-compose
```

---

## Quickstart

```bash
git clone git@github.com:Balagopal2022BCS0058/mlops-assignment-2..git
cd mlops-assignment-2.
python -m venv .venv && source .venv/bin/activate
make install
make mlflow &      # start MLflow on :5001
make data          # generate + prepare data
make train         # train models + register in MLflow
make serve &       # start inference API on :8000
make drift         # generate Evidently drift report
make test          # run all 16 tests
```
