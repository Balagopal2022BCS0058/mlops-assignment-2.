# MLOps Assignment II — Submission Document

**Student:** Balagopal R  
**Email:** balagopal.r@shipsy.io  
**GitHub Repository:** https://github.com/Balagopal2022BCS0058/mlops-assignment-2.

---

## Overview

This assignment replaces the rule-based churn scoring system (Assignment I) with a full ML-based pipeline, adding MLOps tooling for experiment tracking, model registry, inference serving, drift detection, and CI/CD/CT automation.

---

## Stage 1 — Synthetic Data Generation

**File:** `scripts/generate_data.py`

The system generates 1,000 customers with support ticket history and realistic churn labels.  
Churn is driven by: high ticket frequency, negative sentiment, cancellation tickets, and low tenure.

**Command:**
```bash
python scripts/generate_data.py
```

**Output:**
```
Customers: 1000
Tickets  : 3923
Churn rate: 18.50%
```

| File | Description |
|------|-------------|
| `data/raw/customers.csv` | Customer IDs, monthly charge, tenure |
| `data/raw/tickets.csv` | Ticket ID, date, category, sentiment score |
| `data/raw/churn_labels.csv` | Binary churn label per customer |

---

## Stage 2 — Feature Engineering + Data Splitting

**Files:** `src/churn/features/engineering.py`, `src/churn/data/splitter.py`  
**Script:** `scripts/prepare_data.py`

Five feature groups are engineered:

| Feature | Description |
|---------|-------------|
| `ticket_freq_7d / 30d / 90d` | Rolling ticket count in each time window (no future leakage) |
| `ticket_sentiment_score` | Mean VADER sentiment across all tickets |
| `ticket_category_*` | Per-category ticket counts (billing, technical, cancellation, general) |
| `avg_time_between_tickets_days` | Mean gap between consecutive tickets |
| `monthly_charge_delta` | Customer charge minus cohort median |

A single **sklearn Pipeline** serializes the scaler + classifier as one artifact, guaranteeing identical transformations at training and inference time.

**Split:** Stratified 70/15/15 (train/val/test)

```
train: 700 rows | churn rate: 18.57%
val  : 150 rows | churn rate: 18.00%
test : 150 rows | churn rate: 18.67%
```

---

## Stage 3 — Model Training + MLflow Experiment Tracking

**Files:** `src/churn/models/train.py`, `src/churn/models/evaluate.py`  
**Script:** `scripts/train.py`

Two classifiers are trained and compared:

| Model | F1 | ROC-AUC | Precision | Recall |
|-------|-----|---------|-----------|--------|
| XGBoost | 0.1923 | 0.5709 | 0.200 | 0.185 |
| RandomForest | 0.2857 | 0.6212 | 0.276 | 0.296 |

Each MLflow run logs:
- **Parameters:** model type, hyperparameters, feature list, train sample count
- **Metrics:** F1, ROC-AUC, Precision, Recall
- **Artifacts:** ROC curve PNG, PR curve PNG, Confusion Matrix PNG, serialized Pipeline

**MLflow UI:** http://127.0.0.1:5000

To view experiment runs:
```bash
make mlflow   # starts MLflow server
# Open http://localhost:5000 in browser
```

**Screenshot placeholder:** MLflow Experiments view showing both runs with metrics.

---

## Stage 4 — Model Registry + Stage Transitions

**File:** `src/churn/models/registry.py`  
**Script:** `scripts/promote_model.py`

Models move through three stages managed by MLflow Model Registry:

```
None → Staging → Production → Archived
```

| Version | Model | Stage |
|---------|-------|-------|
| v1 | XGBoost | Production (archived after v2) |
| v2 | RandomForest | Staging |

**Commands:**
```bash
python scripts/promote_model.py list
python scripts/promote_model.py staging 2
python scripts/promote_model.py production 2
python scripts/promote_model.py archive 1
```

**Screenshot placeholder:** MLflow Model Registry showing versions and stage transitions.

---

## Stage 5 — Inference API (Replaces Rule Engine)

**Files:** `src/churn/serving/app.py`, `src/churn/serving/routes.py`

FastAPI serves the Production model with three endpoints:

### `GET /health`
```json
{
  "status": "ok",
  "model_name": "churn-classifier",
  "model_version": "1",
  "model_stage": "Production"
}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C0001",
    "ticket_freq_7d": 5,
    "ticket_freq_30d": 12,
    "ticket_freq_90d": 20,
    "ticket_sentiment_score": -0.6,
    "ticket_category_cancellation": 3,
    "monthly_charge_delta": 40.0
  }'
```

**Response:**
```json
{
  "customer_id": "C0001",
  "churn_probability": 0.1658,
  "churn_label": false,
  "model_version": "1"
}
```

### `GET /model-info`
Returns model name, version, stage, and full feature list.

**Start server:**
```bash
make serve
```

**Screenshot placeholder:** FastAPI `/docs` (Swagger UI) showing all endpoints.

---

## Stage 6 — Drift Detection + Automated Retraining

**Files:** `src/churn/monitoring/drift.py`, `src/churn/retraining/trigger.py`  
**Script:** `scripts/generate_drift_report.py`

Evidently compares the training distribution against the test split (simulating production data).

**Output:**
```
Drift score: 0.200 | Threshold: 0.15
Trigger retrain: True
Report saved to reports/drift_report.html
```

When drift exceeds the threshold (0.15), the retraining trigger:
1. Runs `train_and_log()` with fresh data
2. Registers the new model version
3. Promotes it to **Staging** automatically
4. Leaves **Production** promotion as a manual gate

```bash
make drift                              # generate report
python -m churn.retraining.trigger      # run trigger
```

**Screenshot placeholder:** Evidently HTML drift report showing drifted columns.

---

## Stage 7 — Monitoring (Prometheus)

**File:** `src/churn/monitoring/metrics.py`

The API exposes `/metrics` for Prometheus scraping:

| Metric | Type | Description |
|--------|------|-------------|
| `churn_api_requests_total` | Counter | Total requests by endpoint |
| `churn_api_request_latency_seconds` | Histogram | Latency distribution |
| `churn_feature_drift_score` | Gauge | Latest Evidently drift score |
| `churn_model_version` | Gauge | Active model version |

```bash
curl http://localhost:8000/metrics
```

---

## Stage 8 — CI/CD/CT (GitHub Actions)

**Files:** `.github/workflows/ci.yml`, `cd.yml`, `ct.yml`

| Workflow | Trigger | Actions |
|----------|---------|---------|
| **CI** | Every push/PR | ruff lint → mypy type check → pytest (16 tests, 42% coverage) |
| **CD** | Merge to `main` | Build Docker image → push to GHCR |
| **CT** | Weekly (Mon 02:00 UTC) + manual | Generate data → features → drift report → retrain if triggered |

**Screenshot placeholder:** GitHub Actions showing CI workflow passing.

---

## Project Structure

```
mlops-churn/
├── .github/workflows/     # CI, CD, CT pipelines
├── data/raw/              # DVC-tracked raw data
├── src/churn/
│   ├── features/          # Feature engineering + sklearn Pipeline
│   ├── data/              # Loaders + splitter
│   ├── models/            # Training, evaluation, registry
│   ├── serving/           # FastAPI app
│   ├── monitoring/        # Evidently drift + Prometheus metrics
│   └── retraining/        # Drift-triggered retraining
├── tests/                 # 16 unit + integration tests
├── scripts/               # CLI entrypoints
└── docker/                # Dockerfile + docker-compose
```

---

## Quickstart

```bash
git clone git@github.com:Balagopal2022BCS0058/mlops-assignment-2..git
cd mlops-assignment-2.
make install        # install dependencies
make mlflow &       # start MLflow server (port 5000)
make data           # generate data + features
make train          # train models + register in MLflow
make serve &        # start inference API (port 8000)
make drift          # run drift report
make test           # run all tests
```

---

## Key Design Decisions

1. **Single sklearn Pipeline artifact** — scaler + classifier serialized together, loaded identically at training and inference. Eliminates train/serve skew.
2. **Time-based feature windows** — ticket frequency windows use only data prior to a fixed reference date. No future leakage.
3. **MLflow as single source of truth** — one tool handles experiment tracking, artifact storage, and the model registry.
4. **Production promotion is manual** — the CT workflow auto-promotes to Staging but gates Production behind human review.
5. **Pydantic validation at the API boundary** — all requests validated before reaching the model.
