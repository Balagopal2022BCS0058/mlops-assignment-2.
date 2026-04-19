# MLOps Assignment II — ML Churn Prediction

Replaces a rule-based churn scoring engine with a production-grade ML pipeline.

## Stack

| Component | Tool |
|-----------|------|
| ML | scikit-learn + XGBoost |
| Feature pipeline | sklearn Pipeline (train/serve parity) |
| Experiment tracking | MLflow |
| Model registry | MLflow Registry (Staging → Production → Archived) |
| Inference API | FastAPI + Pydantic |
| Drift detection | Evidently AI |
| Monitoring | Prometheus |
| CI/CD/CT | GitHub Actions |
| Containerization | Docker + docker-compose |

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
make install

# 2. Start MLflow (new terminal)
make mlflow

# 3. Generate data, train, serve
make data
make train
make serve

# 4. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"C001","ticket_freq_7d":5,"ticket_freq_30d":12,"ticket_freq_90d":20,"ticket_sentiment_score":-0.6}'

# 5. Drift detection
make drift

# 6. Run tests
make test
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Churn probability for a customer |
| `/health` | GET | Model name, version, stage |
| `/model-info` | GET | Feature list + model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

## Project Structure

```
src/churn/
├── features/      # engineering.py — 10 features, no leakage
├── data/          # loaders.py, splitter.py
├── models/        # train.py, evaluate.py, registry.py
├── serving/       # FastAPI app.py, loader.py
├── monitoring/    # drift.py (Evidently), metrics.py (Prometheus)
└── retraining/    # trigger.py — auto-retrain on drift
```

## CI/CD/CT

- **CI** — lint + type check + tests on every push
- **CD** — Docker image → GHCR on merge to main
- **CT** — weekly scheduled retraining via GitHub Actions cron

See [docs/SUBMISSION.md](docs/SUBMISSION.md) for full stage-by-stage explanation.
