PYTHONPATH=src

.PHONY: install data train serve test lint drift demo

install:
	pip install -e ".[dev]"

data:
	PYTHONPATH=$(PYTHONPATH) python scripts/generate_data.py
	PYTHONPATH=$(PYTHONPATH) python scripts/prepare_data.py

train:
	PYTHONPATH=$(PYTHONPATH) python scripts/train.py

serve:
	PYTHONPATH=$(PYTHONPATH) uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v --cov=src/churn --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/churn --ignore-missing-imports

drift:
	PYTHONPATH=$(PYTHONPATH) python scripts/generate_drift_report.py

mlflow:
	.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlflow-artifacts \
	  --host 0.0.0.0 --port 5001 --allowed-hosts "*"

demo:
	@echo "=== Starting demo: data → train → serve ==="
	$(MAKE) data
	$(MAKE) train
	$(MAKE) drift
	@echo "=== Run 'make serve' to start the API ==="
