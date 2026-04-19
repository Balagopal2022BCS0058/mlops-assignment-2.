"""Stage 2: train + log to MLflow + register."""
import sys

sys.path.insert(0, "src")

from churn.models.train import train_and_log
from churn.models.registry import promote_to_staging, promote_to_production

if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    version = train_and_log(model_key, register=True)
    if version:
        promote_to_staging(version)
        promote_to_production(version)
