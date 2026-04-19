"""Retraining trigger — runs after drift report and decides whether to retrain."""
import json
import sys

from churn.models.registry import promote_to_staging
from churn.models.train import train_and_log


def maybe_retrain(drift_summary_path: str = "reports/drift_summary.json") -> None:
    with open(drift_summary_path) as f:
        summary = json.load(f)

    if not summary.get("trigger_retrain", False):
        print("No drift detected above threshold. Skipping retraining.")
        return

    print(f"Drift {summary['drift_score']:.3f} > threshold {summary['threshold']}. Retraining...")
    version = train_and_log("xgboost", register=True)
    if version:
        promote_to_staging(version)
        print(f"New model v{version} promoted to Staging. Awaiting manual Production promotion.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/drift_summary.json"
    maybe_retrain(path)
