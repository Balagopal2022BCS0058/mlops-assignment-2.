"""Evidently-based feature drift detection."""
import json
import os

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from churn.config import settings
from churn.features.engineering import get_feature_columns
from churn.monitoring.metrics import DRIFT_SCORE


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_dir: str = "reports") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    feature_cols = get_feature_columns()

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
    )

    html_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(html_path)

    result = report.as_dict()
    drift_metrics = result["metrics"][0]["result"]
    drift_score = drift_metrics.get("share_of_drifted_columns", 0.0)

    DRIFT_SCORE.set(drift_score)

    summary = {
        "drift_score": drift_score,
        "drifted_columns": drift_metrics.get("number_of_drifted_columns", 0),
        "total_columns": drift_metrics.get("number_of_columns", 0),
        "threshold": settings.drift_threshold,
        "trigger_retrain": drift_score > settings.drift_threshold,
    }

    with open(os.path.join(output_dir, "drift_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Drift score: {drift_score:.3f} | Threshold: {settings.drift_threshold}")
    print(f"Trigger retrain: {summary['trigger_retrain']}")
    return summary
