"""Generate Evidently drift report comparing train vs test split."""
import sys

sys.path.insert(0, "src")

import pandas as pd

from churn.data.loaders import load_split
from churn.monitoring.drift import generate_drift_report

reference = load_split("train")
current = load_split("test")

summary = generate_drift_report(reference, current)
print(f"Report saved to reports/drift_report.html")
