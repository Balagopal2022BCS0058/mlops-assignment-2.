"""Stage 1: raw → processed features parquet."""
import os
import sys

sys.path.insert(0, "src")

import pandas as pd

from churn.config import settings
from churn.data.loaders import load_customers, load_labels, load_tickets
from churn.data.splitter import split_and_save
from churn.features.engineering import build_features

os.makedirs(settings.data_processed_dir, exist_ok=True)

customers = load_customers()
tickets = load_tickets()
labels = load_labels()

features = build_features(customers, tickets)
features.to_parquet(f"{settings.data_processed_dir}/features.parquet", index=False)
print(f"Features shape: {features.shape}")

split_and_save(features, labels)
print("Data preparation complete.")
