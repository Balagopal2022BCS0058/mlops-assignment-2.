import os

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import settings


def split_and_save(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    merged = features.merge(labels, on="customer_id")
    X = merged.drop(columns=["customer_id", "churn"])
    y = merged["churn"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    os.makedirs(settings.data_splits_dir, exist_ok=True)
    for name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        df = Xs.copy()
        df["churn"] = ys.values
        df.to_parquet(f"{settings.data_splits_dir}/{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows | churn rate: {ys.mean():.2%}")
