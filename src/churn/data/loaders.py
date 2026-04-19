import pandas as pd

from churn.config import settings


def load_customers() -> pd.DataFrame:
    return pd.read_csv(f"{settings.data_raw_dir}/customers.csv")


def load_tickets() -> pd.DataFrame:
    df = pd.read_csv(f"{settings.data_raw_dir}/tickets.csv")
    df["ticket_date"] = pd.to_datetime(df["ticket_date"])
    return df


def load_labels() -> pd.DataFrame:
    return pd.read_csv(f"{settings.data_raw_dir}/churn_labels.csv")


def load_features() -> pd.DataFrame:
    return pd.read_parquet(f"{settings.data_processed_dir}/features.parquet")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(f"{settings.data_splits_dir}/{name}.parquet")
