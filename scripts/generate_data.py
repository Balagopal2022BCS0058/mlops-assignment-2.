"""Generates synthetic churn dataset: customers, tickets, and churn labels."""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_CUSTOMERS = 1000
END_DATE = datetime(2024, 3, 31)
START_DATE = datetime(2023, 1, 1)
CATEGORIES = ["billing", "technical", "cancellation", "general"]

os.makedirs("data/raw", exist_ok=True)


def _make_customers() -> pd.DataFrame:
    records = []
    for i in range(N_CUSTOMERS):
        cid = f"C{i:04d}"
        monthly_charge = round(random.uniform(20, 200), 2)
        tenure_months = random.randint(1, 36)
        records.append({"customer_id": cid, "monthly_charge": monthly_charge,
                        "tenure_months": tenure_months})
    return pd.DataFrame(records)


def _make_tickets(customers: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in customers.iterrows():
        cid = row["customer_id"]
        n_tickets = np.random.poisson(lam=4)
        for _ in range(n_tickets):
            ticket_date = START_DATE + timedelta(
                days=random.randint(0, (END_DATE - START_DATE).days)
            )
            category = random.choices(
                CATEGORIES, weights=[0.3, 0.35, 0.15, 0.2]
            )[0]
            sentiment_raw = random.gauss(-0.1 if category == "cancellation" else 0.1, 0.4)
            sentiment = round(max(-1.0, min(1.0, sentiment_raw)), 4)
            records.append({
                "ticket_id": f"T{len(records):05d}",
                "customer_id": cid,
                "ticket_date": ticket_date.strftime("%Y-%m-%d"),
                "category": category,
                "sentiment_score": sentiment,
            })
    return pd.DataFrame(records)


def _make_labels(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    agg = (
        tickets.groupby("customer_id")
        .agg(total_tickets=("ticket_id", "count"), avg_sentiment=("sentiment_score", "mean"),
             cancel_tickets=("category", lambda x: (x == "cancellation").sum()))
        .reset_index()
    )
    merged = customers.merge(agg, on="customer_id", how="left").fillna(0)

    def _churn(row) -> int:
        score = 0.0
        score += min(row["total_tickets"] / 15, 1.0) * 0.35
        score += max(0, -row["avg_sentiment"]) * 0.30
        score += min(row["cancel_tickets"] / 3, 1.0) * 0.25
        score += (1 - min(row["tenure_months"] / 24, 1.0)) * 0.10
        return int(random.random() < score)

    merged["churn"] = merged.apply(_churn, axis=1)
    return merged[["customer_id", "churn"]]


if __name__ == "__main__":
    customers = _make_customers()
    tickets = _make_tickets(customers)
    labels = _make_labels(customers, tickets)

    customers.to_csv("data/raw/customers.csv", index=False)
    tickets.to_csv("data/raw/tickets.csv", index=False)
    labels.to_csv("data/raw/churn_labels.csv", index=False)

    print(f"Customers: {len(customers)}")
    print(f"Tickets  : {len(tickets)}")
    print(f"Churn rate: {labels['churn'].mean():.2%}")
