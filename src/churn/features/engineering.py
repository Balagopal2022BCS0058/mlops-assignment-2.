"""Feature engineering: derives ML features from raw customers + tickets tables."""
import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2024-03-31")
WINDOWS = {"7d": 7, "30d": 30, "90d": 90}


def build_features(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    tickets = tickets.copy()
    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"])

    # ticket frequency windows — only use data prior to reference date
    for label, days in WINDOWS.items():
        cutoff = REFERENCE_DATE - pd.Timedelta(days=days)
        mask = (tickets["ticket_date"] >= cutoff) & (tickets["ticket_date"] <= REFERENCE_DATE)
        freq = tickets[mask].groupby("customer_id").size().rename(f"ticket_freq_{label}")
        customers = customers.merge(freq, on="customer_id", how="left")
        customers[f"ticket_freq_{label}"] = customers[f"ticket_freq_{label}"].fillna(0).astype(int)

    # avg sentiment per customer
    sentiment = (
        tickets.groupby("customer_id")["sentiment_score"].mean().rename("ticket_sentiment_score")
    )
    customers = customers.merge(sentiment, on="customer_id", how="left")
    customers["ticket_sentiment_score"] = customers["ticket_sentiment_score"].fillna(0.0)

    # ticket category counts
    for cat in ["billing", "technical", "cancellation", "general"]:
        cat_mask = tickets["category"] == cat
        cat_counts = tickets[cat_mask].groupby("customer_id").size().rename(
            f"ticket_category_{cat}"
        )
        customers = customers.merge(cat_counts, on="customer_id", how="left")
        customers[f"ticket_category_{cat}"] = customers[f"ticket_category_{cat}"].fillna(0).astype(int)

    # avg time between tickets (days)
    def _avg_gap(group: pd.DataFrame) -> float:
        dates = group["ticket_date"].sort_values()
        if len(dates) < 2:
            return 0.0
        deltas = dates.diff().dropna().dt.days
        return float(deltas.mean())

    gaps = tickets.groupby("customer_id").apply(_avg_gap, include_groups=False).rename("avg_time_between_tickets_days")
    customers = customers.merge(gaps, on="customer_id", how="left")
    customers["avg_time_between_tickets_days"] = customers["avg_time_between_tickets_days"].fillna(0.0)

    # monthly charge delta — change vs mean (uses customer table only)
    # approximated as charge vs median across cohort
    median_charge = customers["monthly_charge"].median()
    customers["monthly_charge_delta"] = customers["monthly_charge"] - median_charge

    feature_cols = [
        "customer_id",
        "ticket_freq_7d", "ticket_freq_30d", "ticket_freq_90d",
        "ticket_sentiment_score",
        "ticket_category_billing", "ticket_category_technical",
        "ticket_category_cancellation", "ticket_category_general",
        "avg_time_between_tickets_days",
        "monthly_charge_delta",
    ]
    return customers[feature_cols].reset_index(drop=True)


def get_feature_columns() -> list[str]:
    return [
        "ticket_freq_7d", "ticket_freq_30d", "ticket_freq_90d",
        "ticket_sentiment_score",
        "ticket_category_billing", "ticket_category_technical",
        "ticket_category_cancellation", "ticket_category_general",
        "avg_time_between_tickets_days",
        "monthly_charge_delta",
    ]
