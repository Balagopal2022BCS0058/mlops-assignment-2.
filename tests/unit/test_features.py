import pandas as pd
import pytest

from churn.features.engineering import build_features, get_feature_columns


@pytest.fixture()
def sample_data():
    customers = pd.DataFrame([
        {"customer_id": "C001", "monthly_charge": 50.0, "tenure_months": 12},
        {"customer_id": "C002", "monthly_charge": 150.0, "tenure_months": 3},
    ])
    tickets = pd.DataFrame([
        {"ticket_id": "T001", "customer_id": "C001", "ticket_date": "2024-03-28",
         "category": "billing", "sentiment_score": -0.3},
        {"ticket_id": "T002", "customer_id": "C001", "ticket_date": "2024-02-15",
         "category": "technical", "sentiment_score": 0.1},
        {"ticket_id": "T003", "customer_id": "C002", "ticket_date": "2024-03-31",
         "category": "cancellation", "sentiment_score": -0.8},
    ])
    return customers, tickets


def test_feature_columns_match(sample_data):
    customers, tickets = sample_data
    features = build_features(customers, tickets)
    expected = set(["customer_id"] + get_feature_columns())
    assert set(features.columns) == expected


def test_ticket_freq_7d_only_counts_recent(sample_data):
    customers, tickets = sample_data
    features = build_features(customers, tickets)
    c001 = features[features["customer_id"] == "C001"].iloc[0]
    # Only T001 (2024-03-28) is within 7d of reference 2024-03-31
    assert c001["ticket_freq_7d"] == 1
    # T002 is on 2024-02-15 which is 44 days before reference 2024-03-31 — outside 30d window
    assert c001["ticket_freq_30d"] == 1


def test_no_future_data_leakage(sample_data):
    customers, tickets = sample_data
    # Add a future ticket — must not appear in any window
    tickets = pd.concat([tickets, pd.DataFrame([{
        "ticket_id": "T999", "customer_id": "C001",
        "ticket_date": "2024-04-15",  # after reference date
        "category": "general", "sentiment_score": 0.5
    }])])
    features = build_features(customers, tickets)
    c001 = features[features["customer_id"] == "C001"].iloc[0]
    assert c001["ticket_freq_7d"] == 1  # future ticket excluded


def test_sentiment_score_bounds(sample_data):
    customers, tickets = sample_data
    features = build_features(customers, tickets)
    assert features["ticket_sentiment_score"].between(-1.0, 1.0).all()


def test_all_customers_in_output(sample_data):
    customers, tickets = sample_data
    features = build_features(customers, tickets)
    assert set(features["customer_id"]) == {"C001", "C002"}
