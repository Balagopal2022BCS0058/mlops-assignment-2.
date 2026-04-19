import pytest
from pydantic import ValidationError

from churn.schemas import TicketFeatures


def test_valid_payload():
    f = TicketFeatures(
        customer_id="C001",
        ticket_freq_7d=2,
        ticket_freq_30d=5,
        ticket_freq_90d=10,
        ticket_sentiment_score=-0.3,
    )
    assert f.customer_id == "C001"


def test_negative_freq_rejected():
    with pytest.raises(ValidationError):
        TicketFeatures(customer_id="C001", ticket_freq_7d=-1, ticket_freq_30d=5,
                       ticket_freq_90d=10, ticket_sentiment_score=0.0)


def test_sentiment_out_of_range_rejected():
    with pytest.raises(ValidationError):
        TicketFeatures(customer_id="C001", ticket_freq_7d=1, ticket_freq_30d=5,
                       ticket_freq_90d=10, ticket_sentiment_score=1.5)


def test_defaults_applied():
    f = TicketFeatures(customer_id="C001", ticket_freq_7d=0, ticket_freq_30d=0,
                       ticket_freq_90d=0, ticket_sentiment_score=0.0)
    assert f.ticket_category_billing == 0
    assert f.monthly_charge_delta == 0.0
