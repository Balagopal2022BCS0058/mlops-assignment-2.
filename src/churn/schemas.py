from pydantic import BaseModel, Field


class TicketFeatures(BaseModel):
    customer_id: str
    ticket_freq_7d: int = Field(ge=0)
    ticket_freq_30d: int = Field(ge=0)
    ticket_freq_90d: int = Field(ge=0)
    ticket_sentiment_score: float = Field(ge=-1.0, le=1.0)
    ticket_category_billing: int = Field(ge=0, default=0)
    ticket_category_technical: int = Field(ge=0, default=0)
    ticket_category_cancellation: int = Field(ge=0, default=0)
    ticket_category_general: int = Field(ge=0, default=0)
    avg_time_between_tickets_days: float = Field(ge=0.0, default=0.0)
    monthly_charge_delta: float = 0.0


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: bool
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_stage: str
