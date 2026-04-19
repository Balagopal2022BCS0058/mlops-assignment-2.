from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total API requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "churn_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

DRIFT_SCORE = Gauge(
    "churn_feature_drift_score",
    "Latest feature drift score from Evidently",
)

MODEL_VERSION = Gauge(
    "churn_model_version",
    "Currently deployed model version",
)
