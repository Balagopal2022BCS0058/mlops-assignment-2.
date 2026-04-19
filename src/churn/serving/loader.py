import mlflow.sklearn

from churn.config import settings
from churn.models.registry import get_production_model_uri

_model = None
_model_version = "unknown"


def load_model():
    global _model, _model_version
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = get_production_model_uri()
    _model = mlflow.sklearn.load_model(uri)

    from mlflow import MlflowClient
    client = MlflowClient()
    versions = client.get_latest_versions(settings.mlflow_model_name, stages=[settings.model_stage])
    if versions:
        _model_version = versions[0].version
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def get_model_version() -> str:
    return _model_version
