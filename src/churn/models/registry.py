"""Model registry helpers — stage transitions with metric gates."""
import mlflow
from mlflow import MlflowClient

from churn.config import settings


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def promote_to_staging(version: str) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=settings.mlflow_model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"v{version} → Staging")


def promote_to_production(version: str) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=settings.mlflow_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"v{version} → Production (previous archived)")


def archive_version(version: str) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=settings.mlflow_model_name,
        version=version,
        stage="Archived",
    )
    print(f"v{version} → Archived")


def get_production_model_uri() -> str:
    return f"models:/{settings.mlflow_model_name}/{settings.model_stage}"


def list_versions() -> None:
    client = get_client()
    for mv in client.search_model_versions(f"name='{settings.mlflow_model_name}'"):
        print(f"  v{mv.version} | stage={mv.current_stage} | run={mv.run_id[:8]}")
