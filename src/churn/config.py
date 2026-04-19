from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "churn-prediction"
    mlflow_model_name: str = "churn-classifier"
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_splits_dir: str = "data/splits"
    model_stage: str = "Production"
    drift_threshold: float = 0.15
    accuracy_decay_threshold: float = 0.05
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
