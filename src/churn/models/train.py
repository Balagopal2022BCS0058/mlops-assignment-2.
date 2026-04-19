"""Training entrypoint. Logs everything to MLflow and registers the model."""
import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from churn.config import settings
from churn.data.loaders import load_split
from churn.features.engineering import get_feature_columns
from churn.features.pipeline import build_pipeline
from churn.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=3, use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    ),
}

FEATURE_COLS = get_feature_columns()


def train_and_log(model_key: str = "xgboost", register: bool = True) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_df = load_split("train")
    val_df = load_split("val")

    X_train, y_train = train_df[FEATURE_COLS], train_df["churn"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["churn"]

    clf = MODELS[model_key]
    pipeline = build_pipeline(clf)

    with mlflow.start_run(run_name=model_key) as run:
        mlflow.log_param("model_type", model_key)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("feature_list", ",".join(FEATURE_COLS))
        mlflow.log_params(clf.get_params())

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            for fig_fn, fig in [
                ("roc_curve.png", plot_roc_curve(y_val, y_proba)),
                ("pr_curve.png", plot_pr_curve(y_val, y_proba)),
                ("confusion_matrix.png", plot_confusion_matrix(y_val, y_proba)),
            ]:
                path = os.path.join(tmpdir, fig_fn)
                fig.savefig(path)
                mlflow.log_artifact(path)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        if register:
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, settings.mlflow_model_name)
            print(f"Registered {settings.mlflow_model_name} v{mv.version}")
            print(f"Metrics: {metrics}")
            return mv.version

    return run.info.run_id


if __name__ == "__main__":
    train_and_log("xgboost")
