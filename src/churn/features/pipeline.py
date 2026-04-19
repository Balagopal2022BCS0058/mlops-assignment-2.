"""sklearn Pipeline factory — single artifact guarantees train/serve parity."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])
