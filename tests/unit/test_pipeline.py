import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from churn.features.pipeline import build_pipeline


def test_pipeline_fit_predict():
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipe = build_pipeline(clf)
    X = np.random.rand(50, 10)
    y = (X[:, 0] > 0.5).astype(int)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 50


def test_pipeline_predict_proba():
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipe = build_pipeline(clf)
    X = np.random.rand(50, 10)
    y = (X[:, 0] > 0.5).astype(int)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_reproducible():
    clf1 = RandomForestClassifier(n_estimators=5, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=5, random_state=42)
    X = np.random.rand(50, 10)
    y = (X[:, 0] > 0.5).astype(int)
    p1, p2 = build_pipeline(clf1), build_pipeline(clf2)
    p1.fit(X, y)
    p2.fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))
