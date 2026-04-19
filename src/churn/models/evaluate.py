import io

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def plot_roc_curve(y_true, y_pred_proba) -> plt.Figure:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax)
    ax.set_title("ROC Curve")
    return fig


def plot_pr_curve(y_true, y_pred_proba) -> plt.Figure:
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, ax=ax)
    ax.set_title("Precision-Recall Curve")
    return fig


def plot_confusion_matrix(y_true, y_pred_proba, threshold: float = 0.5) -> plt.Figure:
    y_pred = (y_pred_proba >= threshold).astype(int)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    return fig
