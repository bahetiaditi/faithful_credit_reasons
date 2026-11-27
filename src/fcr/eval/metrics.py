"""Discrimination metrics used for every model. `risk` is higher for riskier applicants."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, risk: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def discrimination(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    risk = np.asarray(risk, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("need both outcomes to measure discrimination")
    auc = float(roc_auc_score(y, risk))
    return {
        "cases": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
        "roc_auc": auc, "gini": 2 * auc - 1,
        "pr_auc": float(average_precision_score(y, risk)), "ks": ks_statistic(y, risk),
    }
