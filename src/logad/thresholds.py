
"""Leakage-safe anomaly threshold calibration.

Higher scores mean more anomalous.
Alerts use the strict comparison: score > threshold.

This module contains reusable functions only.
Importing it does not read files, labels, or datasets.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    policy: str
    status: str
    threshold: float
    objective_value: Optional[float]
    reason: Optional[str]
    fallback_used: bool


def _scores(values, name):
    """Validate a non-empty one-dimensional finite score vector."""
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")

    return array


def _labels(values, expected_count):
    """Validate binary labels, with anomaly represented by 1."""
    array = np.asarray(values)

    if array.ndim != 1 or array.size != expected_count:
        raise ValueError("Validation labels have incorrect shape.")

    if not np.isin(array, [0, 1]).all():
        raise ValueError("Validation labels must be binary 0/1.")

    return array.astype(np.int8)


def alert_mask(scores, threshold):
    """Apply the approved strict > alert rule."""
    values = _scores(scores, "scores")

    if not (
        math.isfinite(threshold)
        or threshold == math.inf
    ):
        raise ValueError("Invalid threshold.")

    return values > threshold


def training_reference_threshold(training_scores):
    """99th percentile of training scores, using linear quantile."""
    scores = _scores(training_scores, "training_scores")

    threshold = float(
        np.quantile(scores, 0.99, method="linear")
    )

    return ThresholdResult(
        policy="training_reference",
        status="supported",
        threshold=threshold,
        objective_value=None,
        reason=None,
        fallback_used=False,
    )


def candidate_thresholds(validation_scores):
    """Return thresholds for all achievable strict > alert sets.

    Every distinct score is a candidate threshold.
    The value immediately below the minimum is included
    to represent the all-alert case.
    Positive infinity represents the no-alert case.
    """
    scores = _scores(validation_scores, "validation_scores")

    unique = np.unique(scores)

    below_minimum = np.nextafter(
        unique[0],
        -np.inf,
    )

    candidates = [
        float(below_minimum),
        *[float(value) for value in unique],
        math.inf,
    ]

    return candidates


def _metrics(predicted, labels):
    """Return binary confusion counts and defined metrics."""
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    tn = int(np.sum((~predicted) & (labels == 0)))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else None
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else None
    )

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0
        else None
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def _fallback(policy, training_scores, reason):
    """Use the approved training-reference fallback."""
    reference = training_reference_threshold(training_scores)

    return ThresholdResult(
        policy=policy,
        status="unsupported",
        threshold=reference.threshold,
        objective_value=None,
        reason=reason,
        fallback_used=True,
    )


def validation_f1_threshold(
    training_scores,
    validation_scores,
    validation_labels,
):
    """Maximise anomaly-class validation F1."""
    train = _scores(training_scores, "training_scores")
    scores = _scores(validation_scores, "validation_scores")
    labels = _labels(validation_labels, scores.size)

    if np.unique(labels).size < 2:
        return _fallback(
            "validation_f1",
            train,
            "validation_missing_class",
        )

    best = None

    for threshold in candidate_thresholds(scores):
        metrics = _metrics(scores > threshold, labels)

        objective = metrics["f1"]

        if objective is None:
            continue

        if (
            best is None
            or objective > best[0]
            or (
                objective == best[0]
                and threshold > best[1]
            )
        ):
            best = (objective, threshold)

    if best is None:
        return _fallback(
            "validation_f1",
            train,
            "validation_f1_undefined",
        )

    return ThresholdResult(
        policy="validation_f1",
        status="supported",
        threshold=best[1],
        objective_value=best[0],
        reason=None,
        fallback_used=False,
    )


def false_alert_constrained_threshold(
    training_scores,
    validation_scores,
    validation_labels,
    maximum_fpr=0.01,
):
    """Maximise recall subject to validation FPR <= 0.01."""
    train = _scores(training_scores, "training_scores")
    scores = _scores(validation_scores, "validation_scores")
    labels = _labels(validation_labels, scores.size)

    if maximum_fpr != 0.01:
        raise ValueError("Maximum FPR must match approved 0.01.")

    if np.unique(labels).size < 2:
        return _fallback(
            "false_alert_constrained",
            train,
            "validation_missing_class",
        )

    best = None

    for threshold in candidate_thresholds(scores):
        metrics = _metrics(scores > threshold, labels)

        if metrics["fpr"] is None:
            continue

        if metrics["fpr"] > maximum_fpr:
            continue

        objective = metrics["recall"]

        if objective is None:
            continue

        if (
            best is None
            or objective > best[0]
            or (
                objective == best[0]
                and threshold > best[1]
            )
        ):
            best = (objective, threshold)

    if best is None:
        return _fallback(
            "false_alert_constrained",
            train,
            "no_feasible_validation_threshold",
        )

    return ThresholdResult(
        policy="false_alert_constrained",
        status="supported",
        threshold=best[1],
        objective_value=best[0],
        reason=None,
        fallback_used=False,
    )


def threshold_to_json(value):
    """Convert a threshold to a JSON-safe representation."""
    if value == math.inf:
        return {
            "type": "positive_infinity",
            "value": None,
        }

    if not math.isfinite(value):
        raise ValueError("Invalid threshold.")

    return {
        "type": "finite",
        "value": float(value),
    }


def result_to_json(result):
    """Serialise a threshold result without non-standard JSON values."""
    return {
        "policy": result.policy,
        "status": result.status,
        "threshold": threshold_to_json(result.threshold),
        "objective_value": result.objective_value,
        "reason": result.reason,
        "fallback_used": result.fallback_used,
    }
