
"""
Minimal dissertation alert-management backend.

Operational alerts are NOT ground-truth anomaly labels.

Scientific inputs:
    scores
    frozen threshold
    scorer
    threshold policy
    window identities

Alert rule:
    thresholds.alert_mask(scores, threshold)
    Equivalent to score > threshold.

Severity policy:
    exceedance = score - threshold
    relative_exceedance = exceedance / max(abs(threshold), 1.0)

    LOW:    relative_exceedance <= 0.10
    MEDIUM: relative_exceedance <= 0.50
    HIGH:   relative_exceedance > 0.50

These are operational presentation categories.
They are not statistically calibrated severity estimates.

For threshold +infinity, no alerts are generated.

Alert lifecycle:
    OPEN -> ACKNOWLEDGED -> RESOLVED

No direct OPEN -> RESOLVED transition.
No changes after RESOLVED.

All scientific fields are immutable.
Lifecycle events are stored separately.
No file I/O or external services.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np

from logad import thresholds


SEVERITIES = frozenset({
    "LOW",
    "MEDIUM",
    "HIGH",
})

STATUSES = frozenset({
    "OPEN",
    "ACKNOWLEDGED",
    "RESOLVED",
})

ALLOWED_TRANSITIONS = {
    "OPEN": frozenset({"ACKNOWLEDGED"}),
    "ACKNOWLEDGED": frozenset({"RESOLVED"}),
    "RESOLVED": frozenset(),
}


class AlertError(ValueError):
    pass


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise AlertError(name + " must be a non-empty string")

    return value


def _threshold(value):
    if isinstance(value, (bool, np.bool_)):
        raise AlertError("threshold must be numeric")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise AlertError("threshold must be numeric") from None

    if math.isnan(number) or number == -math.inf:
        raise AlertError("invalid threshold")

    return number


def _score(value):
    if isinstance(value, (bool, np.bool_)):
        raise AlertError("score must be numeric")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise AlertError("score must be numeric") from None

    if not math.isfinite(number):
        raise AlertError("non-finite score")

    return number


def _window_id(value):
    if isinstance(value, bool):
        raise AlertError("invalid window identity")

    if isinstance(value, int) and value >= 0:
        return value

    if isinstance(value, str) and value.strip():
        return value

    raise AlertError("invalid window identity")


def severity_for(score, threshold):
    """
    Deterministic operational severity, not ground-truth severity.

    Call only for an actual alert.
    """
    score = _score(score)
    threshold = _threshold(threshold)

    if not score > threshold:
        raise AlertError("severity requires an alerting score")

    exceedance = score - threshold

    # Both values are finite for an alert because +inf cannot be exceeded.
    # For extreme finite values, subtraction may overflow to +inf.
    scale = max(abs(threshold), 1.0)

    relative = exceedance / scale

    if relative <= 0.10:
        return "LOW"

    if relative <= 0.50:
        return "MEDIUM"

    return "HIGH"


def _identity(run_id, scorer, policy, window_id):
    payload = json.dumps(
        [
            run_id,
            scorer,
            policy,
            window_id,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Alert:
    alert_id: str
    run_id: str
    scorer: str
    policy: str
    window_id: object
    score: float
    threshold: float
    severity: str


@dataclass(frozen=True)
class AuditEvent:
    alert_id: str
    previous_status: str
    new_status: str
    actor: str
    timestamp_utc: str


@dataclass(frozen=True)
class AlertView:
    alert: Alert
    status: str
    events: tuple


def generate_alerts(
    scores,
    threshold,
    *,
    run_id,
    scorer,
    policy,
    window_ids,
):
    """
    Generate immutable operational alerts from supplied frozen scores.

    This function never accesses labels or recalibrates thresholds.
    """
    run_id = _nonempty_string(run_id, "run_id")
    scorer = _nonempty_string(scorer, "scorer")
    policy = _nonempty_string(policy, "policy")

    threshold_value = _threshold(threshold)

    if isinstance(scores, (str, bytes)):
        raise AlertError("scores must be numeric")

    try:
        score_values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise AlertError("invalid scores") from None

    if score_values.ndim != 1 or score_values.size == 0:
        raise AlertError("scores must be a non-empty 1D vector")

    if not np.isfinite(score_values).all():
        raise AlertError("non-finite score")

    try:
        identities = tuple(window_ids)
    except TypeError:
        raise AlertError("window_ids must be iterable") from None

    if len(identities) != len(score_values):
        raise AlertError("window identity count mismatch")

    identities = tuple(
        _window_id(value)
        for value in identities
    )

    if len(set(identities)) != len(identities):
        raise AlertError("duplicate window identities")

    # The frozen scientific implementation defines the alert rule.
    mask = thresholds.alert_mask(
        score_values,
        threshold_value,
    )

    if len(mask) != len(score_values):
        raise AlertError("threshold mask length mismatch")

    alerts = []

    for index, active in enumerate(mask):
        if not bool(active):
            continue

        score = float(score_values[index])
        window_id = identities[index]

        alerts.append(
            Alert(
                alert_id=_identity(
                    run_id,
                    scorer,
                    policy,
                    window_id,
                ),
                run_id=run_id,
                scorer=scorer,
                policy=policy,
                window_id=window_id,
                score=score,
                threshold=threshold_value,
                severity=severity_for(
                    score,
                    threshold_value,
                ),
            )
        )

    return tuple(alerts)


class AlertManager:
    """
    Minimal in-memory operational alert manager.

    No persistence or database is used.
    """

    def __init__(self, alerts, *, clock=None):
        self._alerts = {}
        self._statuses = {}
        self._events = {}

        self._clock = clock or (
            lambda: datetime.now(timezone.utc)
        )

        for alert in alerts:
            if not isinstance(alert, Alert):
                raise AlertError("expected Alert objects")

            if alert.alert_id in self._alerts:
                raise AlertError("duplicate alert identity")

            self._alerts[alert.alert_id] = alert
            self._statuses[alert.alert_id] = "OPEN"
            self._events[alert.alert_id] = []

    def get(self, alert_id):
        if alert_id not in self._alerts:
            raise AlertError("unknown alert")

        return AlertView(
            alert=self._alerts[alert_id],
            status=self._statuses[alert_id],
            events=tuple(self._events[alert_id]),
        )

    def transition(self, alert_id, new_status, *, actor):
        if alert_id not in self._alerts:
            raise AlertError("unknown alert")

        if new_status not in STATUSES:
            raise AlertError("invalid alert status")

        actor = _nonempty_string(actor, "actor")

        previous = self._statuses[alert_id]

        if new_status not in ALLOWED_TRANSITIONS[previous]:
            raise AlertError("invalid lifecycle transition")

        timestamp = self._clock()

        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise AlertError("clock must return timezone-aware datetime")

        stamp = timestamp.astimezone(
            timezone.utc
        ).isoformat()

        event = AuditEvent(
            alert_id=alert_id,
            previous_status=previous,
            new_status=new_status,
            actor=actor,
            timestamp_utc=stamp,
        )

        self._events[alert_id].append(event)
        self._statuses[alert_id] = new_status

        return self.get(alert_id)

    def acknowledge(self, alert_id, *, actor):
        return self.transition(
            alert_id,
            "ACKNOWLEDGED",
            actor=actor,
        )

    def resolve(self, alert_id, *, actor):
        return self.transition(
            alert_id,
            "RESOLVED",
            actor=actor,
        )

    def filter(
        self,
        *,
        severity=None,
        status=None,
        scorer=None,
        policy=None,
        window_id=None,
    ):
        if severity is not None and severity not in SEVERITIES:
            raise AlertError("invalid severity filter")

        if status is not None and status not in STATUSES:
            raise AlertError("invalid status filter")

        if scorer is not None:
            _nonempty_string(scorer, "scorer")

        if policy is not None:
            _nonempty_string(policy, "policy")

        if window_id is not None:
            window_id = _window_id(window_id)

        selected = []

        for alert_id in self._alerts:
            view = self.get(alert_id)
            alert = view.alert

            if severity is not None and alert.severity != severity:
                continue

            if status is not None and view.status != status:
                continue

            if scorer is not None and alert.scorer != scorer:
                continue

            if policy is not None and alert.policy != policy:
                continue

            if window_id is not None and alert.window_id != window_id:
                continue

            selected.append(view)

        return tuple(selected)
