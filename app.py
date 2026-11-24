
# ============================================================
# LOG ANOMALY DETECTION AND ALERT MANAGEMENT
#
# MSc Computer Science Dissertation
#
# PUBLIC SYNTHETIC DEMONSTRATION
#
# Scientific datasets and labels are not loaded.
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

# Portable package discovery.

if (HERE / "src" / "logad").is_dir():
    SOURCE_ROOT = HERE / "src"

elif (HERE.parent / "src" / "logad").is_dir():
    SOURCE_ROOT = HERE.parent / "src"

else:
    raise RuntimeError(
        "Required logad package not found."
    )

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from logad.dashboard_alert_management_candidate_v1 import (
    AlertError,
    AlertManager,
    generate_alerts,
)

st.set_page_config(
    page_title="Log Anomaly Detection Dashboard",
    page_icon="📊",
    layout="wide",
)

NOTICE = (
    "DEMONSTRATION MODE: Synthetic anomaly scores. "
    "Not scientific validation results."
)

SCORERS = [
    "rarity",
    "isolation_forest_seed11",
    "isolation_forest_seed22",
    "isolation_forest_seed33",
    "isolation_forest_seed44",
    "isolation_forest_seed55",
]

POLICIES = {
    "demonstration_policy_a": 0.70,
    "demonstration_policy_b": 0.85,
}

RUN_ID = "synthetic-demonstration-v1"


# ============================================================
# SYNTHETIC SCORE GENERATION
# ============================================================

def synthetic_scores():

    rng = np.random.default_rng(20261124)

    output = {}

    for index, scorer in enumerate(SCORERS):

        values = rng.beta(
            2.0 + index * 0.15,
            5.0,
            size=120,
        )

        for position in (12, 29, 53, 78, 101):

            values[position] = min(
                0.99,
                0.88 + index * 0.012,
            )

        output[scorer] = values.astype(
            np.float64
        )

    return output


# ============================================================
# PERMANENT BACKEND INTEGRATION
# ============================================================

def manager_for(scorer, policy):

    if "d5e_managers" not in st.session_state:
        st.session_state["d5e_managers"] = {}

    managers = st.session_state["d5e_managers"]

    key = f"{scorer}:{policy}"

    if key not in managers:

        scores = synthetic_scores()[scorer]

        threshold = POLICIES[policy]

        alerts = generate_alerts(
            scores,
            threshold,
            run_id=RUN_ID,
            scorer=scorer,
            policy=policy,
            window_ids=list(range(len(scores))),
        )

        managers[key] = AlertManager(alerts)

    return managers[key]


def filtered_views(
    manager,
    scorer,
    policy,
    severities,
    statuses,
):

    selected = []

    for severity in severities:

        for alert_status in statuses:

            views = manager.filter(
                scorer=scorer,
                policy=policy,
                severity=severity,
                status=alert_status,
            )

            selected.extend(views)

    return tuple(selected)


def view_rows(views):

    rows = []

    for view in views:

        alert = view.alert

        rows.append({
            "alert_id": alert.alert_id,
            "window": alert.window_id,
            "score": alert.score,
            "threshold": alert.threshold,
            "scorer": alert.scorer,
            "policy": alert.policy,
            "severity": alert.severity,
            "status": view.status,
        })

    return rows


def audit_rows(managers):

    events = []

    for manager in managers.values():

        for view in manager.filter():

            for event in view.events:

                events.append({
                    "timestamp_utc": event.timestamp_utc,
                    "alert_id": event.alert_id,
                    "previous_status": event.previous_status,
                    "new_status": event.new_status,
                    "actor": event.actor,
                })

    events.sort(
        key=lambda item: item["timestamp_utc"]
    )

    return events


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title(
    "Log Anomaly Detection and Alert Management"
)

st.warning(NOTICE)

st.info(
    "Operational severity is a presentation policy, "
    "not scientifically calibrated severity. "
    "Acknowledgement and resolution are workflow "
    "actions, not anomaly ground-truth labels."
)

tabs = st.tabs([
    "Anomaly Scores",
    "Alert Management",
    "Research Status",
    "Reproducibility",
])

# Shared controls are created outside the tabs so both
# visualization and alert management use identical inputs.

scorer = st.selectbox(
    "Demonstration scorer",
    SCORERS,
    key="d5e_scorer",
)

policy = st.selectbox(
    "Demonstration threshold policy",
    list(POLICIES),
    key="d5e_policy",
)

scores = synthetic_scores()[scorer]

threshold = POLICIES[policy]

manager = manager_for(
    scorer,
    policy,
)

all_views = manager.filter(
    scorer=scorer,
    policy=policy,
)

# ============================================================
# TAB 1: ANOMALY SCORES
# ============================================================

with tabs[0]:

    st.header(
        "Synthetic Anomaly-Score Visualization"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Synthetic windows",
        len(scores),
    )

    c2.metric(
        "Demonstration threshold",
        f"{threshold:.3f}",
    )

    c3.metric(
        "Generated alerts",
        len(all_views),
    )

    chart_data = pd.DataFrame({
        "Window": np.arange(len(scores)),
        "Anomaly score": scores,
        "Threshold": np.full(
            len(scores),
            threshold,
        ),
    })

    st.subheader("Scores by window")

    st.line_chart(
        chart_data.set_index("Window")
    )

    st.subheader("Score distribution")

    counts, edges = np.histogram(
        scores,
        bins=20,
        range=(0.0, 1.0),
    )

    distribution = pd.DataFrame({
        "Score bin": [
            f"{edges[i]:.2f}-{edges[i+1]:.2f}"
            for i in range(len(counts))
        ],
        "Count": counts,
    })

    st.bar_chart(
        distribution.set_index("Score bin")
    )

    st.caption(
        "All scores and thresholds shown here are "
        "synthetic demonstration values. "
        "No frozen validation scores are loaded."
    )


# ============================================================
# TAB 2: ALERT MANAGEMENT
# ============================================================

with tabs[1]:

    st.header(
        "Operational Alert Management"
    )

    st.caption(
        "Alerts, severity, filtering, lifecycle "
        "transitions and audit events are managed "
        "by the permanent alert backend."
    )

    severities = st.multiselect(
        "Severity filter",
        ["LOW", "MEDIUM", "HIGH"],
        default=["LOW", "MEDIUM", "HIGH"],
        key="d5e_severity",
    )

    statuses = st.multiselect(
        "Status filter",
        ["OPEN", "ACKNOWLEDGED", "RESOLVED"],
        default=[
            "OPEN",
            "ACKNOWLEDGED",
            "RESOLVED",
        ],
        key="d5e_status",
    )

    views = filtered_views(
        manager,
        scorer,
        policy,
        severities,
        statuses,
    )

    rows = view_rows(views)

    st.metric(
        "Visible alerts",
        len(rows),
    )

    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
        )

        options = {
            (
                f"{row['alert_id'][:12]} | "
                f"window {row['window']} | "
                f"{row['status']}"
            ): row["alert_id"]
            for row in rows
        }

        selected_label = st.selectbox(
            "Select alert",
            list(options),
            key="d5e_selected_alert",
        )

        selected_id = options[selected_label]

        selected_view = manager.get(
            selected_id
        )

        left, right = st.columns(2)

        with left:

            if st.button(
                "Acknowledge alert",
                key="d5e_ack",
                disabled=(
                    selected_view.status != "OPEN"
                ),
            ):

                try:

                    manager.acknowledge(
                        selected_id,
                        actor="demonstration_operator",
                    )

                    st.rerun()

                except AlertError as exc:
                    st.error(str(exc))

        with right:

            if st.button(
                "Resolve alert",
                key="d5e_resolve",
                disabled=(
                    selected_view.status
                    != "ACKNOWLEDGED"
                ),
            ):

                try:

                    manager.resolve(
                        selected_id,
                        actor="demonstration_operator",
                    )

                    st.rerun()

                except AlertError as exc:
                    st.error(str(exc))

    else:

        st.info(
            "No alerts match the selected filters."
        )

    st.subheader(
        "Timestamped Audit History"
    )

    history = audit_rows(
        st.session_state["d5e_managers"]
    )

    if history:

        st.dataframe(
            pd.DataFrame(history),
            width="stretch",
        )

    else:

        st.caption(
            "No operational actions recorded."
        )

    st.warning(
        "Operational alert state is session-local. "
        "It is not durable across application restarts."
    )


# ============================================================
# TAB 3: RESEARCH STATUS
# ============================================================

with tabs[2]:

    st.header("Research Status")

    st.write(
        "This MSc Computer Science dissertation "
        "investigates a lightweight log-anomaly "
        "detection and alert-management system."
    )

    st.subheader("Research methodology")

    st.markdown(
        "- Log-template rarity baseline\n"
        "- Isolation Forest comparison\n"
        "- Chronological evaluation design\n"
        "- Validation-based threshold selection\n"
        "- Reproducibility and integrity controls"
    )

    st.warning(
        "This public demonstration does not "
        "display private evaluation results, "
        "ground-truth labels, frozen validation "
        "scores or private research manifests."
    )

    st.write(
        "Exact historical score-row alignment "
        "for the frozen validation arrays "
        "has not been verified."
    )

    st.write(
        "Scientific model comparisons and "
        "evaluation metrics must be reported "
        "separately using verified research evidence."
    )


# ============================================================
# TAB 4: REPRODUCIBILITY
# ============================================================

with tabs[3]:

    st.header(
        "Reproducibility and Integrity"
    )

    st.markdown(
        "- Synthetic scores use a fixed random seed.\n"
        "- Alert generation uses the permanent backend.\n"
        "- Thresholds are demonstration constants.\n"
        "- Scientific datasets are not loaded.\n"
        "- Ground-truth labels are not accessed.\n"
        "- No model training is performed.\n"
        "- No threshold recalibration is performed.\n"
        "- Operational changes do not modify scores."
    )

    st.info(
        "Private research evidence remains "
        "outside the public demonstration package."
    )
