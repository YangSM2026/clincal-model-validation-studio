"""Local interface for binary clinical-model validation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cmvs.core import evaluate_predictions, normalize_cell, read_csv, validate

st.set_page_config(page_title="Clinical Model Validation", page_icon="◫", layout="wide")
st.title("Clinical Model Validation")
st.caption("Binary-outcome model evaluation · Python edition")
st.info("Portfolio demonstration, not a clinical decision tool. The included dataset is synthetic. "
        "Do not upload identifiable patient records. Run locally for private data; a hosted app receives uploaded files.")


def fingerprint(raw: bytes, config: dict) -> str:
    return hashlib.sha256(raw + json.dumps(config, sort_keys=True).encode()).hexdigest()


def render_plot(rows: list, series: list[tuple[str, str, str]], xlabel: str, ylabel: str,
                *, diagonal: bool = False, ylim: tuple | None = None) -> None:
    if not rows:
        st.caption("No curve available.")
        return
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for x, y, label in series:
        if x in frame and y in frame:
            ax.plot(frame[x], frame[y], label=label, linewidth=1.8)
    if diagonal:
        ax.plot([0, 1], [0, 1], "--", color="#a0a7ad", linewidth=1, label="Reference")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with st.sidebar:
    st.header("Dataset")
    source = st.radio("Source", ["Synthetic example", "Upload CSV"])
    if source == "Synthetic example":
        demo = Path(__file__).parent / "examples" / "demo.csv"
        if not demo.exists():
            st.error("The bundled example is missing.")
            st.stop()
        raw = demo.read_bytes()
        st.caption("Fictional records; no patient data.")
    else:
        upload = st.file_uploader("CSV file (UTF-8)", type=["csv"])
        if upload is None:
            st.info("Choose a CSV to configure the analysis.")
            st.stop()
        raw = upload.getvalue()
    st.caption("No account, tracking service, or external model API is used by this app.")

try:
    data = read_csv(io.BytesIO(raw))
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

with st.expander("Dataset preview", expanded=False):
    st.caption(f"{len(data):,} rows · {len(data.columns)} columns. Preview includes only the first 10 rows.")
    st.dataframe(data.head(10), hide_index=True, width="stretch")

# Configuration widgets intentionally sit outside a form: outcome changes must refresh
# the eligible predictors and event values. Fitting still requires an explicit button.
with st.sidebar:
    st.header("Analysis specification")
    workflow = st.radio("Workflow", ["Train a logistic baseline", "Evaluate fixed probabilities"])
    fixed = workflow == "Evaluate fixed probabilities"
    columns = list(data.columns)
    outcome = st.selectbox("Binary outcome", columns,
                           index=(columns.index("event_30d") if "event_30d" in columns else len(columns) - 1)
                           if source == "Synthetic example" else None, placeholder="Select the outcome column")
    if outcome is None:
        st.info("Select an outcome column to continue.")
        st.stop()
    values = sorted(data[outcome].map(normalize_cell).dropna().unique().tolist())
    if not values:
        st.error("The outcome contains no observed values.")
        st.stop()
    event = st.selectbox("Positive event", values,
                         index=(values.index("yes") if "yes" in values else len(values) - 1)
                         if source == "Synthetic example" else None, placeholder="Select the event category")
    if event is None:
        st.info("Select which category represents the event.")
        st.stop()
    if len(values) != 2:
        st.warning(f"This baseline requires two outcome categories; found {len(values)}.")
    available = [c for c in columns if c != outcome]
    if fixed:
        probability = st.selectbox("Event probability column", available,
                                  index=available.index("fixed_risk") if source == "Synthetic example" and "fixed_risk" in available else None,
                                  placeholder="Select probabilities between 0 and 1")
        st.caption("Supply probabilities generated without fitting or tuning on these outcome labels. No model is trained here.")
        ready = probability is not None
    else:
        numeric_defaults = [c for c in ["age", "baseline_score"] if c in available] if source == "Synthetic example" else []
        numeric = st.multiselect("Numeric predictors", available, default=numeric_defaults)
        categorical_options = [c for c in available if c not in numeric]
        categorical = st.multiselect("Categorical predictors", categorical_options,
                                     default=[c for c in ["group"] if c in categorical_options] if source == "Synthetic example" else [])
        st.caption("Unselected columns are excluded. Remove IDs, post-outcome measurements, and other leakage sources.")
        test_size = st.slider("Holdout fraction", .15, .40, .25, .05)
        ready = bool(numeric or categorical)
    seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1)
    bootstrap = st.selectbox("Bootstrap resamples", [0, 200, 500, 1000])
    st.caption("Intervals describe uncertainty conditional on the fixed predictions, not the full model-development pipeline.")
    run = st.button("Run validation", type="primary", width="stretch",
                    disabled=len(values) != 2 or not ready)

config = dict(outcome=outcome, event=event, seed=int(seed), bootstrap=bootstrap)
if fixed:
    config.update(probability=probability)
else:
    config.update(numeric=numeric, categorical=categorical, test_size=test_size)
current = fingerprint(raw, config)
if run:
    try:
        with st.spinner("Evaluating fixed probabilities…" if fixed else "Fitting on training rows and evaluating the untouched holdout…"):
            result = evaluate_predictions(data, **config) if fixed else validate(data, **config)
        st.session_state["validation"] = dict(fingerprint=current, result=result, config=config)
    except (ValueError, TypeError, ArithmeticError) as exc:
        st.error(f"Validation could not run: {exc}")

saved = st.session_state.get("validation")
if not saved:
    if fixed:
        st.subheader("Evaluate an existing set of predictions")
        st.write("Select the event and its probability column, then run validation. Probabilities must lie between 0 and 1. No fitting or recalibration is performed.")
        st.warning("This app cannot establish that your predictions are independent of the evaluation outcomes. Confirm their provenance before interpreting performance.")
    else:
        st.subheader("Start with a prespecified baseline")
        st.write("Select the event and predictor types, then run validation. Imputation, encoding, "
                 "and scaling are learned from training rows only. A regularized logistic model is "
                 "compared with the training-event prevalence.")
        st.warning("This interface uses a stratified row split. It is not suitable for repeated observations "
                   "from the same patient, multi-centre independence, or temporal validation without a separate split strategy.")
    st.stop()
if saved["fingerprint"] != current:
    st.warning("Dataset or settings changed. Previous results are hidden; run validation to refresh them.")
    st.stop()

result = saved["result"]
audit = result.get("audit", {})
model_label = "Fixed probabilities" if fixed else "Logistic baseline"
st.subheader("Fixed-probability performance" if fixed else "Held-out performance")
if fixed:
    st.caption("All eligible supplied rows are evaluated without training. Independence of the predictions must be established outside this app.")
else:
    st.caption("One prespecified split. Repeatedly changing settings against this holdout turns it into a tuning set; "
               "reserve a separate independent dataset for final validation.")
metrics = result.get("metrics", {})
for col, (key, label) in zip(st.columns(4), [
    ("roc_auc", "ROC AUC"), ("average_precision", "Average precision"),
    ("brier_score", "Brier score ↓"), ("log_loss", "Log loss ↓")]):
    value = metrics.get(key)
    col.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "Unavailable")
if fixed:
    st.caption(f"Evaluated rows: {len(result.get('predictions', []))}")
else:
    st.caption(f"Training rows: {audit.get('n_train', '—')} · "
               f"Holdout rows: {audit.get('n_test', '—')}")
for warning in audit.get("warnings", []):
    st.warning(warning)

tabs = st.tabs(["Discrimination", "Calibration & utility", "Audit & uncertainty", "Export"])
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.markdown("#### ROC curve")
        render_plot(result.get("roc_curve", []), [("fpr", "tpr", model_label)],
                    "False-positive rate", "True-positive rate", diagonal=True, ylim=(0, 1.02))
    with right:
        st.markdown("#### Precision–recall curve")
        render_plot(result.get("pr_curve", []), [("recall", "precision", model_label)],
                    "Recall", "Precision", ylim=(0, 1.02))
    st.markdown("#### Threshold summary")
    st.caption("A threshold of 0.5 is a descriptive default, not a recommended clinical threshold.")
    st.dataframe(pd.DataFrame(result.get("thresholds", [])), hide_index=True, width="stretch")
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Calibration")
        render_plot(result.get("calibration", []), [("mean_predicted", "observed_rate", "Evaluation bins" if fixed else "Holdout bins")],
                    "Mean predicted probability", "Observed event rate", diagonal=True, ylim=(0, 1.02))
        st.dataframe(pd.DataFrame(result.get("calibration", [])), hide_index=True)
    with right:
        st.markdown("#### Decision curve")
        render_plot(result.get("decision_curve", []), [
            ("threshold", "net_benefit", model_label),
            ("threshold", "treat_all", "Treat all"),
            ("threshold", "treat_none", "Treat none")], "Threshold probability", "Net benefit")
        st.caption("Interpret only over a defensible threshold range. Net benefit alone does not establish clinical usefulness.")
with tabs[2]:
    if not fixed:
        st.markdown("#### Constant-probability comparator")
        st.json(result.get("comparator", {}))
    st.markdown("#### Bootstrap uncertainty")
    st.json(result.get("bootstrap", {}))
    st.markdown("#### Processing audit")
    st.json(audit)
    st.markdown("#### Recorded settings")
    st.json(saved["config"])
with tabs[3]:
    st.caption("Downloads contain model results and outcome/prediction rows. Treat outputs derived from private data as private too.")
    st.download_button("Download validation report (JSON)",
                       json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                       file_name="validation-report.json", mime="application/json")
    predictions = pd.DataFrame(result.get("predictions", []))
    st.download_button("Download evaluated predictions (CSV)" if fixed else "Download holdout predictions (CSV)", predictions.to_csv(index=False),
                       file_name="evaluated-predictions.csv" if fixed else "holdout-predictions.csv", mime="text/csv")
