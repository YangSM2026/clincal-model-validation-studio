"""Training-only preprocessing and held-out binary logistic model assessment.

A single internal split is not external validation. Bootstrap intervals condition
on the fitted model and do not include model-development uncertainty.
"""
from __future__ import annotations

import csv
import io
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

MISSING = {"", "na", "n/a", "null", "nan"}


def read_csv(file_or_path):
    """Read bounded UTF-8 CSV without coercing category identifiers to numbers."""
    raw = (file_or_path.read() if hasattr(file_or_path, "read")
           else Path(file_or_path).read_bytes())
    if isinstance(raw, bytes):
        if len(raw) > 1024 * 1024:
            raise ValueError("CSV must be no larger than 1 MB.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must use UTF-8 encoding.") from exc
    elif isinstance(raw, str):
        text = raw.lstrip("\ufeff")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            raise ValueError("CSV must be no larger than 1 MB.")
    else:
        raise ValueError("Expected a CSV path or readable text/binary stream.")
    try:
        rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(text), strict=True)
                if any(v.strip() for v in row)]
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV: {exc}") from exc
    if len(rows) < 3:
        raise ValueError("CSV requires a header and at least two data rows.")
    header, records = rows[0], rows[1:]
    if not 2 <= len(header) <= 60:
        raise ValueError("Use between 2 and 60 columns.")
    if any(not h or len(h) > 100 for h in header) or len(set(header)) != len(header):
        raise ValueError("Column names must be nonempty, unique and at most 100 characters.")
    if len(records) > 5000:
        raise ValueError("Use at most 5000 rows.")
    for i, row in enumerate(records, 2):
        if len(row) != len(header):
            raise ValueError(f"Record {i} has the wrong number of fields.")
        if any(len(v) > 500 for v in row):
            raise ValueError(f"Record {i} contains a cell longer than 500 characters.")
    return pd.DataFrame(records, columns=header, dtype=object)


def _clean(v):
    if pd.isna(v):
        return None
    value = str(v).strip()
    return None if value.lower() in MISSING else value


def _metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def _confusion(y, p, t):
    positive = p >= t
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    tn = int(np.sum(~positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    ratio = lambda a, b: a / b if b else None
    return {"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "accuracy": (tp + tn) / len(y)}


def validate(data, outcome, event, numeric, categorical, *, test_size=.25, seed=42, bootstrap=0):
    """Assess a fixed ridge-logistic baseline; never fit or tune on holdout rows.

    Returned row indices are zero-based input positions, not DataFrame labels.
    Event and categorical identifiers are matched textually. Missing outcomes
    are excluded; invalid numeric predictor values are treated as missing.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame, for example from read_csv().")
    if data.columns.has_duplicates or any(not isinstance(c, str) or not c.strip() for c in data.columns):
        raise ValueError("Column names must be unique, nonempty strings.")
    if not 2 <= len(data.columns) <= 60 or not 2 <= len(data) <= 5000:
        raise ValueError("Use 2–60 columns and 2–5000 rows.")
    if isinstance(numeric, str) or isinstance(categorical, str):
        raise ValueError("numeric and categorical must be lists of column names.")
    numeric, categorical = list(numeric), list(categorical)
    selected = numeric + categorical
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one predictor; predictor lists must not overlap or repeat.")
    if outcome not in data or any(c not in data for c in selected):
        raise ValueError("Outcome and predictor columns must exist in the data.")
    if outcome in selected:
        raise ValueError("The outcome cannot also be a predictor.")
    if isinstance(test_size, bool) or not isinstance(test_size, (float, int)) or not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction strictly between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1.")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or not 0 <= bootstrap <= 10000:
        raise ValueError("bootstrap must be an integer from 0 to 10000.")
    frame = data.reset_index(drop=True).map(_clean)
    present = frame[outcome].notna().to_numpy()
    ids = np.flatnonzero(present)
    labels = frame.loc[present, outcome].to_numpy()
    levels = sorted(set(labels))
    if len(levels) != 2:
        raise ValueError("Binary validation requires exactly two nonmissing outcome categories.")
    event = str(event).strip()
    if event not in levels:
        raise ValueError("The event must exactly match one outcome category.")
    y_all = (labels == event).astype(int)
    if min(np.bincount(y_all, minlength=2)) < 2:
        raise ValueError("Each outcome class needs at least two rows for stratified splitting.")
    try:
        train, test = train_test_split(ids, test_size=float(test_size), random_state=int(seed), stratify=y_all)
    except ValueError as exc:
        raise ValueError(f"Cannot form a stratified training/holdout split: {exc}") from exc
    train, test = np.sort(train), np.sort(test)
    y_train = (frame.loc[train, outcome].to_numpy() == event).astype(int)
    y_test = (frame.loc[test, outcome].to_numpy() == event).astype(int)
    if min(np.bincount(y_train, minlength=2)) < 5 or len(test) < 4 or len(set(y_test)) != 2:
        raise ValueError("Baseline fitting needs at least five training rows in each class and four holdout rows containing both classes.")
    prep, dropped, features, blocks = {}, [], [], []
    for name in selected:
        values = frame[name]
        if name in numeric:
            nums = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            nums[~np.isfinite(nums)] = np.nan
            valid = nums[train][np.isfinite(nums[train])]
            if len(valid) == 0:
                dropped.append({"name": name, "reason": "No usable numeric values in training"})
                continue
            median = float(np.median(valid))
            filled = np.where(np.isfinite(nums), nums, median)
            with np.errstate(over="ignore", invalid="ignore"):
                mean, scale = float(np.mean(filled[train])), float(np.std(filled[train]))
            if not np.isfinite(mean) or not np.isfinite(scale):
                raise ValueError(f"Numeric values in {name} are too extreme to preprocess safely.")
            if scale < 1e-12:
                dropped.append({"name": name, "reason": "Constant in training"})
                continue
            prep[name] = {"kind": "numeric", "median": median, "mean": mean, "scale": scale,
                          "invalid": int(np.sum(values.notna().to_numpy() & np.isnan(nums))),
                          "missing": int(values.isna().sum())}
            with np.errstate(over="ignore", invalid="ignore"):
                blocks.append(((filled - mean) / scale).reshape(-1, 1))
            features.append(name)
        else:
            known = [v for v in values.iloc[train] if v is not None]
            categories = sorted(set(known))
            if len(categories) < 2:
                dropped.append({"name": name, "reason": "All missing or constant in training"})
                continue
            if len(categories) > 30:
                raise ValueError(f"{name} has more than 30 training categories; exclude identifiers.")
            counts = Counter(known)
            mode = sorted(counts, key=lambda v: (-counts[v], v))[0]
            filled = np.array([mode if v is None else v for v in values], dtype=object)
            unseen = ~np.isin(filled, categories)
            prep[name] = {"kind": "categorical", "mode": mode, "categories": categories,
                          "unseen_holdout": int(unseen[test].sum()), "missing": int(values.isna().sum())}
            blocks.append(np.column_stack([filled == c for c in categories] + [unseen]).astype(float))
            features.extend([f"{name} = {c}" for c in categories] + [f"{name} = [unseen]"])
    if not blocks:
        raise ValueError("No usable predictors remain after training-only preprocessing.")
    if len(features) > 200:
        raise ValueError("More than 200 encoded features; exclude high-cardinality predictors.")
    X = np.column_stack(blocks)
    if not np.isfinite(X[ids]).all():
        raise ValueError("Numeric values are too extreme to preprocess safely.")
    # sklearn sums losses; C below matches upstream mean loss + lambda/2 * ||beta||².
    model = LogisticRegression(C=1 / (len(train) * .05), solver="lbfgs", max_iter=3000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X[train], y_train)
        except ConvergenceWarning as exc:
            raise ValueError("The optimizer did not converge; no validation result was produced.") from exc
    p = model.predict_proba(X[test])[:, 1]
    prevalence = float(np.mean(y_train))
    baseline = np.full(len(test), prevalence)
    result, notes = _assessment(y_test, p, bootstrap, seed, baseline)
    notes.extend(["Single internal split, not external validation.",
                  "No patient-level or site-level grouping is inferred; related observations require a grouped split.",
                  "Do not select models repeatedly on this holdout; reserve independent validation."])
    result.update({
        "config": {"outcome": outcome, "event": event, "numeric": numeric, "categorical": categorical,
                   "test_size": float(test_size), "seed": int(seed), "ridge_lambda": .05},
        "audit": {"n_input": len(data), "n_missing_outcome": int((~present).sum()),
                  "n_train": len(train), "n_test": len(test), "train_rows": train.tolist(), "test_rows": test.tolist(),
                  "training_events": int(y_train.sum()), "holdout_events": int(y_test.sum()),
                  "preprocessing": prep, "dropped_features": dropped, "features": features,
                  "duplicate_rows": int(frame.duplicated().sum()), "duplicates_removed": 0,
                  "warnings": notes, "iterations": int(model.n_iter_[0])},
        "predictions": [{"row": int(row), "observed": int(y), "probability": float(prob)}
                        for row, y, prob in zip(test, y_test, p)]})
    return result


def _assessment(y_test, p, bootstrap, seed, baseline=None):
    metrics = _metrics(y_test, p)
    fpr, tpr, rt = roc_curve(y_test, p)
    precision, recall, pt = precision_recall_curve(y_test, p)
    calibration = []
    # Fixed equal-width bins; omit empty bins rather than inventing observations.
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        mask = bins == b
        if mask.any():
            calibration.append({"mean_predicted": float(p[mask].mean()),
                                "observed_rate": float(y_test[mask].mean()), "n": int(mask.sum())})
    decisions = []
    for t in np.arange(.05, 1, .05):
        c = _confusion(y_test, p, t)
        decisions.append({"threshold": float(t),
                          "net_benefit": float((c["tp"] - c["fp"] * t / (1 - t)) / len(y_test)),
                          "treat_all": float(y_test.mean() - (1 - y_test.mean()) * t / (1 - t)),
                          "treat_none": 0.0})
    samples = {key: [] for key in metrics}
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(bootstrap):
        ix = rng.integers(0, len(y_test), len(y_test))
        if len(set(y_test[ix])) < 2:
            continue
        valid += 1
        for key, value in _metrics(y_test[ix], p[ix]).items():
            samples[key].append(value)
    intervals = {key: {"lower": float(np.quantile(v, .025)), "upper": float(np.quantile(v, .975))}
                 for key, v in samples.items() if len(v) >= 100}
    notes = []
    if bootstrap and valid < 100:
        notes.append("Fewer than 100 valid bootstrap replicates; percentile intervals are suppressed.")
    return {
        "metrics": metrics, "comparator": ({**_metrics(y_test, baseline), "probability": float(baseline[0])} if baseline is not None else None),
        "calibration": calibration,
        "roc_curve": [{"fpr": float(a), "tpr": float(b), "threshold": float(c) if np.isfinite(c) else None}
                      for a, b, c in zip(fpr, tpr, rt)],
        "pr_curve": [{"precision": float(a), "recall": float(b), "threshold": float(pt[i]) if i < len(pt) else None}
                     for i, (a, b) in enumerate(zip(precision, recall))],
        "thresholds": [_confusion(y_test, p, .5)], "decision_curve": decisions,
        "bootstrap": {"requested": int(bootstrap), "valid": valid, "skipped": int(bootstrap) - valid,
                      "method": "Ordinary fixed-prediction percentile bootstrap; 95% intervals",
                      "intervals": intervals},
    }, notes


def evaluate_predictions(data, outcome, event, probability, *, bootstrap=0, seed=42):
    """Evaluate supplied binary probabilities without fitting or recalibration.

    Data provenance determines whether this is internal or external validation;
    this function cannot establish independence of supplied predictions.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if data.columns.has_duplicates or outcome not in data or probability not in data:
        raise ValueError("Outcome/probability columns must exist and column names must be unique.")
    if outcome == probability:
        raise ValueError("Outcome and probability must be different columns.")
    if not 2 <= len(data) <= 5000:
        raise ValueError("Use 2–5000 rows.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1.")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or not 0 <= bootstrap <= 10000:
        raise ValueError("bootstrap must be an integer from 0 to 10000.")
    labels = data[outcome].reset_index(drop=True).map(_clean)
    ids = np.flatnonzero(labels.notna().to_numpy())
    levels = sorted(set(labels.iloc[ids]))
    if len(levels) != 2:
        raise ValueError("Binary evaluation requires exactly two nonmissing outcome categories.")
    event = str(event).strip()
    if event not in levels:
        raise ValueError("The event must exactly match one outcome category.")
    y = (labels.iloc[ids].to_numpy() == event).astype(int)
    p = pd.to_numeric(data[probability].iloc[ids], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("All retained probabilities must be finite numbers between 0 and 1; missing/invalid values are not imputed.")
    result, notes = _assessment(y, p, bootstrap, seed)
    notes.append("Supplied predictions are evaluated without fitting. Independence and validation design must be established from data provenance.")
    result.update({
        "config": {"mode": "fixed_predictions", "outcome": outcome, "event": event,
                   "probability": probability, "seed": int(seed)},
        "audit": {"n_input": len(data), "n_missing_outcome": len(data) - len(ids),
                  "n_evaluated": len(ids), "n_invalid_probability": 0, "warnings": notes},
        "predictions": [{"row": int(row), "observed": int(label), "probability": float(prob)}
                        for row, label, prob in zip(ids, y, p)]})
    return result
