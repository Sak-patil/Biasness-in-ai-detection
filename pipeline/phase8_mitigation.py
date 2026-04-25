"""
Phase 8 — Mitigation
======================
Severity-based bias mitigation: SMOTE, reweighting, ThresholdOptimizer,
ExponentiatedGradient — with automated selection and post-mitigation validation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple, Optional
from .config import PipelineConfig

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


def run_mitigation(
    risk_level: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_train: pd.Series,
    sensitive_test: pd.Series,
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Automatically select and apply mitigation based on risk level.

    LOW    → SMOTE oversampling (pre-processing)
    MEDIUM → Sample reweighting (pre-processing)
    HIGH   → ExponentiatedGradient (in-processing) + ThresholdOptimizer (post-processing)
    """
    results = {
        "risk_level": risk_level,
        "techniques_applied": [],
        "original_accuracy": None,
        "mitigated_accuracy": None,
        "accuracy_change": None,
    }

    # Baseline accuracy
    y_pred_original = model.predict(X_test)
    original_acc = accuracy_score(y_test, y_pred_original)
    results["original_accuracy"] = round(original_acc, 4)

    if risk_level == "LOW":
        results = _apply_smote(
            model, X_train, y_train, X_test, y_test,
            sensitive_train, config, results
        )

    elif risk_level == "MEDIUM":
        results = _apply_reweighting(
            model, X_train, y_train, X_test, y_test,
            sensitive_train, config, results
        )

    elif risk_level == "HIGH":
        # Try ExponentiatedGradient first (strongest fix)
        results = _apply_exponentiated_gradient(
            model, X_train, y_train, X_test, y_test,
            sensitive_train, sensitive_test, config, results
        )

        # Also apply ThresholdOptimizer as a fallback/complement
        results = _apply_threshold_optimizer(
            model, X_train, y_train, X_test, y_test,
            sensitive_train, sensitive_test, config, results
        )

    # Record accuracy change
    if results.get("mitigated_accuracy") is not None:
        results["accuracy_change"] = round(
            results["mitigated_accuracy"] - original_acc, 4
        )
        results["accuracy_within_tolerance"] = (
            abs(results["accuracy_change"]) <= config.mitigation.accuracy_loss_tolerance
        )

    return results


def _apply_smote(
    model, X_train, y_train, X_test, y_test,
    sensitive_train, config, results
) -> Dict:
    """Apply SMOTE oversampling to increase minority group representation."""
    if not SMOTE_AVAILABLE:
        results["techniques_applied"].append({
            "name": "SMOTE", "status": "skipped",
            "reason": "imbalanced-learn not installed"
        })
        return results

    try:
        smote = SMOTE(k_neighbors=config.mitigation.smote_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Retrain model on resampled data
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_resampled, y_resampled)

        y_pred = model_copy.predict(X_test)
        new_acc = accuracy_score(y_test, y_pred)

        results["techniques_applied"].append({
            "name": "SMOTE Oversampling",
            "status": "applied",
            "original_samples": len(X_train),
            "resampled_samples": len(X_resampled),
            "new_accuracy": round(new_acc, 4),
        })
        results["mitigated_accuracy"] = round(new_acc, 4)
        results["mitigated_model"] = model_copy

    except Exception as e:
        results["techniques_applied"].append({
            "name": "SMOTE", "status": "error", "error": str(e)
        })

    return results


def _apply_reweighting(
    model, X_train, y_train, X_test, y_test,
    sensitive_train, config, results
) -> Dict:
    """Apply sample reweighting to penalise errors on underrepresented groups."""
    try:
        # Compute weights inversely proportional to group frequency
        group_counts = sensitive_train.value_counts()
        total = len(sensitive_train)
        n_groups = len(group_counts)

        weights = sensitive_train.map(
            lambda g: total / (n_groups * group_counts.get(g, 1))
        )

        # Retrain with sample weights
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train, sample_weight=weights)

        y_pred = model_copy.predict(X_test)
        new_acc = accuracy_score(y_test, y_pred)

        results["techniques_applied"].append({
            "name": "Sample Reweighting",
            "status": "applied",
            "group_weights": {str(k): round(total / (n_groups * v), 4)
                             for k, v in group_counts.items()},
            "new_accuracy": round(new_acc, 4),
        })
        results["mitigated_accuracy"] = round(new_acc, 4)
        results["mitigated_model"] = model_copy

    except Exception as e:
        results["techniques_applied"].append({
            "name": "Sample Reweighting", "status": "error", "error": str(e)
        })

    return results


def _apply_exponentiated_gradient(
    model, X_train, y_train, X_test, y_test,
    sensitive_train, sensitive_test, config, results
) -> Dict:
    """
    ExponentiatedGradient: rewrites the training objective to minimise
    prediction error SUBJECT TO a hard fairness constraint.
    Most powerful but most expensive.
    """
    if not FAIRLEARN_AVAILABLE:
        results["techniques_applied"].append({
            "name": "ExponentiatedGradient", "status": "skipped",
            "reason": "Fairlearn not installed"
        })
        return results

    try:
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(
            estimator=model.__class__(**model.get_params()),
            constraints=constraint,
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

        y_pred = mitigator.predict(X_test)
        new_acc = accuracy_score(y_test, y_pred)

        # Check fairness improvement
        rates = {}
        for g in sensitive_test.unique():
            mask = sensitive_test == g
            rates[str(g)] = y_pred[mask].mean()

        dp_gap = max(rates.values()) - min(rates.values()) if rates else 0

        results["techniques_applied"].append({
            "name": "ExponentiatedGradient (in-processing)",
            "status": "applied",
            "constraint": "DemographicParity",
            "new_accuracy": round(new_acc, 4),
            "new_dp_gap": round(dp_gap, 4),
            "approval_rates": {k: round(v * 100, 2) for k, v in rates.items()},
        })
        results["mitigated_accuracy"] = round(new_acc, 4)
        results["mitigated_model"] = mitigator

    except Exception as e:
        results["techniques_applied"].append({
            "name": "ExponentiatedGradient", "status": "error", "error": str(e)
        })

    return results


def _apply_threshold_optimizer(
    model, X_train, y_train, X_test, y_test,
    sensitive_train, sensitive_test, config, results
) -> Dict:
    """
    ThresholdOptimizer: adjusts per-group decision thresholds AFTER training
    to equalise error rates with minimum accuracy loss.
    """
    if not FAIRLEARN_AVAILABLE:
        results["techniques_applied"].append({
            "name": "ThresholdOptimizer", "status": "skipped",
            "reason": "Fairlearn not installed"
        })
        return results

    try:
        postprocess = ThresholdOptimizer(
            estimator=model,
            constraints="demographic_parity",
            prefit=True,
        )
        postprocess.fit(X_train, y_train, sensitive_features=sensitive_train)

        y_pred = postprocess.predict(X_test, sensitive_features=sensitive_test)
        new_acc = accuracy_score(y_test, y_pred)

        results["techniques_applied"].append({
            "name": "ThresholdOptimizer (post-processing)",
            "status": "applied",
            "constraint": "demographic_parity",
            "new_accuracy": round(new_acc, 4),
        })

        # Only update if this is better than ExponentiatedGradient
        if results.get("mitigated_accuracy") is None or new_acc > results["mitigated_accuracy"]:
            results["mitigated_accuracy"] = round(new_acc, 4)

    except Exception as e:
        results["techniques_applied"].append({
            "name": "ThresholdOptimizer", "status": "error", "error": str(e)
        })

    return results
