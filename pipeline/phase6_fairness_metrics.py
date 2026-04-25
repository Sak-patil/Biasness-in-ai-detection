"""
Phase 6 — Model-level Fairness Metrics
========================================
Demographic parity, equalized odds, disparate impact ratio,
calibration per group, and individual fairness.
"""

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from typing import Dict, List, Any
from .config import PipelineConfig

# Fairlearn is optional — graceful fallback
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


def run_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
    sensitive_features: pd.DataFrame,
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Compute all model-level fairness metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        sensitive_features: DataFrame with sensitive attribute columns
        config: Pipeline configuration
    """
    results = {}

    # --- 1. Demographic Parity ---
    results["demographic_parity"] = _demographic_parity(
        y_true, y_pred, sensitive_features, config
    )

    # --- 2. Equalized Odds ---
    results["equalized_odds"] = _equalized_odds(
        y_true, y_pred, sensitive_features, config
    )

    # --- 3. Disparate Impact Ratio ---
    results["disparate_impact"] = _disparate_impact(
        y_pred, sensitive_features, config
    )

    # --- 4. Calibration per Group ---
    results["calibration"] = _calibration_per_group(
        y_true, y_prob, sensitive_features, config
    )

    # --- 5. Individual Fairness ---
    results["individual_fairness"] = _individual_fairness_check(
        y_pred, y_prob, sensitive_features, config
    )

    # --- Summary ---
    results["summary"] = _build_summary(results, config)

    return results


def _demographic_parity(
    y_true, y_pred, sensitive_features, config
) -> Dict[str, Any]:
    """
    Compute demographic parity: does the model approve at similar rates
    across all groups?
    """
    dp_results = {}

    for attr in config.sensitive_attributes:
        if attr not in sensitive_features.columns:
            continue

        groups = sensitive_features[attr].dropna().unique()
        rates = {}
        for g in groups:
            mask = sensitive_features[attr] == g
            rates[str(g)] = round(y_pred[mask].mean() * 100, 2)

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        gap = round(max_rate - min_rate, 2)

        # Use Fairlearn if available
        dp_diff = None
        if FAIRLEARN_AVAILABLE:
            try:
                dp_diff = round(demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features[attr]
                ), 4)
            except Exception:
                pass

        dp_results[attr] = {
            "approval_rates": rates,
            "gap_percentage_points": gap,
            "fairlearn_dp_difference": dp_diff,
            "above_threshold": gap / 100 > config.fairness_metrics.demographic_parity_gap_threshold,
            "interpretation": (
                f"The model approves {max(rates, key=rates.get)} at {max_rate}% "
                f"but {min(rates, key=rates.get)} at only {min_rate}%. "
                f"That is a {gap}-point gap"
                + (f" — it should be within {config.fairness_metrics.demographic_parity_gap_threshold*100:.0f} points. 🔴"
                   if gap / 100 > config.fairness_metrics.demographic_parity_gap_threshold
                   else " — within acceptable range. 🟢")
            )
        }

    return dp_results


def _equalized_odds(y_true, y_pred, sensitive_features, config) -> Dict[str, Any]:
    """
    Equalized odds: separate the error types.
    Checks FPR and FNR per group.
    """
    eo_results = {}

    for attr in config.sensitive_attributes:
        if attr not in sensitive_features.columns:
            continue

        groups = sensitive_features[attr].dropna().unique()
        group_metrics = {}

        for g in groups:
            mask = sensitive_features[attr] == g
            yt = y_true[mask]
            yp = y_pred[mask]

            tp = ((yt == 1) & (yp == 1)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            tn = ((yt == 0) & (yp == 0)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            group_metrics[str(g)] = {
                "true_positive_rate": round(tpr, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            }

        # Compute gaps
        tprs = [m["true_positive_rate"] for m in group_metrics.values()]
        fprs = [m["false_positive_rate"] for m in group_metrics.values()]

        eo_results[attr] = {
            "per_group": group_metrics,
            "tpr_gap": round(max(tprs) - min(tprs), 4),
            "fpr_gap": round(max(fprs) - min(fprs), 4),
            "above_threshold": (
                max(tprs) - min(tprs) > config.fairness_metrics.equalized_odds_threshold or
                max(fprs) - min(fprs) > config.fairness_metrics.equalized_odds_threshold
            )
        }

    return eo_results


def _disparate_impact(y_pred, sensitive_features, config) -> Dict[str, Any]:
    """
    Disparate impact ratio: minority rate / majority rate.
    Below 0.80 is legally presumptive discrimination (4/5ths rule).
    """
    di_results = {}

    for attr in config.sensitive_attributes:
        if attr not in sensitive_features.columns:
            continue

        groups = sensitive_features[attr].dropna().unique()
        rates = {}
        for g in groups:
            mask = sensitive_features[attr] == g
            rates[str(g)] = y_pred[mask].mean()

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 0

        di_results[attr] = {
            "approval_rates": {k: round(v * 100, 2) for k, v in rates.items()},
            "disparate_impact_ratio": di_ratio,
            "legal_threshold": config.fairness_metrics.disparate_impact_threshold,
            "passes_legal_test": di_ratio >= config.fairness_metrics.disparate_impact_threshold,
            "interpretation": (
                f"DI ratio = {di_ratio:.3f}. "
                + (f"Below {config.fairness_metrics.disparate_impact_threshold} — "
                   f"legally presumptive discrimination under 4/5ths rule. 🔴"
                   if di_ratio < config.fairness_metrics.disparate_impact_threshold
                   else "Passes the legal standard. 🟢")
            )
        }

    return di_results


def _calibration_per_group(y_true, y_prob, sensitive_features, config) -> Dict[str, Any]:
    """
    Check if the model's confidence scores are trustworthy per group.
    """
    cal_results = {}

    for attr in config.sensitive_attributes:
        if attr not in sensitive_features.columns:
            continue

        groups = sensitive_features[attr].dropna().unique()
        group_cal = {}

        for g in groups:
            mask = sensitive_features[attr] == g
            yt = y_true[mask]
            yp = y_prob[mask]

            if len(yt) < 20:
                group_cal[str(g)] = {"status": "too few samples for calibration"}
                continue

            try:
                prob_true, prob_pred = calibration_curve(yt, yp, n_bins=5)
                cal_error = np.mean(np.abs(prob_true - prob_pred))
                group_cal[str(g)] = {
                    "mean_calibration_error": round(cal_error, 4),
                    "well_calibrated": cal_error < 0.1,
                    "prob_true": [round(p, 4) for p in prob_true],
                    "prob_pred": [round(p, 4) for p in prob_pred],
                }
            except Exception:
                group_cal[str(g)] = {"status": "calibration computation failed"}

        cal_results[attr] = group_cal

    return cal_results


def _individual_fairness_check(y_pred, y_prob, sensitive_features, config) -> Dict[str, Any]:
    """
    Simplified individual fairness: compare outcomes for records that
    differ only on sensitive attributes.
    """
    if_results = {}

    for attr in config.sensitive_attributes:
        if attr not in sensitive_features.columns:
            continue

        groups = sensitive_features[attr].dropna().unique()
        if len(groups) != 2:
            if_results[attr] = {"status": "skipped — requires exactly 2 groups"}
            continue

        g1_mask = sensitive_features[attr] == groups[0]
        g2_mask = sensitive_features[attr] == groups[1]

        # Compare probability distributions
        g1_probs = y_prob[g1_mask]
        g2_probs = y_prob[g2_mask]

        # Wasserstein distance between probability distributions
        from scipy.stats import wasserstein_distance
        w_dist = wasserstein_distance(g1_probs, g2_probs)

        if_results[attr] = {
            "wasserstein_distance": round(w_dist, 4),
            "interpretation": (
                f"Probability distribution gap between groups: {w_dist:.4f}. "
                + ("Significant individual-level fairness concern. 🔴" if w_dist > 0.1
                   else "Acceptable individual fairness. 🟢")
            )
        }

    return if_results


def _build_summary(results: Dict, config: PipelineConfig) -> Dict:
    """Build summary of all fairness metrics."""
    dp_flags = sum(1 for v in results["demographic_parity"].values() if v.get("above_threshold"))
    eo_flags = sum(1 for v in results["equalized_odds"].values() if v.get("above_threshold"))
    di_flags = sum(1 for v in results["disparate_impact"].values() if not v.get("passes_legal_test"))

    return {
        "demographic_parity_violations": dp_flags,
        "equalized_odds_violations": eo_flags,
        "disparate_impact_violations": di_flags,
        "total_violations": dp_flags + eo_flags + di_flags,
        "status": (
            "🔴 FAIRNESS VIOLATIONS DETECTED"
            if (dp_flags + eo_flags + di_flags) > 0
            else "🟢 ALL METRICS PASS"
        )
    }
