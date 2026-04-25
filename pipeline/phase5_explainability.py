"""
Phase 5 — Explainability (NEW)
===============================
Uses SHAP to explain WHY bias exists, not just THAT it exists.
Produces per-group feature importance and root cause reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .config import PipelineConfig

# SHAP is optional — graceful fallback if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def run_explainability(
    df: pd.DataFrame,
    model: Any,
    config: PipelineConfig,
    feature_columns: list = None
) -> Dict[str, Any]:
    """
    Explain model predictions per group using SHAP values.

    Args:
        df: DataFrame with features and sensitive attributes
        model: Trained sklearn-compatible model
        config: Pipeline configuration
        feature_columns: List of feature column names (if None, auto-detect)

    Returns:
        Dictionary with per-group SHAP analysis and root cause report
    """
    if not SHAP_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "SHAP not installed. Run: pip install shap",
            "shap_available": False
        }

    results = {}
    target = config.target_column

    if feature_columns is None:
        feature_columns = [c for c in df.columns
                          if c not in config.sensitive_attributes and c != target]

    X = df[feature_columns]

    # --- 1. Compute SHAP values ---
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        results["shap_computed"] = True
    except Exception as e:
        # Fallback to TreeExplainer or KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            results["shap_computed"] = True
            results["explainer_type"] = "TreeExplainer"
        except Exception:
            try:
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X, nsamples=100)
                results["shap_computed"] = True
                results["explainer_type"] = "KernelExplainer"
            except Exception as e2:
                return {
                    "status": "error",
                    "error": str(e2),
                    "shap_computed": False
                }

    # --- 2. Per-group SHAP analysis ---
    results["per_group_shap"] = _per_group_shap(df, shap_values, feature_columns, config)

    # --- 3. SHAP divergence between groups ---
    results["shap_divergence"] = _shap_divergence(df, shap_values, feature_columns, config)

    # --- 4. Root cause report ---
    results["root_cause_report"] = _generate_root_cause_report(results, config)

    # --- 5. Explainability Disparity Score ---
    results["explainability_disparity_score"] = _compute_disparity_score(results)

    return results


def _per_group_shap(
    df: pd.DataFrame,
    shap_values: Any,
    feature_columns: list,
    config: PipelineConfig
) -> Dict[str, Dict]:
    """Compute average SHAP values per group for each sensitive attribute."""
    per_group = {}

    sv = _extract_shap_array(shap_values)
    shap_df = pd.DataFrame(sv, columns=feature_columns, index=df.index)

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        groups = df[attr].dropna().unique()
        group_means = {}

        for group_val in groups:
            mask = df[attr] == group_val
            group_shap = shap_df.loc[mask].mean()
            group_means[str(group_val)] = {
                col: round(val, 4) for col, val in group_shap.items()
            }

        per_group[attr] = group_means

    return per_group


def _extract_shap_array(shap_values: Any) -> np.ndarray:
    """
    Extract a 2D numpy array from various SHAP value formats.
    Handles Explanation objects, lists (old API), and 3D arrays
    from classifiers (picks positive class index 1).
    """
    if hasattr(shap_values, 'values'):
        sv = shap_values.values
    elif isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    # Handle 3D arrays: (n_samples, n_features, n_classes) → pick positive class
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]  # Positive class SHAP values

    return sv


def _shap_divergence(
    df: pd.DataFrame,
    shap_values: Any,
    feature_columns: list,
    config: PipelineConfig
) -> Dict[str, list]:
    """
    Find features where SHAP values diverge most between groups.
    These are the root causes of bias.
    """
    sv = _extract_shap_array(shap_values)
    shap_df = pd.DataFrame(sv, columns=feature_columns, index=df.index)
    divergence = {}

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        groups = df[attr].dropna().unique()
        if len(groups) < 2:
            continue

        # Compute mean SHAP per group
        group_means = {}
        for g in groups:
            mask = df[attr] == g
            group_means[str(g)] = shap_df.loc[mask].mean()

        # Compute max divergence per feature across all group pairs
        feature_divergences = []
        group_list = list(group_means.keys())

        for col in feature_columns:
            values = [group_means[g][col] for g in group_list]
            max_div = max(values) - min(values)
            feature_divergences.append({
                "feature": col,
                "divergence": round(abs(max_div), 4),
                "helps_most": group_list[np.argmax(values)],
                "hurts_most": group_list[np.argmin(values)],
            })

        # Sort by divergence (biggest gaps first)
        feature_divergences.sort(key=lambda x: x["divergence"], reverse=True)
        divergence[attr] = feature_divergences[:10]  # Top 10 divergent features

    return divergence


def _generate_root_cause_report(results: Dict, config: PipelineConfig) -> str:
    """
    Generate a plain-language root cause report.
    Example: "The model is biased primarily because Feature X (zip code)
    encodes neighbourhood demographics correlated with caste..."
    """
    report_lines = ["## Root Cause Analysis\n"]

    divergence = results.get("shap_divergence", {})

    for attr, features in divergence.items():
        if not features:
            continue

        report_lines.append(f"### Bias drivers for {attr}:\n")

        for i, feat in enumerate(features[:5], 1):
            report_lines.append(
                f"{i}. **{feat['feature']}** — This feature helps "
                f"'{feat['helps_most']}' (SHAP +{feat['divergence']:.3f}) but "
                f"hurts '{feat['hurts_most']}'. "
                f"Divergence: {feat['divergence']:.4f}"
            )

        top_feats = [f["feature"] for f in features[:3]]
        report_lines.append(
            f"\n**Recommendation:** Investigate and potentially remove or decorrelate "
            f"{', '.join(top_feats)} to reduce {attr}-based bias.\n"
        )

    return "\n".join(report_lines) if len(report_lines) > 1 else "No significant bias drivers found."


def _compute_disparity_score(results: Dict) -> float:
    """
    Compute a 0-1 score measuring how differently SHAP values
    distribute across groups. Used in the composite risk score.
    """
    all_divergences = []
    for attr, features in results.get("shap_divergence", {}).items():
        for feat in features:
            all_divergences.append(feat["divergence"])

    if not all_divergences:
        return 0.0

    # Normalise: average of top-5 divergences, capped at 1
    top5 = sorted(all_divergences, reverse=True)[:5]
    return round(min(np.mean(top5), 1.0), 4)
