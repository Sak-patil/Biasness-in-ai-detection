"""
Phase 3 — Label Audit (NEW — Critical Addition)
=================================================
Verifies that the ground truth labels themselves are not biased.
If historical labels encode discrimination, every downstream metric
is computed against a corrupted reference.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from .config import PipelineConfig


def run_label_audit(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Audit the target labels for embedded historical bias.

    Returns:
      - label_distribution: approval rates per group
      - label_disparity: gap between highest and lowest group approval rates
      - counterfactual_signals: features where flipping group changes outcome
      - label_bias_score: 0-1 composite score
      - recommendation: what to do if labels are biased
    """
    results = {}
    target = config.target_column

    if target not in df.columns:
        return {"error": f"Target column '{target}' not found in data."}

    # --- 1. Label Distribution per Group ---
    results["label_distribution"] = _label_distribution(df, config)

    # --- 2. Label Disparity ---
    results["label_disparity"] = _label_disparity(results["label_distribution"])

    # --- 3. Statistical Test: Is label assignment independent of group? ---
    results["independence_tests"] = _label_independence_tests(df, config)

    # --- 4. Counterfactual Signal Detection ---
    results["counterfactual_signals"] = _counterfactual_check(df, config)

    # --- 5. Composite Label Bias Score ---
    results["label_bias_score"] = _compute_label_bias_score(results)

    # --- 6. Recommendation ---
    score = results["label_bias_score"]
    if score > 0.6:
        results["recommendation"] = (
            "🔴 HIGH label bias detected. Labels likely encode historical discrimination. "
            "Recommend: (a) Expert relabelling of borderline cases, "
            "(b) Using real-world outcomes instead of historical decisions as labels, or "
            "(c) Applying label-bias adjustment weights before proceeding."
        )
        results["should_halt"] = True
    elif score > 0.3:
        results["recommendation"] = (
            "🟡 MODERATE label bias detected. Some groups have notably different label rates. "
            "Recommend: Manual review of a stratified sample of borderline decisions."
        )
        results["should_halt"] = False
    else:
        results["recommendation"] = (
            "🟢 LOW label bias. Label distribution appears reasonably balanced across groups."
        )
        results["should_halt"] = False

    return results


def _label_distribution(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """Compute positive label rates per group for each sensitive attribute."""
    target = config.target_column
    distribution = {}

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue
        grouped = df.groupby(attr)[target].agg(["mean", "count", "sum"])
        grouped.columns = ["approval_rate", "total_count", "approved_count"]
        grouped["rejected_count"] = grouped["total_count"] - grouped["approved_count"]
        grouped["approval_rate"] = (grouped["approval_rate"] * 100).round(2)
        distribution[attr] = grouped

    return distribution


def _label_disparity(label_dist: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Find the maximum gap in approval rates between groups."""
    disparities = {}
    for attr, dist_df in label_dist.items():
        rates = dist_df["approval_rate"]
        disparities[attr] = {
            "max_rate": round(rates.max(), 2),
            "max_group": rates.idxmax(),
            "min_rate": round(rates.min(), 2),
            "min_group": rates.idxmin(),
            "gap": round(rates.max() - rates.min(), 2),
            "ratio": round(rates.min() / rates.max(), 4) if rates.max() > 0 else 0
        }
    return disparities


def _label_independence_tests(df: pd.DataFrame, config: PipelineConfig) -> List[Dict]:
    """Test if label assignment is statistically independent of sensitive attributes."""
    target = config.target_column
    results = []

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        ct = pd.crosstab(df[attr], df[target])

        # Choose test based on sample size
        min_expected = stats.chi2_contingency(ct)[3].min()
        if min_expected < config.bias_detection.fisher_cell_cutoff and ct.shape == (2, 2):
            # Fisher's exact test for small samples
            odds_ratio, p_value = stats.fisher_exact(ct)
            test_used = "Fisher's exact test (small sample)"
        else:
            chi2, p_value, dof, expected = stats.chi2_contingency(ct)
            test_used = "Chi-square test"

        results.append({
            "attribute": attr,
            "test": test_used,
            "p_value": round(p_value, 6),
            "significant": p_value < config.bias_detection.chi_square_alpha,
            "interpretation": (
                f"Label assignment IS statistically dependent on {attr} (p={p_value:.4f})"
                if p_value < config.bias_detection.chi_square_alpha
                else f"No significant relationship between labels and {attr}"
            )
        })

    return results


def _counterfactual_check(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Simplified counterfactual analysis:
    For each sensitive attribute, compare records that are identical on all
    non-sensitive features but differ on the sensitive attribute.
    If outcomes differ systematically, labels encode attribute-based bias.
    """
    target = config.target_column
    signals = {}

    # Use non-sensitive features as matching keys
    non_sensitive = [c for c in df.columns
                     if c not in config.sensitive_attributes and c != target]

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        groups = df[attr].dropna().unique()
        if len(groups) != 2:
            signals[attr] = {
                "status": "skipped",
                "reason": f"Counterfactual check requires exactly 2 groups, found {len(groups)}"
            }
            continue

        g1, g2 = groups[0], groups[1]
        df1 = df[df[attr] == g1]
        df2 = df[df[attr] == g2]

        # Compare average outcome for each group
        rate1 = df1[target].mean()
        rate2 = df2[target].mean()
        gap = abs(rate1 - rate2)

        signals[attr] = {
            "group_1": str(g1),
            "group_1_rate": round(rate1 * 100, 2),
            "group_2": str(g2),
            "group_2_rate": round(rate2 * 100, 2),
            "gap_percentage": round(gap * 100, 2),
            "potential_label_bias": gap > 0.15,
            "interpretation": (
                f"If you changed someone's {attr} from {g1} to {g2}, their average label "
                f"would shift by {gap*100:.1f} percentage points. "
                + ("This suggests labels encode bias." if gap > 0.15
                   else "This is within acceptable range.")
            )
        }

    return signals


def _compute_label_bias_score(results: Dict) -> float:
    """
    Compute a 0-1 composite label bias score.
    Higher = more biased labels.
    """
    scores = []

    # Contribution from label disparity
    for attr, disp in results.get("label_disparity", {}).items():
        gap_normalized = min(disp["gap"] / 50.0, 1.0)  # 50% gap = max score
        scores.append(gap_normalized)

    # Contribution from independence tests
    for test in results.get("independence_tests", []):
        if test["significant"]:
            scores.append(0.8)  # Significant dependence = high bias signal
        else:
            scores.append(0.1)

    # Contribution from counterfactual signals
    for attr, signal in results.get("counterfactual_signals", {}).items():
        if isinstance(signal, dict) and signal.get("potential_label_bias"):
            scores.append(0.9)
        elif isinstance(signal, dict) and "gap_percentage" in signal:
            scores.append(min(signal["gap_percentage"] / 30.0, 1.0))

    return round(np.mean(scores) if scores else 0.0, 4)
