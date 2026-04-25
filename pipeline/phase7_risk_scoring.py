"""
Phase 7 — Risk Scoring
========================
Composite bias risk score with configurable weights,
Pareto frontier analysis, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .config import PipelineConfig

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


def run_risk_scoring(
    phase_results: Dict[str, Any],
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Combine all phase results into a single composite risk score (0-1).

    Args:
        phase_results: Dictionary with keys matching phase names, containing
                       their respective results.
        config: Pipeline configuration
    """
    results = {}

    # --- 1. Extract component scores ---
    components = _extract_components(phase_results, config)
    results["components"] = components

    # --- 2. Compute composite score ---
    score, level = _compute_composite_score(components, config)
    results["composite_score"] = score
    results["risk_level"] = level

    # --- 3. Sensitivity analysis ---
    results["sensitivity"] = _sensitivity_analysis(components, config)

    # --- 4. Generate recommendations ---
    results["recommendations"] = _generate_recommendations(results, config)

    return results


def _extract_components(phase_results: Dict, config: PipelineConfig) -> Dict[str, float]:
    """Extract normalised (0-1) component scores from all phase results."""
    components = {}

    # Cramér's V — max across all sensitive attributes
    bias_tests = phase_results.get("bias_detection", {}).get("chi_square_tests", [])
    if bias_tests:
        max_v = max(t.get("cramers_v", 0) for t in bias_tests)
        components["cramers_v"] = min(max_v, 1.0)
    else:
        components["cramers_v"] = 0.0

    # Demographic parity gap — max across attributes
    dp = phase_results.get("fairness_metrics", {}).get("demographic_parity", {})
    if dp:
        max_gap = max(v.get("gap_percentage_points", 0) for v in dp.values()) / 100
        components["dp_gap"] = min(max_gap, 1.0)
    else:
        components["dp_gap"] = 0.0

    # Imbalance ratio — from data audit
    underrep = phase_results.get("data_audit", {}).get("underrepresented_groups", [])
    if underrep:
        # Normalise: more underrepresented groups = higher score
        components["imbalance_ratio"] = min(len(underrep) * 0.2, 1.0)
    else:
        components["imbalance_ratio"] = 0.0

    # Missing data gap — from data audit
    gaps = phase_results.get("data_audit", {}).get("missing_data_gaps", [])
    if gaps:
        max_gap = max(g.get("gap_percentage", 0) for g in gaps) / 100
        components["missing_gap"] = min(max_gap, 1.0)
    else:
        components["missing_gap"] = 0.0

    # Explainability disparity — from Phase 5
    expl = phase_results.get("explainability", {})
    components["explainability_disparity"] = expl.get("explainability_disparity_score", 0.0)

    # Label bias — from Phase 3
    label = phase_results.get("label_audit", {})
    components["label_bias"] = label.get("label_bias_score", 0.0)

    return components


def _compute_composite_score(
    components: Dict[str, float], config: PipelineConfig
) -> Tuple[float, str]:
    """Compute the weighted composite risk score."""
    weights = config.risk_scoring.weights
    thresholds = config.risk_scoring.thresholds

    score = sum(
        weights.get(key, 0) * value
        for key, value in components.items()
    )
    score = round(min(score, 1.0), 4)

    if score < thresholds["low"]:
        level = "LOW"
    elif score < thresholds["high"]:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return score, level


def _sensitivity_analysis(
    components: Dict[str, float], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Run sensitivity analysis: perturb weights ±20% and check
    if the risk classification changes. If it changes frequently,
    the result is "borderline" and needs manual review.
    """
    n_runs = config.risk_scoring.sensitivity_runs
    perturbation = config.risk_scoring.sensitivity_perturbation
    base_weights = config.risk_scoring.weights
    thresholds = config.risk_scoring.thresholds

    # Base classification
    base_score, base_level = _compute_composite_score(components, config)

    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    all_scores = []

    for _ in range(n_runs):
        # Perturb each weight by ±perturbation
        perturbed = {}
        for key, w in base_weights.items():
            noise = np.random.uniform(-perturbation, perturbation)
            perturbed[key] = max(0, w + noise * w)

        # Normalise weights to sum to 1
        total = sum(perturbed.values())
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}

        # Compute score with perturbed weights
        score = sum(perturbed.get(k, 0) * v for k, v in components.items())
        score = min(score, 1.0)
        all_scores.append(score)

        if score < thresholds["low"]:
            level_counts["LOW"] += 1
        elif score < thresholds["high"]:
            level_counts["MEDIUM"] += 1
        else:
            level_counts["HIGH"] += 1

    # Check stability
    dominant_level = max(level_counts, key=level_counts.get)
    stability = level_counts[dominant_level] / n_runs

    return {
        "base_score": base_score,
        "base_level": base_level,
        "score_range": [round(min(all_scores), 4), round(max(all_scores), 4)],
        "score_std": round(np.std(all_scores), 4),
        "level_distribution": level_counts,
        "stability": round(stability * 100, 1),
        "is_borderline": stability < 0.85,
        "recommendation": (
            f"⚠️ BORDERLINE — classification changes in {100-stability*100:.0f}% of weight "
            f"perturbations. Manual review recommended."
            if stability < 0.85
            else f"✅ STABLE — classification holds in {stability*100:.0f}% of perturbations."
        )
    }


def compute_pareto_frontier(
    model_class,
    X_train, y_train, X_test, y_test,
    sensitive_train, sensitive_test,
    config: PipelineConfig,
    n_models: int = None
) -> Dict[str, Any]:
    """
    Train multiple models with different fairness constraint strengths
    and plot the Pareto frontier (accuracy vs fairness).

    Returns the set of Pareto-optimal models.
    """
    if not FAIRLEARN_AVAILABLE:
        return {"status": "skipped", "reason": "Fairlearn not installed"}

    if n_models is None:
        n_models = config.mitigation.pareto_models

    from sklearn.metrics import accuracy_score

    # Generate constraint strengths from loose to tight
    epsilons = np.linspace(0.01, 0.5, n_models)
    model_results = []

    for eps in epsilons:
        try:
            constraint = DemographicParity(difference_bound=eps)
            mitigator = ExponentiatedGradient(model_class, constraints=constraint)
            mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

            y_pred = mitigator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Compute fairness metric
            rates = {}
            for g in sensitive_test.unique():
                mask = sensitive_test == g
                rates[str(g)] = y_pred[mask].mean()
            dp_gap = max(rates.values()) - min(rates.values()) if rates else 0

            model_results.append({
                "epsilon": round(eps, 4),
                "accuracy": round(acc, 4),
                "dp_gap": round(dp_gap, 4),
                "model": mitigator,
            })
        except Exception as e:
            continue

    # Find Pareto frontier
    pareto = _find_pareto_optimal(model_results)

    return {
        "all_models": [
            {"epsilon": m["epsilon"], "accuracy": m["accuracy"], "dp_gap": m["dp_gap"]}
            for m in model_results
        ],
        "pareto_optimal": [
            {"epsilon": m["epsilon"], "accuracy": m["accuracy"], "dp_gap": m["dp_gap"]}
            for m in pareto
        ],
        "best_balanced": _select_best_balanced(pareto) if pareto else None,
        "n_models_trained": len(model_results),
    }


def _find_pareto_optimal(models: List[Dict]) -> List[Dict]:
    """Find Pareto-optimal models: maximise accuracy, minimise dp_gap."""
    pareto = []
    for m in models:
        dominated = False
        for other in models:
            if (other["accuracy"] >= m["accuracy"] and
                other["dp_gap"] <= m["dp_gap"] and
                (other["accuracy"] > m["accuracy"] or other["dp_gap"] < m["dp_gap"])):
                dominated = True
                break
        if not dominated:
            pareto.append(m)
    return pareto


def _select_best_balanced(pareto: List[Dict]) -> Dict:
    """Select the model with the best balance of accuracy and fairness."""
    if not pareto:
        return None
    # Minimise the combined normalised distance
    accs = [m["accuracy"] for m in pareto]
    gaps = [m["dp_gap"] for m in pareto]
    max_acc, min_acc = max(accs), min(accs)
    max_gap, min_gap = max(gaps), min(gaps)

    best = None
    best_dist = float("inf")
    for m in pareto:
        norm_acc = (m["accuracy"] - min_acc) / (max_acc - min_acc + 1e-10)
        norm_gap = (m["dp_gap"] - min_gap) / (max_gap - min_gap + 1e-10)
        dist = np.sqrt((1 - norm_acc) ** 2 + norm_gap ** 2)
        if dist < best_dist:
            best_dist = dist
            best = m

    return {
        "epsilon": best["epsilon"],
        "accuracy": best["accuracy"],
        "dp_gap": best["dp_gap"]
    } if best else None


def _generate_recommendations(results: Dict, config: PipelineConfig) -> List[str]:
    """Generate prioritised plain-language recommendations."""
    recs = []
    level = results["risk_level"]
    score = results["composite_score"]
    components = results["components"]

    if level == "HIGH":
        recs.append("🔴 DO NOT DEPLOY this model. Significant bias has been detected.")
    elif level == "MEDIUM":
        recs.append("🟡 REVIEW REQUIRED before deployment. Moderate bias signals found.")
    else:
        recs.append("🟢 Model passes basic fairness checks. Continue monitoring after deployment.")

    # Component-specific recommendations
    if components.get("cramers_v", 0) > 0.3:
        recs.append(
            f"Statistical bias (Cramér's V = {components['cramers_v']:.2f}) is strong. "
            "Apply in-processing mitigation (ExponentiatedGradient)."
        )
    if components.get("dp_gap", 0) > 0.1:
        recs.append(
            f"Demographic parity gap is {components['dp_gap']*100:.0f}%. "
            "Consider ThresholdOptimizer for post-processing correction."
        )
    if components.get("label_bias", 0) > 0.3:
        recs.append(
            "Label bias detected. Consider expert relabelling before retraining."
        )
    if components.get("imbalance_ratio", 0) > 0.2:
        recs.append(
            "Underrepresented groups detected. Apply SMOTE oversampling to training data."
        )
    if results.get("sensitivity", {}).get("is_borderline"):
        recs.append(
            "⚠️ Risk classification is borderline. Schedule manual fairness review."
        )

    return recs
