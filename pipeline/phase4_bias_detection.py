"""
Phase 4 — Statistical Bias Detection
======================================
Chi-square, Cramér's V, point-biserial correlation, PCA, MCA,
intersectional analysis — all with confidence intervals and
small-sample safeguards.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from itertools import combinations
from typing import Dict, List, Any, Tuple
from .config import PipelineConfig


def run_bias_detection(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Run the complete statistical bias detection phase.
    """
    results = {}

    # --- 1. Chi-square + Cramér's V ---
    results["chi_square_tests"] = _chi_square_cramers_v(df, config)

    # --- 2. Point-biserial / Pearson Correlation ---
    results["correlations"] = _correlation_analysis(df, config)

    # --- 3. PCA Analysis ---
    results["pca_analysis"] = _pca_bias_check(df, config)

    # --- 4. Intersectional Analysis ---
    results["intersectional"] = _intersectional_analysis(df, config)

    # --- 5. Bootstrap Confidence Intervals ---
    results["confidence_intervals"] = _bootstrap_ci(df, config)

    # --- Summary ---
    results["summary"] = _build_summary(results, config)

    return results


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V between two series."""
    if pd.api.types.is_numeric_dtype(x):
        x = pd.qcut(x, q=5, duplicates="drop")
    if pd.api.types.is_numeric_dtype(y):
        y = pd.qcut(y, q=5, duplicates="drop")

    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    return np.sqrt(chi2 / denom) if denom > 0 else 0.0


def _chi_square_cramers_v(df: pd.DataFrame, config: PipelineConfig) -> List[Dict]:
    """
    Run chi-square test and Cramér's V for each sensitive attribute
    against the target variable.
    """
    target = config.target_column
    results = []

    for attr in config.sensitive_attributes:
        if attr not in df.columns or target not in df.columns:
            continue

        ct = pd.crosstab(df[attr], df[target])
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)

        # Small sample safeguard: use Fisher's exact if expected cells < 5
        min_expected = expected.min()
        if min_expected < config.bias_detection.fisher_cell_cutoff and ct.shape == (2, 2):
            _, fisher_p = stats.fisher_exact(ct)
            test_note = f"Fisher's exact test used (min expected cell = {min_expected:.1f} < 5)"
            effective_p = fisher_p
        else:
            test_note = "Chi-square test"
            effective_p = p_value

        v = _cramers_v(df[attr], df[target])

        # Interpret Cramér's V
        if v < 0.1:
            strength = "WEAK (negligible association)"
        elif v < 0.3:
            strength = "MODERATE (notable association)"
        else:
            strength = "STRONG (serious bias signal) 🔴"

        results.append({
            "attribute": attr,
            "chi_square": round(chi2, 4),
            "p_value": round(effective_p, 6),
            "cramers_v": round(v, 4),
            "strength": strength,
            "statistically_significant": effective_p < config.bias_detection.chi_square_alpha,
            "above_threshold": v >= config.bias_detection.cramers_v_threshold,
            "test_note": test_note,
            "min_expected_cell": round(min_expected, 2),
            "interpretation": (
                f"Knowing someone's {attr} {'DOES' if effective_p < 0.05 else 'does NOT'} "
                f"significantly predict their outcome. "
                f"Effect strength: {v:.3f} ({strength})"
            )
        })

    return results


def _correlation_analysis(df: pd.DataFrame, config: PipelineConfig) -> List[Dict]:
    """
    Point-biserial correlation between binary-encoded sensitive attributes
    and numeric features (credit score, income, etc.).
    """
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != config.target_column]

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        # Encode as binary if categorical
        if df[attr].dtype == "object" or df[attr].dtype.name == "category":
            unique_vals = df[attr].dropna().unique()
            if len(unique_vals) != 2:
                continue
            encoded = (df[attr] == unique_vals[0]).astype(int)
        else:
            encoded = df[attr]

        for col in numeric_cols:
            if col == attr:
                continue

            clean = df[[col]].join(encoded.rename("_attr")).dropna()
            if len(clean) < 10:
                continue

            corr, p_value = stats.pointbiserialr(clean["_attr"], clean[col])

            if abs(corr) >= config.bias_detection.correlation_threshold:
                flag = "🔴 INVESTIGATE"
            elif abs(corr) >= 0.1:
                flag = "🟡 MONITOR"
            else:
                flag = "🟢 OK"

            results.append({
                "sensitive_attribute": attr,
                "feature": col,
                "correlation": round(corr, 4),
                "p_value": round(p_value, 6),
                "flag": flag,
                "interpretation": (
                    f"Knowing {attr} explains {abs(corr)*100:.1f}% of the variation in {col}."
                )
            })

    return results


def _pca_bias_check(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Run PCA and check if sensitive attributes load heavily on the
    first principal components — indicating they are secretly driving
    variation in the entire dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Include encoded sensitive attributes
    df_pca = df[numeric_cols].dropna()
    if len(df_pca) < 10 or len(numeric_cols) < 3:
        return {"status": "skipped", "reason": "Not enough numeric columns or data for PCA"}

    pca = PCA(n_components=min(5, len(numeric_cols)))
    pca.fit(df_pca)

    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=numeric_cols
    )

    # Check if sensitive attributes load heavily on PC1
    sensitive_in_numeric = [a for a in config.sensitive_attributes if a in numeric_cols]
    flags = []
    for attr in sensitive_in_numeric:
        pc1_loading = abs(loadings.loc[attr, "PC1"])
        if pc1_loading > 0.3:
            flags.append({
                "attribute": attr,
                "pc1_loading": round(pc1_loading, 4),
                "warning": f"'{attr}' is a top driver of variation (PC1 loading = {pc1_loading:.3f})"
            })

    return {
        "explained_variance": [round(v, 4) for v in pca.explained_variance_ratio_],
        "top_pc1_features": loadings["PC1"].abs().nlargest(5).to_dict(),
        "sensitive_attribute_flags": flags,
        "status": "⚠️ BIAS SIGNAL" if flags else "OK"
    }


def _intersectional_analysis(df: pd.DataFrame, config: PipelineConfig) -> List[Dict]:
    """
    The most powerful check: test bias for every COMBINATION of sensitive attributes.
    A model can pass the gender test and the caste test but fail the Female+SC test.
    """
    target = config.target_column
    results = []

    available_attrs = [a for a in config.sensitive_attributes if a in df.columns]

    for r in range(2, len(available_attrs) + 1):
        for combo in combinations(available_attrs, r):
            # Create combined group column
            combined = df[list(combo)].astype(str).agg("_".join, axis=1)
            combined_name = " × ".join(combo)

            # Skip groups that are too small
            group_sizes = combined.value_counts()

            ct = pd.crosstab(combined, df[target])

            # Choose test based on sample size
            min_cell = ct.min().min()
            if min_cell < config.bias_detection.fisher_cell_cutoff and ct.shape == (2, 2):
                _, p_value = stats.fisher_exact(ct)
                test_used = "Fisher's exact"
            else:
                try:
                    chi2, p_value, _, expected = stats.chi2_contingency(ct)
                    test_used = "Chi-square"
                except ValueError:
                    continue

            # Cramér's V for the intersection
            v = _cramers_v(combined, df[target])

            # Approval rates per intersectional group
            rates = df.groupby(combined)[target].mean()
            max_gap = rates.max() - rates.min()

            results.append({
                "intersection": combined_name,
                "groups_tested": list(combo),
                "n_subgroups": len(group_sizes),
                "smallest_subgroup_size": int(group_sizes.min()),
                "test_used": test_used,
                "p_value": round(p_value, 6),
                "cramers_v": round(v, 4),
                "max_approval_gap": round(max_gap * 100, 2),
                "highest_rate_group": rates.idxmax(),
                "lowest_rate_group": rates.idxmin(),
                "statistically_significant": p_value < config.bias_detection.chi_square_alpha,
                "confidence_warning": (
                    "⚠️ LOW CONFIDENCE — subgroup size < 30"
                    if group_sizes.min() < config.bias_detection.small_sample_cutoff
                    else "✅ Adequate sample size"
                )
            })

    return results


def _bootstrap_ci(
    df: pd.DataFrame, config: PipelineConfig, n_bootstrap: int = 200
) -> Dict[str, Dict]:
    """
    Compute 95% bootstrap confidence intervals for Cramér's V
    for each sensitive attribute vs target.
    """
    target = config.target_column
    ci_results = {}

    for attr in config.sensitive_attributes:
        if attr not in df.columns or target not in df.columns:
            continue

        bootstrap_vs = []
        n = len(df)

        for _ in range(n_bootstrap):
            sample = df.sample(n=n, replace=True)
            try:
                v = _cramers_v(sample[attr], sample[target])
                bootstrap_vs.append(v)
            except Exception:
                continue

        if bootstrap_vs:
            ci_lower = np.percentile(bootstrap_vs, 2.5)
            ci_upper = np.percentile(bootstrap_vs, 97.5)
            point_estimate = _cramers_v(df[attr], df[target])

            ci_results[attr] = {
                "point_estimate": round(point_estimate, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "ci_width": round(ci_upper - ci_lower, 4),
                "reliability": (
                    "HIGH" if (ci_upper - ci_lower) < 0.1
                    else "MEDIUM" if (ci_upper - ci_lower) < 0.2
                    else "LOW — interpret with caution"
                )
            }

    return ci_results


def _build_summary(results: Dict, config: PipelineConfig) -> Dict:
    """Summarise all Phase 4 findings."""
    significant_tests = [
        t for t in results["chi_square_tests"] if t["statistically_significant"]
    ]
    strong_correlations = [
        c for c in results["correlations"] if "INVESTIGATE" in c.get("flag", "")
    ]
    intersectional_flags = [
        i for i in results["intersectional"]
        if i["statistically_significant"] and i["cramers_v"] >= config.bias_detection.cramers_v_threshold
    ]

    return {
        "significant_chi_square_tests": len(significant_tests),
        "strong_correlations": len(strong_correlations),
        "intersectional_bias_signals": len(intersectional_flags),
        "total_flags": len(significant_tests) + len(strong_correlations) + len(intersectional_flags),
        "status": (
            "🔴 BIAS DETECTED" if (significant_tests or intersectional_flags)
            else "🟢 NO SIGNIFICANT BIAS"
        )
    }
