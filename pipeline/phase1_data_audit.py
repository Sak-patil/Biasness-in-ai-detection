"""
Phase 1 — Data Audit
=====================
Inspects the raw dataset for representation gaps, missing data patterns,
and missingness mechanisms (MCAR / MAR / NMAR).
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from .config import PipelineConfig


def run_data_audit(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Run the complete data audit phase.

    Returns a dictionary with:
      - group_representation: counts and percentages per sensitive attribute
      - underrepresented_groups: groups below the threshold
      - missing_data_analysis: per-group missingness rates
      - missing_data_gaps: gap in missingness between groups
      - outcome_distribution: approval/rejection rates per group
      - missingness_type: MCAR / MAR / NMAR classification per column
    """
    results = {}

    # --- 1. Group Representation ---
    results["group_representation"] = _group_representation(df, config)

    # --- 2. Underrepresented Groups ---
    results["underrepresented_groups"] = _find_underrepresented(
        results["group_representation"], config.data_audit.underrepresentation_threshold
    )

    # --- 3. Missing Data Analysis per Group ---
    results["missing_data_analysis"] = _missing_data_per_group(df, config)

    # --- 4. Missing Data Gaps ---
    results["missing_data_gaps"] = _missing_data_gaps(
        results["missing_data_analysis"], config
    )

    # --- 5. Outcome Distribution per Group ---
    results["outcome_distribution"] = _outcome_distribution(df, config)

    # --- 6. Missingness Classification ---
    results["missingness_type"] = _classify_missingness(df, config)

    # --- Summary ---
    results["summary"] = _build_summary(results, config)

    return results


def _group_representation(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """Count and percentage of records per value of each sensitive attribute."""
    rep = {}
    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue
        counts = df[attr].value_counts()
        pcts = df[attr].value_counts(normalize=True)
        rep[attr] = pd.DataFrame({
            "count": counts,
            "percentage": (pcts * 100).round(2)
        })
    return rep


def _find_underrepresented(
    group_rep: Dict[str, pd.DataFrame], threshold: float
) -> List[Dict[str, Any]]:
    """Identify groups below the representation threshold."""
    flagged = []
    for attr, rep_df in group_rep.items():
        for group_val, row in rep_df.iterrows():
            if row["percentage"] / 100 < threshold:
                flagged.append({
                    "attribute": attr,
                    "group": group_val,
                    "percentage": row["percentage"],
                    "count": int(row["count"]),
                    "severity": "HIGH" if row["percentage"] / 100 < threshold / 2 else "MEDIUM"
                })
    return flagged


def _missing_data_per_group(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """Compute the percentage of missing values per column, split by each sensitive group."""
    analysis = {}
    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue
        groups = df[attr].dropna().unique()
        missing_rates = {}
        for group_val in groups:
            subset = df[df[attr] == group_val]
            missing_pct = (subset.isnull().sum() / len(subset) * 100).round(2)
            missing_rates[group_val] = missing_pct
        analysis[attr] = pd.DataFrame(missing_rates)
    return analysis


def _missing_data_gaps(
    missing_analysis: Dict[str, pd.DataFrame], config: PipelineConfig
) -> List[Dict[str, Any]]:
    """Find columns where missingness differs significantly between groups."""
    gaps = []
    threshold = config.data_audit.missing_data_gap_threshold * 100  # Convert to percentage

    for attr, miss_df in missing_analysis.items():
        for col in miss_df.index:
            values = miss_df.loc[col]
            max_gap = values.max() - values.min()
            if max_gap > threshold:
                gaps.append({
                    "attribute": attr,
                    "column": col,
                    "max_group": values.idxmax(),
                    "min_group": values.idxmin(),
                    "gap_percentage": round(max_gap, 2),
                    "max_missing_pct": round(values.max(), 2),
                    "min_missing_pct": round(values.min(), 2),
                })
    return gaps


def _outcome_distribution(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """Compute approval/positive-outcome rates per group for each sensitive attribute."""
    target = config.target_column
    if target not in df.columns:
        return {}

    dist = {}
    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue
        grouped = df.groupby(attr)[target].agg(["mean", "count"])
        grouped.columns = ["approval_rate", "count"]
        grouped["approval_rate"] = (grouped["approval_rate"] * 100).round(2)
        dist[attr] = grouped
    return dist


def _classify_missingness(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, str]:
    """
    Classify each column's missingness mechanism:
      - MCAR: Missing Completely At Random (Little's test approximation)
      - MAR:  Missing At Random (depends on observed variables)
      - NMAR: Not Missing At Random (depends on the missing value itself)

    Uses a simplified heuristic:
      - If missingness is uncorrelated with ALL other columns → MCAR
      - If missingness correlates with other observed columns → MAR
      - If missingness correlates with the sensitive attributes → potential NMAR (flag)
    """
    classifications = {}
    cols_with_missing = [c for c in df.columns if df[c].isnull().any()]

    for col in cols_with_missing:
        missing_indicator = df[col].isnull().astype(int)

        # Check correlation with sensitive attributes
        sensitive_corr = False
        for attr in config.sensitive_attributes:
            if attr in df.columns and attr != col:
                # Convert to numeric if needed
                if df[attr].dtype == "object":
                    encoded = pd.factorize(df[attr])[0]
                else:
                    encoded = df[attr].fillna(0)

                corr, p_val = stats.pointbiserialr(missing_indicator, encoded)
                if p_val < 0.05 and abs(corr) > 0.1:
                    sensitive_corr = True
                    break

        # Check correlation with other observed columns
        other_corr = False
        for other_col in df.select_dtypes(include=[np.number]).columns:
            if other_col != col and not df[other_col].isnull().all():
                clean = df[[other_col]].dropna()
                if len(clean) > 10:
                    indicator_clean = missing_indicator.loc[clean.index]
                    corr, p_val = stats.pointbiserialr(indicator_clean, clean[other_col])
                    if p_val < 0.05 and abs(corr) > 0.1:
                        other_corr = True
                        break

        if sensitive_corr:
            classifications[col] = "NMAR (⚠️ correlated with protected attribute — highest risk)"
        elif other_corr:
            classifications[col] = "MAR (depends on other variables — manageable)"
        else:
            classifications[col] = "MCAR (random — safe to handle normally)"

    return classifications


def _build_summary(results: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """Build a human-readable summary of Phase 1 findings."""
    n_underrep = len(results["underrepresented_groups"])
    n_gaps = len(results["missing_data_gaps"])
    nmar_cols = [k for k, v in results["missingness_type"].items() if "NMAR" in v]

    issues = []
    if n_underrep > 0:
        issues.append(f"{n_underrep} underrepresented group(s) detected")
    if n_gaps > 0:
        issues.append(f"{n_gaps} column(s) with significant missing data gaps between groups")
    if nmar_cols:
        issues.append(f"{len(nmar_cols)} column(s) with NMAR missingness (highest risk): {nmar_cols}")

    return {
        "total_records": None,  # Will be filled by the caller
        "total_issues": n_underrep + n_gaps + len(nmar_cols),
        "issues": issues,
        "status": "PASS" if not issues else "ISSUES FOUND"
    }
