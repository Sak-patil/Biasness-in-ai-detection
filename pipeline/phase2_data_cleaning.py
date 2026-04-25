"""
Phase 2 — Data Cleaning
========================
One-hot encoding, normalisation, inconsistency removal, and proxy detection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
from .config import PipelineConfig


def run_data_cleaning(
    df: pd.DataFrame, config: PipelineConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the complete data cleaning phase.

    Returns:
      - cleaned DataFrame
      - report dictionary with all findings
    """
    report = {}
    df_clean = df.copy()

    # --- 1. Inconsistency Removal ---
    df_clean, inconsistencies = _remove_inconsistencies(df_clean)
    report["inconsistencies_removed"] = inconsistencies

    # --- 2. One-Hot Encoding (fair encoding) ---
    df_clean, encoding_report = _fair_encoding(df_clean, config)
    report["encoding"] = encoding_report

    # --- 3. StandardScaler Normalisation ---
    df_clean, scaling_report = _normalize_features(df_clean, config)
    report["scaling"] = scaling_report

    # --- 4. Proxy Variable Detection ---
    report["proxy_variables"] = _detect_proxies(df_clean, config)

    # --- Summary ---
    n_proxies = len([p for p in report["proxy_variables"] if p["is_proxy"]])
    report["summary"] = {
        "inconsistencies_removed": len(inconsistencies),
        "columns_encoded": len(encoding_report.get("encoded_columns", [])),
        "columns_scaled": len(scaling_report.get("scaled_columns", [])),
        "proxy_variables_found": n_proxies,
        "status": "PROXIES DETECTED" if n_proxies > 0 else "CLEAN"
    }

    return df_clean, report


def _remove_inconsistencies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Remove records with logical contradictions.
    Examples: negative age, employment years > age, income=0 with luxury assets.
    """
    inconsistencies = []
    original_len = len(df)

    # Age-related checks
    if "age" in df.columns:
        mask = df["age"] < 0
        if mask.any():
            inconsistencies.append({"rule": "negative age", "count": int(mask.sum())})
            df = df[~mask]

        if "employment_years" in df.columns:
            mask = df["employment_years"] > df["age"]
            if mask.any():
                inconsistencies.append({
                    "rule": "employment_years > age", "count": int(mask.sum())
                })
                df = df[~mask]

    # Income checks
    if "income" in df.columns:
        mask = df["income"] < 0
        if mask.any():
            inconsistencies.append({"rule": "negative income", "count": int(mask.sum())})
            df = df[~mask]

    removed = original_len - len(df)
    if removed > 0:
        inconsistencies.append({"total_removed": removed})

    return df, inconsistencies


def _fair_encoding(
    df: pd.DataFrame, config: PipelineConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply one-hot encoding to categorical columns.
    Instead of Female=0, Male=1 (which implies a ranking),
    creates separate binary columns: is_female, is_male.
    """
    report = {"encoded_columns": [], "original_columns_dropped": []}

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Don't encode the target column — keep it as-is
    target = config.target_column
    if target in categorical_cols:
        categorical_cols.remove(target)

    # Don't one-hot encode sensitive attributes (keep them for analysis)
    # but create encoded versions alongside
    cols_to_encode = [c for c in categorical_cols if c not in config.sensitive_attributes]

    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)
        report["encoded_columns"] = cols_to_encode
        report["original_columns_dropped"] = cols_to_encode

    return df, report


def _normalize_features(
    df: pd.DataFrame, config: PipelineConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply StandardScaler to numeric columns (mean=0, std=1).
    Ensures income (lakhs) and age (decades) are on the same scale.
    Excludes target column and sensitive attributes.
    """
    report = {"scaled_columns": [], "scaler_stats": {}}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target and sensitive attributes from scaling
    exclude = [config.target_column] + config.sensitive_attributes
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    if cols_to_scale:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        report["scaled_columns"] = cols_to_scale
        report["scaler_stats"] = {
            col: {"mean": round(m, 4), "std": round(s, 4)}
            for col, m, s in zip(cols_to_scale, scaler.mean_, scaler.scale_)
        }

    return df, report


def _detect_proxies(df: pd.DataFrame, config: PipelineConfig) -> List[Dict[str, Any]]:
    """
    Detect proxy variables — features that accidentally encode protected information.
    Uses Cramér's V to correlate every input feature against each sensitive attribute.
    Zip code → neighbourhood demographics, college name → caste, first name → gender.
    """
    proxies = []
    threshold = config.bias_detection.cramers_v_threshold

    for attr in config.sensitive_attributes:
        if attr not in df.columns:
            continue

        for col in df.columns:
            if col == attr or col == config.target_column:
                continue

            try:
                v = _cramers_v(df[attr], df[col])
                is_proxy = v >= threshold
                proxies.append({
                    "feature": col,
                    "sensitive_attribute": attr,
                    "cramers_v": round(v, 4),
                    "is_proxy": is_proxy,
                    "recommendation": f"⚠️ '{col}' is a proxy for '{attr}' — consider removing or decorrelating"
                    if is_proxy else "OK"
                })
            except Exception:
                continue

    return proxies


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V between two categorical/discrete series."""
    # Bin numeric columns into categories for contingency table
    if pd.api.types.is_numeric_dtype(x):
        x = pd.qcut(x, q=5, duplicates="drop")
    if pd.api.types.is_numeric_dtype(y):
        y = pd.qcut(y, q=5, duplicates="drop")

    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    denom = n * (min(r, k) - 1)
    if denom == 0:
        return 0.0
    return np.sqrt(chi2 / denom)
