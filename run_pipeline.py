"""
Unbias AI Decision — Main Pipeline Runner
===========================================
Orchestrates all 8 phases end-to-end and produces a complete bias report.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config
from pipeline.phase1_data_audit import run_data_audit
from pipeline.phase2_data_cleaning import run_data_cleaning
from pipeline.phase3_label_audit import run_label_audit
from pipeline.phase4_bias_detection import run_bias_detection
from pipeline.phase5_explainability import run_explainability
from pipeline.phase6_fairness_metrics import run_fairness_metrics
from pipeline.phase7_risk_scoring import run_risk_scoring
from pipeline.phase8_mitigation import run_mitigation


def run_full_pipeline(data_path: str = None, config_path: str = None):
    """
    Run the complete Unbias AI Decision pipeline.
    """
    print("=" * 70)
    print("  UNBIAS AI DECISION — Full Pipeline Run")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load Config ---
    config = load_config(config_path)
    print(f"\n✅ Config loaded. Sensitive attributes: {config.sensitive_attributes}")

    # --- Load Data ---
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded from {data_path}: {len(df)} records, {len(df.columns)} columns")
    else:
        print("📊 No data file provided. Generating demo dataset...")
        from generate_demo_data import generate_demo_dataset
        df = generate_demo_dataset(n_samples=2000)
        print(f"✅ Generated {len(df)} demo records with embedded biases")

    all_results = {"timestamp": datetime.now().isoformat(), "data_shape": list(df.shape)}

    # ================================================================
    # PHASE 1 — Data Audit
    # ================================================================
    print("\n" + "─" * 50)
    print("📋 PHASE 1 — Data Audit")
    print("─" * 50)

    audit_results = run_data_audit(df, config)
    all_results["data_audit"] = _serialize(audit_results)

    print(f"  Underrepresented groups: {len(audit_results['underrepresented_groups'])}")
    print(f"  Missing data gaps: {len(audit_results['missing_data_gaps'])}")
    for issue in audit_results["summary"]["issues"]:
        print(f"  ⚠️  {issue}")

    # ================================================================
    # PHASE 2 — Data Cleaning
    # ================================================================
    print("\n" + "─" * 50)
    print("🧹 PHASE 2 — Data Cleaning")
    print("─" * 50)

    df_clean, cleaning_report = run_data_cleaning(df, config)
    all_results["data_cleaning"] = _serialize(cleaning_report)

    print(f"  Inconsistencies removed: {cleaning_report['summary']['inconsistencies_removed']}")
    print(f"  Columns encoded: {cleaning_report['summary']['columns_encoded']}")
    print(f"  Proxy variables found: {cleaning_report['summary']['proxy_variables_found']}")

    for proxy in cleaning_report["proxy_variables"]:
        if proxy["is_proxy"]:
            print(f"  🔴 PROXY: {proxy['feature']} ↔ {proxy['sensitive_attribute']} "
                  f"(Cramér's V = {proxy['cramers_v']})")

    # ================================================================
    # PHASE 3 — Label Audit
    # ================================================================
    print("\n" + "─" * 50)
    print("🏷️  PHASE 3 — Label Audit")
    print("─" * 50)

    label_results = run_label_audit(df, config)
    all_results["label_audit"] = _serialize(label_results)

    print(f"  Label bias score: {label_results.get('label_bias_score', 'N/A')}")
    print(f"  {label_results.get('recommendation', '')}")
    if label_results.get("should_halt"):
        print("  ⛔ PIPELINE HALTED — Label bias too high. Fix labels before proceeding.")
        print("     (Continuing anyway for demonstration purposes...)")

    # ================================================================
    # PHASE 4 — Statistical Bias Detection
    # ================================================================
    print("\n" + "─" * 50)
    print("📊 PHASE 4 — Statistical Bias Detection")
    print("─" * 50)

    bias_results = run_bias_detection(df, config)
    all_results["bias_detection"] = _serialize(bias_results)

    for test in bias_results["chi_square_tests"]:
        symbol = "🔴" if test["above_threshold"] else "🟢"
        print(f"  {symbol} {test['attribute']}: Cramér's V = {test['cramers_v']} "
              f"({test['strength']})")

    if bias_results["intersectional"]:
        significant = [i for i in bias_results["intersectional"] if i["statistically_significant"]]
        print(f"  Intersectional tests: {len(bias_results['intersectional'])} run, "
              f"{len(significant)} significant")
        for item in significant[:3]:
            print(f"    ↳ {item['intersection']}: gap = {item['max_approval_gap']}%, "
                  f"Cramér's V = {item['cramers_v']}")

    # ================================================================
    # PHASE 5 — Train Model + Explainability
    # ================================================================
    print("\n" + "─" * 50)
    print("🔍 PHASE 5 — Model Training + Explainability")
    print("─" * 50)

    # Prepare features for model training
    target = config.target_column
    feature_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                    if c != target and c not in config.sensitive_attributes]

    X = df_clean[feature_cols].fillna(df_clean[feature_cols].median())
    y = df_clean[target]
    sensitive = df[config.sensitive_attributes[0]]  # Primary sensitive attribute

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    sens_train = sensitive.loc[X_train.index]
    sens_test = sensitive.loc[X_test.index]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    base_acc = accuracy_score(y_test, y_pred)
    print(f"  Model accuracy: {base_acc:.4f}")

    # Explainability
    expl_results = run_explainability(df_clean, model, config, feature_cols)
    all_results["explainability"] = _serialize(expl_results)

    if expl_results.get("shap_computed"):
        print(f"  SHAP computed successfully")
        print(f"  Explainability disparity score: "
              f"{expl_results.get('explainability_disparity_score', 'N/A')}")

        # Print top bias drivers
        for attr, features in expl_results.get("shap_divergence", {}).items():
            if features:
                print(f"  Top bias drivers for {attr}:")
                for f in features[:3]:
                    print(f"    ↳ {f['feature']}: divergence = {f['divergence']}")
    else:
        print(f"  SHAP: {expl_results.get('reason', expl_results.get('error', 'unknown'))}")

    # ================================================================
    # PHASE 6 — Model-level Fairness Metrics
    # ================================================================
    print("\n" + "─" * 50)
    print("⚖️  PHASE 6 — Fairness Metrics")
    print("─" * 50)

    sens_features_df = df.loc[X_test.index, config.sensitive_attributes]
    fairness_results = run_fairness_metrics(
        y_test, pd.Series(y_pred, index=X_test.index),
        pd.Series(y_prob, index=X_test.index),
        sens_features_df, config
    )
    all_results["fairness_metrics"] = _serialize(fairness_results)

    for attr, dp in fairness_results["demographic_parity"].items():
        print(f"  {dp['interpretation']}")

    for attr, di in fairness_results["disparate_impact"].items():
        print(f"  {di['interpretation']}")

    print(f"  Overall: {fairness_results['summary']['status']}")

    # ================================================================
    # PHASE 7 — Risk Scoring
    # ================================================================
    print("\n" + "─" * 50)
    print("📈 PHASE 7 — Risk Scoring")
    print("─" * 50)

    risk_results = run_risk_scoring(all_results, config)
    all_results["risk_scoring"] = _serialize(risk_results)

    print(f"  Composite Risk Score: {risk_results['composite_score']}")
    print(f"  Risk Level: {risk_results['risk_level']}")
    print(f"  Sensitivity: {risk_results['sensitivity']['recommendation']}")

    print(f"\n  Components:")
    for comp, val in risk_results["components"].items():
        print(f"    {comp}: {val:.4f}")

    print(f"\n  Recommendations:")
    for rec in risk_results["recommendations"]:
        print(f"    → {rec}")

    # ================================================================
    # PHASE 8 — Mitigation
    # ================================================================
    print("\n" + "─" * 50)
    print("🔧 PHASE 8 — Mitigation")
    print("─" * 50)

    mitigation_results = run_mitigation(
        risk_results["risk_level"],
        model, X_train, y_train, X_test, y_test,
        sens_train, sens_test, config
    )
    all_results["mitigation"] = _serialize(mitigation_results)

    print(f"  Original accuracy: {mitigation_results['original_accuracy']}")
    print(f"  Mitigated accuracy: {mitigation_results.get('mitigated_accuracy', 'N/A')}")
    print(f"  Accuracy change: {mitigation_results.get('accuracy_change', 'N/A')}")

    for tech in mitigation_results["techniques_applied"]:
        print(f"  ↳ {tech['name']}: {tech['status']}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    # Save JSON report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📄 Full report saved to: {report_path}")
    print(f"🖥️  To view the dashboard, run: streamlit run dashboard.py")

    return all_results


def _serialize(obj):
    """Make results JSON-serializable by converting pandas objects."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()
                if not k.endswith("_model") and k != "mitigated_model"}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unbias AI Decision Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    run_full_pipeline(data_path=args.data, config_path=args.config)
