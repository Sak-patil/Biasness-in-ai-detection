# ⚖️ Unbias AI Decision

> **Ensuring Fairness and Detecting Bias in Automated Decisions**

Computer programs now make life-changing decisions — who gets a job, a bank loan, or medical care. If these programs learn from flawed historical data, they repeat and amplify those exact same discriminatory mistakes at massive scale.

**Unbias AI** is an automated 8-phase pipeline that detects, explains, and fixes algorithmic bias in any dataset or ML model — before it impacts real people.

---

## 🚀 Live Demo

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate a demo hiring dataset (no Male/Female — uses race, age_group, education)
python generate_demo_data.py

# 3. Launch the interactive dashboard
streamlit run dashboard.py
```

Open **http://localhost:8501** → Upload `demo_data.csv` → Select sensitive columns → Click **▶ Run Analysis**

---

## 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| **CSV Upload** | Upload any dataset — no coding required |
| **Auto-detection** | Automatically suggests sensitive columns (race, age, zip, religion, etc.) |
| **Dynamic KPI cards** | Approval gap, Legal Safety (4/5ths), Proxy count — update per attribute |
| **Approval Rate Chart** | Visual comparison across all groups |
| **Intersectional Analysis** | Checks every combination (e.g. Black + 41-50 + HighSchool) |
| **Proxy Variable Detection** | Finds hidden backdoors using Cramér's V |
| **Risk Breakdown** | Shows which signal is driving the bias score |
| **Plain-language Recommendations** | Actionable next steps for non-technical stakeholders |

---

## 🔬 The 8-Phase Pipeline

```
Phase 1 → Data Audit          Who is in the data? Missing data gaps? MCAR/MAR/NMAR?
Phase 2 → Data Cleaning        Fair encoding, normalisation, proxy detection (Cramér's V)
Phase 3 → Label Audit          Are the training labels themselves biased?
Phase 4 → Bias Detection       Chi-square, Cramér's V, intersectional analysis, bootstrap CIs
Phase 5 → Explainability       SHAP — WHY does the bias exist? Which features drive it?
Phase 6 → Fairness Metrics     Demographic parity, equalized odds, disparate impact, calibration
Phase 7 → Risk Scoring         Composite 0-1 risk score with Pareto frontier + sensitivity analysis
Phase 8 → Mitigation           SMOTE / Reweighting / ExponentiatedGradient / ThresholdOptimizer
```

---

## 📁 Project Structure

```
unbias ai/
│
├── dashboard.py              ← Streamlit interactive dashboard (main entry point)
├── run_pipeline.py           ← Full 8-phase CLI pipeline runner
├── generate_demo_data.py     ← Synthetic hiring dataset generator
├── demo_data.csv             ← Pre-generated demo dataset (2,000 rows)
├── config.yaml               ← All thresholds & weights (no hardcoding)
├── requirements.txt          ← Full dependency list
│
└── pipeline/
    ├── config.py             ← Config loader (dataclasses)
    ├── phase1_data_audit.py  ← Representation, missingness, MCAR/MAR/NMAR
    ├── phase2_data_cleaning.py ← Encoding, normalisation, proxy detection
    ├── phase3_label_audit.py ← Label bias scoring, counterfactual check
    ├── phase4_bias_detection.py ← Chi-square, Cramér's V, intersectional, bootstrap
    ├── phase5_explainability.py ← SHAP per-group, divergence, root cause report
    ├── phase6_fairness_metrics.py ← DP, EO, disparate impact, calibration
    ├── phase7_risk_scoring.py ← Composite score, Pareto frontier, sensitivity
    └── phase8_mitigation.py  ← SMOTE, reweighting, ExponentiatedGradient, ThresholdOptimizer
```

---

## 🎯 How to Verify Bias — 5-Step Checklist

```
Step 1  Check the badge         🟢 Green=OK  🟡 Yellow=Review  🔴 Red=Stop

Step 2  Read the KPI cards
        ├── Approval Gap > 10%?         → Bias found
        ├── Legal Safety < 0.80?        → Legally discriminatory (4/5ths rule)
        └── Proxy variables > 0?        → Hidden backdoor detected

Step 3  Switch attributes (top selector)
        └── Each attribute shows its own gap, DI ratio, Cramér's V

Step 4  Check Intersectional tab
        └── Gap > 10% in any combination?  → Blind spot found

Step 5  Read "What To Do" tab
        └── Follow numbered recommendations
```

---

## 🗂️ Demo Dataset

The included `demo_data.csv` is a **synthetic job hiring dataset** with intentionally embedded biases:

| Attribute | Bias Embedded |
|-----------|--------------|
| `race` | White/Asian ~59% hired, Black/Hispanic ~36% hired |
| `age_group` | 41-60 group hired ~18% less despite more experience |
| `education` | Fair — higher education legitimately predicts hiring |
| `college_tier` | **Proxy variable** — correlates with race (Cramér's V ≈ 0.75) |
| `skill_score` | **Missing more** for Black/Hispanic (~20%) vs White (~4%) |

No Male/Female. No gender. A completely different domain to demonstrate the system works on **any** dataset.

---

## ⚙️ Configuration

All thresholds are in `config.yaml` — change them without touching code:

```yaml
bias_detection:
  cramers_v_threshold: 0.30    # Flag bias above this
  min_subgroup_size: 30        # Minimum group size for reliable stats

risk_scoring:
  weights:
    cramers_v: 0.25
    dp_gap: 0.30
    label_bias: 0.20
    imbalance_ratio: 0.15
    explainability_disparity: 0.10
  thresholds:
    low: 0.30
    high: 0.60

mitigation:
  accuracy_loss_tolerance: 0.05   # Max acceptable accuracy drop after fixing bias
```

---

## 📦 Requirements

```
pandas, numpy, scipy, scikit-learn
fairlearn          ← In-processing & post-processing mitigation
shap               ← Explainability (root cause analysis)
imbalanced-learn   ← SMOTE oversampling
streamlit          ← Interactive dashboard
plotly             ← Charts
missingno          ← Missing data visualisation
pyyaml             ← Config loading
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📊 Key Metrics Explained

| Metric | What it means | Threshold |
|--------|--------------|-----------|
| **Cramér's V** | Statistical strength of association between a feature and a protected attribute | Flag if > 0.30 |
| **Demographic Parity Gap** | Difference in approval rates between groups | Flag if > 10% |
| **Disparate Impact Ratio** | Minority rate ÷ majority rate | Must be ≥ 0.80 (legal standard) |
| **Label Bias Score** | How biased are the training labels themselves | Flag if > 0.30 |
| **SHAP Divergence** | How differently a feature influences decisions across groups | Higher = bigger bias driver |
| **Composite Risk Score** | Weighted combination of all signals (0–1) | Low < 0.30, Medium 0.30–0.60, High > 0.60 |

---

## ⚖️ Legal Standards

This system checks against the **4/5ths (80%) rule** — the primary legal standard for employment discrimination in the US (EEOC Uniform Guidelines) and similar frameworks in the EU AI Act:

> If the selection rate for any group is less than **4/5ths (80%)** of the group with the highest selection rate, this is considered evidence of adverse impact.

A **Disparate Impact Ratio below 0.80** means the system is **presumptively discriminatory** under this standard.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as part of the Unbiased AI Decision project — ensuring fairness before systems impact real people.*
