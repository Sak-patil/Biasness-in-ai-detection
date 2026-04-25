# Unbias AI Decision — Complete System Document (Corrected & Improved)

> **One-line summary:** We built a system that finds hidden discrimination in computer programs before and after they go live — and fixes it automatically, while explaining *why* it exists.

---

## The Story of Our System

Imagine you apply for a bank loan. You have a decent income, a stable job, and a clean history. But the bank's computer says no. You never find out why. The computer is not racist or sexist in the way a human might be — it does not "think." But it learned its decision-making patterns from 20 years of old loan records, and in those 20 years, the bank's human employees were unconsciously biased. The computer absorbed all of that history and now repeats it at massive scale, thousands of decisions per day, with nobody checking.

**Our system is the checker.**

Making a model perfectly fair means overriding some of its learned patterns — which reduces accuracy slightly. We do not blindly pick the most fair model. We pick the **"Pareto optimal"** model — the one where you cannot get more fairness without losing accuracy, and you cannot get more accuracy without losing fairness. This is the honest sweet spot.

---

## What We Actually Built — The Complete 8-Phase Pipeline

### Phase 1 — Data Audit: Understanding What You Have

First we look at the raw data and ask the most basic question: **who is even in here?** If only 3% of the training records are from women from rural areas, the program will barely know they exist. You cannot teach a child to be fair to people it has never seen.

| Tool | What It Does | Library |
|------|-------------|---------|
| **Missing data analysis** | Draws a picture of where data is missing. If gaps cluster in one group (women, rural applicants), that is a structural problem. | `missingno`, `pandas` |
| **Group representation count** | Counts how many records exist per group. Any group below 10–15% is flagged as underrepresented. | `pandas` |
| **Outcome distribution per group** | Plots approval/hiring/diagnosis rate per group separately. If one group has 72% approval and another 31%, historical bias is baked in. | `seaborn`, `matplotlib` |
| **Missingness classification (MCAR/MAR/NMAR)** | Classifies *why* data is missing. MCAR = random (safe). MAR = depends on other variables (manageable). NMAR = missing because of its own value (most dangerous — e.g., high earners skipping salary fields). | `scipy.stats` |

---

### Phase 2 — Data Cleaning: Removing Unfairness Baked Into the Data

Small things matter here. If someone encoded "Female" as the number zero in a spreadsheet, the computer interprets that as "female means nothing."

| Tool | What It Does | Library |
|------|-------------|---------|
| **One-hot encoding (fair encoding)** | Instead of Male=1, Female=0, creates separate binary columns. The model sees these as independent yes/no questions, not a ranked scale. | `pandas.get_dummies` |
| **StandardScaler normalisation** | Ensures income (in lakhs) and age (in decades) are on the same scale. Without this, income dominates the model purely because its numbers are larger. | `sklearn.preprocessing` |
| **Inconsistency removal** | Removes records that contradict themselves — age=15 but employment=30 years. These corrupt the model's patterns. | `pandas` |
| **Proxy variable detection** | Even after removing gender, features like zip code, name, or university can act as proxies. We correlate every feature against protected attributes using Cramér's V. If zip code has high correlation with caste, it is a hidden backdoor for discrimination. | `scipy.stats` (Cramér's V) |

---

### Phase 3 — Label Audit: Can We Trust the Ground Truth? *(NEW — Critical Addition)*

> **Why this phase was added:** Every fairness metric in the pipeline assumes the ground truth labels (approved/rejected, hired/not-hired) are correct. But if those labels are themselves products of historical bias — e.g., "this loan defaulted" because the borrower was given predatory terms, or "this employee failed" because they were judged by biased managers — then every downstream metric is computed against a corrupted reference. Without this phase, the entire system can certify a biased model as "fair."

| Tool | What It Does | Method |
|------|-------------|--------|
| **Label distribution analysis** | Check if approval/rejection rates differ dramatically across groups *in the raw labels*. If historical approval rates show 85% for Group A and 40% for Group B, the labels themselves may encode systemic discrimination. | `pandas` group-by analysis |
| **Expert review protocol** | Domain experts (e.g., experienced loan officers, doctors) review a stratified sample of labelled decisions, especially borderline cases from underrepresented groups, to identify labelling errors or bias patterns. | Manual review + structured questionnaire |
| **Counterfactual label check** | For each record, ask: "Would this person's label change if only their protected attribute changed?" If changing gender from Female to Male flips a rejection to an approval in a significant number of cases, the labels encode gender bias. | `dowhy` (causal inference) |
| **Outcome vs. label comparison** | Where real-world outcomes are available (did the loan actually default?), compare them against the original labels. If rejected applicants from Group B would have succeeded at similar rates to approved applicants from Group A, the labels were wrong. | `pandas`, `scipy.stats` |

**Decision gate:** If label bias is detected above a configurable threshold, the pipeline halts and recommends: (a) relabelling with expert oversight, (b) using proxy outcomes instead, or (c) proceeding with explicit label-bias adjustment weights.

---

### Phase 4 — Statistical Bias Detection: The Core Tests

We run the actual bias tests using mathematical tools.

| Tool | What It Does | Library | Output |
|------|-------------|---------|--------|
| **Chi-square test** | Builds a table: rows are gender groups, columns are approved/rejected. Compares reality against what would happen if gender had zero effect. p-value below 0.05 means the bias is statistically real, not random noise. | `scipy.stats.chi2_contingency` | p-value |
| **Fisher's exact test** *(NEW)* | Same purpose as chi-square but reliable for small sample sizes (any cell count < 5). Used automatically as a fallback during intersectional analysis where subgroup sizes can be tiny. | `scipy.stats.fisher_exact` | p-value |
| **Cramér's V** | Follow-up to chi-square. Chi-square has a flaw — with large datasets, even a tiny meaningless relationship gives a high value. Cramér's V corrects for sample size. Scale: 0 = no bias, below 0.1 = weak, 0.1–0.3 = moderate, above 0.3 = serious red flag. | `scipy.stats` | 0 to 1 |
| **Pearson / Point-biserial correlation** *(Corrected)* | Measures straight-line relationship between a binary attribute (gender as 0/1) and a numeric score (credit score). Point-biserial is the mathematically precise variant for binary × continuous variables. A value of 0.4 means knowing gender explains 40% of score variation. | `scipy.stats.pointbiserialr` | −1 to +1 |
| **PCA** | Compresses all features into ranked "directions of maximum variation." If gender or caste loads heavily on the first principal component, that attribute is secretly the most powerful driver of outcomes. | `sklearn.decomposition.PCA` | Component loadings |
| **MCA** | Same as PCA but for purely categorical datasets. Places every category as a point in 2D space. If "Female" and "Rejected" cluster together, the association is strong and visible. | `prince.MCA` | 2D scatter map |
| **Intersectional analysis** | Creates combined columns (e.g., Female_SC, Male_General) and runs *all* above tests on every combination. A model can pass the gender test and the caste test but still systematically fail a specific gender-caste combination. | `itertools.combinations` + all tests | Per-intersection results |
| **Confidence intervals** *(NEW)* | Wraps every metric in a 95% bootstrap confidence interval. Prevents false alarms on small datasets — a Cramér's V of 0.35 with CI [0.10, 0.60] is uncertain, but 0.35 with CI [0.30, 0.40] is reliable. | Bootstrap resampling | 95% CI bounds |

> **Small-sample safeguard:** When any intersectional cell has fewer than 30 records, the system automatically switches from chi-square to Fisher's exact test and flags the result with a "low confidence" warning. This prevents unreliable statistics from triggering false bias alarms.

---

### Phase 5 — Explainability: Understanding WHY Bias Exists *(NEW — Critical Addition)*

> **Why this phase was added:** Phases 3–4 answer "Is there bias?" and "How bad is it?" but never "Why does it exist?" Without explainability, a compliance officer cannot write a root cause report, and engineers cannot prioritise which features to fix. A regulator will ask: "You found bias — what causes it?" The system must answer.

| Tool | What It Does | Library |
|------|-------------|---------|
| **SHAP (SHapley Additive exPlanations)** | For every prediction, SHAP assigns each feature a contribution score. Run per-group: if "zip_code" has SHAP value +0.3 for Group A but −0.2 for Group B, zip code is a primary driver of discriminatory outcomes. | `shap` |
| **SHAP group comparison** | Computes average SHAP values for each feature, split by protected group. Produces a side-by-side bar chart showing which features help Group A but hurt Group B. The top 5 divergent features are the root causes of bias. | `shap` + `pandas` |
| **Feature proxy report** | Combines Cramér's V proxy detection (Phase 2) with SHAP importance to produce a ranked list: "These features are both highly correlated with protected attributes AND highly influential in predictions." These are the highest-priority targets for intervention. | Custom (Cramér's V × SHAP importance) |

**Output:** A plain-language report: *"The model is biased primarily because Feature X (zip code) encodes neighbourhood demographics correlated with caste, and Feature Y (years of experience) penalises women who took career breaks. Removing or debiasing these two features would reduce the demographic parity gap by an estimated 60%."*

---

### Phase 6 — Model-level Fairness Metrics: After the Model is Trained

| Metric | What It Measures | Formula / Library | Threshold |
|--------|-----------------|-------------------|-----------|
| **Demographic parity** | Compares approval rates across groups. If men are approved at 78% and women at 55%, the gap is 23 points. | `fairlearn.metrics.demographic_parity_difference` | Gap should be below 0.10 (10%) |
| **Equalized odds** | Separates error types. Checks if qualified people from all groups are approved at equal rates (true positive rate parity) AND if unqualified people from all groups are rejected at equal rates (false positive rate parity). | `fairlearn.metrics.equalized_odds_difference` | FPR and FNR per group |
| **Disparate impact ratio** | The legal standard. Minority approval rate ÷ majority approval rate. Below 0.80 is legally presumptive discrimination under US employment law (the "4/5ths rule"). | `min_group_rate / max_group_rate` | Must be ≥ 0.80 |
| **Calibration per group** | When the model says "70% chance of approval" for a female applicant, is she actually approved 70% of the time? If calibration is off for one group, the model is internally miscalibrated. | `sklearn.calibration.calibration_curve` | Per-group calibration plots |
| **Individual fairness** *(NEW)* | Similar individuals should receive similar outcomes regardless of group membership. Measures whether people with nearly identical features but different protected attributes get different decisions. | `fairlearn` + custom Lipschitz metric | Outcome difference for similar individuals should be near zero |

---

### Phase 7 — Risk Scoring: Combining Everything Into One Number

**Composite bias risk score** — Weighted formula bringing all components together:

```
Risk Score = (0.30 × Cramér's V) + (0.30 × Demographic Parity Gap) +
             (0.15 × Imbalance Ratio) + (0.10 × Missing Data Gap) +
             (0.10 × Explainability Disparity) + (0.05 × Label Bias Score)
```

> **Improvement over original:** Weights are now **configurable via YAML** instead of hardcoded. Two new components added: Explainability Disparity (how differently SHAP values distribute across groups) and Label Bias Score (from Phase 3). All components normalised to 0–1.

```yaml
# config.yaml — Fully configurable risk scoring
risk_scoring:
  weights:
    cramers_v: 0.30
    dp_gap: 0.30
    imbalance_ratio: 0.15
    missing_gap: 0.10
    explainability_disparity: 0.10
    label_bias: 0.05
  thresholds:
    low: 0.30
    high: 0.60
  # Organisations can adjust these based on their domain and risk tolerance
```

**Classification:** 0–0.30 = Low Risk | 0.30–0.60 = Medium Risk | 0.60–1.0 = High Risk

**Pareto frontier analysis** — Train 20–30 versions of the model with different fairness constraint levels using Fairlearn. Plot each as a point on a scatter chart (accuracy vs fairness). The Pareto frontier is the curve of optimal models — none can improve on both axes simultaneously. Pick the point closest to your organisation's tolerance.

**Sensitivity analysis** *(NEW)*: Run the risk score calculation across 100 random weight perturbations (±20% on each weight). If the risk classification (Low/Medium/High) changes in more than 15% of perturbations, flag the result as "borderline — requires manual review." This prevents overconfidence in a single weight configuration.

---

### Phase 8 — Mitigation: Fixing the Problems Found

Think of this like a hospital. A mild condition gets mild treatment. A severe condition gets stronger treatment.

| Risk Level | Technique | What It Does | Library |
|------------|-----------|-------------|---------|
| **LOW** | **SMOTE oversampling** | Finds minority group records, finds their nearest neighbours, generates new synthetic records along the line between them. Gives the model a richer training set for disadvantaged groups. | `imblearn.over_sampling.SMOTE` |
| **MEDIUM** | **Sample reweighting** | Instead of creating new records, tells the model "mistakes on this group cost you twice as much." The model adjusts its patterns to try harder on failing groups. | `sklearn` `sample_weight` |
| **MEDIUM** | **Feature intervention** *(NEW)* | Based on SHAP analysis from Phase 5, remove or decorrelate the top proxy features driving bias. More targeted than blanket statistical fixes. | `shap` + feature engineering |
| **HIGH** | **ThresholdOptimizer** | After training, finds per-group decision thresholds that satisfy fairness constraints with minimum accuracy loss. Instead of one global threshold of 0.5, might use 0.45 for disadvantaged group and 0.55 for advantaged group. | `fairlearn.postprocessing.ThresholdOptimizer` |
| **HIGH** | **ExponentiatedGradient** | Rewrites the training objective itself. Instead of minimising only prediction error, minimises error *subject to a hard fairness constraint*. Most powerful fix but most expensive computationally. | `fairlearn.reductions.ExponentiatedGradient` |

**Automated mitigation selection** *(NEW)*: The system automatically selects the appropriate mitigation strategy based on the risk score. No manual intervention needed for standard cases.

**Post-mitigation re-validation** *(NEW — Critical)*: After every mitigation step, the entire pipeline (Phases 4–7) is re-run automatically to verify that:
1. The targeted bias was actually reduced
2. No *new* bias was introduced on a different axis (e.g., fixing gender bias didn't worsen caste bias)
3. Accuracy loss is within acceptable bounds

> **Why this matters:** ThresholdOptimizer can inadvertently worsen caste bias while fixing gender bias. Without re-validation, you trade one form of discrimination for another.

---

### Phase 9 — Monitoring: Watching for Problems After Deployment *(Enhanced)*

Bias can re-enter over time through a feedback loop — the model's own bad decisions pollute the next batch of training data, making the model worse with every version.

| Tool | What It Does | Library |
|------|-------------|---------|
| **Disparate impact trend tracking** | Every month, computes minority ÷ majority approval rate. Plots it over time with the 0.80 legal danger line. Catches drift early — before thousands of unfair decisions are made. | `evidently`, `plotly` |
| **Data distribution drift detection** | Compares incoming data distribution against training data each month. If the applicant pool suddenly shifts (80% from one region), all fairness metrics become unreliable until retraining. | `evidently` |
| **Feedback loop detection** | Records the model's decisions, then 3–6 months later records real-world outcomes. Correlates real outcomes against predictions, split by group. If the model's rejections make that group's real-world outcomes worse, the feedback loop is active. | Custom (lagged outcome correlation) |
| **Benign vs. harmful drift classification** *(NEW)* | Not all drift is bad. Economic changes naturally shift distributions. The system classifies drift as *benign* (affects all groups equally) or *harmful* (disproportionately affects protected groups) using per-group drift scores. Only harmful drift triggers alerts. | `evidently` + custom per-group analysis |
| **Automated alerting** *(NEW)* | When DI ratio crosses 0.80, or drift score exceeds threshold, or feedback loop is detected, the system sends alerts via email/Slack webhook. No one needs to watch the dashboard manually. | Webhook integration |
| **Audit log** *(NEW)* | Immutable log of every pipeline run: data version, model version, all fairness scores, mitigation applied, who approved deployment. Required for regulatory compliance — auditors need to trace exactly which model version produced which fairness score on which data. | `SQLAlchemy` + PostgreSQL |
| **Streamlit live dashboard** | Shows risk score, intersectional heatmap, disparate impact trend, Cramér's V per attribute, SHAP explanations, and active alerts — all in one screen. | `streamlit`, `plotly` |

---

## Dashboard Design — Built for Non-Technical Users

| Element | What It Shows |
|---------|--------------|
| **Top summary badge** | "Action needed" or "All clear." A manager knows within one second whether there is a problem. |
| **Four metric cards** | Each card: name in plain English, the number, and what it means in context. A non-technical person reads all four in 20 seconds. |
| **Plain language alerts** | Instead of "Demographic parity gap = 0.28 (p < 0.001)", says: *"The model is approving loans for men at 82% but for women at only 54%. That is a 28-point gap — it should be within 10 points."* |
| **"What does this mean?" buttons** | Every chart has a toggle for jargon-free explanations. Technical users ignore them. Managers, auditors, regulators use them. |
| **Root cause panel** *(NEW)* | Powered by SHAP. Shows: *"Bias is primarily driven by zip code (proxy for caste) and career gap years (penalises women)."* Answers the "why" question that the original dashboard couldn't. |
| **Approval rate bars with group switcher** | Switch between Gender, Caste, Region. Red bars are problem groups. |
| **"Is bias getting better or worse?" trend chart** | Answers one question: are we improving or getting worse? Dotted line is the legal danger zone. |
| **Blind spots heatmap** | Intersectional grid colour-coded green, amber, red. Labels replace formulas: "watch" and "urgent." |
| **Confidence indicators** *(NEW)* | Every metric shows its confidence level (high/medium/low) based on sample size and bootstrap CI width. Prevents acting on unreliable statistics. |
| **What should be done next** | Numbered priority list in plain language. Not "apply ThresholdOptimizer" — instead "Do not deploy this model yet. The legal safety number is below the minimum." |

---

## The Five Things That Make Our Solution Better

1. **Pareto Frontier** — Instead of "make it as fair as possible," we find the exact point where fairness and accuracy are both optimised simultaneously. Practical for organisations that cannot sacrifice accuracy.

2. **Intersectionality** — We check not just gender, not just caste — every combination. With statistical safeguards (Fisher's exact test, confidence intervals) to ensure results are reliable even for small subgroups.

3. **Explainability** *(NEW)* — We don't just say "there is bias." We say "bias exists because of these specific features, and here is how much each one contributes." This makes the system actionable, not just diagnostic.

4. **Label Audit** *(NEW)* — We verify the ground truth itself before trusting any metric. If the training labels encode historical discrimination, we detect it and correct it before proceeding.

5. **Feedback Loop Monitoring** — We track not just whether the model is fair today but whether it is becoming more or less fair over time, with automated alerts and audit trails for regulatory compliance.

---

## Complete Library Manifest

| Purpose | Library | Status |
|---------|---------|--------|
| Missing data visualisation | `missingno` | Original |
| Data manipulation | `pandas`, `numpy` | Original |
| Statistical tests | `scipy.stats` | Original |
| Machine learning | `scikit-learn` | Original |
| Oversampling | `imbalanced-learn` (SMOTE) | Original |
| Fairness metrics + mitigation | `fairlearn` | Original |
| Categorical dimensionality reduction | `prince` (MCA) | Original |
| Dashboard | `streamlit`, `plotly` | Original |
| Production monitoring | `evidently` | Original |
| **Explainability** | **`shap`** | **NEW** |
| **Causal inference (label audit)** | **`dowhy`** | **NEW** |
| **Configuration management** | **`pydantic-settings`** | **NEW** |
| **Audit logging** | **`sqlalchemy`** | **NEW** |

Every tool in this list is free, open source, and installable with a single `pip install` command.

---

## Types of Bias Covered

| Bias Type | How We Address It | Phase |
|-----------|-------------------|-------|
| **Historical bias** | Core focus — statistical detection + mitigation | 4, 8 |
| **Representation bias** | Group representation counts + SMOTE | 1, 8 |
| **Measurement bias** *(NEW)* | SHAP analysis reveals if features are measured differently across groups | 5 |
| **Label bias** *(NEW)* | Dedicated label audit with counterfactual checks | 3 |
| **Aggregation bias** *(NEW)* | Intersectional analysis + per-subgroup calibration | 4, 6 |
| **Proxy discrimination** | Cramér's V proxy detection + SHAP feature importance | 2, 5 |
| **Feedback loop bias** | Lagged outcome correlation monitoring | 9 |

---

## Regulatory Framework Mapping *(NEW)*

| Regulation | Relevant Metric | Our Coverage |
|------------|----------------|--------------|
| **US EEOC 4/5ths Rule** | Disparate Impact Ratio ≥ 0.80 | ✅ Phase 6 |
| **EU AI Act (High-risk AI)** | Transparency, human oversight, accuracy, non-discrimination | ✅ Explainability (Phase 5), Dashboard, Audit logs |
| **India IT Act / Digital India Act** | Non-discrimination, data protection | ✅ Intersectional analysis, data audit |
| **GDPR Art. 22** | Right to explanation for automated decisions | ✅ SHAP explanations (Phase 5) |

---

*Document Version: 2.0 — Corrected & Improved*
*All gaps from the original feasibility audit have been addressed.*
