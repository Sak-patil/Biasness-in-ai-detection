"""
Unbias AI Decision — Interactive Dashboard
Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Unbias AI", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }
.header-box {
    background: linear-gradient(135deg,#1a1a2e,#0f3460);
    padding:2rem; border-radius:16px; margin-bottom:1.5rem;
    border:1px solid rgba(255,255,255,0.1);
}
.header-box h1 { color:#e2e8f0; margin:0; font-size:1.8rem; }
.header-box p  { color:#94a3b8; margin:0.4rem 0 0; }
.kpi { background:#1e293b; border:1px solid rgba(255,255,255,0.08);
       border-radius:12px; padding:1.2rem; text-align:center; }
.kpi .label { color:#94a3b8; font-size:.75rem; text-transform:uppercase;
              letter-spacing:1px; margin-bottom:.4rem; }
.kpi .val   { font-size:1.8rem; font-weight:700; }
.kpi .sub   { color:#475569; font-size:.78rem; margin-top:.25rem; }
.tip { background:rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.3);
       border-radius:8px; padding:.9rem; color:#93c5fd; font-size:.88rem; }
.explain-box {
    background: linear-gradient(135deg, rgba(16,185,129,.08), rgba(59,130,246,.08));
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    color: #cbd5e1; font-size: .9rem; line-height: 1.6;
}
.explain-box b, .explain-box strong { color: #6ee7b7; }
.explain-box .simple { color: #fbbf24; font-weight: 600; font-size: .95rem; }
.explain-box .tech { color: #94a3b8; font-size: .82rem; font-style: italic; margin-top: .4rem; }
.next-step {
    background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.3);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    color: #fde68a; font-size: .9rem; line-height: 1.6;
}
.next-step b { color: #fbbf24; }
.metric-help {
    background: rgba(139,92,246,.08); border: 1px solid rgba(139,92,246,.25);
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    color: #c4b5fd; font-size: .85rem; line-height: 1.5;
}
.metric-help .plain { color: #e2e8f0; font-weight: 600; }
.metric-help .technical { color: #94a3b8; font-size: .8rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────
from scipy import stats

def cramers_v(x, y):
    if pd.api.types.is_numeric_dtype(x):
        x = pd.qcut(x, q=5, duplicates="drop")
    if pd.api.types.is_numeric_dtype(y):
        y = pd.qcut(y, q=5, duplicates="drop")
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    d = n * (min(r,k)-1)
    return float(np.sqrt(chi2/d)) if d>0 else 0.0

def auto_detect_sensitive(df):
    """Heuristic: flag columns likely to be sensitive attributes."""
    hints = ["gender","sex","caste","race","ethnicity","religion","region",
             "age","disability","nationality","zip","pincode","income",
             "education","marital","language","tribe"]
    found = []
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in hints):
            found.append(c)
    return found

def auto_detect_target(df):
    """Heuristic: pick the most likely binary outcome column."""
    preferred_names = ["hired", "approved", "approve", "decision", "target", "label", "outcome", "result"]

    for name in preferred_names:
        for c in df.columns:
            cl = c.lower()
            if cl == name or cl.endswith(f"_{name}") or name in cl:
                return c

    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if binary_cols:
        return binary_cols[-1]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]

    return df.columns[-1]

def prepare_binary_target(df, target):
    """Return a binary target series or raise a clear error for invalid targets."""
    series = df[target]
    non_null = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        if non_null.nunique() <= 2:
            return series.astype(int), None
        median = non_null.median()
        return (series > median).astype(int), f"Target binarised at median ({median})"

    if non_null.nunique() <= 2:
        encoded, uniques = pd.factorize(series)
        return pd.Series(encoded, index=series.index).astype(int), f"Target encoded from labels: {list(uniques)}"

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        median = numeric.median()
        return (numeric > median).astype(int), f"Target binarised at median ({median})"

    raise ValueError(
        f"Selected target '{target}' does not look like a binary outcome column. "
        "Choose a label such as hired/approved/decision, not an identifier like customer_id."
    )

def group_approval_rates(df, attr, target):
    return df.groupby(attr)[target].mean().mul(100).round(2)

def disparate_impact(df, attr, target):
    rates = group_approval_rates(df, attr, target)
    if rates.max() == 0: return 0.0
    return round(rates.min()/rates.max(), 4)

def dp_gap(df, attr, target):
    r = group_approval_rates(df, attr, target)
    return round(r.max()-r.min(), 2)

def intersectional_gaps(df, attrs, target):
    from itertools import combinations
    rows=[]
    available = [a for a in attrs if a in df.columns]
    for r in range(2, len(available)+1):
        for combo in combinations(available, r):
            combined = df[list(combo)].astype(str).agg(" × ".join, axis=1)
            rates = df.groupby(combined)[target].mean().mul(100)
            gap = round(rates.max()-rates.min(),2)
            rows.append({
                "Intersection":" × ".join(combo),
                "Max Gap (%)":gap,
                "Worst Group":rates.idxmin(),
                "Best Group":rates.idxmax(),
                "Status":"🔴 urgent" if gap>20 else "🟡 watch" if gap>10 else "🟢 ok"
            })
    return pd.DataFrame(rows)

def proxy_report(df, sensitive_attrs, target, threshold=0.3):
    rows=[]
    non_sens = [c for c in df.columns if c not in sensitive_attrs and c!=target]
    for attr in sensitive_attrs:
        if attr not in df.columns: continue
        for col in non_sens:
            try:
                v = cramers_v(df[attr].dropna(), df[col].dropna())
                if v >= threshold:
                    rows.append({"Feature":col,"Sensitive Attr":attr,
                                 "Cramér's V":round(v,4),"Risk":"🔴 Proxy"})
            except: pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def missing_gap(df, attr, col):
    g = df.groupby(attr)[col].apply(lambda x: x.isnull().mean()*100).round(2)
    return round(g.max()-g.min(),2) if len(g)>1 else 0.0

# ── SIDEBAR — DATA INPUT ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Unbias AI")
    st.markdown("---")

    mode = st.radio("Data source", ["Upload my CSV","Use demo dataset"])
    df_raw = None

    if mode == "Upload my CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    else:
        if st.button("Generate demo dataset"):
            from generate_demo_data import generate_demo_dataset
            df_raw = generate_demo_dataset(2000)
            st.session_state["demo_df"] = df_raw
        if "demo_df" in st.session_state:
            df_raw = st.session_state["demo_df"]
            st.success("Demo dataset ready (2,000 rows)")

    if df_raw is not None:
        st.markdown("---")
        st.markdown("### Column Setup")

        all_cols = df_raw.columns.tolist()
        default_target = auto_detect_target(df_raw)
        target = st.selectbox(
            "Target column (what the model decides):",
            all_cols,
            index=all_cols.index(default_target)
        )

        auto = auto_detect_sensitive(df_raw)
        sensitive = st.multiselect(
            "Sensitive attributes to audit:",
            [c for c in all_cols if c!=target],
            default=[c for c in auto if c!=target],
            help="Pick ANY columns that should not drive decisions — gender, age, race, zip code, religion, etc."
        )

        threshold_cv = st.slider("Cramér's V threshold (bias flag)", 0.1, 0.5, 0.3, 0.05)
        threshold_gap = st.slider("Max allowed approval gap (%)", 5, 30, 10, 5)

        run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    else:
        run = False
        target = None
        sensitive = []
        threshold_cv = 0.3
        threshold_gap = 10

# ── MAIN CONTENT ─────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>⚖️ Unbias AI Decision Dashboard</h1>
  <p>Upload <b>any</b> CSV, pick <b>any</b> columns as sensitive attributes, and instantly verify bias.</p>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.info("👈  Upload your CSV or generate the demo dataset in the sidebar to begin.")
    st.markdown("""
    ### What columns can be 'sensitive'?
    Bias can come from **any** demographic or contextual attribute:

    | Domain | Examples of Sensitive Columns |
    |--------|-------------------------------|
    | **Demographics** | gender, age, race, caste, ethnicity, religion, nationality |
    | **Geography** | region, zip_code, pincode, city, country |
    | **Socioeconomic** | income_bracket, education_level, employment_type |
    | **Health** | disability_status, chronic_illness |
    | **Identity** | marital_status, language, tribe |

    You are **not limited** to gender/caste/region. Pick every column that should NOT influence the model's decision.
    """)
    st.stop()

if not run and "results" not in st.session_state:
    st.info("👈  Configure columns in the sidebar, then click **Run Analysis**.")
    st.stop()

# ── RUN ANALYSIS ─────────────────────────────────────────────────
if run:
    if not sensitive:
        st.error("Please select at least one sensitive attribute.")
        st.stop()
    if target not in df_raw.columns:
        st.error("Target column not found.")
        st.stop()

    with st.spinner("Running full bias analysis…"):
        results = {}
        df = df_raw.copy()

        # Make target binary if needed
        try:
            df[target], target_note = prepare_binary_target(df, target)
            if target_note:
                results["target_binarised"] = target_note
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        # 1. Representation
        rep = {}
        for attr in sensitive:
            if attr in df.columns:
                vc = df[attr].value_counts(normalize=True)*100
                rep[attr] = vc.round(2)
        results["representation"] = rep

        # 2. Approval rates
        approval = {}
        for attr in sensitive:
            if attr in df.columns:
                approval[attr] = group_approval_rates(df, attr, target)
        results["approval"] = approval

        # 3. Disparate impact
        di = {}
        for attr in sensitive:
            if attr in df.columns:
                di[attr] = disparate_impact(df, attr, target)
        results["di"] = di

        # 4. DP gap
        gaps = {}
        for attr in sensitive:
            if attr in df.columns:
                gaps[attr] = dp_gap(df, attr, target)
        results["gaps"] = gaps

        # 5. Cramér's V
        cv = {}
        for attr in sensitive:
            if attr in df.columns:
                cv[attr] = round(cramers_v(df[attr], df[target]), 4)
        results["cv"] = cv

        # 6. Proxy detection
        results["proxies"] = proxy_report(df, sensitive, target, threshold_cv)

        # 7. Missing data gaps
        miss = {}
        for attr in sensitive:
            for col in df.columns:
                if col != attr and col != target:
                    g = missing_gap(df, attr, col)
                    if g > 15:
                        miss[f"{attr}→{col}"] = g
        results["missing"] = miss

        # 8. Intersectional
        if len(sensitive) >= 2:
            results["intersectional"] = intersectional_gaps(df, sensitive, target)
        else:
            results["intersectional"] = pd.DataFrame()

        # 9. Composite risk score
        max_gap_norm = min(max(gaps.values(), default=0)/100, 1.0)
        min_di_val   = min(di.values(), default=1.0)
        di_norm      = max(0, 1 - min_di_val)
        max_cv       = min(max(cv.values(), default=0), 1.0)
        underrep     = sum(1 for series in rep.values() for pct in list(series) if pct < 10)
        imb_norm     = min(underrep*0.15, 1.0)
        risk_score   = round(0.35*max_cv + 0.35*max_gap_norm + 0.20*di_norm + 0.10*imb_norm, 4)
        results["risk_score"] = risk_score
        results["risk_level"] = "HIGH" if risk_score>0.6 else "MEDIUM" if risk_score>0.3 else "LOW"

        st.session_state["results"] = results
        st.session_state["df"] = df
        st.session_state["sensitive"] = sensitive
        st.session_state["target"] = target
        st.session_state["threshold_gap"] = threshold_gap
        st.session_state["threshold_cv"] = threshold_cv

# ── DISPLAY RESULTS ───────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

R  = st.session_state["results"]
df = st.session_state["df"]
sensitive     = st.session_state["sensitive"]
target        = st.session_state["target"]
threshold_gap = st.session_state["threshold_gap"]
threshold_cv_r= st.session_state["threshold_cv"]

risk_score = R["risk_score"]
risk_level = R["risk_level"]

badge = {"HIGH":"🔴 HIGH RISK — DO NOT DEPLOY",
         "MEDIUM":"🟡 REVIEW REQUIRED",
         "LOW":"🟢 ALL CLEAR — MONITOR AFTER DEPLOYMENT"}[risk_level]
col_badge = {"HIGH":"#dc2626","MEDIUM":"#d97706","LOW":"#059669"}[risk_level]

badge_explain = {
    "HIGH": "This system is <b>unfairly treating</b> some groups of people. "
            "If you deploy this AI, it will discriminate — some people will get rejected "
            "not because of their qualifications, but because of <b>who they are</b> (their gender, race, age, etc.).",
    "MEDIUM": "There are <b>warning signs</b> of unfair treatment. The AI might be slightly favoring "
              "some groups over others. It needs a closer look before you can trust its decisions.",
    "LOW": "The AI appears to be treating all groups <b>roughly equally</b>. "
           "No major unfairness detected — but keep checking regularly, because bias can creep in over time."
}[risk_level]

st.markdown(f"""
<div style="background:{col_badge}22;border:2px solid {col_badge};
            border-radius:12px;padding:1rem 1.5rem;margin-bottom:0.5rem;
            display:flex;align-items:center;gap:1rem">
  <span style="font-size:1.3rem;font-weight:700;color:{col_badge}">{badge}</span>
</div>
<div class="explain-box">
  <span class="simple">💬 What does this mean?</span><br>
  {badge_explain}
  <div class="tech">Technical: Composite risk score = {risk_score:.4f} (0 = no bias, 1 = maximum bias). Threshold: LOW &lt; 0.30, MEDIUM 0.30–0.60, HIGH &gt; 0.60.</div>
</div>
""", unsafe_allow_html=True)

# ── "How to Read This Dashboard" guide ─────────────────────────────
with st.expander("📖 First time here? How to read this dashboard (click to expand)", expanded=False):
    st.markdown("""
**This dashboard checks one simple question: Is the AI treating everyone fairly?**

Here's what to look at, step by step:

| Step | What to check | Where to look |
|------|--------------|---------------|
| **1. Check the badge above** | 🟢 Green = Fair, 🟡 Yellow = Suspicious, 🔴 Red = Unfair | The colored banner at the top |
| **2. Read the 4 number cards below** | They tell you *how unfair* the AI is, and for *which group* | The four boxes right below this |
| **3. Pick different groups to inspect** | Use the dropdown to switch between gender, race, age, etc. | The "Attribute" selector below |
| **4. Check each tab for details** | Each tab digs deeper into one type of unfairness | The 5 tabs further below |
| **5. Read "What To Do"** | Clear next steps — what to fix and how | The last tab |

**Key idea:** An AI can be unfair even if nobody programmed it to be. It learns from historical data,
and if that data contains past discrimination (e.g., fewer women were hired in the past),
the AI will repeat those same patterns — unless we catch it here.
    """)

# ── Attribute selector controls ALL KPI cards ──────────────────────
st.markdown("### 🔎 Select a group to inspect")
sel_col, info_col = st.columns([2, 3])
with sel_col:
    active_attr = st.selectbox(
        "Which group do you want to check?", sensitive, key="active_attr",
        help="Pick a sensitive attribute like gender, race, or age — the numbers below will update to show if the AI treats that group fairly"
    )
with info_col:
    st.markdown(
        f"<div class='tip'>📊 The number cards below now show fairness results for "
        f"<b>{active_attr}</b>. Try switching to other groups to compare. "
        f"Card 1 (Overall Risk) always shows the worst result across <i>all</i> groups.</div>",
        unsafe_allow_html=True
    )

# Per-attribute values — these CHANGE when you switch the selector
attr_gap  = R["gaps"].get(active_attr, 0)
attr_di   = R["di"].get(active_attr, 1.0)
attr_cv   = R["cv"].get(active_attr, 0)
n_proxies = len(R["proxies"])

k1,k2,k3,k4 = st.columns(4)
def kpi(col, label, val, sub, color="#e2e8f0"):
    col.markdown(f"""<div class="kpi">
        <div class="label">{label}</div>
        <div class="val" style="color:{color}">{val}</div>
        <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

# Card 1: Overall Risk
risk_plain = ("The AI is seriously biased" if risk_score > 0.6
              else "Some bias detected — needs review" if risk_score > 0.3
              else "Looks fair — keep monitoring")
kpi(k1, "Overall Risk Score", risk_score,
    risk_plain,
    "#ef4444" if risk_score>0.6 else "#f59e0b" if risk_score>0.3 else "#22c55e")

# Card 2: Approval Gap
gap_plain = (f"Some {active_attr} groups are approved {attr_gap:.0f}% more than others — that's too much"
             if attr_gap > threshold_gap
             else f"Groups within {active_attr} are treated similarly ✓")
kpi(k2, f"Approval Gap ({active_attr})", f"{attr_gap:.1f}%",
    gap_plain,
    "#ef4444" if attr_gap>threshold_gap else "#22c55e")

# Card 3: Legal Safety
di_plain = ("⚠ Fails the legal fairness test — could be considered discriminatory"
            if attr_di < 0.8
            else "Passes the legal fairness test ✓")
kpi(k3, f"Legal Safety ({active_attr})", f"{attr_di:.3f}",
    di_plain,
    "#ef4444" if attr_di<0.8 else "#22c55e")

# Card 4: Proxy Variables
proxy_plain = (f"{n_proxies} hidden shortcut(s) found — AI may be using indirect clues to discriminate"
               if n_proxies > 0
               else "No hidden shortcuts found ✓")
kpi(k4, "Hidden Shortcuts (Proxies)", n_proxies,
    proxy_plain,
    "#ef4444" if n_proxies>0 else "#22c55e")

# ── Plain-language explainers under the KPI row ──────────────────
with st.expander("🤔 What do these 4 numbers actually mean? (click to understand)", expanded=False):
    st.markdown(f"""
| Card | Simple Explanation | Technical Detail |
|------|-------------------|-----------------|
| **Overall Risk Score** | A single number from 0 to 1 that combines all the checks below. **Think of it like a health checkup score** — higher = more problems found. | Weighted composite: 35% statistical association + 35% approval gap + 20% legal ratio + 10% data imbalance |
| **Approval Gap** | If the AI approves 80% of Men but only 55% of Women, the gap is 25%. **A big gap = some people are unfairly rejected.** Currently for **{active_attr}**: the gap is **{attr_gap:.1f}%** (your limit is {threshold_gap}%). | Demographic Parity Gap = max(group approval rates) − min(group approval rates) |
| **Legal Safety** | Governments use the "4/5ths rule" — if the lowest group's approval rate is less than 80% of the highest group's rate, **it's legally considered discriminatory.** Currently: **{attr_di:.3f}** (must be ≥ 0.800). | Disparate Impact Ratio = min(group rate) ÷ max(group rate). Below 0.800 = presumptive adverse impact under EEOC guidelines. |
| **Hidden Shortcuts** | Even if you remove "gender" from the data, the AI might use zip code, university name, or other columns as a **secret shortcut** to guess gender anyway. These are called proxies. Currently: **{n_proxies}** found. | Proxy detection via Cramér's V — measures statistical association between non-sensitive features and sensitive attributes. Threshold: {threshold_cv_r}. |
    """)

st.divider()

# ── Tab layout ────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 Who Gets Approved?","🔍 Hidden Blind Spots","🚨 Secret Shortcuts",
    "📈 Where Is the Bias Coming From?","✅ What Should You Do Next?"
])

# ═══════════════════════════════════════════════════════════════════
# Tab 1: Approval rates — "Who Gets Approved?"
# ═══════════════════════════════════════════════════════════════════
with t1:
    st.markdown("""<div class="explain-box">
    <span class="simple">💡 What this tab shows:</span><br>
    This chart compares <b>how often different groups get a positive outcome</b>
    (hired, approved, selected, etc.). If one group's bar is much shorter than another's,
    the AI is favoring one group over the other — that's unfair, regardless of intent.
    </div>""", unsafe_allow_html=True)

    view_attr = active_attr   # synced to the main selector above
    if view_attr in R["approval"]:
        rate_series = R["approval"][view_attr]
        rdf = rate_series.reset_index()
        rdf.columns = ["Group","Approval Rate (%)"]
        rdf = rdf.sort_values("Approval Rate (%)")
        fig = px.bar(rdf, x="Approval Rate (%)", y="Group", orientation="h",
                     color="Approval Rate (%)",
                     color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                     range_color=[0,100], height=350)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          xaxis_range=[0,100])
        fig.add_vline(x=50, line_dash="dash", line_color="#475569")
        st.plotly_chart(fig, use_container_width=True)

        gap = R["gaps"].get(view_attr,0)
        di_val = R["di"].get(view_attr,1.0)
        cv_val = R["cv"].get(view_attr,0)

        # Find best and worst groups for this attribute
        best_group = rate_series.idxmax()
        worst_group = rate_series.idxmin()
        best_rate = rate_series.max()
        worst_rate = rate_series.min()

        c1,c2,c3 = st.columns(3)
        c1.metric("Approval Gap", f"{gap:.1f}%",
                  delta=f"{'⚠ Over limit' if gap>threshold_gap else '✓ Within limit'}",
                  delta_color="inverse" if gap>threshold_gap else "normal")
        c2.metric("Legal Safety (4/5ths Rule)", f"{di_val:.3f}",
                  delta="⚠ Fails legal test" if di_val<0.8 else "✓ Passes",
                  delta_color="inverse" if di_val<0.8 else "normal")
        c3.metric("Bias Strength (Cramér's V)", f"{cv_val:.4f}",
                  delta="⚠ Strong bias signal" if cv_val>threshold_cv_r else "✓ Weak/none",
                  delta_color="inverse" if cv_val>threshold_cv_r else "normal")

        # ── Plain-language output interpretation ──
        st.markdown(f"""<div class="explain-box">
        <span class="simple">📋 What the output tells you:</span><br>
        For <b>{view_attr}</b>, the group "<b>{best_group}</b>" gets approved at <b>{best_rate:.1f}%</b>,
        but "<b>{worst_group}</b>" gets approved at only <b>{worst_rate:.1f}%</b>.
        That's a gap of <b>{gap:.1f} percentage points</b>.
        {"<br><br>🔴 <b>This gap is too large.</b> If 100 people from each group applied with the same qualifications, roughly " + str(int(gap)) + " more people from '" + str(best_group) + "' would be approved just because of their " + str(view_attr) + "." if gap > threshold_gap else "<br><br>🟢 <b>The gap is within acceptable limits.</b> The AI appears to treat different " + str(view_attr) + " groups roughly equally."}
        <div class="tech">Technical: Demographic Parity Gap = {gap:.2f}%, Disparate Impact Ratio = {di_val:.4f}
        (4/5ths rule threshold = 0.800), Cramér's V = {cv_val:.4f} (threshold = {threshold_cv_r})</div>
        </div>""", unsafe_allow_html=True)

        # ── What to do next ──
        if gap > threshold_gap or di_val < 0.8:
            st.markdown(f"""<div class="next-step">
            <b>⏭ What should you do next?</b><br>
            • <b>Don't use this AI for {view_attr}-related decisions</b> until the gap is fixed.<br>
            • Check if the training data has an equal number of examples from each {view_attr} group — imbalanced data is the #1 cause.<br>
            • Review the <b>"Secret Shortcuts"</b> tab — the AI might be using another column to guess {view_attr} indirectly.<br>
            • Go to the <b>"What Should You Do Next?"</b> tab for detailed fix instructions.
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🥧 Who is in the training data?")
    st.markdown("""<div class="metric-help">
    <span class="plain">Why does this matter?</span>
    An AI learns from examples. If it saw 1,000 examples of Group A but only 50 examples of Group B,
    it barely knows Group B exists — and will make worse decisions for them.
    <div class="technical">This shows the percentage representation of each group in the training dataset.
    Groups below 10% are flagged as underrepresented.</div>
    </div>""", unsafe_allow_html=True)

    if view_attr in R["representation"]:
        rep_s = R["representation"][view_attr]
        rep_df = rep_s.reset_index()
        rep_df.columns = ["Group","Share (%)"]
        fig2 = px.pie(rep_df, values="Share (%)", names="Group",
                      title=f"Who is in the training data? ({view_attr})",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        under = rep_s[rep_s<10]
        if not under.empty:
            under_names = ', '.join(str(x) for x in under.index)
            st.warning(f"⚠️ **Underrepresented groups (below 10%): {under_names}** — the AI has barely seen these groups, so its decisions for them are unreliable.")
            st.markdown(f"""<div class="next-step">
            <b>⏭ What to do about this:</b> Collect more data for {under_names}, or use techniques like
            oversampling (creating synthetic examples) to balance the training set.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 2: Intersectional — "Hidden Blind Spots"
# ═══════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### 🔍 Hidden Blind Spots — Checking Every Combination")
    st.markdown("""<div class="explain-box">
    <span class="simple">💡 What this tab shows:</span><br>
    Sometimes an AI can pass the fairness test for <b>gender</b> and pass for <b>race</b> —
    but still be unfair to a specific combination like <b>"older Black women"</b> or <b>"young rural men"</b>.
    <br><br>
    <b>Real-world example:</b> A hiring AI might hire men at 60% and women at 58% — looks fair!
    And it might hire White candidates at 62% and Black candidates at 57% — still close.
    But when you check <b>"Black women"</b> specifically, their rate drops to 35%.
    That's a blind spot that only shows up when you check combinations.
    <div class="tech">Technical: Intersectional analysis computes approval rates for every combination of
    sensitive attributes and reports the maximum gap within each intersection.</div>
    </div>""", unsafe_allow_html=True)

    idf = R["intersectional"]
    if not idf.empty:
        st.dataframe(
            idf.sort_values("Max Gap (%)", ascending=False).style
               .background_gradient(subset=["Max Gap (%)"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )

        # ── Plain output interpretation ──
        worst = idf.loc[idf["Max Gap (%)"].idxmax()]
        urgent_count = len(idf[idf["Status"].str.contains("urgent")])
        watch_count = len(idf[idf["Status"].str.contains("watch")])

        st.markdown(f"""<div class="explain-box">
        <span class="simple">📋 What the output tells you:</span><br>
        The <b>worst blind spot</b> is in the combination <b>{worst['Intersection']}</b>,
        where the gap between the best-treated and worst-treated group is <b>{worst['Max Gap (%)']:.1f}%</b>.
        <br><br>
        Specifically, the group "<b>{worst['Worst Group']}</b>" is being significantly disadvantaged
        compared to "<b>{worst['Best Group']}</b>".
        <br><br>
        🔴 <b>{urgent_count} combination(s)</b> need urgent attention (gap > 20%).
        🟡 <b>{watch_count} combination(s)</b> should be monitored (gap > 10%).
        <div class="tech">Technical: Each row represents a unique intersection of sensitive attributes.
        Status flags: 🔴 urgent (gap > 20%), 🟡 watch (gap > 10%), 🟢 ok (gap ≤ 10%).</div>
        </div>""", unsafe_allow_html=True)

        if urgent_count > 0:
            st.markdown(f"""<div class="next-step">
            <b>⏭ What should you do next?</b><br>
            • The combination "{worst['Intersection']}" has a severe fairness problem.<br>
            • Check if the group "{worst['Worst Group']}" has enough training data — they might be underrepresented.<br>
            • This kind of discrimination is the hardest to catch and the most damaging — regulators increasingly look for it.<br>
            • Consider retraining the model with fairness constraints applied to these specific intersections.
            </div>""", unsafe_allow_html=True)
    elif len(sensitive)<2:
        st.info("👆 Select at least 2 sensitive attributes (e.g., gender AND race) in the sidebar to run this analysis.")
    else:
        st.success("✅ No major blind spots found — the AI treats all combinations roughly equally.")

# ═══════════════════════════════════════════════════════════════════
# Tab 3: Proxy Variables — "Secret Shortcuts"
# ═══════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 🚨 Secret Shortcuts — Does the AI Discriminate Indirectly?")
    st.markdown("""<div class="explain-box">
    <span class="simple">💡 What this tab shows:</span><br>
    Imagine you tell the AI: "Don't look at gender when making decisions."
    The AI says "OK" — but then it notices that <b>zip code</b> can predict gender with 85% accuracy.
    So it starts using zip code as a <b>secret shortcut</b> to guess gender and discriminate anyway.
    <br><br>
    <b>Common examples:</b><br>
    • <b>Zip code / Pincode</b> → can predict race, caste, or income level<br>
    • <b>University / College name</b> → can predict caste or socioeconomic status<br>
    • <b>First name</b> → can predict gender or ethnicity<br>
    • <b>Browser type or phone model</b> → can predict income bracket
    <div class="tech">Technical: Proxy detection uses Cramér's V — a statistical measure of association
    (0 = no connection, 1 = perfect connection) between each non-sensitive feature and each sensitive attribute.
    Features above the threshold ({threshold_cv_r}) are flagged as proxies.</div>
    </div>""", unsafe_allow_html=True)

    pdf = R["proxies"]
    if not pdf.empty:
        st.dataframe(pdf.sort_values("Cramér's V",ascending=False), use_container_width=True, hide_index=True)

        # ── Plain output interpretation ──
        st.markdown("""<div class="explain-box">
        <span class="simple">📋 What the output tells you:</span>
        </div>""", unsafe_allow_html=True)

        for _,row in pdf.iterrows():
            cv_val_p = row["Cramér's V"]
            feat = row["Feature"]
            sens_a = row["Sensitive Attr"]
            strength = "very strongly" if cv_val_p > 0.6 else "strongly" if cv_val_p > 0.4 else "moderately"

            st.markdown(f"""<div class="explain-box" style="border-color: rgba(239,68,68,.4); background: rgba(239,68,68,.06);">
            🔴 <b>"{feat}"</b> is {strength} connected to <b>"{sens_a}"</b>.
            <br><br>
            <span class="simple">What this means:</span> Even if you remove "{sens_a}" from the AI's input,
            it can still figure out someone's {sens_a} by looking at their "{feat}" value.
            <b>Removing {sens_a} alone does NOT make the AI fair.</b>
            <div class="tech">Technical: Cramér's V = {cv_val_p:.4f} (threshold: {threshold_cv_r}).
            This feature has a statistically significant association with the protected attribute.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="next-step">
        <b>⏭ What should you do next?</b><br>
        • <b>Don't just remove the sensitive column</b> — you must also handle its proxies.<br>
        • Options: (a) Remove the proxy feature too, (b) Add noise to break the connection,
          or (c) Use fairness-aware training that accounts for proxies.<br>
        • The features listed above should be reviewed by someone who understands the domain
          (e.g., an HR expert for hiring data, a loan officer for lending data).
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No secret shortcuts detected — the non-sensitive features don't appear to encode sensitive information.")
        st.markdown("""<div class="explain-box">
        <span class="simple">📋 What this means:</span><br>
        The AI doesn't seem to be using any indirect clues to guess someone's gender, race, age, etc.
        This is a good sign — but it doesn't guarantee complete fairness (check the other tabs too).
        </div>""", unsafe_allow_html=True)

    if R["missing"]:
        st.markdown("### 📉 Missing Data Gaps")
        st.markdown("""<div class="explain-box">
        <span class="simple">💡 What this shows:</span><br>
        If certain groups have <b>more missing information</b> than others, the AI gets a
        <b>clearer picture of some people and a blurry picture of others</b>.
        For example, if income data is missing for 30% of women but only 5% of men,
        the AI knows men better and makes worse guesses about women.
        <div class="tech">Technical: Shows features where the percentage of missing (null) values differs
        by more than 15 percentage points between groups of a sensitive attribute.</div>
        </div>""", unsafe_allow_html=True)

        for k,v in R["missing"].items():
            parts = k.split("→")
            st.warning(f"⚠️ **{parts[1].strip()}** has **{v:.1f}%** more missing data in some **{parts[0].strip()}** groups than others — the AI knows some groups better than others.")

        st.markdown("""<div class="next-step">
        <b>⏭ What should you do next?</b><br>
        • Investigate <b>why</b> the data is missing — is it a data collection problem?<br>
        • Don't just fill in missing values with averages — that can hide real differences.<br>
        • Consider collecting more complete data from the underrepresented groups.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: Risk Breakdown — "Where Is the Bias Coming From?"
# ═══════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 📈 Where Is the Bias Coming From?")
    st.markdown("""<div class="explain-box">
    <span class="simple">💡 What this tab shows:</span><br>
    The overall risk score is made up of several ingredients. This chart shows
    <b>which ingredient is contributing the most</b> to the bias problem.
    Think of it like a doctor telling you "your cholesterol is the main issue" — so you
    know what to fix first.
    <div class="tech">Technical: The composite risk score is a weighted sum of 4 normalised components:
    Cramér's V (35%), Approval Gap (35%), Disparate Impact deviation (20%), and Representation Imbalance (10%).
    Each component is scored 0–1.</div>
    </div>""", unsafe_allow_html=True)

    comp_data = {
        "How strongly decisions depend\non protected groups": min(max(R["cv"].values(),default=0),1.0),
        "Gap in approval rates\nbetween groups": min(max(R["gaps"].values(),default=0)/100,1.0),
        "Legal fairness test\nfailure degree": max(0,1-min(R["di"].values(),default=1.0)),
        "Unequal group sizes\nin training data": min(
            sum(1 for series in R["representation"].values()
                for pct in list(series) if pct < 10)*0.15, 1.0),
    }
    # Also store original technical labels for the table below
    comp_tech_labels = {
        "How strongly decisions depend\non protected groups": "Cramér's V (statistical association)",
        "Gap in approval rates\nbetween groups": "Demographic Parity Gap",
        "Legal fairness test\nfailure degree": "1 − Disparate Impact Ratio",
        "Unequal group sizes\nin training data": "Representation Imbalance",
    }
    comp_df = pd.DataFrame({
        "What's causing the bias?":list(comp_data.keys()),
        "How bad is it (0=none, 1=worst)":list(comp_data.values())
    }).sort_values("How bad is it (0=none, 1=worst)",ascending=True)
    fig3 = go.Figure(go.Bar(
        x=comp_df["How bad is it (0=none, 1=worst)"],
        y=comp_df["What's causing the bias?"], orientation="h",
        marker=dict(color=comp_df["How bad is it (0=none, 1=worst)"],
                    colorscale=["#22c55e","#f59e0b","#ef4444"],
                    cmin=0,cmax=1),
        text=[f"{v:.3f}" for v in comp_df["How bad is it (0=none, 1=worst)"]],
        textposition="auto"
    ))
    fig3.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",xaxis_range=[0,1],height=350,
                       xaxis_title="Score (0 = no problem, 1 = severe problem)",
                       margin=dict(l=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Plain output interpretation ──
    worst_component = max(comp_data, key=comp_data.get)
    worst_score = comp_data[worst_component]
    st.markdown(f"""<div class="explain-box">
    <span class="simple">📋 What the output tells you:</span><br>
    The <b>biggest contributor</b> to the bias risk is: <b>"{worst_component.replace(chr(10), ' ')}"</b>
    with a score of <b>{worst_score:.3f}</b> out of 1.0.
    {"<br><br>🔴 This is a significant problem and should be the top priority to fix." if worst_score > 0.5 else "<br><br>🟡 This is moderate — worth investigating but not critical." if worst_score > 0.2 else "<br><br>🟢 All components are low — the overall risk is minimal."}
    <div class="tech">Technical: Worst component = {comp_tech_labels[worst_component]} ({worst_score:.4f}).
    The composite risk score weights these: Cramér's V × 0.35 + Gap × 0.35 + DI_deviation × 0.20 + Imbalance × 0.10 = {risk_score:.4f}.</div>
    </div>""", unsafe_allow_html=True)

    # Per-attribute summary table
    st.markdown("### 📋 Summary for Each Group")
    st.markdown("""<div class="metric-help">
    <span class="plain">How to read this table:</span>
    Each row is one of the sensitive attributes you selected. The "Status" column gives you a quick
    verdict — 🔴 means the AI is biased for that group, 🟢 means it looks fair.
    <div class="technical">Cramér's V = strength of statistical association. Approval Gap = max difference in
    positive outcome rates. Disparate Impact = minority rate ÷ majority rate.</div>
    </div>""", unsafe_allow_html=True)

    rows=[]
    for attr in sensitive:
        if attr not in df.columns: continue
        a_gap = R["gaps"].get(attr,0)
        a_di = R["di"].get(attr,1.0)
        a_cv = R["cv"].get(attr,0)
        biased = a_gap>threshold_gap or a_di<0.8 or a_cv>threshold_cv_r
        rows.append({
            "Group Checked":attr,
            "Bias Strength (Cramér's V)":a_cv,
            "Approval Gap (%)":a_gap,
            "Legal Safety Score":a_di,
            "Verdict":("🔴 UNFAIR — Needs fixing" if biased else "🟢 FAIR — Looks OK")
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 5: Recommendations — "What Should You Do Next?"
# ═══════════════════════════════════════════════════════════════════
with t5:
    st.markdown("### ✅ What Should You Do Next?")

    # ── Overall verdict in plain language ──
    if risk_level=="HIGH":
        st.markdown("""<div class="explain-box" style="border-color: rgba(239,68,68,.4); background: rgba(239,68,68,.06);">
        <span class="simple">🔴 STOP — This AI system is NOT safe to use right now.</span><br><br>
        The analysis found <b>significant unfairness</b>. If you use this AI to make decisions about people
        (hiring, lending, admissions, etc.), it will <b>systematically disadvantage</b> certain groups.
        This could harm real people and expose your organization to legal risk.
        <br><br><b>Do not deploy this model until the issues below are fixed.</b>
        </div>""", unsafe_allow_html=True)
    elif risk_level=="MEDIUM":
        st.markdown("""<div class="explain-box" style="border-color: rgba(251,191,36,.4); background: rgba(251,191,36,.06);">
        <span class="simple">🟡 CAUTION — This AI has some fairness concerns.</span><br><br>
        The analysis found <b>moderate warning signs</b>. The AI isn't severely biased, but there are
        gaps that could grow over time or affect specific groups. Review the specific issues below
        before deciding whether to use this system.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="explain-box" style="border-color: rgba(34,197,94,.4); background: rgba(34,197,94,.06);">
        <span class="simple">🟢 LOOKS GOOD — The AI passes basic fairness checks.</span><br><br>
        No major unfairness detected. But fairness is not a one-time check — AI systems can develop
        bias over time as new data comes in. <b>Set up regular monitoring</b> (monthly or quarterly)
        to catch any emerging issues early.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📝 Specific Issues Found & What to Do")

    issue_num = 0
    for attr in sensitive:
        gap=R["gaps"].get(attr,0)
        di_val=R["di"].get(attr,1.0)
        cv_val=R["cv"].get(attr,0)

        if gap>threshold_gap:
            issue_num += 1
            rates = R["approval"].get(attr, pd.Series())
            best_g = rates.idxmax() if not rates.empty else "?"
            worst_g = rates.idxmin() if not rates.empty else "?"
            st.markdown(f"""
**Issue {issue_num}: Unequal approval rates for {attr}** 🔴

| What we found | Details |
|---|---|
| **The problem** | "{best_g}" gets approved much more often than "{worst_g}" — a gap of **{gap:.1f}%** |
| **Why it matters** | If you're using this AI to decide who gets hired/approved, people from the "{worst_g}" group are being unfairly rejected |
| **What to do** | 1) Check if your training data has equal representation from all {attr} groups. 2) Look at the "Secret Shortcuts" tab to see if other features are encoding {attr} indirectly. 3) Consider retraining with fairness constraints. |

*Technical: Demographic Parity Gap = {gap:.2f}% (threshold: {threshold_gap}%)*
            """)

        if di_val<0.8:
            issue_num += 1
            st.markdown(f"""
**Issue {issue_num}: Fails legal fairness standard for {attr}** ⚖️

| What we found | Details |
|---|---|
| **The problem** | The legal safety score is **{di_val:.3f}** — it must be at least **0.800** to be considered fair |
| **Why it matters** | Under the "4/5ths rule" used by governments (US EEOC, EU AI Act), this AI would be considered **discriminatory** against certain {attr} groups. This means legal liability. |
| **What to do** | 1) This is the most serious type of finding — consult your legal/compliance team. 2) The AI needs to be retrained with fairness constraints before any deployment. 3) Document this finding for audit purposes. |

*Technical: Disparate Impact Ratio = {di_val:.4f} (legal threshold: 0.800, per EEOC Uniform Guidelines)*
            """)

        if cv_val>threshold_cv_r:
            issue_num += 1
            st.markdown(f"""
**Issue {issue_num}: AI decisions are linked to {attr}** 📊

| What we found | Details |
|---|---|
| **The problem** | The AI's decisions are statistically connected to {attr} — meaning {attr} is influencing who gets approved/rejected |
| **Why it matters** | Decisions should be based on qualifications, not on {attr}. A strong connection means the AI has "learned" to treat groups differently. |
| **What to do** | 1) Check which features in the data are correlated with {attr} (see "Secret Shortcuts" tab). 2) Remove or modify those features. 3) Retrain and retest. |

*Technical: Cramér's V = {cv_val:.4f} (threshold: {threshold_cv_r}). Scale: 0 = no association, 1 = perfect association.*
            """)

    if not R["proxies"].empty:
        issue_num += 1
        proxy_feats = R["proxies"]["Feature"].tolist()
        proxy_details = []
        for _,row in R["proxies"].iterrows():
            cv_proxy = row["Cramér's V"]
            proxy_details.append(f"• **{row['Feature']}** is a shortcut for **{row['Sensitive Attr']}** (connection strength: {cv_proxy:.3f})")
        st.markdown(f"""
**Issue {issue_num}: Hidden shortcuts detected** 🔍

The AI is using these features as indirect ways to discriminate:

{chr(10).join(proxy_details)}

**What to do:** Simply removing the sensitive column (like "gender") is NOT enough.
You must also remove or modify these shortcut features, otherwise the AI will
still discriminate — just through a back door.

*Technical: Proxy features detected via Cramér's V above threshold {threshold_cv_r}.*
        """)

    if issue_num == 0:
        st.success("✅ No specific issues found! The AI appears to treat all checked groups fairly.")

    st.divider()
    st.markdown("### 🔍 Detailed Verdict for Each Group")
    st.markdown("""<div class="metric-help">
    <span class="plain">Click on any group below to see its detailed fairness report card.</span>
    </div>""", unsafe_allow_html=True)

    for attr in sensitive:
        gap=R["gaps"].get(attr,0)
        di_val=R["di"].get(attr,1.0)
        cv_val=R["cv"].get(attr,0)
        biased = gap>threshold_gap or di_val<0.8 or cv_val>threshold_cv_r
        with st.expander(f"{'🔴 UNFAIR' if biased else '🟢 FAIR'} — {attr}"):
            st.markdown(f"""
| What we checked | Result | Simple Explanation |
|--------|--------|----------|
| Approval Gap | {gap:.1f}% {'🔴' if gap>threshold_gap else '🟢'} | {"Different " + attr + " groups have very different approval rates — that's unfair" if gap>threshold_gap else "All " + attr + " groups get approved at similar rates — looks fair"} |
| Legal Safety | {di_val:.3f} {'🔴' if di_val<0.8 else '🟢'} | {"Fails the government's fairness standard (4/5ths rule) — legally risky" if di_val<0.8 else "Passes the government's fairness standard — legally safe"} |
| Bias Strength | {cv_val:.4f} {'🔴' if cv_val>threshold_cv_r else '🟢'} | {"The AI's decisions are noticeably linked to " + attr + " — it shouldn't be" if cv_val>threshold_cv_r else "No strong connection between " + attr + " and the AI's decisions — good"} |
""")
            rates = R["approval"].get(attr, pd.Series())
            if not rates.empty:
                best = rates.idxmax()
                worst = rates.idxmin()
                st.info(f"**In plain numbers:** '{best}' is approved at **{rates.max():.1f}%** while '{worst}' is approved at only **{rates.min():.1f}%**.")

st.divider()
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:.8rem'>"
    "⚖️ Unbias AI Decision — Making AI fairness understandable for everyone"
    "</div>", unsafe_allow_html=True
)

