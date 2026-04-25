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
        target = st.selectbox("Target column (what the model decides):", all_cols,
                              index=all_cols.index("approved") if "approved" in all_cols else 0)

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
        if df[target].nunique() > 2:
            median = df[target].median()
            df[target] = (df[target] > median).astype(int)
            results["target_binarised"] = f"Target binarised at median ({median})"
        else:
            df[target] = df[target].astype(int)

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

st.markdown(f"""
<div style="background:{col_badge}22;border:2px solid {col_badge};
            border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;
            display:flex;align-items:center;gap:1rem">
  <span style="font-size:1.3rem;font-weight:700;color:{col_badge}">{badge}</span>
</div>
""", unsafe_allow_html=True)

# ── Attribute selector controls ALL KPI cards ──────────────────────
st.markdown("### 🔎 Select attribute to inspect")
sel_col, info_col = st.columns([2, 3])
with sel_col:
    active_attr = st.selectbox(
        "Attribute:", sensitive, key="active_attr",
        help="All four metric cards update when you change this"
    )
with info_col:
    st.markdown(
        f"<div class='tip'>Cards 2–4 now show numbers specifically for "
        f"<b>{active_attr}</b>. Change the selector to compare across attributes. "
        f"Card 1 (Overall Risk) always reflects the worst case across all attributes.</div>",
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

kpi(k1, "Overall Risk Score", risk_score,
    "worst case · all attributes",
    "#ef4444" if risk_score>0.6 else "#f59e0b" if risk_score>0.3 else "#22c55e")
kpi(k2, f"Approval Gap", f"{attr_gap:.1f}%",
    f"{active_attr} · threshold: {threshold_gap}%",
    "#ef4444" if attr_gap>threshold_gap else "#22c55e")
kpi(k3, f"Legal Safety (4/5ths)", f"{attr_di:.3f}",
    f"{active_attr} · must be ≥ 0.800",
    "#ef4444" if attr_di<0.8 else "#22c55e")
kpi(k4, "Proxy Variables", n_proxies,
    "hidden backdoors · all attributes",
    "#ef4444" if n_proxies>0 else "#22c55e")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 Approval Rates","🔍 Intersectional","🚨 Proxy Variables",
    "📈 Risk Breakdown","✅ What To Do"
])

# Tab 1: Approval rates
with t1:
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

        c1,c2,c3 = st.columns(3)
        c1.metric("Approval Gap", f"{gap:.1f}%",
                  delta=f"{'Over limit' if gap>threshold_gap else 'OK'}",
                  delta_color="inverse" if gap>threshold_gap else "normal")
        c2.metric("Disparate Impact Ratio", f"{di_val:.3f}",
                  delta="Fails 4/5ths rule" if di_val<0.8 else "Passes",
                  delta_color="inverse" if di_val<0.8 else "normal")
        c3.metric("Cramér's V", f"{cv_val:.4f}",
                  delta="Above threshold" if cv_val>threshold_cv_r else "OK",
                  delta_color="inverse" if cv_val>threshold_cv_r else "normal")

        st.markdown(f"""<div class="tip">
        <b>Plain English:</b> For <b>{view_attr}</b>, the biggest approval gap is
        <b>{gap:.1f} percentage points</b>. The legal 4/5ths ratio is <b>{di_val:.3f}</b>
        (must be above 0.80). Cramér's V = {cv_val:.4f}
        (above {threshold_cv_r} = strong bias signal).
        </div>""", unsafe_allow_html=True)

    st.markdown("### Representation in Training Data")
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
            st.warning(f"⚠️ Underrepresented groups (below 10%): **{', '.join(str(x) for x in under.index)}** — the model barely learns from these groups.")

# Tab 2: Intersectional
with t2:
    st.markdown("### Blind Spots — Every Combination of Sensitive Attributes")
    st.markdown("""
    <div class="tip">A model can pass the gender test AND the caste test individually,
    but still systematically fail <b>Female + SC</b>. This table checks every combination.
    </div>""", unsafe_allow_html=True)
    idf = R["intersectional"]
    if not idf.empty:
        st.dataframe(
            idf.sort_values("Max Gap (%)", ascending=False).style
               .background_gradient(subset=["Max Gap (%)"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )
        worst = idf.loc[idf["Max Gap (%)"].idxmax()]
        st.error(f"**Worst blind spot:** `{worst['Intersection']}` — gap of **{worst['Max Gap (%)']:.1f}%**. "
                 f"Group '{worst['Worst Group']}' is severely disadvantaged.")
    elif len(sensitive)<2:
        st.info("Select at least 2 sensitive attributes to run intersectional analysis.")
    else:
        st.success("No intersectional issues found.")

# Tab 3: Proxy Variables
with t3:
    st.markdown("### Proxy Variable Detection")
    st.markdown("""
    <div class="tip">
    <b>What is a proxy?</b> Even after you remove a sensitive column like 'caste',
    other columns like zip_code, college_name, or first_name can secretly encode
    the same information and smuggle discrimination back into the model.
    Anything with Cramér's V above the threshold is a hidden backdoor.
    </div>""", unsafe_allow_html=True)
    pdf = R["proxies"]
    if not pdf.empty:
        st.dataframe(pdf.sort_values("Cramér's V",ascending=False), use_container_width=True, hide_index=True)
        for _,row in pdf.iterrows():
            cv_val_p = row["Cramér's V"]
            feat = row["Feature"]
            sens_a = row["Sensitive Attr"]
            st.error(f"🔴 **{feat}** is a proxy for **{sens_a}** "
                     f"(Cramér's V = {cv_val_p:.4f}). "
                     f"Removing '{sens_a}' from your model is NOT enough — "
                     f"'{feat}' will still smuggle that information in.")
    else:
        st.success("✅ No proxy variables detected above the threshold.")

    if R["missing"]:
        st.markdown("### Missing Data Gaps")
        st.markdown("These columns have significantly more missing data in some groups — a structural bias in how data was collected.")
        for k,v in R["missing"].items():
            st.warning(f"⚠️ **{k}**: {v:.1f}% missing data gap between groups")

# Tab 4: Risk Breakdown
with t4:
    st.markdown("### What Is Driving the Risk Score?")
    comp_data = {
        "Cramér's V (statistical bias)": min(max(R["cv"].values(),default=0),1.0),
        "Approval Gap": min(max(R["gaps"].values(),default=0)/100,1.0),
        "Disparate Impact": max(0,1-min(R["di"].values(),default=1.0)),
        "Representation Imbalance": min(
            sum(1 for series in R["representation"].values()
                for pct in list(series) if pct < 10)*0.15, 1.0),
    }
    comp_df = pd.DataFrame({
        "Component":list(comp_data.keys()),
        "Score":list(comp_data.values())
    }).sort_values("Score",ascending=True)
    fig3 = go.Figure(go.Bar(
        x=comp_df["Score"], y=comp_df["Component"], orientation="h",
        marker=dict(color=comp_df["Score"],colorscale=["#22c55e","#f59e0b","#ef4444"],
                    cmin=0,cmax=1),
        text=[f"{v:.3f}" for v in comp_df["Score"]], textposition="auto"
    ))
    fig3.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",xaxis_range=[0,1],height=300,
                       xaxis_title="Score (0=no bias, 1=max bias)")
    st.plotly_chart(fig3, use_container_width=True)

    # Per-attribute summary table
    rows=[]
    for attr in sensitive:
        if attr not in df.columns: continue
        rows.append({
            "Attribute":attr,
            "Cramér's V":R["cv"].get(attr,0),
            "Approval Gap (%)":R["gaps"].get(attr,0),
            "Disparate Impact":R["di"].get(attr,1.0),
            "Status":(
                "🔴 BIASED" if (R["gaps"].get(attr,0)>threshold_gap or
                                R["di"].get(attr,1.0)<0.8 or
                                R["cv"].get(attr,0)>threshold_cv_r)
                else "🟢 OK"
            )
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Tab 5: Recommendations
with t5:
    st.markdown("### What Should You Do?")
    recs=[]
    if risk_level=="HIGH":
        recs.append("🔴 **DO NOT DEPLOY this model.** Significant bias detected.")
    elif risk_level=="MEDIUM":
        recs.append("🟡 **Review required before deployment.** Moderate bias signals found.")
    else:
        recs.append("🟢 **Model passes basic fairness checks.** Continue monitoring after deployment.")

    for attr in sensitive:
        gap=R["gaps"].get(attr,0)
        di_val=R["di"].get(attr,1.0)
        cv_val=R["cv"].get(attr,0)
        if gap>threshold_gap:
            recs.append(f"📌 **{attr}**: Approval gap is **{gap:.1f}%** (above {threshold_gap}% limit). Apply ThresholdOptimizer to equalise approval rates.")
        if di_val<0.8:
            recs.append(f"⚖️ **{attr}**: Disparate Impact = **{di_val:.3f}** — fails the legal 4/5ths rule. This system is presumptively discriminatory for {attr}.")
        if cv_val>threshold_cv_r:
            recs.append(f"📊 **{attr}**: Cramér's V = **{cv_val:.4f}** — strong statistical bias signal. The model's decisions are not independent of {attr}.")

    if not R["proxies"].empty:
        proxy_feats = R["proxies"]["Feature"].tolist()
        recs.append(f"🔍 **Proxy variables detected:** {', '.join(proxy_feats)}. Remove or decorrelate these features before retraining.")

    for i,r in enumerate(recs,1):
        st.markdown(f"**{i}.** {r}")

    st.divider()
    st.markdown("### Verdicts by Attribute")
    for attr in sensitive:
        gap=R["gaps"].get(attr,0)
        di_val=R["di"].get(attr,1.0)
        cv_val=R["cv"].get(attr,0)
        biased = gap>threshold_gap or di_val<0.8 or cv_val>threshold_cv_r
        with st.expander(f"{'🔴' if biased else '🟢'} {attr}"):
            st.markdown(f"""
| Check | Value | Result |
|-------|-------|--------|
| Approval Gap | {gap:.1f}% | {'🔴 Fails' if gap>threshold_gap else '🟢 Passes'} (threshold: {threshold_gap}%) |
| Disparate Impact | {di_val:.3f} | {'🔴 Fails' if di_val<0.8 else '🟢 Passes'} (must be ≥ 0.80) |
| Cramér's V | {cv_val:.4f} | {'🔴 Flagged' if cv_val>threshold_cv_r else '🟢 OK'} (threshold: {threshold_cv_r}) |
""")
            rates = R["approval"].get(attr, pd.Series())
            if not rates.empty:
                best = rates.idxmax()
                worst = rates.idxmin()
                st.info(f"'{best}' is approved at **{rates.max():.1f}%** but '{worst}' at only **{rates.min():.1f}%**.")

st.divider()
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:.8rem'>"
    "⚖️ Unbias AI Decision — Works on any dataset, any sensitive attribute"
    "</div>", unsafe_allow_html=True
)
