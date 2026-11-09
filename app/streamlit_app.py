"""Streamlit app. Run with: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from jobchange import config as C
from jobchange.predict import explain_candidate, load_bundle, score

st.set_page_config(page_title="Smart Recruitment Assistant", layout="wide")

PRIMARY, ORANGE, GREY = "#2563eb", "#f97316", "#94a3b8"


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_metrics():
    return json.loads(C.METRICS_PATH.read_text())


@st.cache_data
def get_demo_candidates():
    return pd.read_csv(C.TEST_CSV)


@st.cache_data
def get_cities():
    return pd.read_csv(C.TRAIN_CSV, usecols=["city"])["city"].value_counts().index.tolist()


CITIES = get_cities()

if not C.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m jobchange.train` first (see README).")
    st.stop()

bundle, metrics = get_bundle(), get_metrics()
H = metrics["holdout"]

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Smart Recruitment Assistant")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Score a candidate", "Batch screening", "Model insights"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    f"Model: **{metrics['selected_model']}**  \n"
    f"Decision threshold: **{bundle['threshold']:.2f}**  \n"
    f"Trained on {metrics['n_train']:,} candidates"
)


def gauge(value: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PRIMARY},
            "steps": [{"range": [0, threshold * 100], "color": "#e2e8f0"},
                      {"range": [threshold * 100, 100], "color": "#fed7aa"}],
            "threshold": {"line": {"color": ORANGE, "width": 4}, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ----------------------------------------------------------------------------- overview
if page == "Overview":
    st.title("Smart Recruitment Assistant")
    st.markdown(
        "Ranks candidates by how likely they are to be **looking for a new job**, so recruiters can focus on "
        "the people most likely to be open to a move. Trained on the HR Analytics: Job Change of Data "
        "Scientists dataset (19,158 candidates)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (hold-out)", f"{H['roc_auc']:.3f}")
    c2.metric("Recall", f"{H['recall']:.0%}", help="Share of real job-seekers the model catches.")
    c3.metric("Precision", f"{H['precision']:.0%}", help="Share of flagged candidates who really are looking.")
    c4.metric("Lift, top 10%", f"{metrics['lift_top_decile']:.1f}x",
              help="How much better than random the top-decile of ranked candidates performs.")

    st.subheader("How it works")
    a, b, c = st.columns(3)
    a.markdown("**1. Clean and engineer**  \nOrdinal encoding of education and company size, parsing of `>20`, `<1` "
               "and `never`, plus a missing-employer flag.")
    b.markdown("**2. Compare models**  \nLogistic regression, decision tree, random forest, XGBoost and LightGBM, "
               "compared with stratified 5-fold CV. Class weights handle the imbalance.")
    c.markdown("**3. Pick the threshold**  \nThe cut-off maximises F1 on out-of-fold predictions, so the "
               "hold-out set is only used for the final scores.")
    st.info("Use the sidebar to score one candidate, screen a CSV of applicants, or look at model quality.")

# ----------------------------------------------------------------------------- single candidate
elif page == "Score a candidate":
    st.title("Score a candidate")
    with st.form("candidate"):
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("City", CITIES, help="Cities seen in training; rare cities are grouped together by the model.")
        city_dev = c1.slider("City development index", 0.40, 0.95, 0.85, 0.001)
        edu = c1.selectbox("Education level", C.EDU_ORDER, index=2)
        major = c1.selectbox("Major discipline", C.MAJOR_OPTIONS)
        exp = c2.selectbox("Years of experience", C.EXPERIENCE_OPTIONS, index=8)
        rel = c2.radio("Relevant experience", ["Has relevent experience", "No relevent experience"], horizontal=False)
        enrolled = c2.selectbox("University enrollment", C.ENROLLMENT_OPTIONS)
        employed = c3.checkbox("Currently employed", value=True)
        size = c3.selectbox("Company size", C.SIZE_ORDER, index=2, disabled=False)
        ctype = c3.selectbox("Company type", C.COMPANY_TYPE_OPTIONS)
        last = c3.selectbox("Years since last job change", C.LAST_JOB_OPTIONS, index=1)
        hours = st.slider("Training hours completed", 1, 336, 48)
        submitted = st.form_submit_button("Score candidate", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "city": city, "city_development_index": city_dev, "relevent_experience": rel,
            "enrolled_university": enrolled, "education_level": edu, "major_discipline": major,
            "experience": exp, "company_size": size if employed else None,
            "company_type": ctype if employed else None, "last_new_job": last, "training_hours": hours,
        }])
        res = score(bundle, row).iloc[0]
        p = float(res["priority_score"])
        left, right = st.columns([1, 1.3])
        with left:
            st.plotly_chart(gauge(p, bundle["threshold"]), use_container_width=True)
            if p >= bundle["threshold"]:
                st.success(f"**{res['recommendation']}**: likely to be open to a move.")
            else:
                st.warning(f"**{res['recommendation']}**: less likely to be looking right now.")
            st.caption(f"Orange line = decision threshold ({bundle['threshold']:.0%}).")
        with right:
            contrib = explain_candidate(bundle, row, top_n=7).reset_index()
            contrib.columns = ["Feature", "Contribution"]
            contrib["Direction"] = contrib["Contribution"].map(lambda v: "Raises score" if v > 0 else "Lowers score")
            fig = px.bar(contrib.iloc[::-1], x="Contribution", y="Feature", orientation="h", color="Direction",
                         color_discrete_map={"Raises score": ORANGE, "Lowers score": PRIMARY},
                         title="What drove this score (SHAP, log-odds)")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- batch
elif page == "Batch screening":
    st.title("Batch screening")
    src = st.radio("Applicant data", ["Demo set (2,129 unlabeled candidates)", "Upload a CSV"], horizontal=True)
    df = None
    if src.startswith("Demo"):
        df = get_demo_candidates()
    else:
        up = st.file_uploader("CSV with the same columns as the dataset", type="csv")
        if up:
            df = pd.read_csv(up)
    if df is not None:
        try:
            res = score(bundle, df).sort_values("priority_score", ascending=False).reset_index(drop=True)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        c1, c2 = st.columns(2)
        n_top = c1.slider("Shortlist size", 5, min(200, len(res)), 25)
        thr = c2.slider("Decision threshold", 0.10, 0.90, float(round(bundle["threshold"], 2)), 0.01)
        advance = int((res["priority_score"] >= thr).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Applicants scored", f"{len(res):,}")
        m2.metric("Recommended to advance", f"{advance:,}", f"{advance / len(res):.0%} of pool", delta_color="off")
        m3.metric("Shortlist avg. score", f"{res['priority_score'].head(n_top).mean():.0%}")
        res["recommendation"] = res["priority_score"].map(lambda v: "Advance to next stage" if v >= thr else "Lower priority")
        fig = px.histogram(res, x="priority_score", nbins=40, color_discrete_sequence=[PRIMARY],
                           title="Distribution of priority scores")
        fig.add_vline(x=thr, line_color=ORANGE, line_width=3, annotation_text="threshold")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"Top {n_top} candidates")
        st.dataframe(
            res.head(n_top), use_container_width=True, hide_index=True,
            column_config={"priority_score": st.column_config.ProgressColumn(
                "Priority score", min_value=0.0, max_value=1.0, format="%.2f")},
        )
        st.download_button("Download full ranking (CSV)", res.to_csv(index=False).encode(),
                           "ranked_candidates.csv", "text/csv")

# ----------------------------------------------------------------------------- insights
else:
    st.title("Model insights")
    t1, t2, t3 = st.tabs(["Performance", "Drivers", "Limitations & ethics"])
    fig_dir = C.FIGURES_DIR
    with t1:
        cmp_ = pd.DataFrame(metrics["cv_comparison"]).T[["roc_auc", "pr_auc", "f1", "precision", "recall"]]
        st.markdown("**Cross-validated model comparison** (training split, 5-fold)")
        st.dataframe(cmp_.style.format("{:.3f}"), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.image(str(fig_dir / "roc_pr_curves.png"), caption="Hold-out ROC and precision-recall curves")
        c2.image(str(fig_dir / "cumulative_gain.png"), caption="Cumulative gain on the hold-out set")
        c3, c4 = st.columns(2)
        c3.image(str(fig_dir / "confusion_matrix.png"))
        pk = pd.Series(metrics["precision_at_k"], name="Precision@k").rename_axis("k").reset_index()
        c4.markdown(f"**Precision @ k** (random baseline: {metrics['positive_rate']:.0%})")
        c4.dataframe(pk.style.format({"Precision@k": "{:.0%}"}), hide_index=True)
    with t2:
        st.image(str(fig_dir / "feature_importance.png"))
        st.caption("Mean absolute SHAP value across a sample of held-out candidates.")
    with t3:
        st.markdown(
            """
- The score says similar past candidates were looking. It does not say this person will accept an offer.
- Gender is not used as a feature, but others such as city can still act as proxies, so this needs an audit before real use.
- Use the score to prioritise outreach, not to reject anyone automatically.
- The data is a 2020 snapshot of trainees from one programme, so it may not carry over to other roles or markets.
- The threshold is a business choice: lower it to catch more job seekers, raise it for a tighter shortlist.
            """
        )
