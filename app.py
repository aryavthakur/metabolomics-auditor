import os
import re
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

from main import main as run_audit


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Metabolomics Validity Auditor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Global CSS (Cursor-ish)
# ----------------------------
st.markdown(
    """
<style>
/* Hide Streamlit default chrome a bit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
  background:
    radial-gradient(1200px 600px at 20% 0%, rgba(120, 58, 255, 0.25), transparent 55%),
    radial-gradient(900px 500px at 90% 10%, rgba(0, 220, 180, 0.18), transparent 60%),
    radial-gradient(700px 400px at 60% 90%, rgba(255, 200, 0, 0.08), transparent 60%),
    linear-gradient(180deg, #05060a 0%, #070a12 35%, #05060a 100%);
  color: #E9ECF1;
}

/* Typography */
h1, h2, h3 {
  letter-spacing: -0.02em;
}
.small-muted {
  color: rgba(233, 236, 241, 0.72);
  font-size: 0.95rem;
}

/* Glass card */
.glass {
  border: 1px solid rgba(255,255,255,0.09);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
}

/* Hero */
.hero-wrap {
  max-width: 1180px;
  margin: 0 auto;
  padding-top: 22px;
}
.hero-title {
  font-size: 3.2rem;
  line-height: 1.05;
  margin-bottom: 10px;
}
.badge {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  color: rgba(233, 236, 241, 0.85);
  font-size: 0.9rem;
}
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  color: rgba(233, 236, 241, 0.85);
  font-size: 0.85rem;
}
.pill-high { border-color: rgba(255, 80, 80, 0.35); background: rgba(255, 80, 80, 0.10); }
.pill-med  { border-color: rgba(255, 200, 0, 0.35); background: rgba(255, 200, 0, 0.10); }
.pill-low  { border-color: rgba(80, 200, 255, 0.35); background: rgba(80, 200, 255, 0.10); }

/* Buttons look a bit more premium */
.stButton>button {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  color: #E9ECF1;
  padding: 10px 14px;
}
.stButton>button:hover {
  border-color: rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.10);
}

/* File uploader */
[data-testid="stFileUploader"] section {
  border-radius: 16px !important;
  border: 1px dashed rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.03) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  padding: 10px 14px;
}
.stTabs [aria-selected="true"] {
  border-color: rgba(120, 58, 255, 0.40) !important;
  background: rgba(120, 58, 255, 0.14) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def detect_columns(df: pd.DataFrame) -> dict:
    """Heuristic detection for common metabolomics result columns."""
    cols = [c.strip() for c in df.columns]
    lower = {c: c.lower() for c in cols}

    def find_any(keywords):
        for c in cols:
            lc = lower[c]
            for k in keywords:
                if k in lc:
                    return c
        return None

    feature = None
    # often first column is metabolite name/id
    if len(cols) > 0:
        feature = cols[0]

    fc = find_any(["fold change", "fold_change", "fc"])
    logfc = find_any(["log2(fc", "log2fc", "log_fc", "log2 fold", "log fold"])
    p = find_any(["p-value", "pvalue", "p value", "p_val", "pval"])
    fdr = find_any(["fdr", "q-value", "qvalue", "adj p", "padj", "adjusted p"])

    return {"feature": feature, "fc": fc, "logfc": logfc, "p": p, "fdr": fdr}


def parse_score_from_report(md_text: str) -> int | None:
    """Try to extract a score like '60 / 100' from the markdown."""
    m = re.search(r"(\d{1,3})\s*/\s*100", md_text)
    if not m:
        return None
    score = int(m.group(1))
    if 0 <= score <= 100:
        return score
    return None


def build_issues(colmap: dict) -> list[dict]:
    issues = []
    if not colmap.get("p"):
        issues.append(
            {
                "severity": "high",
                "title": "No p-values detected",
                "why": "You can’t assess statistical significance; results are exploratory only.",
                "fix": "Export stats from your tool (t-test/ANOVA) and include a p-value column.",
            }
        )
    if not colmap.get("fdr"):
        issues.append(
            {
                "severity": "med",
                "title": "No FDR/q-values detected",
                "why": "Multiple-testing correction isn’t available in this file.",
                "fix": "Add an FDR/q-value column if you tested many metabolites.",
            }
        )
    if not colmap.get("fc") and not colmap.get("logfc"):
        issues.append(
            {
                "severity": "high",
                "title": "No fold-change column detected",
                "why": "Without FC/logFC, effect size isn’t interpretable.",
                "fix": "Include Fold Change or log2FC in the results export.",
            }
        )
    if len(issues) == 0:
        issues.append(
            {
                "severity": "low",
                "title": "No major structural issues detected",
                "why": "Basic columns look present.",
                "fix": "You can still review missingness and naming consistency.",
            }
        )
    return issues


def severity_pill(sev: str) -> str:
    if sev == "high":
        return '<span class="pill pill-high">HIGH</span>'
    if sev == "med":
        return '<span class="pill pill-med">MED</span>'
    return '<span class="pill pill-low">LOW</span>'


# ----------------------------
# Session state (prevents showing stale data before upload)
# ----------------------------
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "last_run_report" not in st.session_state:
    st.session_state.last_run_report = None
if "last_run_path" not in st.session_state:
    st.session_state.last_run_path = None


# ----------------------------
# Hero header
# ----------------------------
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
c1, c2 = st.columns([0.72, 0.28], gap="large")

with c1:
    st.markdown(
        """
        <div class="badge">🧪 <b>Metabolomics Validity Auditor</b> <span style="opacity:.7;">•</span> local Streamlit app</div>
        <div style="height:12px;"></div>
        <div class="hero-title">Make metabolomics outputs<br/>auditable in seconds.</div>
        <div class="small-muted">
          Upload a CSV → run checks → get a clean report you can attach to figures, supplements, or lab notes.
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("**Quick actions**")
    st.caption("Tip: If you see weird behavior, refresh the page once.")
    if st.button("🧹 Clear current session"):
        st.session_state.uploaded_df = None
        st.session_state.uploaded_name = None
        st.session_state.last_run_report = None
        st.session_state.last_run_path = None
        st.success("Cleared. Upload a new file.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)


# ----------------------------
# Upload + Step flow
# ----------------------------
top = st.container()
with top:
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### Upload")
        uploaded = st.file_uploader("Upload a metabolomics results CSV", type=["csv"], label_visibility="collapsed")
        st.caption("Supported: .csv (up to 200MB). Your file is processed locally.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### Workflow")
        step = 1
        if st.session_state.uploaded_df is not None:
            step = 2
        if st.session_state.last_run_report is not None:
            step = 4

        st.progress({1: 0.25, 2: 0.55, 3: 0.75, 4: 1.0}[step])
        st.markdown(
            f"""
            <div class="small-muted">
            <b>Step 1:</b> Upload CSV<br/>
            <b>Step 2:</b> Preview + detect columns<br/>
            <b>Step 3:</b> Run audit<br/>
            <b>Step 4:</b> Export report
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# When a new file is uploaded, load it and reset old results
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.session_state.uploaded_df = df
    st.session_state.uploaded_name = uploaded.name
    st.session_state.last_run_report = None
    st.session_state.last_run_path = None

    os.makedirs("inputs", exist_ok=True)
    with open("inputs/results.csv", "wb") as f:
        f.write(uploaded.getbuffer())


# ----------------------------
# Nothing else should render unless a file exists
# ----------------------------
if st.session_state.uploaded_df is None:
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="glass">
            <h3 style="margin-top:0;">👆 Upload a CSV to begin</h3>
            <div class="small-muted">
              Once you upload, you’ll see a preview, detected column mapping, and a “Run audit” button.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ----------------------------
# Post-upload UI
# ----------------------------
df = st.session_state.uploaded_df
colmap = detect_columns(df)
issues = build_issues(colmap)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

a, b, c = st.columns([0.34, 0.33, 0.33], gap="large")

with a:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Preview")
    st.caption(f"File: **{st.session_state.uploaded_name}**")
    st.dataframe(df.head(8), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Detected column mapping")
    st.caption("Heuristic detection — improves as we add more formats.")
    st.json(colmap)
    st.markdown("</div>", unsafe_allow_html=True)

with c:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Quick stats")
    st.metric("Rows", f"{df.shape[0]:,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    missing = int(df.isna().sum().sum())
    st.metric("Missing cells", f"{missing:,}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

run_left, run_right = st.columns([0.7, 0.3], gap="large")
with run_left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Run audit")
    st.caption("Runs your existing `main.py` audit and reads `outputs/validity_report.md`.")
    if st.button("▶ Run Audit", use_container_width=True):
        os.makedirs("outputs", exist_ok=True)
        with st.spinner("Running audit..."):
            run_audit()

        report_path = "outputs/validity_report.md"
        if os.path.exists(report_path):
            md = open(report_path, "r", encoding="utf-8", errors="ignore").read()
            st.session_state.last_run_report = md
            st.session_state.last_run_path = report_path
            st.success("Done ✅ Audit report created.")
        else:
            st.error("Audit ran, but no report was created in outputs/. Check main.py for OUTPUT_PATH.")
    st.markdown("</div>", unsafe_allow_html=True)

with run_right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Top issues")
    # show first 3 issues
    for it in issues[:3]:
        st.markdown(
            f"""
            {severity_pill(it["severity"])} <b>{it["title"]}</b><br/>
            <span class="small-muted">{it["why"]}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Results area (only after audit)
# ----------------------------
if st.session_state.last_run_report is None:
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.info("Run the audit to generate the report + export options.")
    st.stop()

report_text = st.session_state.last_run_report
score = parse_score_from_report(report_text)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

tab_summary, tab_report, tab_data = st.tabs(["✨ Summary", "📄 Full Report", "🧾 Data"])

with tab_summary:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## Summary")

    # Score card row
    x1, x2, x3 = st.columns([0.25, 0.45, 0.30], gap="large")
    with x1:
        if score is None:
            st.metric("Confidence score", "—")
        else:
            st.metric("Confidence score", f"{score} / 100")

    with x2:
        st.markdown("**Key findings**")
        for it in issues[:3]:
            st.markdown(
                f"- **{it['title']}** — {it['why']}",
            )

    with x3:
        st.markdown("**Recommended next actions**")
        for it in issues[:3]:
            st.markdown(f"- {it['fix']}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Simple plot: histogram of FC or logFC if available
    plot_col = colmap.get("logfc") or colmap.get("fc")
    if plot_col and pd.api.types.is_numeric_dtype(df[plot_col]):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### Effect size distribution")
        st.caption(f"Histogram of **{plot_col}**")
        series = df[plot_col].replace([np.inf, -np.inf], np.nan).dropna()
        st.bar_chart(series.value_counts(bins=20).sort_index(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="glass"><b>Effect size plot</b><div class="small-muted">No numeric FC/logFC column detected for plotting.</div></div>',
            unsafe_allow_html=True,
        )

with tab_report:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Export buttons
    e1, e2, e3 = st.columns([0.34, 0.33, 0.33], gap="large")
    with e1:
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=report_text,
            file_name="validity_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with e2:
        # also let them download the uploaded CSV again
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download uploaded CSV",
            data=csv_bytes,
            file_name=st.session_state.uploaded_name or "results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e3:
        # Timestamped copy suggestion (just text)
        st.caption("Tip")
        st.code(f"outputs/validity_report.md", language="bash")

    st.markdown("---")
    st.markdown(report_text)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## Data")
    st.caption("This renders only after you upload. (No stale preview.)")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
