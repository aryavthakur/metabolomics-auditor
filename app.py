# app.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from main import run_audit
from schema_mapper import detect_schema

INPUT_PATH = "inputs/results.csv"
REPORT_MD_PATH = "outputs/validity_report.md"
REPORT_JSON_PATH = "outputs/validity_report.json"


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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
  background:
    radial-gradient(1200px 600px at 20% 0%, rgba(120, 58, 255, 0.25), transparent 55%),
    radial-gradient(900px 500px at 90% 10%, rgba(0, 220, 180, 0.18), transparent 60%),
    radial-gradient(700px 400px at 60% 90%, rgba(255, 200, 0, 0.08), transparent 60%),
    linear-gradient(180deg, #05060a 0%, #070a12 35%, #05060a 100%);
  color: #E9ECF1;
}

h1, h2, h3 { letter-spacing: -0.02em; }

.small-muted {
  color: rgba(233, 236, 241, 0.72);
  font-size: 0.95rem;
}

.glass {
  border: 1px solid rgba(255,255,255,0.09);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
}

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

[data-testid="stFileUploader"] section {
  border-radius: 16px !important;
  border: 1px dashed rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.03) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 10px; }
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
def parse_score_from_report(md_text: str) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*/\s*100", md_text)
    if not m:
        return None
    score = int(m.group(1))
    return score if 0 <= score <= 100 else None


def severity_pill(sev: str) -> str:
    sev = (sev or "").lower().strip()
    if sev == "high":
        return '<span class="pill pill-high">HIGH</span>'
    if sev == "med":
        return '<span class="pill pill-med">MED</span>'
    return '<span class="pill pill-low">LOW</span>'


def safe_makedirs(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ----------------------------
# Session state
# ----------------------------
for k, v in {
    "uploaded_df": None,
    "uploaded_name": None,
    "last_run_report": None,
    "last_run_path": None,
    "last_run_json": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
          Upload a CSV → run checks → export a report you can attach to figures, supplements, or lab notes.
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
        st.session_state.last_run_json = None
        # clear files too (optional but nice)
        for p in [INPUT_PATH, REPORT_MD_PATH, REPORT_JSON_PATH]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
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
        uploaded = st.file_uploader(
            "Upload a metabolomics results CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
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
            <b>Step 2:</b> Preview + detect schema<br/>
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
    st.session_state.last_run_json = None

    safe_makedirs(INPUT_PATH)
    with open(INPUT_PATH, "wb") as f:
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
              Once you upload, you’ll see a preview, schema mapping, and a “Run audit” button.
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
sm = detect_schema(df)

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
    st.markdown("### Detected schema mapping")
    st.caption("Header + data-aware detection (shows ambiguity when unsure).")
    st.json(
        {
            "canonical_to_original": sm.canonical_to_original,
            "missing": sm.missing,
            "ambiguities": sm.ambiguities,
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Quick stats")
    st.metric("Rows", f"{df.shape[0]:,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    missing_cells = int(df.isna().sum().sum())
    st.metric("Missing cells", f"{missing_cells:,}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

run_left, run_right = st.columns([0.7, 0.3], gap="large")
with run_left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Run audit")
    st.caption("Runs `main.run_audit()` and reads `outputs/validity_report.md`.")

    if st.button("▶ Run Audit", use_container_width=True):
        safe_makedirs(REPORT_MD_PATH)
        safe_makedirs(REPORT_JSON_PATH)

        with st.spinner("Running audit..."):
            try:
                report_text = run_audit(
                    csv_path=INPUT_PATH,
                    report_path=REPORT_MD_PATH,
                    json_path=REPORT_JSON_PATH,
                )
            except Exception as e:
                st.error(f"Audit crashed: {e}")
                st.stop()

        st.session_state.last_run_report = report_text
        st.session_state.last_run_path = REPORT_MD_PATH
        st.session_state.last_run_json = read_text(REPORT_JSON_PATH)
        st.success("Done ✅ Audit report created.")

    st.markdown("</div>", unsafe_allow_html=True)

with run_right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Top issues (pre-check)")
    # lightweight pre-check based on detected schema
    # (final truth comes from audit flags)
    pre = []
    if "p_value" in sm.missing:
        pre.append(("high", "No p-values detected"))
    if "fdr" in sm.missing:
        pre.append(("med", "No FDR/q-values detected"))
    if ("fold_change" in sm.missing) and ("log2fc" in sm.missing):
        pre.append(("high", "No effect size (FC/log2FC) detected"))
    if not pre:
        pre.append(("low", "No major structural issues detected"))

    for sev, title in pre[:3]:
        st.markdown(
            f"""{severity_pill(sev)} <b>{title}</b>""",
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

    x1, x2, x3 = st.columns([0.25, 0.45, 0.30], gap="large")
    with x1:
        st.metric("Confidence score", "—" if score is None else f"{score} / 100")

    with x2:
        st.markdown("**What this means**")
        st.markdown("- Use the report to justify what is / isn’t statistically interpretable.")
        st.markdown("- Ambiguities mean headers were unclear (the app shows them above).")

    with x3:
        st.markdown("**Next actions**")
        st.markdown("- If p-values are missing → export stats output.")
        st.markdown("- If FDR missing → add BH-adjusted p-values.")
        st.markdown("- If effect size missing → include FC/log2FC.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Simple plot: histogram of log2fc preferred, else fold_change
    plot_col = sm.canonical_to_original.get("log2fc") or sm.canonical_to_original.get("fold_change")
    if plot_col and pd.api.types.is_numeric_dtype(pd.to_numeric(df[plot_col], errors="coerce")):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### Effect size distribution")
        st.caption(f"Histogram of **{plot_col}**")
        series = pd.to_numeric(df[plot_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        st.bar_chart(series.value_counts(bins=20).sort_index(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="glass"><b>Effect size plot</b><div class="small-muted">No numeric FC/log2FC column detected for plotting.</div></div>',
            unsafe_allow_html=True,
        )

with tab_report:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

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
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download uploaded CSV",
            data=csv_bytes,
            file_name=st.session_state.uploaded_name or "results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e3:
        if st.session_state.last_run_json:
            st.download_button(
                "⬇️ Download audit JSON",
                data=st.session_state.last_run_json,
                file_name="validity_report.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Tip")
            st.code("outputs/validity_report.md", language="bash")

    st.markdown("---")
    st.markdown(report_text)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## Data")
    st.caption("This renders only after you upload.")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
