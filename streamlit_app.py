# app.py — Validex (Fancy UI) • context-aware • schema-aware • Streamlit Cloud friendly
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Prefer the new API (context-aware). Fall back to legacy main() if needed.
try:
    from main import run_audit as run_audit_fn  # new
except Exception:
    run_audit_fn = None  # type: ignore

try:
    from main import main as legacy_main  # old
except Exception:
    legacy_main = None  # type: ignore

# Schema mapper is optional for UI mapping; the audit itself uses it in main.py.
try:
    from schema_mapper import detect_schema  # type: ignore
except Exception:
    detect_schema = None  # type: ignore


# ----------------------------
# Constants / paths
# ----------------------------
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
INPUT_PATH = os.path.join(INPUT_DIR, "results.csv")
REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "validity_report.md")
REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "validity_report.json")


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Validex",
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
def _ensure_dirs() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _clear_paths() -> None:
    for p in [INPUT_PATH, REPORT_MD_PATH, REPORT_JSON_PATH]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def parse_score_from_report(md_text: str) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*/\s*100", md_text)
    if not m:
        return None
    score = int(m.group(1))
    return score if 0 <= score <= 100 else None


def severity_pill(sev: str) -> str:
    sev = (sev or "").lower()
    if sev == "high":
        return '<span class="pill pill-high">HIGH</span>'
    if sev == "med":
        return '<span class="pill pill-med">MED</span>'
    return '<span class="pill pill-low">LOW</span>'


def extract_flags_from_report(md_text: str) -> list[dict]:
    """
    Best-effort parse for the '## Flags' section produced by main.py.
    If absent, returns [].
    """
    flags: list[dict] = []
    if "## Flags" not in md_text:
        return flags

    section = md_text.split("## Flags", 1)[1]
    # stop at next header if any
    for stopper in ["\n## ", "\n# "]:
        if stopper in section:
            section = section.split(stopper, 1)[0]

    lines = [ln.strip() for ln in section.splitlines() if ln.strip().startswith("- **")]
    for ln in lines:
        try:
            sev = re.search(r"\*\*(HIGH|MED|LOW)\*\*", ln)
            sev_txt = (sev.group(1).lower() if sev else "low")

            title_m = re.search(r"—\s*(.*?):", ln)
            title = title_m.group(1).strip() if title_m else "Flag"

            why_m = re.search(r":\s*(.*?)(?:\s{2,}|$)", ln)
            why = why_m.group(1).strip() if why_m else ""

            flags.append({"severity": sev_txt, "title": title, "why": why, "fix": ""})
        except Exception:
            continue
    return flags


def build_context_ui(defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Context panel (reviewer decision tree style).
    Stored in st.session_state.context and passed to main.run_audit(context=...).
    """
    defaults = defaults or {}

    st.markdown("### Context")
    st.caption("This changes what Validex expects (like a reviewer decision tree).")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        design_groups = st.selectbox(
            "Number of groups",
            ["two", "multi"],
            index=0 if defaults.get("design_groups", "two") == "two" else 1,
            help="two = 2 groups (control vs disease). multi = 3+ groups/timepoints.",
        )
        paired = st.checkbox(
            "Paired / repeated measures",
            value=bool(defaults.get("paired", False)),
            help="Same subject measured multiple times (before/after, matched pairs).",
        )
        longitudinal = st.checkbox(
            "Longitudinal / time-series",
            value=bool(defaults.get("longitudinal", False)),
            help="Multiple timepoints per subject or correlated measurements over time.",
        )

    with c2:
        targeted = st.checkbox(
            "Targeted metabolomics",
            value=bool(defaults.get("targeted", False)),
            help="Small metabolite panel, hypothesis-driven (often less multiple-testing burden).",
        )
        goal = st.selectbox(
            "Study goal",
            ["confirmatory", "exploratory"],
            index=0 if defaults.get("goal", "confirmatory") == "confirmatory" else 1,
            help="confirmatory = stricter stats; exploratory = patterns + validation emphasis.",
        )
        batch_expected = st.checkbox(
            "Batch effects likely",
            value=bool(defaults.get("batch_expected", False)),
            help="Multiple runs/days/instruments or strong technical variation expected.",
        )

    with c3:
        transform = st.selectbox(
            "Transform",
            ["unknown", "none", "log", "log2", "auto"],
            index=["unknown", "none", "log", "log2", "auto"].index(defaults.get("transform", "unknown")),
            help="What transform was applied before stats (often log/log2 in metabolomics).",
        )
        alpha_raw = st.text_input("Alpha", value=str(defaults.get("alpha", "0.05")))
        comparison_label = st.text_input(
            "Comparison label",
            value=str(defaults.get("comparison_label", "")),
            placeholder="e.g., Control vs Disease",
        )

    notes = st.text_area(
        "Notes (optional)",
        value=str(defaults.get("notes", "")),
        placeholder="Anything important about design/statistics/QC…",
        height=90,
    )

    # Parse alpha safely
    try:
        alpha_val = float(alpha_raw)
    except Exception:
        alpha_val = 0.05

    return {
        "design_groups": design_groups,
        "paired": paired,
        "longitudinal": longitudinal,
        "targeted": targeted,
        "goal": goal,
        "batch_expected": batch_expected,
        "transform": transform,
        "alpha": alpha_val,
        "comparison_label": comparison_label.strip(),
        "notes": notes.strip(),
    }


# ----------------------------
# Session state
# ----------------------------
for k, v in {
    "uploaded_df": None,
    "uploaded_name": None,
    "last_run_report": None,
    "last_run_path": None,
    "last_run_json": None,
    "context": {},
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
        <div class="badge">🧪 <b>Validex</b><span style="opacity:.7;">•</span> Streamlit app</div>
        <div style="height:12px;"></div>
        <div class="hero-title">Validex<br/>Metabolomics Audit Engine.</div>
        <div class="small-muted">
          Upload a CSV → specify context → run audit → export report.
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("**Quick actions**")
    st.caption("Tip: If you see weird behavior, refresh once.")
    if st.button("🧹 Clear current session", use_container_width=True):
        st.session_state.uploaded_df = None
        st.session_state.uploaded_name = None
        st.session_state.last_run_report = None
        st.session_state.last_run_path = None
        st.session_state.last_run_json = None
        st.session_state.context = {}
        _clear_paths()
        st.success("Cleared. Upload a new file.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)


# ----------------------------
# Upload + workflow
# ----------------------------
_ensure_dirs()

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Upload")
    uploaded = st.file_uploader(
        "Upload a metabolomics results CSV",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.caption("Supported: .csv (up to 200MB). Processed locally in the app environment.")
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
        """
        <div class="small-muted">
        <b>Step 1:</b> Upload CSV<br/>
        <b>Step 2:</b> Preview + schema mapping<br/>
        <b>Step 3:</b> Add context + run audit<br/>
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

    _clear_paths()
    os.makedirs(INPUT_DIR, exist_ok=True)
    with open(INPUT_PATH, "wb") as f:
        f.write(uploaded.getbuffer())


# Nothing else should render unless a file exists
if st.session_state.uploaded_df is None:
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="glass">
            <h3 style="margin-top:0;">👆 Upload a CSV to begin</h3>
            <div class="small-muted">
              After upload: you’ll see preview, schema mapping, context input, and “Run audit”.
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

# Schema mapping (preferred)
schema_payload: dict = {}
if detect_schema is not None:
    try:
        sm = detect_schema(df)
        schema_payload = {
            "canonical_to_original": getattr(sm, "canonical_to_original", {}),
            "missing": getattr(sm, "missing", []),
            "ambiguities": getattr(sm, "ambiguities", {}),
        }
    except Exception:
        schema_payload = {}
else:
    schema_payload = {}

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

a, b, c = st.columns([0.34, 0.33, 0.33], gap="large")

with a:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Preview")
    st.caption(f"File: **{st.session_state.uploaded_name}**")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Detected schema mapping")
    st.caption("Uses schema_mapper when available (canonical fields + missing + ambiguities).")
    if schema_payload:
        st.json(schema_payload)
    else:
        st.info("schema_mapper.detect_schema() not available. (UI will still run.)")
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

# Context + run
run_left, run_right = st.columns([0.68, 0.32], gap="large")

with run_left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Context + Run audit")
    st.session_state.context = build_context_ui(st.session_state.context or {})

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.caption("Runs the audit and reads `outputs/validity_report.md` (and optional JSON).")

    if st.button("▶ Run Audit", use_container_width=True):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with st.spinner("Running audit..."):
            try:
                if run_audit_fn is not None:
                    # New API (supports context)
                    _ = run_audit_fn(
                        csv_path=INPUT_PATH,
                        report_path=REPORT_MD_PATH,
                        json_path=REPORT_JSON_PATH,
                        context=st.session_state.context,
                    )
                elif legacy_main is not None:
                    # Old API fallback (no context)
                    legacy_main()
                else:
                    raise RuntimeError("Neither main.run_audit nor main.main could be imported.")
            except TypeError:
                # If their run_audit doesn't accept context yet, rerun without it
                if run_audit_fn is not None:
                    _ = run_audit_fn(
                        csv_path=INPUT_PATH,
                        report_path=REPORT_MD_PATH,
                        json_path=REPORT_JSON_PATH,
                    )
                else:
                    raise
            except Exception as e:
                st.error(f"Audit crashed: {e}")
                st.stop()

        if os.path.exists(REPORT_MD_PATH):
            md = open(REPORT_MD_PATH, "r", encoding="utf-8", errors="ignore").read()
            st.session_state.last_run_report = md
            st.session_state.last_run_path = REPORT_MD_PATH
            st.session_state.last_run_json = REPORT_JSON_PATH if os.path.exists(REPORT_JSON_PATH) else None
            st.success("Done ✅ Audit report created.")
        else:
            st.error("Audit ran, but no report was created in outputs/. Check main.py OUTPUT_PATH.")
    st.markdown("</div>", unsafe_allow_html=True)

with run_right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Top flags")
    if st.session_state.last_run_report:
        flags = extract_flags_from_report(st.session_state.last_run_report)[:3]
        if flags:
            for f in flags:
                st.markdown(
                    f"""
                    {severity_pill(f.get("severity","low"))} <b>{f.get("title","Flag")}</b><br/>
                    <span class="small-muted">{f.get("why","")}</span>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        else:
            st.caption("No Flags section found in the report (or none detected).")
    else:
        st.caption("Run the audit to populate flags.")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Results area
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
        st.markdown("**Context used**")
        ctx = st.session_state.context or {}
        ctx_lines = [
            f"- Groups: `{ctx.get('design_groups','two')}`",
            f"- Paired: `{ctx.get('paired', False)}`",
            f"- Longitudinal: `{ctx.get('longitudinal', False)}`",
            f"- Targeted: `{ctx.get('targeted', False)}`",
            f"- Goal: `{ctx.get('goal','confirmatory')}`",
            f"- Batch expected: `{ctx.get('batch_expected', False)}`",
        ]
        st.markdown("\n".join(ctx_lines))

    with x3:
        st.markdown("**Quick takeaways**")
        flags = extract_flags_from_report(report_text)[:3]
        if flags:
            for f in flags:
                st.markdown(f"- **{f.get('title','Flag')}**")
        else:
            st.markdown("- No major flags detected (or Flags section not present).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Simple effect-size plot if a likely FC/log2FC exists (best-effort)
    plot_col = None
    if schema_payload and schema_payload.get("canonical_to_original"):
        c2o = schema_payload["canonical_to_original"]
        plot_col = c2o.get("log2fc") or c2o.get("fold_change")

    if plot_col is None:
        cols = list(df.columns)
        lower = {c: str(c).lower() for c in cols}
        for c in cols:
            lc = lower[c]
            if "log2fc" in lc or "logfc" in lc or "log2 fold" in lc:
                plot_col = c
                break
        if plot_col is None:
            for c in cols:
                lc = lower[c]
                if "fold" in lc and "change" in lc:
                    plot_col = c
                    break

    if plot_col and pd.api.types.is_numeric_dtype(df[plot_col]):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### Effect size distribution")
        st.caption(f"Histogram of **{plot_col}**")
        series = df[plot_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) > 0:
            st.bar_chart(series.value_counts(bins=20).sort_index(), use_container_width=True)
        else:
            st.caption("No numeric values available for plotting.")
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
        if os.path.exists(REPORT_JSON_PATH):
            json_bytes = open(REPORT_JSON_PATH, "rb").read()
            st.download_button(
                "⬇️ Download report (JSON)",
                data=json_bytes,
                file_name="validity_report.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("JSON report not found (optional output).")

    st.markdown("---")
    st.markdown(report_text)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## Data")
    st.caption("Renders only after you upload (no stale preview).")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
