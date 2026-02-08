import os
import shutil
import streamlit as st
import pandas as pd

from audit import run_audit
from schema_mapper import detect_schema

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
INPUT_PATH = os.path.join(INPUT_DIR, "results.csv")
REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "validity_report.md")
REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "validity_report.json")


def ensure_dirs():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_outputs():
    for p in [INPUT_PATH, REPORT_MD_PATH, REPORT_JSON_PATH]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


st.set_page_config(page_title="Metabolomics Validity Auditor", layout="wide")
st.title("🧪 Metabolomics Validity Auditor")
st.caption("Upload a metabolomics results CSV. Get an audit + a clean downloadable report (local-first).")

ensure_dirs()

with st.sidebar:
    st.header("Controls")
    if st.button("Reset / Clear"):
        clear_outputs()
        st.success("Cleared inputs/outputs.")


uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    # Save upload
    clear_outputs()
    with open(INPUT_PATH, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("Uploaded successfully.")

    # Preview
    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Schema mapping
    st.subheader("Detected Schema Mapping")
    sm = detect_schema(df)
    st.json(
        {
            "canonical_to_original": sm.canonical_to_original,
            "missing": sm.missing,
            "ambiguities": sm.ambiguities,
        }
    )

    # Run audit button
    if st.button("Run Audit", type="primary"):
        try:
            run_audit(csv_path=INPUT_PATH, report_path=REPORT_MD_PATH, json_path=REPORT_JSON_PATH)
            st.success("Audit complete.")
        except Exception as e:
            st.error(f"Audit crashed: {e}")
            st.stop()

# Render report if present
if os.path.exists(REPORT_MD_PATH):
    st.subheader("Report")
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        report_md = f.read()
    st.markdown(report_md)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download report (.md)",
            data=report_md.encode("utf-8"),
            file_name="validity_report.md",
            mime="text/markdown",
        )
    with col2:
        if os.path.exists(REPORT_JSON_PATH):
            with open(REPORT_JSON_PATH, "rb") as f:
                st.download_button(
                    "Download report (.json)",
                    data=f.read(),
                    file_name="validity_report.json",
                    mime="application/json",
                )
else:
    st.info("Upload a CSV and run the audit to see the report here.")
