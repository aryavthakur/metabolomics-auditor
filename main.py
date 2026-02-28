# main.py
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from schema_mapper import apply_canonical_schema, detect_schema

# Default IO (your Streamlit UI writes to these)
INPUT_PATH = "inputs/results.csv"
OUTPUT_MD_PATH = "outputs/validity_report.md"
OUTPUT_JSON_PATH = "outputs/validity_report.json"
CONTEXT_JSON_PATH = "inputs/context.json"  # optional (only used if present)


# ----------------------------
# Small utilities
# ----------------------------
def _safe_makedirs_for_file(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _read_optional_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _as_percent(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "—"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Forgiving numeric coercion (handles '<0.001', commas, blanks)."""
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[<>]=?\s*", "", regex=True)  # "<0.001" -> "0.001"
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def _numeric_parse_rate(series: pd.Series) -> float:
    num = _coerce_numeric(series)
    return float(num.notna().mean()) if len(num) else 0.0


def _fraction_in_range(series: pd.Series, lo: float, hi: float) -> float:
    num = _coerce_numeric(series)
    if len(num) == 0:
        return 0.0
    mask = num.notna()
    if mask.sum() == 0:
        return 0.0
    return float(((num[mask] >= lo) & (num[mask] <= hi)).mean())


def _detect_pvalue_like_issues(series: pd.Series) -> Tuple[float, float]:
    """
    Returns (numeric_rate, in_0_1_rate).
    Useful for p-values and FDR.
    """
    nr = _numeric_parse_rate(series)
    r01 = _fraction_in_range(series, 0.0, 1.0)
    return nr, r01


def _is_probably_log2fc(colname: Optional[str]) -> bool:
    if not colname:
        return False
    s = colname.lower()
    return "log2" in s or "log2fc" in s or "log_fc" in s or "logfc" in s


# ----------------------------
# Core audit
# ----------------------------
def run_audit(
    csv_path: str = INPUT_PATH,
    report_md_path: str = OUTPUT_MD_PATH,
    report_json_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Validex audit entrypoint.

    - Reads csv_path
    - Detects schema via schema_mapper
    - Generates report markdown (and optional JSON)
    - Returns markdown text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    # Optional context: argument wins, else inputs/context.json if present
    ctx = context or _read_optional_json(CONTEXT_JSON_PATH)

    # Detect schema on raw df (so we can report original column names)
    sm = detect_schema(df)

    # Canonicalize columns for easier downstream checks (non-destructive)
    canon_df, _ = apply_canonical_schema(df)

    # Pull detected columns (original names)
    feature_col = sm.canonical_to_original.get("feature")
    p_col = sm.canonical_to_original.get("p_value")
    fdr_col = sm.canonical_to_original.get("fdr")
    fc_col = sm.canonical_to_original.get("fold_change")
    log2fc_col = sm.canonical_to_original.get("log2fc")

    # Safety: sometimes fold_change alias list includes log2fc-ish headers.
    # Prefer log2fc_col for log2, and keep fc_col as "FC" if it doesn't look log-like.
    if fc_col and _is_probably_log2fc(fc_col) and not log2fc_col:
        log2fc_col = fc_col
        fc_col = None

    # ----------------------------
    # Scoring + flags
    # ----------------------------
    confidence = 100
    interpretations: list[str] = []
    recommendations: list[str] = []
    flags: list[Dict[str, Any]] = []

    def flag(severity: str, title: str, why: str, fix: str) -> None:
        flags.append({"severity": severity, "title": title, "why": why, "fix": fix})

    # Feature column
    if not feature_col:
        confidence -= 10
        interpretations.append(
            "No clear metabolite/feature identifier column was detected; results are harder to audit and cite."
        )
        recommendations.append(
            "Add a column like 'Metabolite', 'Compound', or 'Feature' as a human-readable identifier."
        )
        flag(
            "med",
            "No metabolite/feature ID detected",
            "A feature identifier improves interpretability and export consistency.",
            "Add a metabolite/feature name/ID column.",
        )

    # p-values
    if not p_col:
        confidence -= 35
        interpretations.append(
            "No p-values were detected. Without p-values, statistical significance cannot be assessed (exploratory output)."
        )
        recommendations.append(
            "Export your univariate test output (t-test/ANOVA/mixed model) including a p-value column."
        )
        flag(
            "high",
            "No p-values detected",
            "You can’t assess statistical significance; results are exploratory only.",
            "Include a p-value column from your statistical pipeline.",
        )
    else:
        p_nr, p_r01 = _detect_pvalue_like_issues(df[p_col])
        if p_nr < 0.8:
            confidence -= 10
            interpretations.append(
                f"P-value column '{p_col}' has low numeric parse rate ({_as_percent(p_nr)})."
            )
            recommendations.append(
                "Clean p-values to numeric (e.g., 0.001 or 1e-5). Remove text/junk and unify formatting."
            )
            flag(
                "med",
                "P-value column may be messy",
                f"Only {_as_percent(p_nr)} of values look numeric.",
                "Standardize p-values to numeric values.",
            )
        if p_r01 < 0.9:
            confidence -= 10
            interpretations.append(
                f"P-value column '{p_col}' contains many values outside [0,1] ({_as_percent(1 - p_r01)} out-of-range)."
            )
            recommendations.append(
                "Verify that the detected column truly contains p-values (not -log10(p), scores, or test statistics)."
            )
            flag(
                "med",
                "P-values out of range",
                "Many values are outside the valid p-value range [0,1].",
                "Check if this column is -log10(p) or another statistic; rename/export correctly.",
            )

    # FDR/q-values (only meaningful if p-values exist)
    if p_col and not fdr_col:
        confidence -= 20
        interpretations.append(
            "P-values were detected without multiple-testing correction (FDR/q-values). This increases false positive risk in high-dimensional metabolomics."
        )
        recommendations.append(
            "Add BH-FDR (q-values) when testing many metabolites/features."
        )
        flag(
            "med",
            "No FDR/q-values detected",
            "Multiple-testing correction isn’t available in this file.",
            "Add an FDR/q-value column (e.g., Benjamini–Hochberg).",
        )
    elif fdr_col:
        fdr_nr, fdr_r01 = _detect_pvalue_like_issues(df[fdr_col])
        if fdr_nr < 0.8:
            confidence -= 5
            interpretations.append(
                f"FDR/q-value column '{fdr_col}' has low numeric parse rate ({_as_percent(fdr_nr)})."
            )
            recommendations.append("Clean q-values to numeric values (0..1).")
            flag(
                "low",
                "FDR column may be messy",
                f"Only {_as_percent(fdr_nr)} of values look numeric.",
                "Standardize q-values to numeric values.",
            )
        if fdr_r01 < 0.9:
            confidence -= 5
            interpretations.append(
                f"FDR/q-value column '{fdr_col}' contains many values outside [0,1]."
            )
            recommendations.append(
                "Verify the q-value/FDR export (q-values should be between 0 and 1)."
            )
            flag(
                "low",
                "FDR values out of range",
                "Some values are outside [0,1].",
                "Check export and column selection; ensure true q-values.",
            )

    # Effect size (FC/log2FC)
    if not fc_col and not log2fc_col:
        confidence -= 20
        interpretations.append(
            "No fold-change (FC) or log2FC column was detected. Without effect size, directionality and practical significance are harder to interpret."
        )
        recommendations.append("Include FC and/or log2FC in your exported results.")
        flag(
            "high",
            "No effect size detected (FC/log2FC)",
            "Without FC/log2FC, effect size isn’t interpretable.",
            "Include Fold Change (FC) or log2FC in the export.",
        )
    else:
        eff_col = log2fc_col or fc_col
        if eff_col:
            nr = _numeric_parse_rate(df[eff_col])
            if nr < 0.8:
                confidence -= 5
                interpretations.append(
                    f"Effect size column '{eff_col}' has low numeric parse rate ({_as_percent(nr)})."
                )
                recommendations.append(
                    "Ensure effect sizes are numeric (remove annotations/strings)."
                )
                flag(
                    "low",
                    "Effect size column may be messy",
                    f"Only {_as_percent(nr)} of values look numeric.",
                    "Standardize effect sizes to numeric values.",
                )

    # Common “bad export” pattern: effect size present, no p-values
    if (fc_col or log2fc_col) and not p_col:
        interpretations.append(
            "Effect sizes are present without p-values; treat this as exploratory reporting (not statistical evidence)."
        )

    # Context-aware nudges (optional, lightweight)
    if ctx:
        # These are *recommendation nudges*; don’t hard-fail.
        design = str(ctx.get("design", "")).lower()  # e.g., "paired", "independent", "repeated"
        n_groups = ctx.get("n_groups", None)
        targeted = str(ctx.get("mode", "")).lower()  # e.g., "targeted", "untargeted"

        if targeted == "untargeted" and p_col and not fdr_col:
            # already flagged, but reinforce as “context mismatch”
            interpretations.append(
                "Context: Untargeted metabolomics typically requires multiple-testing correction; missing FDR is a common reviewer red flag."
            )

        if design in ("paired", "repeated", "longitudinal"):
            recommendations.append(
                "Context note: If samples are paired/repeated measures, ensure your stats use paired tests or mixed models (not independent tests)."
            )

        if isinstance(n_groups, int) and n_groups >= 3:
            recommendations.append(
                "Context note: For 3+ groups, ANOVA/Welch ANOVA/Kruskal–Wallis (or linear models) are typical; avoid multiple pairwise t-tests without correction."
            )

    confidence = max(0, min(100, confidence))

    # ----------------------------
    # Build Markdown report
    # ----------------------------
    _safe_makedirs_for_file(report_md_path)

    md: list[str] = []
    md.append("# Metabolomics Validity Report\n")
    md.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    md.append("## Dataset Overview")
    md.append(f"- Rows (features): {n_rows}")
    md.append(f"- Columns: {n_cols}\n")

    md.append("## Detected Statistical Columns (schema-mapped)")
    md.append(f"- Feature / metabolite ID: {feature_col}")
    md.append(f"- Fold change (FC): {fc_col}")
    md.append(f"- log2FC: {log2fc_col}")
    md.append(f"- p-value: {p_col}")
    md.append(f"- FDR / q-value: {fdr_col}\n")

    if sm.ambiguities:
        md.append("## Schema Ambiguities")
        for canon, cands in sm.ambiguities.items():
            md.append(f"- **{canon}** matched multiple columns: {', '.join(cands)}")
        md.append("")

    if sm.missing:
        md.append("## Missing Canonical Fields")
        md.append("- " + ", ".join(sm.missing) + "\n")

    if ctx:
        md.append("## Context (user-provided)")
        # Keep this compact and readable
        for k, v in ctx.items():
            md.append(f"- {k}: {v}")
        md.append("")

    md.append("## Scientific Interpretation")
    if interpretations:
        for i in interpretations:
            md.append(f"- {i}")
    else:
        md.append("- No major statistical issues detected.")
    md.append("")

    md.append("## Recommendations")
    if recommendations:
        # de-dupe while preserving order
        seen = set()
        recs = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                recs.append(r)
        for r in recs:
            md.append(f"- {r}")
    else:
        md.append("- No immediate corrective actions required.")
    md.append("")

    md.append("## Overall Confidence Score")
    md.append(f"**{confidence} / 100**\n")

    if flags:
        md.append("## Flags")
        for f in flags:
            md.append(
                f"- **{f['severity'].upper()}** — {f['title']}: {f['why']}  \n"
                f"  _Fix_: {f['fix']}"
            )
        md.append("")

    report_text = "\n".join(md)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ----------------------------
    # Optional JSON output
    # ----------------------------
    if report_json_path:
        _safe_makedirs_for_file(report_json_path)

        payload: Dict[str, Any] = {
            "input": {"csv_path": csv_path},
            "overview": {"n_rows": n_rows, "n_cols": n_cols},
            "detected": {
                "feature": feature_col,
                "fold_change": fc_col,
                "log2fc": log2fc_col,
                "p_value": p_col,
                "fdr": fdr_col,
            },
            "schema": {
                "canonical_to_original": sm.canonical_to_original,
                "missing": sm.missing,
                "ambiguities": sm.ambiguities,
                # scores exists in your rewritten schema_mapper.py
                "scores": {k: [(c, float(s)) for c, s in v] for k, v in sm.scores.items()},
            },
            "context": ctx,
            "analysis": {
                "confidence": confidence,
                "interpretations": interpretations,
                "recommendations": recommendations,
                "flags": flags,
            },
        }

        with open(report_json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2)

    return report_text


def main() -> None:
    """
    CLI entrypoint.
    Streamlit should call run_audit(), not this.
    """
    print("🔍 Starting metabolomics auditor...")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input not found: {INPUT_PATH}")
        return

    _safe_makedirs_for_file(OUTPUT_MD_PATH)
    run_audit(
        csv_path=INPUT_PATH,
        report_md_path=OUTPUT_MD_PATH,
        report_json_path=OUTPUT_JSON_PATH,
    )
    print(f"✅ Report written: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
