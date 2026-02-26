# main.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from schema_mapper import apply_canonical_schema, detect_schema

INPUT_PATH = "inputs/results.csv"
OUTPUT_PATH = "outputs/validity_report.md"


def _safe_makedirs(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _as_percent(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "—"


def _numeric_coerce_rate(series: pd.Series) -> float:
    """Approximate fraction of values coercible to numeric."""
    if series is None or len(series) == 0:
        return 0.0
    if pd.api.types.is_numeric_dtype(series):
        return float(series.notna().mean())
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[<>]=?\s*", "", regex=True)  # "<0.001" -> "0.001"
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    num = pd.to_numeric(s, errors="coerce")
    return float(num.notna().mean())


def run_audit(
    csv_path: str = INPUT_PATH,
    report_path: str = OUTPUT_PATH,
    json_path: Optional[str] = None,
) -> str:
    """
    Validex audit entrypoint.

    - Reads csv_path
    - Detects schema via schema_mapper (header + data-aware scoring)
    - Writes markdown report to report_path
    - Optionally writes a machine-readable JSON payload to json_path
    - Returns markdown report text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    # Detect schema (with scores/ambiguities)
    sm = detect_schema(df)

    # Canonicalize columns for downstream checks (non-destructive)
    canonical_df, sm2 = apply_canonical_schema(df)

    # Resolve detected columns (original names) for report display
    feature_col = sm.canonical_to_original.get("feature")
    p_col = sm.canonical_to_original.get("p_value")
    fdr_col = sm.canonical_to_original.get("fdr")
    fc_col = sm.canonical_to_original.get("fold_change")
    log2fc_col = sm.canonical_to_original.get("log2fc")

    # ----------------------------
    # Scientific interpretation / scoring
    # ----------------------------
    confidence = 100
    interpretations = []
    recommendations = []
    flags: list[dict[str, Any]] = []

    def flag(severity: str, title: str, why: str, fix: str) -> None:
        flags.append({"severity": severity, "title": title, "why": why, "fix": fix})

    # Feature / identifier column
    if not feature_col:
        confidence -= 10
        interpretations.append(
            "No clear metabolite/feature identifier column was detected; interpretation may be harder for humans."
        )
        recommendations.append(
            "Include a first column with metabolite/feature names (e.g., 'Metabolite', 'Compound', 'Feature')."
        )
        flag(
            "med",
            "No metabolite/feature ID detected",
            "A clear feature identifier improves interpretability and export consistency.",
            "Add a metabolite/feature name column.",
        )

    # p-values
    if not p_col:
        confidence -= 35
        interpretations.append(
            "No p-values were detected. Without p-values, statistical significance cannot be assessed (exploratory-only output)."
        )
        recommendations.append(
            "Export statistical test results (t-test/ANOVA/etc.) that include a p-value column."
        )
        flag(
            "high",
            "No p-values detected",
            "You can’t assess statistical significance; results are exploratory only.",
            "Include a p-value column from your statistical test output.",
        )
    else:
        # Data sanity: p-values should be mostly numeric and in [0,1]
        p_series = df[p_col]
        numeric_rate = _numeric_coerce_rate(p_series)
        if numeric_rate < 0.8:
            confidence -= 10
            interpretations.append(
                f"The detected p-value column ('{p_col}') has low numeric parse rate ({_as_percent(numeric_rate)})."
            )
            recommendations.append(
                "Clean the p-value column (remove text/junk; keep numeric values like 0.001 or 1e-5)."
            )
            flag(
                "med",
                "P-value column may be messy",
                f"Only {_as_percent(numeric_rate)} of values look numeric.",
                "Clean/standardize p-values to numeric values.",
            )

    # FDR / q-values
    if p_col and not fdr_col:
        confidence -= 20
        interpretations.append(
            "P-values were detected without a multiple-testing correction (FDR/q-values). This increases false positive risk in high-dimensional metabolomics."
        )
        recommendations.append(
            "Add FDR/q-values (e.g., Benjamini–Hochberg) when many metabolites are tested."
        )
        flag(
            "med",
            "No FDR/q-values detected",
            "Multiple-testing correction isn’t available in this file.",
            "Add an FDR/q-value column (e.g., BH-adjusted p-values).",
        )
    elif p_col and fdr_col:
        interpretations.append(
            "Both p-values and FDR/q-values were detected, indicating statistically interpretable results with multiple-testing control."
        )

    # Effect size (FC/log2FC)
    if not fc_col and not log2fc_col:
        confidence -= 20
        interpretations.append(
            "No fold-change or log2 fold-change column was detected. Without effect size, practical significance and directionality are harder to interpret."
        )
        recommendations.append(
            "Include Fold Change (FC) or log2FC in your exported results."
        )
        flag(
            "high",
            "No effect size detected (FC/log2FC)",
            "Without FC/log2FC, effect size isn’t interpretable.",
            "Include Fold Change or log2FC in the results export.",
        )

    # If effect size exists but p-values do not, that's a common “bad” output
    if (fc_col or log2fc_col) and not p_col:
        # already penalized above, but add a targeted interpretation
        interpretations.append(
            "Effect-size values are present without p-values; this is typical of exploratory exports and should not be treated as statistical evidence."
        )
        recommendations.append(
            "Run a statistical test and include p-values (and ideally FDR) alongside effect sizes."
        )

    confidence = max(0, min(100, confidence))

    # ----------------------------
    # Build report (Markdown)
    # ----------------------------
    _safe_makedirs(report_path)

    md: list[str] = []
    md.append("# Metabolomics Validity Report\n")

    md.append("## Dataset Overview")
    md.append(f"- Number of rows (features): {n_rows}")
    md.append(f"- Number of columns: {n_cols}\n")

    md.append("## Detected Statistical Columns (schema-mapped)")
    md.append(f"- Feature / metabolite ID: {feature_col}")
    md.append(f"- Fold change (FC): {fc_col}")
    md.append(f"- log2FC: {log2fc_col}")
    md.append(f"- p-value: {p_col}")
    md.append(f"- FDR / q-value: {fdr_col}\n")

    # Show schema uncertainties so the app is “fail-soft” and transparent
    if sm.ambiguities:
        md.append("## Schema Ambiguities")
        for canon, cands in sm.ambiguities.items():
            md.append(f"- **{canon}** matched multiple columns: {', '.join(cands)}")
        md.append("")  # spacing

    if sm.missing:
        md.append("## Missing Canonical Fields")
        md.append("- " + ", ".join(sm.missing) + "\n")

    md.append("## Scientific Interpretation")
    if interpretations:
        for i in interpretations:
            md.append(f"- {i}")
    else:
        md.append("- No major statistical issues detected.")
    md.append("")

    md.append("## Recommendations")
    if recommendations:
        for r in recommendations:
            md.append(f"- {r}")
    else:
        md.append("- No immediate corrective actions required.")
    md.append("")

    md.append("## Overall Confidence Score")
    md.append(f"**{confidence} / 100**\n")

    # Optional: add a concise flags section (useful for UI summaries)
    if flags:
        md.append("## Flags")
        for f in flags:
            md.append(f"- **{f['severity'].upper()}** — {f['title']}: {f['why']}  \n  _Fix_: {f['fix']}")
        md.append("")

    report_text = "\n".join(md)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ----------------------------
    # Optional JSON output (machine-readable)
    # ----------------------------
    if json_path:
        _safe_makedirs(json_path)
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
                "scores": {k: [(c, float(s)) for c, s in v] for k, v in sm.scores.items()},
            },
            "analysis": {
                "confidence": confidence,
                "interpretations": interpretations,
                "recommendations": recommendations,
                "flags": flags,
            },
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2)

    return report_text


def main() -> None:
    """
    CLI-style entrypoint.
    Keeps compatibility with existing app.py that does: from main import main as run_audit
    """
    print("🔍 Starting metabolomics auditor...")
    print(f"Checking for input file at: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ results.csv not found")
        return

    _safe_makedirs(OUTPUT_PATH)
    run_audit(csv_path=INPUT_PATH, report_path=OUTPUT_PATH)
    print("✅ Validity report written to outputs/validity_report.md")


if __name__ == "__main__":
    main()