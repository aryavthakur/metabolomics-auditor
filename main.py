# main.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from schema_mapper import apply_canonical_schema, detect_schema

# NEW: context support
from context_engine import Context, context_expectations, context_narrative, default_context

INPUT_PATH = "inputs/results.csv"
OUTPUT_PATH = "outputs/validity_report.md"


def _safe_makedirs(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _as_percent(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
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
    context: Optional[dict] = None,
) -> str:
    """
    Validex audit entrypoint.

    - Reads csv_path
    - Detects schema via schema_mapper (header + data-aware scoring)
    - Writes markdown report to report_path
    - Optionally writes a machine-readable JSON payload to json_path
    - Accepts an optional user-provided context dict that changes expectations/scoring
    - Returns markdown report text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    # ----------------------------
    # Context (fail-soft)
    # ----------------------------
    ctx_obj = default_context()
    if isinstance(context, dict):
        try:
            ctx_obj = Context(**context)
        except Exception:
            # Keep defaults if invalid/partial context
            ctx_obj = default_context()

    cx = context_expectations(ctx_obj)

    # ----------------------------
    # Detect + canonicalize schema
    # ----------------------------
    sm = detect_schema(df)

    # Canonicalize columns for downstream checks (non-destructive)
    canonical_df, sm2 = apply_canonical_schema(df)  # noqa: F841  (kept for future downstream checks)

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
    interpretations: list[str] = []
    recommendations: list[str] = []
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

    # p-values (context-aware)
    if cx["expect_p_values"] and not p_col:
        confidence -= 35
        interpretations.append(
            "Context suggests inferential claims are expected, but no p-values were detected. Treat results as exploratory only."
        )
        recommendations.append(
            "Export statistical test results that include a p-value column (and specify the test used)."
        )
        flag(
            "high",
            "Missing p-values (context-required)",
            "Given the study context, inferential statistics are expected.",
            "Include p-values from your statistical test output.",
        )
    elif p_col:
        # Data sanity: p-values should be mostly numeric
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

    # FDR / q-values (context-aware)
    if cx["expect_fdr"] and p_col and not fdr_col:
        confidence -= 20
        interpretations.append(
            "Context implies high-dimensional testing (untargeted), but no multiple-testing correction (FDR/q-values) was detected."
        )
        recommendations.append(
            "Add FDR/q-values (e.g., Benjamini–Hochberg) when many metabolites/features are tested."
        )
        flag(
            "med",
            "Missing FDR/q-values (context-required)",
            "Untargeted metabolomics typically requires multiple-testing correction.",
            "Add an FDR/q-value column (BH-adjusted p-values).",
        )
    elif p_col and fdr_col:
        interpretations.append(
            "Both p-values and FDR/q-values were detected, indicating multiple-testing control is present."
        )

    # Effect size (FC/log2FC) (context-aware)
    if cx["expect_effect_size"] and not fc_col and not log2fc_col:
        confidence -= 20
        interpretations.append(
            "No fold-change or log2 fold-change column was detected. Without effect size, practical significance and directionality are harder to interpret."
        )
        recommendations.append("Include Fold Change (FC) or log2FC in your exported results.")
        flag(
            "high",
            "Missing effect size (FC/log2FC)",
            "Reviewers expect effect size alongside significance in metabolomics reporting.",
            "Include Fold Change or log2FC in the results export.",
        )

    # If effect size exists but p-values do not, add targeted note
    if (fc_col or log2fc_col) and not p_col and cx["expect_p_values"]:
        interpretations.append(
            "Effect-size values are present without p-values; do not treat this as statistical evidence without inferential testing."
        )
        recommendations.append(
            "Run a statistical test and include p-values (and ideally FDR) alongside effect sizes."
        )

    # Paired/longitudinal caution note (context-only)
    if cx["expect_paired_awareness"]:
        interpretations.append(
            "Design is paired/longitudinal: ensure your statistical test accounts for within-subject correlation (paired tests or mixed-effects models)."
        )
        recommendations.append(
            "If not already done, use paired t-test / repeated-measures ANOVA / mixed-effects model instead of independent tests."
        )
        flag(
            "low",
            "Paired/longitudinal design check",
            "Independent-sample tests on repeated measures inflate error.",
            "Confirm paired/mixed modeling was used and documented.",
        )

    # Batch effects caution note (context-only)
    if cx["expect_batch_handling"]:
        interpretations.append(
            "Batch effects are likely: ensure normalization/batch correction (or QC-based correction) is documented to avoid instrument-driven differences."
        )
        recommendations.append(
            "Document batch correction approach (QC-based normalization, ComBat, mixed models, etc.)."
        )
        flag(
            "low",
            "Batch effects check",
            "Uncorrected batches can masquerade as biological signal.",
            "Add batch correction/normalization details.",
        )

    # Small-n caution note
    if cx["small_n_risk"]:
        confidence -= 5
        interpretations.append(
            "Small sample size increases instability and false discoveries; emphasize effect sizes and conservative interpretation."
        )
        recommendations.append(
            "Consider nonparametric tests/bootstrapping and report confidence intervals where possible."
        )
        flag(
            "med",
            "Small-n risk",
            "Small n reduces power and increases variance in estimates.",
            "Use conservative stats + effect sizes + CIs.",
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

    md.append("## Context (user-provided)")
    for line in context_narrative(ctx_obj):
        md.append(f"- {line}")
    md.append("")

    md.append("## Detected Statistical Columns (schema-mapped)")
    md.append(f"- Feature / metabolite ID: {feature_col}")
    md.append(f"- Fold change (FC): {fc_col}")
    md.append(f"- log2FC: {log2fc_col}")
    md.append(f"- p-value: {p_col}")
    md.append(f"- FDR / q-value: {fdr_col}\n")

    # Transparent “fail-soft” mapping notes
    if getattr(sm, "ambiguities", None):
        if sm.ambiguities:
            md.append("## Schema Ambiguities")
            for canon, cands in sm.ambiguities.items():
                md.append(f"- **{canon}** matched multiple columns: {', '.join(cands)}")
            md.append("")

    if getattr(sm, "missing", None):
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

    if flags:
        md.append("## Flags")
        for f in flags:
            md.append(
                f"- **{f['severity'].upper()}** — {f['title']}: {f['why']}  \n"
                f"  _Fix_: {f['fix']}"
            )
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
            "context": ctx_obj.to_dict(),
            "context_expectations": cx,
            "detected": {
                "feature": feature_col,
                "fold_change": fc_col,
                "log2fc": log2fc_col,
                "p_value": p_col,
                "fdr": fdr_col,
            },
            "schema": {
                "canonical_to_original": getattr(sm, "canonical_to_original", {}),
                "missing": getattr(sm, "missing", []),
                "ambiguities": getattr(sm, "ambiguities", {}),
                "scores": {
                    k: [(c, float(s)) for c, s in v]
                    for k, v in getattr(sm, "scores", {}).items()
                },
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
    Streamlit should import run_audit directly.
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
