# main.py — Validex (context-aware + Streamlit-friendly)
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from schema_mapper import apply_canonical_schema, detect_schema

# Default IO (Streamlit writes/reads these)
INPUT_PATH = "inputs/results.csv"
OUTPUT_MD_PATH = "outputs/validity_report.md"
OUTPUT_JSON_PATH = "outputs/validity_report.json"
CONTEXT_JSON_PATH = "inputs/context.json"  # optional fallback (if present)


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
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_percent(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
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
    """Returns (numeric_rate, in_0_1_rate). Useful for p-values and FDR/q-values."""
    nr = _numeric_parse_rate(series)
    r01 = _fraction_in_range(series, 0.0, 1.0)
    return nr, r01


def _is_probably_log2fc(colname: Optional[str]) -> bool:
    if not colname:
        return False
    s = colname.lower()
    return ("log2" in s) or ("log2fc" in s) or ("log_fc" in s) or ("logfc" in s)


def _norm_ctx(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize the Streamlit UI context to a stable schema.
    Expected keys from your fancy app.py:
      design_groups: "two" | "multi"
      paired: bool
      longitudinal: bool
      targeted: bool
      goal: "confirmatory" | "exploratory"
      batch_expected: bool
      transform: "unknown"|"none"|"log"|"log2"|"auto"
      alpha: float
      comparison_label: str
      notes: str
    """
    ctx = context or {}
    out: Dict[str, Any] = {}

    out["design_groups"] = str(ctx.get("design_groups", "two")).lower()
    if out["design_groups"] not in ("two", "multi"):
        out["design_groups"] = "two"

    out["paired"] = bool(ctx.get("paired", False))
    out["longitudinal"] = bool(ctx.get("longitudinal", False))
    out["targeted"] = bool(ctx.get("targeted", False))

    out["goal"] = str(ctx.get("goal", "confirmatory")).lower()
    if out["goal"] not in ("confirmatory", "exploratory"):
        out["goal"] = "confirmatory"

    out["batch_expected"] = bool(ctx.get("batch_expected", False))

    out["transform"] = str(ctx.get("transform", "unknown")).lower()
    if out["transform"] not in ("unknown", "none", "log", "log2", "auto"):
        out["transform"] = "unknown"

    try:
        out["alpha"] = float(ctx.get("alpha", 0.05))
    except Exception:
        out["alpha"] = 0.05

    out["comparison_label"] = str(ctx.get("comparison_label", "")).strip()
    out["notes"] = str(ctx.get("notes", "")).strip()

    return out


def _context_expectations(ctx: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    """
    Convert context into scoring expectations (this is what makes the score change).
    """
    targeted = bool(ctx.get("targeted", False))
    goal = str(ctx.get("goal", "confirmatory"))
    design_groups = str(ctx.get("design_groups", "two"))

    # "High-dimensional" heuristic: untargeted OR just lots of rows
    high_dim = (not targeted) and (n_features >= 50)

    # Base expectations
    require_p = True
    require_effect = True
    require_fdr = high_dim  # untargeted/high-dim should have correction
    # Exploratory can be slightly looser about effect size (but still preferred)
    effect_penalty_scale = 0.6 if goal == "exploratory" else 1.0
    fdr_penalty_scale = 0.25 if targeted else 1.0  # targeted = small penalty if missing

    # If multi-group confirmatory: strongly prefer multiple-testing correction too
    if design_groups == "multi" and goal == "confirmatory":
        require_fdr = True

    return {
        "high_dim": high_dim,
        "require_p": require_p,
        "require_effect": require_effect,
        "require_fdr": require_fdr,
        "effect_penalty_scale": effect_penalty_scale,
        "fdr_penalty_scale": fdr_penalty_scale,
    }


# ----------------------------
# Core audit
# ----------------------------
def run_audit(
    csv_path: str = INPUT_PATH,
    report_path: str = OUTPUT_MD_PATH,
    json_path: Optional[str] = OUTPUT_JSON_PATH,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,  # allows older callers to pass report_md_path/report_json_path without crashing
) -> str:
    """
    Validex audit entrypoint (matches your Streamlit UI call signature):

      run_audit(csv_path=..., report_path=..., json_path=..., context=...)

    Back-compat:
      - report_md_path / report_json_path are accepted via kwargs.
    """
    # Backward-compat arg names
    if "report_md_path" in kwargs and report_path == OUTPUT_MD_PATH:
        report_path = str(kwargs["report_md_path"])
    if "report_json_path" in kwargs and (json_path == OUTPUT_JSON_PATH or json_path is None):
        json_path = kwargs["report_json_path"]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    # Optional context: explicit argument wins, else inputs/context.json if present
    ctx_raw = context or _read_optional_json(CONTEXT_JSON_PATH)
    ctx = _norm_ctx(ctx_raw)

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

    # Safety: sometimes fold_change alias list matches log2fc-like headers
    if fc_col and _is_probably_log2fc(fc_col) and not log2fc_col:
        log2fc_col = fc_col
        fc_col = None

    # ----------------------------
    # Context expectations (THIS is what makes score change)
    # ----------------------------
    exp = _context_expectations(ctx, n_features=n_rows)

    # ----------------------------
    # Scoring + flags
    # ----------------------------
    confidence = 100
    interpretations: list[str] = []
    recommendations: list[str] = []
    flags: list[Dict[str, Any]] = []

    def add_flag(severity: str, title: str, why: str, fix: str, penalty: int = 0) -> None:
        nonlocal confidence
        if penalty:
            confidence -= penalty
        flags.append(
            {
                "severity": severity,
                "title": title,
                "why": why,
                "fix": fix,
                "penalty": penalty,
            }
        )

    # ---- Feature column (mostly readability / auditability)
    if not feature_col:
        add_flag(
            "med",
            "No metabolite/feature ID detected",
            "A feature identifier improves interpretability and export consistency.",
            "Add a column like 'Metabolite', 'Compound', or 'Feature' as a human-readable identifier.",
            penalty=10,
        )
        interpretations.append(
            "No clear metabolite/feature identifier column was detected; results are harder to audit and cite."
        )

    # ---- p-values (almost always required)
    if exp["require_p"] and not p_col:
        add_flag(
            "high",
            "No p-values detected",
            "You can’t assess statistical significance; results are exploratory only.",
            "Include a p-value column from your statistical pipeline (t-test/ANOVA/linear/mixed model).",
            penalty=45,
        )
        interpretations.append(
            "No p-values were detected. Without p-values, statistical significance cannot be assessed."
        )
    elif p_col:
        p_nr, p_r01 = _detect_pvalue_like_issues(df[p_col])
        if p_nr < 0.80:
            add_flag(
                "med",
                "P-value column may be messy",
                f"Only {_as_percent(p_nr)} of values look numeric.",
                "Standardize p-values to numeric values (e.g., 0.001 or 1e-5). Remove text/junk.",
                penalty=10,
            )
        if p_r01 < 0.90:
            add_flag(
                "med",
                "P-values out of range",
                "Many values are outside the valid p-value range [0,1].",
                "Check if this column is -log10(p) or another statistic; export true p-values.",
                penalty=10,
            )

    # ---- FDR/q-values (context dependent)
    if p_col and exp["require_fdr"] and not fdr_col:
        base_pen = 20
        scaled = int(round(base_pen * float(exp["fdr_penalty_scale"])))
        add_flag(
            "med",
            "No FDR/q-values detected",
            "Given your context (high-dimensional / untargeted / multi-group confirmatory), missing correction is a common reviewer red flag.",
            "Add an FDR/q-value column (e.g., Benjamini–Hochberg).",
            penalty=scaled,
        )
        interpretations.append(
            "P-values were detected without multiple-testing correction (FDR/q-values), which increases false positive risk."
        )
    elif fdr_col:
        fdr_nr, fdr_r01 = _detect_pvalue_like_issues(df[fdr_col])
        if fdr_nr < 0.80:
            add_flag(
                "low",
                "FDR column may be messy",
                f"Only {_as_percent(fdr_nr)} of values look numeric.",
                "Standardize q-values/FDR values to numeric values between 0 and 1.",
                penalty=5,
            )
        if fdr_r01 < 0.90:
            add_flag(
                "low",
                "FDR values out of range",
                "Some values are outside [0,1].",
                "Ensure this is truly an adjusted p-value / q-value column (0..1).",
                penalty=5,
            )

    # ---- Effect size (context dependent severity)
    eff_col = log2fc_col or fc_col
    if exp["require_effect"] and not eff_col:
        base_pen = 25
        scaled = int(round(base_pen * float(exp["effect_penalty_scale"])))
        add_flag(
            "high" if ctx.get("goal") == "confirmatory" else "med",
            "No effect size detected (FC/log2FC)",
            "Without effect size (FC/log2FC), directionality and practical significance are harder to interpret.",
            "Include Fold Change (FC) and/or log2FC in the export.",
            penalty=scaled,
        )
        interpretations.append(
            "No fold-change (FC) or log2FC column was detected; practical significance is harder to interpret."
        )
    elif eff_col:
        nr = _numeric_parse_rate(df[eff_col])
        if nr < 0.80:
            add_flag(
                "low",
                "Effect size column may be messy",
                f"Only {_as_percent(nr)} of values look numeric.",
                "Ensure effect sizes are numeric (remove annotations/strings).",
                penalty=5,
            )

    # ---- Context-only mismatch checks (these directly change score)
    # Paired/longitudinal: we can't *prove* the test used, but we can flag missing design metadata + empty notes
    if (ctx.get("paired") or ctx.get("longitudinal")):
        if not ctx.get("notes"):
            add_flag(
                "low",
                "Design requires paired/repeated modeling",
                "Context indicates paired/repeated measures, but no notes were provided to confirm paired tests/mixed models were used.",
                "Add a short note documenting paired t-test / repeated measures ANOVA / mixed-effects model used.",
                penalty=10,
            )
            recommendations.append(
                "Context note: Paired/repeated designs should use paired tests or mixed-effects models (not independent tests)."
            )

    # Batch effects expected: encourage documentation or batch/QC columns
    if ctx.get("batch_expected") and not ctx.get("notes"):
        add_flag(
            "low",
            "Batch effects expected",
            "Context indicates batch effects are likely, but no notes were provided to confirm correction/normalization.",
            "Add a note describing batch correction/QC normalization (or include batch/QC fields if available).",
            penalty=7,
        )

    # Transform unknown/none: small assumption-risk penalty
    if ctx.get("transform") in ("unknown", "none"):
        add_flag(
            "low",
            "Transform not specified",
            "Metabolomics often uses log/log2 transforms; unspecified transform makes assumption checking harder.",
            "Document transform (log/log2/none) used prior to hypothesis testing.",
            penalty=5,
        )

    # De-dupe recommendations (preserve order)
    if ctx.get("design_groups") == "multi":
        recommendations.append(
            "Context note: For 3+ groups, ANOVA/Welch ANOVA/Kruskal–Wallis (or linear models) are typical; avoid many pairwise t-tests without correction."
        )
    if exp["high_dim"] and p_col and not fdr_col:
        recommendations.append(
            "High-dimensional context: Apply multiple-testing correction (BH-FDR) when testing many metabolites/features."
        )
    if not eff_col:
        recommendations.append("Include FC/log2FC so effect size and directionality can be interpreted.")
    if not feature_col:
        recommendations.append("Add a metabolite/feature identifier column for auditability.")
    if not p_col:
        recommendations.append("Export univariate/modeled results including p-values (and preferably q-values).")

    seen = set()
    recommendations = [r for r in recommendations if not (r in seen or seen.add(r))]

    confidence = max(0, min(100, confidence))

    # ----------------------------
    # Build Markdown report
    # ----------------------------
    _safe_makedirs_for_file(report_path)

    md: list[str] = []
    md.append("# Validex Audit Report\n")
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

    # Always include context + expectations so you can verify it’s being used
    md.append("## Context (used for scoring)")
    if ctx_raw:
        for k, v in ctx.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- (none provided)")
    md.append("")
    md.append("## Context-derived expectations")
    md.append(f"- High-dimensional expectation: {exp['high_dim']}")
    md.append(f"- Require p-values: {exp['require_p']}")
    md.append(f"- Require FDR/q-values: {exp['require_fdr']} (penalty scale: {exp['fdr_penalty_scale']})")
    md.append(f"- Require effect size (FC/log2FC): {exp['require_effect']} (penalty scale: {exp['effect_penalty_scale']})")
    md.append("")

    md.append("## Scientific Interpretation")
    if flags:
        # show key takeaways from flags (readable)
        for f in flags:
            md.append(f"- {f['title']}: {f['why']}")
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
                f"  _Fix_: {f['fix']}  \n"
                f"  _Penalty_: {f.get('penalty', 0)}"
            )
        md.append("")

    report_text = "\n".join(md)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ----------------------------
    # Optional JSON output
    # ----------------------------
    if json_path:
        _safe_makedirs_for_file(str(json_path))
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
                "scores": {k: [(c, float(s)) for c, s in v] for k, v in getattr(sm, "scores", {}).items()},
            },
            "context": ctx_raw,
            "context_normalized": ctx,
            "expectations": exp,
            "analysis": {
                "confidence": confidence,
                "flags": flags,
                "recommendations": recommendations,
            },
        }
        with open(str(json_path), "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2)

    return report_text


def main() -> None:
    """
    CLI entrypoint.
    Streamlit should call run_audit(), not this.
    """
    print("🔍 Starting Validex...")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input not found: {INPUT_PATH}")
        return

    _safe_makedirs_for_file(OUTPUT_MD_PATH)
    run_audit(
        csv_path=INPUT_PATH,
        report_path=OUTPUT_MD_PATH,
        json_path=OUTPUT_JSON_PATH,
        context=_read_optional_json(CONTEXT_JSON_PATH),
    )
    print(f"✅ Report written: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()