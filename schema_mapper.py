# schema_mapper.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Canonical fields Validex cares about
# ----------------------------
# NOTE: Avoid single-letter aliases ("p", "q") — too many false matches in real exports.
KNOWN_ALIASES: Dict[str, List[str]] = {
    "feature": [
        "metabolite", "metabolites", "compound", "compound_name", "name", "analyte",
        "feature", "feature_name", "metabolite_name", "id", "identifier",
    ],
    "p_value": [
        "pvalue", "p_value", "p-value", "p val", "p.val", "p_val", "pval", "p_val_raw",
        "p_uncorrected", "p_unadj", "unadjusted_p", "raw_p",
    ],
    "fdr": [
        "fdr", "fdr_bh", "bh_fdr", "qvalue", "q_value", "q-value", "q val", "q.val",
        "padj", "p_adj", "p-adj", "adj_p", "adj.p", "adjusted_p", "adjusted_pvalue",
        "adj_p_value", "adj_pvalue",
    ],
    # Keep linear FC separate from log2FC to avoid semantic errors.
    "fold_change": [
        "foldchange", "fold_change", "fold change", "fc", "ratio", "treatment_control_ratio",
    ],
    "log2fc": [
        "log2fc", "log2_fc", "log2 fold change", "log2_fold_change", "log2(fc)", "log2 fc",
        "log2ratio", "log2_ratio",
    ],
}


# Canonical numeric expectations (used for plausibility scoring)
_NUMERIC_EXPECTATIONS = {
    "p_value": {"min": 0.0, "max": 1.0},
    "fdr": {"min": 0.0, "max": 1.0},
    # fold_change/log2fc can vary widely; we score them differently.
}


def _norm_header(s: str) -> str:
    """
    Normalize a column header to improve matching across vendor/tools:
    - lowercase
    - trim
    - convert common separators to underscores
    - remove non-alphanumeric/underscore
    - collapse multiple underscores
    """
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^\w]+", "_", s)          # keep alnum + underscore
    s = re.sub(r"_+", "_", s).strip("_")   # collapse underscores
    return s


def _coerce_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
    """
    Try to coerce a series to numeric, returning (numeric_series, numeric_rate).
    Handles common lab/table artifacts like '<0.001', '1e-5', commas, etc.
    """
    if series is None:
        return pd.Series(dtype="float64"), 0.0

    s = series.copy()

    # If already numeric dtype, keep it
    if pd.api.types.is_numeric_dtype(s):
        numeric_rate = float(s.notna().mean()) if len(s) else 0.0
        return s.astype("float64"), numeric_rate

    # Strip common junk
    s = s.astype(str).str.strip()

    # Convert things like "<0.001" -> "0.001"
    s = s.str.replace(r"^[<>]=?\s*", "", regex=True)

    # Remove commas in numbers (rare but occurs)
    s = s.str.replace(",", "", regex=False)

    # Empty strings -> NA
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})

    num = pd.to_numeric(s, errors="coerce")
    numeric_rate = float(num.notna().mean()) if len(num) else 0.0
    return num, numeric_rate


def _range_score(num: pd.Series, expected_min: float, expected_max: float) -> float:
    """
    Score how plausible a numeric column is for a bounded metric (p-values, FDR).
    Returns 0..1.
    """
    if len(num) == 0:
        return 0.0
    x = num.dropna()
    if len(x) == 0:
        return 0.0

    in_range = ((x >= expected_min) & (x <= expected_max)).mean()
    return float(in_range)


def _fc_score(num: pd.Series) -> float:
    """
    Plausibility score for linear fold-change.
    Typical FC is positive; values near 0 or negative are suspicious for linear FC.
    Returns 0..1.
    """
    if len(num) == 0:
        return 0.0
    x = num.dropna()
    if len(x) == 0:
        return 0.0

    # Prefer mostly positive and not all ~0
    positive_rate = (x > 0).mean()
    nontrivial_rate = (x.abs() > 1e-9).mean()
    return float(0.7 * positive_rate + 0.3 * nontrivial_rate)


def _log2fc_score(num: pd.Series) -> float:
    """
    Plausibility score for log2FC.
    Log2FC can be negative; having both signs is common.
    Returns 0..1.
    """
    if len(num) == 0:
        return 0.0
    x = num.dropna()
    if len(x) == 0:
        return 0.0

    has_neg = (x < 0).any()
    has_pos = (x > 0).any()
    nontrivial_rate = (x.abs() > 1e-9).mean()

    # Strong signal if both pos and neg exist.
    sign_score = 1.0 if (has_neg and has_pos) else 0.6 if (has_neg or has_pos) else 0.0
    return float(0.7 * sign_score + 0.3 * nontrivial_rate)


def _header_match_score(norm_col: str, norm_alias: str) -> float:
    """
    Score header match strength between a normalized column name and a normalized alias.
    Returns 0..1.
    """
    if norm_col == norm_alias:
        return 1.0
    if norm_col.startswith(norm_alias) or norm_col.endswith(norm_alias):
        return 0.85
    if norm_alias in norm_col:
        return 0.7
    return 0.0


def _feature_score(series: pd.Series) -> float:
    """
    Heuristic plausibility score for a feature/metabolite identifier column.
    Prefer non-numeric, mostly unique, not empty.
    Returns 0..1.
    """
    if series is None or len(series) == 0:
        return 0.0

    # If numeric dtype, it's probably not a feature ID.
    if pd.api.types.is_numeric_dtype(series):
        return 0.0

    s = series.astype(str).str.strip()
    nonempty = (s != "").mean()

    # Uniqueness: features are often mostly unique.
    nunique = s.nunique(dropna=True)
    uniq_rate = nunique / max(len(s), 1)

    # Penalize columns that look like sample group labels (very low cardinality)
    # e.g., "case/control", "A/B", etc.
    low_cardinality_penalty = 0.0
    if nunique <= 10 and len(s) >= 20:
        low_cardinality_penalty = 0.25

    score = 0.6 * float(nonempty) + 0.4 * float(min(1.0, uniq_rate * 1.5))
    score = max(0.0, score - low_cardinality_penalty)
    return float(min(1.0, score))


@dataclass
class SchemaMap:
    """
    Result of schema detection.
    """
    canonical_to_original: Dict[str, str]
    canonical_to_normed: Dict[str, str]
    ambiguities: Dict[str, List[str]]         # canonical -> candidate original cols
    missing: List[str]                        # canonicals with no match
    scores: Dict[str, List[Tuple[str, float]]]  # canonical -> [(col, score), ...]

    def rename_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df with matched columns renamed to canonical names.
        Only renames matches; does not drop anything.
        """
        rename_map = {orig: canon for canon, orig in self.canonical_to_original.items()}
        return df.rename(columns=rename_map).copy()


def detect_schema(df: pd.DataFrame, aliases: Optional[Dict[str, List[str]]] = None) -> SchemaMap:
    """
    Detect canonical columns in df using:
      1) normalized header exact/contains matching
      2) numeric plausibility scoring (p/FDR in [0,1], etc.)
      3) deterministic tie-breaking
    """
    aliases = aliases or KNOWN_ALIASES

    cols: List[str] = [str(c) for c in df.columns]
    norm_cols: Dict[str, str] = {c: _norm_header(c) for c in cols}

    canonical_to_original: Dict[str, str] = {}
    canonical_to_normed: Dict[str, str] = {}
    ambiguities: Dict[str, List[str]] = {}
    missing: List[str] = []
    scores: Dict[str, List[Tuple[str, float]]] = {}

    for canonical, alias_list in aliases.items():
        norm_aliases = [_norm_header(canonical)] + [_norm_header(a) for a in alias_list]

        # Score every column against this canonical
        cand_scores: List[Tuple[str, float]] = []
        for col in cols:
            ncol = norm_cols[col]

            # Header score: max over aliases
            header_score = 0.0
            for na in norm_aliases:
                header_score = max(header_score, _header_match_score(ncol, na))

            if header_score == 0.0:
                continue

            # Data plausibility score
            data_score = 0.5  # neutral baseline
            if canonical in ("p_value", "fdr", "fold_change", "log2fc", "feature"):
                if canonical == "feature":
                    data_score = _feature_score(df[col])
                else:
                    num, numeric_rate = _coerce_numeric(df[col])

                    # Weighted by numeric rate
                    if canonical in ("p_value", "fdr"):
                        r = _range_score(num, _NUMERIC_EXPECTATIONS[canonical]["min"], _NUMERIC_EXPECTATIONS[canonical]["max"])
                        data_score = 0.7 * r + 0.3 * numeric_rate
                    elif canonical == "fold_change":
                        data_score = 0.7 * _fc_score(num) + 0.3 * numeric_rate
                    elif canonical == "log2fc":
                        data_score = 0.7 * _log2fc_score(num) + 0.3 * numeric_rate

            # Combine: header matters a lot, but data can break ties
            final = 0.65 * header_score + 0.35 * data_score
            cand_scores.append((col, float(final)))

        # Sort deterministically by score desc, then by column name
        cand_scores.sort(key=lambda x: (-x[1], x[0].lower()))
        scores[canonical] = cand_scores

        if not cand_scores:
            missing.append(canonical)
            continue

        # Collect candidates that are "close" to best score
        best_col, best_score = cand_scores[0]
        close = [c for c, s in cand_scores if s >= best_score - 0.08]  # within 0.08 of best

        if len(close) > 1:
            ambiguities[canonical] = close

        canonical_to_original[canonical] = best_col
        canonical_to_normed[canonical] = norm_cols[best_col]

    return SchemaMap(
        canonical_to_original=canonical_to_original,
        canonical_to_normed=canonical_to_normed,
        ambiguities=ambiguities,
        missing=missing,
        scores=scores,
    )


def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Backward-compatible API:
    returns canonical -> original column mapping.
    """
    sm = detect_schema(df)
    return sm.canonical_to_original


def apply_canonical_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMap]:
    """
    Convenience helper: detects schema and returns (renamed_df, schema_map).
    """
    sm = detect_schema(df)
    return sm.rename_df(df), sm