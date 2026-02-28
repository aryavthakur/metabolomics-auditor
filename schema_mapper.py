# schema_mapper.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import pandas as pd


# ------------------------------------------------------------
# Canonical fields Validex cares about (expand over time)
# ------------------------------------------------------------
# NOTE:
# - "fold_change" and "log2fc" are separated so we can prefer log2fc when present,
#   but still accept FC-only exports.
# - "feature" is optional but strongly recommended for readability / auditability.
KNOWN_ALIASES: Dict[str, List[str]] = {
    "feature": [
        "feature", "metabolite", "metabolite_name", "compound", "compound_name",
        "name", "id", "identifier", "annotation", "analyte", "biochemical",
        "peak", "peak_id", "m/z", "mz", "rt", "retention_time"
    ],
    "p_value": ["p", "pval", "pvalue", "p-value", "p.val", "p_value", "p value", "p_val"],
    "fdr": ["fdr", "q", "qval", "qvalue", "q-value", "q value", "adj.p", "padj", "adj_p", "adjusted_p", "adj p"],
    "fold_change": ["fc", "fold", "foldchange", "fold_change", "fold change"],
    "log2fc": ["log2fc", "log_fc", "log2_fold_change", "log2 fold change", "log2(fc)", "log2 fc", "log fold change", "logfc"],
}


# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------
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


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce to numeric in a forgiving way:
    - handles strings like "<0.001"
    - strips commas
    - treats blank-ish strings as NA
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[<>]=?\s*", "", regex=True)  # "<0.001" -> "0.001"
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def _numeric_rate(series: pd.Series) -> float:
    """Fraction of non-null values that can be parsed as numeric."""
    if series is None or len(series) == 0:
        return 0.0
    num = _coerce_numeric(series)
    return float(num.notna().mean())


def _pvalue_plausibility(series: pd.Series) -> float:
    """
    Score p-value plausibility:
    - numeric rate is required
    - values should mostly fall in [0,1]
    Returns 0..1.
    """
    num = _coerce_numeric(series)
    if len(num) == 0:
        return 0.0
    nr = float(num.notna().mean())
    if nr == 0.0:
        return 0.0
    in_range = float(((num >= 0) & (num <= 1)).mean())
    # Weight numeric-ness heavily, but require range plausibility too
    return 0.75 * nr + 0.25 * in_range


def _effect_size_plausibility(series: pd.Series) -> float:
    """
    Score effect-size plausibility:
    - numeric rate is required
    - we don't require a fixed range (FC can be huge), but we downweight absurd parse rates
    Returns 0..1.
    """
    nr = _numeric_rate(series)
    return nr


def _feature_plausibility(series: pd.Series) -> float:
    """
    Score feature/identifier plausibility:
    - prefer non-numeric-ish columns
    - prefer high uniqueness (but not required)
    Returns 0..1.
    """
    if series is None or len(series) == 0:
        return 0.0
    # If it's highly numeric, probably not a name column
    nr = _numeric_rate(series)
    uniqueness = float(series.astype(str).nunique(dropna=True) / max(1, len(series)))
    # We want low numeric rate, moderate uniqueness
    score = (1.0 - min(1.0, nr)) * 0.7 + min(1.0, uniqueness) * 0.3
    return float(max(0.0, min(1.0, score)))


# ------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------
@dataclass
class SchemaMap:
    """
    Result of schema detection.

    canonical_to_original:
      chosen best original column for each canonical (if any)
    canonical_to_normed:
      normalized form of chosen original column
    ambiguities:
      canonical -> list of original column names that were plausible matches
    missing:
      list of canonicals that had no plausible match
    scores:
      canonical -> list of (original_col, score) sorted high->low.
      This lets main.py serialize scores for transparency (optional).
    """
    canonical_to_original: Dict[str, str]
    canonical_to_normed: Dict[str, str]
    ambiguities: Dict[str, List[str]]
    missing: List[str]
    scores: Dict[str, List[Tuple[str, float]]]

    def rename_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df with matched columns renamed to canonical names.
        Only renames matches; does not drop anything.
        """
        rename_map = {orig: canon for canon, orig in self.canonical_to_original.items()}
        return df.rename(columns=rename_map).copy()


# ------------------------------------------------------------
# Core schema detection
# ------------------------------------------------------------
def detect_schema(df: pd.DataFrame, aliases: Optional[Dict[str, List[str]]] = None) -> SchemaMap:
    """
    Detect canonical columns in df using:
    1) robust header normalization + alias sets
    2) data-aware scoring to choose the best match when multiple candidates exist

    Returns SchemaMap:
      - canonical_to_original mapping
      - ambiguities (if multiple candidates match)
      - missing canonicals
      - scores per canonical (candidate ranking)
    """
    aliases = aliases or KNOWN_ALIASES

    cols: List[str] = [str(c) for c in df.columns]
    normed_cols: Dict[str, List[str]] = {}
    for c in cols:
        normed_cols.setdefault(_norm_header(c), []).append(c)

    canonical_to_original: Dict[str, str] = {}
    canonical_to_normed: Dict[str, str] = {}
    ambiguities: Dict[str, List[str]] = {}
    missing: List[str] = []
    scores: Dict[str, List[Tuple[str, float]]] = {}

    def header_match_candidates(canon: str, alias_list: Iterable[str]) -> List[str]:
        """Return all original columns whose normalized header matches any alias."""
        candidates = [canon] + list(alias_list)
        found: List[str] = []
        for a in candidates:
            a_norm = _norm_header(a)
            if a_norm in normed_cols:
                found.extend(normed_cols[a_norm])
        # Deduplicate preserving order
        seen = set()
        out = []
        for x in found:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def score_candidate(canon: str, col: str) -> float:
        """
        Data-aware scoring 0..1:
        - p_value / fdr: prefer numeric + range plausibility
        - fold_change / log2fc: prefer numeric
        - feature: prefer non-numeric + reasonably unique
        """
        s = df[col] if col in df.columns else None
        if s is None:
            return 0.0

        if canon == "p_value":
            return _pvalue_plausibility(s)
        if canon == "fdr":
            # fdr behaves like p-values: usually [0,1]
            return _pvalue_plausibility(s)
        if canon in ("fold_change", "log2fc"):
            return _effect_size_plausibility(s)
        if canon == "feature":
            return _feature_plausibility(s)

        # fallback: prefer numeric-ish columns slightly
        return _numeric_rate(s) * 0.5

    for canon, alias_list in aliases.items():
        cands = header_match_candidates(canon, alias_list)

        if not cands:
            missing.append(canon)
            scores[canon] = []
            continue

        ranked = [(c, float(score_candidate(canon, c))) for c in cands]
        ranked.sort(key=lambda x: x[1], reverse=True)

        scores[canon] = ranked

        # Determine ambiguity: more than one candidate with decent score
        # Thresholds:
        # - if top score is weak, still pick deterministic but mark as ambiguous if others close
        top_col, top_score = ranked[0]

        # "plausible" means score >= 0.35 for stat cols, >= 0.25 for feature
        plausible_cut = 0.25 if canon == "feature" else 0.35
        plausible = [c for c, sc in ranked if sc >= plausible_cut]

        if len(plausible) > 1:
            ambiguities[canon] = plausible

        canonical_to_original[canon] = top_col
        canonical_to_normed[canon] = _norm_header(top_col)

    return SchemaMap(
        canonical_to_original=canonical_to_original,
        canonical_to_normed=canonical_to_normed,
        ambiguities=ambiguities,
        missing=missing,
        scores=scores,
    )


# ------------------------------------------------------------
# Backward-compatible helpers
# ------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Backward-compatible API:
    returns canonical -> original column mapping.
    """
    return detect_schema(df).canonical_to_original


def apply_canonical_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMap]:
    """
    Convenience helper: detects schema and returns (renamed_df, schema_map).
    """
    sm = detect_schema(df)
    return sm.rename_df(df), sm
