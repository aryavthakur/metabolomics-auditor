# schema_mapper.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ------------------------------------------------------------
# Canonical fields Validex cares about (expand over time)
# ------------------------------------------------------------
# Notes:
# - We keep FC and log2FC separate.
# - We intentionally include a lot of real-world export header variants (MetaboAnalyst,
#   Skyline/other pipelines, vendor outputs, etc.).
KNOWN_ALIASES: Dict[str, List[str]] = {
    "feature": [
        "feature",
        "metabolite",
        "metabolite_name",
        "metabolite name",
        "compound",
        "compound_name",
        "compound name",
        "analyte",
        "biochemical",
        "biochemical name",
        "name",
        "identifier",
        "id",
        "annotation",
        "feature_id",
        "feature id",
        "peak",
        "peak_id",
        "peak id",
        "library_id",
        "library id",
        "hmdb",
        "kegg",
        "inchi",
        "smiles",
    ],
    "p_value": [
        "p",
        "pval",
        "pvalue",
        "p_value",
        "p value",
        "p-val",
        "p-value",
        "p.val",
        "p.value",
        "p_val",
        "pvalue (t-test)",
        "p_value (t-test)",
        "raw_p",
        "raw p",
        "unadjusted p",
        "unadjusted_p",
    ],
    "fdr": [
        "fdr",
        "fdr_adj",
        "fdr adjusted",
        "fdr_adjusted",
        "fdr adjusted p-value",
        "fdr adjusted p value",
        "fdr_adjusted_p_value",
        "fdr_adjusted_pvalue",
        "q",
        "qval",
        "qvalue",
        "q_value",
        "q value",
        "q-value",
        "adj.p",
        "adj_p",
        "adj p",
        "adj p-value",
        "adj p value",
        "adj_p_value",
        "adjusted_p",
        "adjusted p",
        "adjusted p-value",
        "adjusted p value",
        "adjusted_p_value",
        "padj",
        "p_adj",
        "p-adj",
        "p adj",
        "bh",
        "bh_fdr",
        "bh fdr",
        "bh_adj",
        "bh adj",
        "fdr_bh",
        "fdr bh",
        "adj.p.val",
        "adj.p.val.",
    ],
    "fold_change": [
        "fc",
        "fold",
        "foldchange",
        "fold_change",
        "fold change",
        "fold-change",
        "ratio",
        "group_ratio",
        "group ratio",
        "mean_ratio",
        "mean ratio",
    ],
    "log2fc": [
        "log2fc",
        "log2_fc",
        "log2 fold change",
        "log2_fold_change",
        "log2(fold_change)",
        "log2(fc)",
        "log2 fc",
        "log_fc",
        "logfc",
        "log fold change",
        "log_fold_change",
    ],
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
    s = re.sub(r"[^\w]+", "_", s)  # keep alnum + underscore
    s = re.sub(r"_+", "_", s).strip("_")
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
    """Fraction of values coercible to numeric (0..1)."""
    if series is None or len(series) == 0:
        return 0.0
    num = _coerce_numeric(series)
    return float(num.notna().mean())


def _pvalue_plausibility(series: pd.Series) -> float:
    """
    Score p-value plausibility:
    - numeric rate
    - values mostly in [0,1]
    Returns 0..1.
    """
    num = _coerce_numeric(series)
    if len(num) == 0:
        return 0.0
    nr = float(num.notna().mean())
    if nr == 0.0:
        return 0.0
    in_range = float(((num >= 0) & (num <= 1)).mean())
    return float(max(0.0, min(1.0, 0.75 * nr + 0.25 * in_range)))


def _effect_size_plausibility(series: pd.Series) -> float:
    """
    Score effect-size plausibility:
    - numeric rate is required
    Returns 0..1.
    """
    return float(max(0.0, min(1.0, _numeric_rate(series))))


def _feature_plausibility(series: pd.Series) -> float:
    """
    Score feature/identifier plausibility:
    - prefer non-numeric-ish columns
    - prefer reasonably unique values
    Returns 0..1.
    """
    if series is None or len(series) == 0:
        return 0.0

    nr = _numeric_rate(series)
    # uniqueness among non-null string values
    s = series.dropna().astype(str)
    uniq = float(s.nunique() / max(1, len(s))) if len(s) else 0.0

    # Encourage low numeric rate + moderate uniqueness
    score = (1.0 - min(1.0, nr)) * 0.7 + min(1.0, uniq) * 0.3
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
    """
    canonical_to_original: Dict[str, str]
    canonical_to_normed: Dict[str, str]
    ambiguities: Dict[str, List[str]]
    missing: List[str]
    scores: Dict[str, List[Tuple[str, float]]]

    def rename_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with matched columns renamed to canonical names."""
        rename_map = {orig: canon for canon, orig in self.canonical_to_original.items()}
        return df.rename(columns=rename_map).copy()


# ------------------------------------------------------------
# Core schema detection
# ------------------------------------------------------------
def detect_schema(
    df: pd.DataFrame,
    aliases: Optional[Dict[str, List[str]]] = None,
) -> SchemaMap:
    """
    Detect canonical columns in df using:
    1) robust header normalization + alias sets
    2) fuzzy header matching (substring containment)
    3) data-aware scoring to choose the best match when multiple candidates exist

    Returns SchemaMap with:
      - canonical_to_original mapping
      - ambiguities (if multiple candidates match)
      - missing canonicals
      - scores per canonical (candidate ranking)
    """
    aliases = aliases or KNOWN_ALIASES
    cols: List[str] = [str(c) for c in df.columns]
    normed: Dict[str, str] = {c: _norm_header(c) for c in cols}

    # Reverse index for exact normalized header matches
    norm_to_originals: Dict[str, List[str]] = {}
    for c in cols:
        norm_to_originals.setdefault(normed[c], []).append(c)

    canonical_to_original: Dict[str, str] = {}
    canonical_to_normed: Dict[str, str] = {}
    ambiguities: Dict[str, List[str]] = {}
    missing: List[str] = []
    scores: Dict[str, List[Tuple[str, float]]] = {}

    def header_score(col_norm: str, alias_norm: str) -> float:
        """
        Header similarity score (0..1).
        - exact match => 1.0
        - substring match (alias inside col or col inside alias) => 0.65
        - token overlap (weak) => 0.45
        """
        if col_norm == alias_norm:
            return 1.0
        if alias_norm and (alias_norm in col_norm or col_norm in alias_norm):
            return 0.65

        col_tokens = set(col_norm.split("_"))
        alias_tokens = set(alias_norm.split("_"))
        if not col_tokens or not alias_tokens:
            return 0.0
        overlap = len(col_tokens & alias_tokens) / max(1, len(alias_tokens))
        if overlap >= 0.6:
            return 0.45
        if overlap >= 0.4:
            return 0.25
        return 0.0

    def candidate_columns_for(canon: str, alias_list: Iterable[str]) -> List[Tuple[str, float]]:
        """
        Generate (col, best_header_score) candidates for a canonical field.
        We search:
        - exact normalized matches against aliases
        - fuzzy matches by scanning all columns (small df, ok)
        """
        alias_norms = {_norm_header(canon)} | {_norm_header(a) for a in alias_list}

        cand: Dict[str, float] = {}

        # Exact matches first
        for a_norm in alias_norms:
            for orig in norm_to_originals.get(a_norm, []):
                cand[orig] = max(cand.get(orig, 0.0), 1.0)

        # Fuzzy scan
        for col in cols:
            cn = normed[col]
            best = 0.0
            for a_norm in alias_norms:
                best = max(best, header_score(cn, a_norm))
            if best > 0.0:
                cand[col] = max(cand.get(col, 0.0), best)

        # Return sorted by header score (desc) so we evaluate stronger header matches first
        out = [(c, float(hs)) for c, hs in cand.items()]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def data_score(canon: str, col: str) -> float:
        s = df[col] if col in df.columns else None
        if s is None:
            return 0.0

        if canon == "p_value":
            return _pvalue_plausibility(s)
        if canon == "fdr":
            return _pvalue_plausibility(s)
        if canon in ("fold_change", "log2fc"):
            return _effect_size_plausibility(s)
        if canon == "feature":
            return _feature_plausibility(s)

        return 0.0

    def combined_score(canon: str, col: str, hscore: float) -> float:
        """
        Combine header score + data plausibility.
        Weighting:
        - header matters (prevents grabbing random numeric columns)
        - data plausibility matters more for stat columns
        """
        dscore = data_score(canon, col)

        # feature headers can vary wildly; let data plausibility drive more
        if canon == "feature":
            w_h, w_d = 0.35, 0.65
        else:
            w_h, w_d = 0.45, 0.55

        return float(max(0.0, min(1.0, w_h * hscore + w_d * dscore)))

    # minimum confidence to accept a mapping at all
    min_accept: Dict[str, float] = {
        "feature": 0.22,
        "p_value": 0.35,
        "fdr": 0.35,
        "fold_change": 0.30,
        "log2fc": 0.30,
    }

    for canon, alias_list in aliases.items():
        cands = candidate_columns_for(canon, alias_list)

        if not cands:
            missing.append(canon)
            scores[canon] = []
            continue

        ranked = [(col, combined_score(canon, col, h)) for col, h in cands]
        ranked.sort(key=lambda x: x[1], reverse=True)
        scores[canon] = [(c, float(s)) for c, s in ranked]

        top_col, top_score = ranked[0]
        if top_score < min_accept.get(canon, 0.30):
            missing.append(canon)
            continue

        # Ambiguity: if there are multiple plausible close contenders
        plausible_cut = max(min_accept.get(canon, 0.30), top_score - 0.10)
        plausible = [c for c, sc in ranked if sc >= plausible_cut]
        if len(plausible) > 1:
            ambiguities[canon] = plausible

        canonical_to_original[canon] = top_col
        canonical_to_normed[canon] = normed[top_col]

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
    """Backward-compatible API: returns canonical -> original mapping."""
    return detect_schema(df).canonical_to_original


def apply_canonical_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMap]:
    """Convenience helper: returns (renamed_df, schema_map)."""
    sm = detect_schema(df)
    return sm.rename_df(df), sm
