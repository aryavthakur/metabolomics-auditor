# schema_mapper.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Canonical fields your auditor cares about (expand over time)
KNOWN_ALIASES: Dict[str, List[str]] = {
    "p_value": ["p", "pval", "pvalue", "p-value", "p.val", "p_value", "p value"],
    "fdr": ["fdr", "q", "qval", "qvalue", "q-value", "adj.p", "padj", "adj_p", "q value"],
    "fold_change": ["fc", "fold", "foldchange", "fold_change", "fold change", "logfc", "log2fc", "log2(fc)", "log2 fc"],
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
    s = s.strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^\w]+", "_", s)          # keep alnum + underscore
    s = re.sub(r"_+", "_", s).strip("_")   # collapse underscores
    return s


@dataclass
class SchemaMap:
    """
    Result of schema detection.
    """
    canonical_to_original: Dict[str, str]
    canonical_to_normed: Dict[str, str]
    ambiguities: Dict[str, List[str]]      # canonical -> list of original cols that matched
    missing: List[str]                     # canonicals with no match

    def rename_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df with matched columns renamed to canonical names.
        Only renames matches; does not drop anything.
        """
        rename_map = {orig: canon for canon, orig in self.canonical_to_original.items()}
        return df.rename(columns=rename_map).copy()


def detect_schema(df: pd.DataFrame, aliases: Optional[Dict[str, List[str]]] = None) -> SchemaMap:
    """
    Detect canonical columns in df based on robust normalization and aliases.
    Returns a SchemaMap containing:
      - canonical_to_original mapping
      - ambiguities (if multiple cols match a canonical)
      - missing canonicals
    """
    aliases = aliases or KNOWN_ALIASES

    # Build normalized lookup: normed_header -> [original_headers]
    normed_to_originals: Dict[str, List[str]] = {}
    for c in df.columns:
        normed_to_originals.setdefault(_norm_header(str(c)), []).append(str(c))

    canonical_to_original: Dict[str, str] = {}
    canonical_to_normed: Dict[str, str] = {}
    ambiguities: Dict[str, List[str]] = {}
    missing: List[str] = []

    for canonical, alias_list in aliases.items():
        # Consider canonical itself as an alias too
        candidates = [canonical] + list(alias_list)

        matched_originals: List[str] = []
        matched_normed: List[str] = []

        for a in candidates:
            a_norm = _norm_header(a)
            if a_norm in normed_to_originals:
                matched_normed.append(a_norm)
                matched_originals.extend(normed_to_originals[a_norm])

        # Deduplicate while preserving order
        seen = set()
        matched_originals = [x for x in matched_originals if not (x in seen or seen.add(x))]

        if not matched_originals:
            missing.append(canonical)
            continue

        if len(matched_originals) > 1:
            ambiguities[canonical] = matched_originals

        # Choose the first match deterministically
        canonical_to_original[canonical] = matched_originals[0]
        canonical_to_normed[canonical] = _norm_header(matched_originals[0])

    return SchemaMap(
        canonical_to_original=canonical_to_original,
        canonical_to_normed=canonical_to_normed,
        ambiguities=ambiguities,
        missing=missing,
    )


def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Backward-compatible API:
    returns canonical -> original column mapping (same shape as your old function).
    """
    sm = detect_schema(df)
    return sm.canonical_to_original


def apply_canonical_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMap]:
    """
    Convenience helper: detects schema and returns (renamed_df, schema_map).
    """
    sm = detect_schema(df)
    return sm.rename_df(df), sm
