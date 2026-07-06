"""Search ASHG abstracts (local CSV) by gene name + category keywords."""
import re
import pandas as pd

from config import (
    ASHG_FILE,
    CLINICAL_KEYWORDS, MODEL_ORGANISM_KEYWORDS, BIOLOGICAL_KEYWORDS,
    CLINICAL_RETMAX, MODEL_ORGANISM_RETMAX, BIOLOGICAL_RETMAX,
)

_ashg_df = None
_text_col = None

CATEGORY_KEYWORDS = {
    "clinical": CLINICAL_KEYWORDS,
    "model_organism": MODEL_ORGANISM_KEYWORDS,
    "biological": BIOLOGICAL_KEYWORDS,
}

CATEGORY_RETMAX = {
    "clinical": CLINICAL_RETMAX,
    "model_organism": MODEL_ORGANISM_RETMAX,
    "biological": BIOLOGICAL_RETMAX,
}


def _load_ashg():
    global _ashg_df, _text_col
    if _ashg_df is None:
        _ashg_df = pd.read_csv(ASHG_FILE, dtype=str).fillna("")
        _text_col = (_ashg_df["title"] + " " + _ashg_df["abstract"]).str.lower()
    return _ashg_df, _text_col


def search_ashg_category(gene_name: str, category: str) -> list[dict]:
    """
    Search ASHG abstracts for a gene + category keywords.
    Uses word-boundary matching for gene name, then keyword regex on matches.
    Returns at most retmax results, ranked by year (most recent first).
    """
    df, text_lower = _load_ashg()
    keywords = CATEGORY_KEYWORDS[category]
    retmax = CATEGORY_RETMAX.get(category, 20)

    gene_pat = r'\b' + re.escape(gene_name.lower()) + r'\b'
    gene_mask = text_lower.str.contains(gene_pat, regex=True)
    gene_subset = df[gene_mask]
    if gene_subset.empty:
        return []

    subset_text = text_lower[gene_mask]
    kw_pat = "|".join(re.escape(kw) for kw in keywords)
    kw_mask = subset_text.str.contains(kw_pat, case=False, regex=True)
    matches = gene_subset[kw_mask.values].copy()

    if matches.empty:
        return []

    matches["_year_int"] = pd.to_numeric(matches["year"], errors="coerce").fillna(0).astype(int)
    matches = matches.sort_values("_year_int", ascending=False).head(retmax)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "affiliations": row.get("affiliations", ""),
            "abstract": row.get("abstract", ""),
            "year": row.get("year", ""),
            "source": "ashg",
            "pmid": None,
        })
    return results
