"""PubMed API client for searching and fetching articles by evidence category."""
import time
import requests
from xml.etree import ElementTree

from config import BASE_URL_NCBI, NCBI_API_KEY, PUBMED_COOLDOWN
from config import (
    CLINICAL_KEYWORDS, MODEL_ORGANISM_KEYWORDS, BIOLOGICAL_KEYWORDS,
    CLINICAL_RETMAX, MODEL_ORGANISM_RETMAX, BIOLOGICAL_RETMAX,
)

AMBIGUOUS_GENE_NAMES = {
    "APC", "BAD", "BAX", "BID", "CA2", "CAMP", "EGF", "EGFR", "FOS", "GC", "HR",
    "JUN", "MTOR", "MYC", "NGF", "PIP", "SRC", "TNF", "VHL",
    "CAD", "DMD", "FAP", "SCD",
    "APP", "ARC", "CAT", "IMPACT", "KIN", "KIT", "MAG", "MAX", "MET",
    "NODAL", "RAN", "REST", "SET", "SON", "TANK", "TUB", "VIM",
    "AR", "C2", "C3", "C5", "C6", "C7", "C9", "CP", "CS", "F2", "F3", "F5",
    "F7", "F8", "F9", "FH", "GK", "HP", "IK", "KL", "KY", "MB", "PC", "SI",
    "TF", "TG", "TH", "XG", "XK",
}

CATEGORY_CONFIG = {
    "clinical": {"keywords": CLINICAL_KEYWORDS, "retmax": CLINICAL_RETMAX},
    "model_organism": {"keywords": MODEL_ORGANISM_KEYWORDS, "retmax": MODEL_ORGANISM_RETMAX},
    "biological": {"keywords": BIOLOGICAL_KEYWORDS, "retmax": BIOLOGICAL_RETMAX},
}


def _gene_term(gene_name: str) -> str:
    if gene_name.upper() in AMBIGUOUS_GENE_NAMES:
        return f'"{gene_name} gene"[Title/Abstract]'
    return f'"{gene_name}"[Title/Abstract]'


def _api_request(url, params, timeout=30):
    """HTTP GET with aggressive retry (mirrors existing pipeline)."""
    delays = [0.1] * 10 + [0.5] * 10 + [1] * 10 + [2, 5, 10, 20, 30, 60]
    for i in range(len(delays) + 1):
        try:
            if PUBMED_COOLDOWN > 0:
                time.sleep(PUBMED_COOLDOWN)
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException:
            if i < len(delays):
                time.sleep(delays[i])
            else:
                print(f"ERROR: All PubMed API attempts failed for {url}")
                return None
    return None


def fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch abstract text for a list of PMIDs via efetch."""
    if not pmids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = _api_request(f"{BASE_URL_NCBI}efetch.fcgi", params)
    if not resp:
        return {}

    abstracts = {}
    try:
        root = ElementTree.fromstring(resp.content)
        for article in root.findall('.//PubmedArticle'):
            pmid_el = article.find('.//PMID')
            if pmid_el is None:
                continue
            pmid = pmid_el.text
            abstract_parts = article.findall('.//Abstract/AbstractText')
            if abstract_parts:
                text = " ".join("".join(part.itertext()) for part in abstract_parts)
                abstracts[pmid] = text
            else:
                abstracts[pmid] = ""
    except Exception as e:
        print(f"ERROR: Could not parse efetch XML: {e}")
    return abstracts


def search_pubmed_category(gene_name: str, category: str) -> list[dict]:
    """
    Search PubMed for a gene in a specific evidence category.
    Returns list of dicts with keys: title, pmid, pubdate, abstract, source.
    """
    cfg = CATEGORY_CONFIG[category]
    gene_term = _gene_term(gene_name)
    kw_clause = " OR ".join(f'"{kw}"' for kw in cfg["keywords"])
    query = f'({gene_term}) AND ({kw_clause})'

    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": cfg["retmax"],
        "sort": "relevance",
        "retmode": "json",
    }
    if NCBI_API_KEY:
        esearch_params["api_key"] = NCBI_API_KEY

    resp = _api_request(f"{BASE_URL_NCBI}esearch.fcgi", esearch_params)
    if not resp:
        return []

    try:
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []
    if not id_list:
        return []

    esummary_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json",
    }
    if NCBI_API_KEY:
        esummary_params["api_key"] = NCBI_API_KEY

    resp = _api_request(f"{BASE_URL_NCBI}esummary.fcgi", esummary_params)
    if not resp:
        return []

    try:
        summary_data = resp.json().get("result", {})
    except Exception:
        return []

    articles = []
    for pmid in id_list:
        info = summary_data.get(pmid, {})
        articles.append({
            "title": info.get("title", "No title"),
            "pmid": pmid,
            "pubdate": info.get("pubdate", ""),
            "abstract": None,
            "source": "pubmed",
        })

    # Fetch abstracts for all articles
    pmids = [a["pmid"] for a in articles if a["pmid"]]
    if pmids:
        abs_map = fetch_abstracts(pmids)
        for a in articles:
            a["abstract"] = abs_map.get(a["pmid"], "")

    return articles
