"""Europe PMC full-text helpers (adapted from PESTO).

- search_body_only: articles naming the gene only in their body, tables or
  supplement, with disease keywords in their title or abstract. PubMed cannot
  return them, since the gene is absent from their title and abstract.
- add_excerpts: append open-access passages naming the gene to articles.

Every network failure returns empty, so the abstract-only behaviour is the floor.
"""
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT = 45


def _get(url, params=None, tries=2):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
    return None


def _search(query, page_size, result_type="lite"):
    params = {"query": query, "format": "json", "pageSize": page_size, "resultType": result_type}
    r = _get(SEARCH, params)
    data = r.json() if r is not None else {}
    # Europe PMC sometimes answers with a bare {"version"} that looks like an empty result.
    if "hitCount" not in data:
        r = _get(SEARCH, params, tries=1)
        data = r.json() if r is not None else {}
    return (data.get("resultList") or {}).get("result") or []


def search_body_only(gene, keywords, exclude_pmids=(), limit=10):
    kws = [k.replace('"', "") for k in keywords if k]
    body = " OR ".join(f'{f}:"{gene}"' for f in ("BODY", "TABLE", "SUPPL"))
    where = " OR ".join(f'{f}:"{k}"' for f in ("TITLE", "ABSTRACT") for k in kws)
    query = f'({body}) AND ({where}) AND NOT (TITLE:"{gene}" OR ABSTRACT:"{gene}") AND SRC:MED'
    exclude = set(map(str, exclude_pmids))
    out = []
    for a in _search(query, 25, "core"):
        pmid = (a.get("pmid") or "").strip()
        if not pmid or pmid in exclude or not a.get("pmcid"):
            continue
        out.append({
            "title": a.get("title") or "",
            "pmid": pmid,
            "pmcid": a["pmcid"],
            "pubdate": str(a.get("pubYear") or ""),
            "abstract": re.sub(r"<[^>]+>", " ", a.get("abstractText") or "").strip(),
            "source": "pubmed",
            "found_by": "fulltext",
        })
        if len(out) >= limit:
            break
    return out


def pmcids_for(pmids):
    out = {}
    pmids = [str(p) for p in pmids if p]
    for i in range(0, len(pmids), 50):
        chunk = pmids[i:i + 50]
        # Parentheses matter: Europe PMC binds AND tighter than OR.
        query = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED"
        for a in _search(query, len(chunk)):
            if a.get("pmcid"):
                out[(a.get("pmid") or "").strip()] = a["pmcid"]
    return out


def _plain(xml_text):
    """Article body and tables as plain text, without front matter or references."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml_text))
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("front", "ref-list"):
                parent.remove(child)
    parts = []
    for el in root.iter():
        if el.tag in ("td", "th"):
            parts.append((el.text or "").strip() + " |")
        elif el.text:
            parts.append(el.text)
        if el.tail:
            parts.append(el.tail)
    return re.sub(r"\s+", " ", " ".join(parts))


def fetch_fulltext(pmcid):
    r = _get(FULLTEXT.format(pmcid=pmcid), tries=1)
    if r is not None and r.text.lstrip().startswith("<"):
        return _plain(r.text)
    # Europe PMC answers 403 for some author manuscripts that NCBI serves.
    r = _get(EFETCH, {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "xml"}, tries=1)
    return _plain(r.text) if r is not None else ""


def gene_excerpts(pmcid, gene, window=420, most=3):
    text = fetch_fulltext(pmcid)
    if not text:
        return []
    # Case-sensitive: symbols such as SET or IK are ordinary words in lower case.
    spans = [m.start() for m in re.finditer(rf"(?<![A-Za-z0-9-]){re.escape(gene)}(?![A-Za-z0-9-])", text)]
    out, last_end = [], -1
    for pos in spans:
        start, end = max(0, pos - window // 2), pos + window // 2
        if start < last_end:
            continue
        piece = text[start:end].strip()
        out.append(("..." if start else "") + piece + ("..." if end < len(text) else ""))
        last_end = end
        if len(out) >= most:
            break
    return out


def add_excerpts(articles, gene, workers=8):
    """Append full-text passages naming the gene to each article that has an open-access copy."""
    need = [a for a in articles if a.get("pmid") and not a.get("pmcid")]
    if need:
        ids = pmcids_for([a["pmid"] for a in need])
        for a in need:
            a["pmcid"] = ids.get(str(a["pmid"]), "")
    todo = [a for a in articles if a.get("pmcid")]
    with ThreadPoolExecutor(workers) as ex:
        found = list(ex.map(lambda a: gene_excerpts(a["pmcid"], gene), todo))
    for a, ex_ in zip(todo, found):
        a["n_excerpts"] = len(ex_)
        if ex_:
            head = (a.get("abstract") or "").strip()
            tag = "[Found by full-text search: gene named only in body/tables/supplement] " if a.get("found_by") == "fulltext" else ""
            a["abstract"] = f"{tag}{head}\n    [Full-text passages naming {gene}] " + " // ".join(ex_)
    return articles
