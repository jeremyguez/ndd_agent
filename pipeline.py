#!/usr/bin/env python3
"""
NDD Gene Evidence Report Pipeline.

For each gene in NDD_genes_novel.txt, searches PubMed + ASHG abstracts
across 3 evidence categories, runs specialized LLM agents, and produces
a CSV summary + per-gene text reports.

Usage:
    python pipeline.py                          # all 172 genes
    python pipeline.py --genes SATB2 TCF4       # specific genes
    python pipeline.py --n_core 4               # parallel
    python pipeline.py --resume                 # skip already-done genes
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is None:
            class _Fake:
                def update(self, n=1): pass
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Fake()
        return iterable

from config import APP_ROOT, GENES_FILE, OUTPUT_DIR, PROMPTS_DIR, DEFAULT_MODEL, DEFAULT_MAX_PAPERS, CLINICAL_KEYWORDS, CLINICAL_RETMAX, DISEASES

CATEGORIES = ["clinical", "model_organism", "biological"]

CSV_COLUMNS = [
    "gene", "evidence_category", "citation", "pmid", "source",
    "variant_type", "variant_detail", "phenotype",
    "num_patients", "diagnostic_reason", "center_country", "cohort",
    "organism", "knockout_type", "mendelian_ratio", "lethality", "infertility",
    "sample_size", "significance", "evidence_subtype", "summary",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_genes(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, name)
    with open(path) as f:
        return f.read()


def format_articles_for_prompt(articles: list[dict]) -> str:
    if not articles:
        return "No articles found for this category."
    lines = []
    for i, a in enumerate(articles, 1):
        src = a.get("source", "unknown")
        if src == "pubmed":
            header = f"[{i}] {a['title']} (PMID: {a.get('pmid', '?')})"
        else:
            header = f"[{i}] ASHG {a.get('year', '?')} - {a['title']}"
            if a.get("authors"):
                header += f" ({a['authors']})"
        lines.append(header)
        abstract = a.get("abstract", "")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from LLM response, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"WARNING: Could not parse JSON from LLM response ({len(text)} chars)")
        return []


# ---------------------------------------------------------------------------
# Per-gene pipeline (runs in worker process)
# ---------------------------------------------------------------------------

def apply_search_overrides(clinical_keywords: list[str] | None, retmax: int | None):
    """Patch PubMed/ASHG search settings in the current process.

    Must also run inside each worker: with the forkserver/spawn start methods
    (default on Linux since Python 3.14), workers re-import the services and
    do not inherit changes made in the parent process.
    """
    from services.pubmed_service import CATEGORY_CONFIG
    from services.ashg_service import CATEGORY_KEYWORDS, CATEGORY_RETMAX
    if clinical_keywords:
        CATEGORY_CONFIG["clinical"]["keywords"] = list(clinical_keywords)
        CATEGORY_KEYWORDS["clinical"] = list(clinical_keywords)
    if retmax:
        for cat in CATEGORY_CONFIG:
            CATEGORY_CONFIG[cat]["retmax"] = retmax
        for cat in CATEGORY_RETMAX:
            CATEGORY_RETMAX[cat] = retmax


def gather_articles(gene_name: str, category: str, verbose: bool = False) -> list[dict]:
    """Search PubMed + ASHG for a gene in one evidence category."""
    import time
    from services.pubmed_service import search_pubmed_category
    from services.ashg_service import search_ashg_category

    t0 = time.time()
    pubmed_articles = search_pubmed_category(gene_name, category)
    ashg_articles = search_ashg_category(gene_name, category)
    if verbose:
        print(f"  [{gene_name}] {category}: {len(pubmed_articles)} PubMed + {len(ashg_articles)} ASHG ({time.time()-t0:.1f}s)", flush=True)
    return pubmed_articles + ashg_articles


def run_agent(gene_name: str, category: str, articles: list[dict], model: str, max_papers: int = DEFAULT_MAX_PAPERS, verbose: bool = False, mouse_phenotypes: str = "") -> list[dict]:
    """Run the LLM agent for one evidence category. Returns parsed JSON list."""
    import time
    from services.llm_service import call_llm

    prompt_file = {
        "clinical": "clinical_agent.txt",
        "model_organism": "model_organism_agent.txt",
        "biological": "biological_agent.txt",
    }[category]

    template = load_prompt(prompt_file)
    articles_text = format_articles_for_prompt(articles)
    fmt_kwargs = {"gene_name": gene_name, "articles": articles_text, "max_papers": max_papers}
    if category == "model_organism":
        fmt_kwargs["mouse_phenotypes_suffix"] = f", {mouse_phenotypes}" if mouse_phenotypes else ""
    prompt = template.format(**fmt_kwargs)

    if verbose:
        print(f"  [{gene_name}] LLM {category} (~{len(prompt)//4} tokens)...", flush=True)
    t0 = time.time()
    response = call_llm(prompt, model=model)
    elapsed = time.time() - t0
    if not response:
        if verbose:
            print(f"  [{gene_name}] LLM {category}: no response ({elapsed:.1f}s)", flush=True)
        return []
    entries = parse_json_response(response)
    if verbose:
        print(f"  [{gene_name}] LLM {category}: {len(entries)} entries ({elapsed:.1f}s)", flush=True)
    return entries


def run_synthesis(gene_name: str, clinical_json: str, model_org_json: str, bio_json: str, model: str, verbose: bool = False, disease_label: str = "neurodevelopmental disorder") -> str:
    """Run the synthesis agent. Returns the text report."""
    import time
    from services.llm_service import call_llm

    template = load_prompt("synthesis_agent.txt")
    prompt = template.format(
        gene_name=gene_name,
        clinical_json=clinical_json,
        model_organism_json=model_org_json,
        biological_json=bio_json,
        disease_label=disease_label,
    )
    if verbose:
        print(f"  [{gene_name}] LLM synthesis ({len(prompt)} chars)...", flush=True)
    t0 = time.time()
    result = call_llm(prompt, model=model) or ""
    if verbose:
        print(f"  [{gene_name}] Synthesis done ({time.time()-t0:.1f}s)", flush=True)
    return result


def flatten_clinical_rows(gene_name: str, entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        variants = entry.get("variants") or [{}]
        for var in variants:
            phenotypes = var.get("phenotypes", [])
            rows.append({
                "gene": gene_name,
                "evidence_category": "clinical",
                "citation": entry.get("citation", ""),
                "pmid": _extract_pmid(entry.get("citation", "")),
                "source": "ashg" if "ASHG" in entry.get("citation", "") else "pubmed",
                "variant_type": var.get("type", ""),
                "variant_detail": var.get("description", ""),
                "phenotype": "; ".join(phenotypes) if isinstance(phenotypes, list) else str(phenotypes),
                "num_patients": entry.get("num_patients"),
                "diagnostic_reason": entry.get("diagnostic_reason", ""),
                "center_country": entry.get("center_country", ""),
                "cohort": entry.get("cohort", ""),
                "organism": "",
                "knockout_type": "",
                "mendelian_ratio": "",
                "lethality": "",
                "infertility": "",
                "sample_size": "",
                "significance": "",
                "evidence_subtype": "",
                "summary": "",
            })
    return rows


def flatten_model_organism_rows(gene_name: str, entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        phenotypes = entry.get("phenotypes", [])
        rows.append({
            "gene": gene_name,
            "evidence_category": "model_organism",
            "citation": entry.get("citation", ""),
            "pmid": _extract_pmid(entry.get("citation", "")),
            "source": "ashg" if "ASHG" in entry.get("citation", "") else "pubmed",
            "variant_type": "",
            "variant_detail": "",
            "phenotype": "; ".join(phenotypes) if isinstance(phenotypes, list) else str(phenotypes),
            "num_patients": "",
            "diagnostic_reason": "",
            "center_country": "",
            "cohort": "",
            "organism": entry.get("organism", ""),
            "knockout_type": entry.get("knockout_type", ""),
            "mendelian_ratio": entry.get("mendelian_ratio", ""),
            "lethality": entry.get("lethality", ""),
            "infertility": str(entry.get("infertility", "")),
            "sample_size": str(entry.get("sample_size", "")),
            "significance": str(entry.get("significance", "")),
            "evidence_subtype": "",
            "summary": "",
        })
    return rows


def flatten_biological_rows(gene_name: str, entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        rows.append({
            "gene": gene_name,
            "evidence_category": "biological",
            "citation": entry.get("citation", ""),
            "pmid": _extract_pmid(entry.get("citation", "")),
            "source": "ashg" if "ASHG" in entry.get("citation", "") else "pubmed",
            "variant_type": "",
            "variant_detail": "",
            "phenotype": "",
            "num_patients": "",
            "diagnostic_reason": "",
            "center_country": "",
            "cohort": "",
            "organism": "",
            "knockout_type": "",
            "mendelian_ratio": "",
            "lethality": "",
            "infertility": "",
            "sample_size": str(entry.get("sample_size", "")),
            "significance": str(entry.get("significance", "")),
            "evidence_subtype": entry.get("evidence_subtype", ""),
            "summary": entry.get("summary", ""),
        })
    return rows


def _extract_pmid(citation: str) -> str:
    m = re.search(r'PMID:\s*(\d+)', citation)
    return m.group(1) if m else ""


def _append_references(report: str, all_agent_results: dict) -> str:
    """Extract PMIDs cited in the synthesis report and append a reference list."""
    # Build PMID -> (citation, category) lookup from all agent results
    pmid_to_citation = {}
    category_labels = {
        "clinical": "Clinical Evidence",
        "model_organism": "Model Organism Evidence",
        "biological": "Biological Evidence",
    }
    for cat, entries in all_agent_results.items():
        for entry in entries:
            citation = entry.get("citation", "")
            pmid = _extract_pmid(citation)
            if pmid and pmid not in pmid_to_citation:
                pmid_to_citation[pmid] = (citation, category_labels.get(cat, cat))

    # Find PMIDs mentioned in the report
    cited_pmids = re.findall(r'(?:PMID[:\s]*)?(\d{7,9})', report)
    cited_pmids = list(dict.fromkeys(cited_pmids))  # dedupe, preserve order

    # Match against known citations
    refs_by_category = {}
    for pmid in cited_pmids:
        if pmid in pmid_to_citation:
            citation, cat_label = pmid_to_citation[pmid]
            refs_by_category.setdefault(cat_label, []).append(citation)

    if not refs_by_category:
        return report

    lines = ["\n\n## REFERENCES\n"]
    for cat_label in ["Clinical Evidence", "Model Organism Evidence", "Biological Evidence"]:
        refs = refs_by_category.get(cat_label, [])
        if refs:
            lines.append(f"\n### {cat_label}")
            for i, ref in enumerate(refs, 1):
                lines.append(f"{i}. {ref}")

    return report + "\n".join(lines)


def process_gene(payload: dict) -> dict:
    """Full pipeline for one gene. Designed to run in a worker process."""
    import time as _time
    gene_name = payload["gene_name"]
    model = payload["model"]
    max_papers = payload.get("max_papers", DEFAULT_MAX_PAPERS)
    verbose = payload.get("verbose", False)
    mouse_phenotypes = payload.get("mouse_phenotypes", "")
    disease_label = payload.get("disease_label", "neurodevelopmental disorder")
    apply_search_overrides(payload.get("clinical_keywords"), payload.get("retmax"))
    gene_start = _time.time()

    try:
        all_articles = {}
        all_agent_results = {}
        for cat in CATEGORIES:
            articles = gather_articles(gene_name, cat, verbose=verbose)
            all_articles[cat] = articles
            entries = run_agent(gene_name, cat, articles, model, max_papers=max_papers, verbose=verbose, mouse_phenotypes=mouse_phenotypes)
            all_agent_results[cat] = entries

        csv_rows = []
        csv_rows.extend(flatten_clinical_rows(gene_name, all_agent_results["clinical"]))
        csv_rows.extend(flatten_model_organism_rows(gene_name, all_agent_results["model_organism"]))
        csv_rows.extend(flatten_biological_rows(gene_name, all_agent_results["biological"]))

        report = run_synthesis(
            gene_name,
            json.dumps(all_agent_results["clinical"], indent=2),
            json.dumps(all_agent_results["model_organism"], indent=2),
            json.dumps(all_agent_results["biological"], indent=2),
            model,
            verbose=verbose,
            disease_label=disease_label,
        )

        report = _append_references(report, all_agent_results)

        elapsed = _time.time() - gene_start
        from services.llm_service import get_token_usage, reset_token_usage
        token_usage = get_token_usage()
        reset_token_usage()
        result = {
            "status": "ok",
            "gene_name": gene_name,
            "csv_rows": csv_rows,
            "report": report,
            "token_usage": token_usage,
            "articles": all_articles,
            "raw": {
                "articles_found": {cat: len(all_articles[cat]) for cat in CATEGORIES},
                "agent_results": all_agent_results,
            },
        }
        return result

    except Exception as e:
        elapsed = _time.time() - gene_start
        from services.llm_service import get_token_usage, reset_token_usage
        token_usage = get_token_usage()
        reset_token_usage()
        print(f"[{gene_name}] ERROR after {elapsed:.1f}s: {e}", flush=True)
        traceback.print_exc()
        return {
            "status": "error",
            "gene_name": gene_name,
            "csv_rows": [],
            "report": "",
            "raw": {"error": str(e), "traceback": traceback.format_exc()},
        }


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def find_or_create_run(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    existing = [d for d in os.listdir(output_dir) if d.startswith("run_")]
    if existing:
        nums = []
        for d in existing:
            m = re.match(r'run_(\d+)', d)
            if m:
                nums.append(int(m.group(1)))
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
    run_name = f"run_{next_num:03d}"
    run_path = os.path.join(output_dir, run_name)
    os.makedirs(os.path.join(run_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(run_path, "reports"), exist_ok=True)
    return run_path


def find_latest_run(output_dir: str) -> str | None:
    if not os.path.exists(output_dir):
        return None
    existing = [d for d in os.listdir(output_dir) if d.startswith("run_")]
    if not existing:
        return None
    nums = []
    for d in existing:
        m = re.match(r'run_(\d+)', d)
        if m:
            nums.append((int(m.group(1)), d))
    if not nums:
        return None
    _, latest = max(nums, key=lambda x: x[0])
    return os.path.join(output_dir, latest)


def get_completed_genes(run_path: str) -> set[str]:
    results_dir = os.path.join(run_path, "results")
    if not os.path.exists(results_dir):
        return set()
    done = set()
    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            done.add(f[:-5])
    return done


def save_gene_result(run_path: str, result: dict):
    gene = result["gene_name"]
    # Save JSON
    json_path = os.path.join(run_path, "results", f"{gene}.json")
    with open(json_path, "w") as f:
        json.dump(result["raw"], f, indent=2, default=str)
    # Save the articles actually submitted to the LLM (title + PMID only), for traceability
    if result.get("articles") is not None:
        slim = {
            cat: [{"title": a.get("title"), "pmid": a.get("pmid")} for a in articles]
            for cat, articles in result["articles"].items()
        }
        articles_path = os.path.join(run_path, "results", f"{gene}_articles.json")
        with open(articles_path, "w") as f:
            json.dump(slim, f, indent=2, default=str)
    # Save report
    if result.get("report"):
        report_path = os.path.join(run_path, "reports", f"{gene}.txt")
        with open(report_path, "w") as f:
            f.write(result["report"])


def append_csv_rows(csv_path: str, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.reindex(columns=CSV_COLUMNS)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df.to_csv(csv_path, mode="a", index=False, header=not file_exists)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(args):
    start_time = time.time()

    # Resolve disease preset (--disorder drives genes, clinical keywords, mouse
    # phenotypes, synthesis label, and the default run name).
    disease_label = "neurodevelopmental disorder"
    mouse_phenotypes = ""
    if getattr(args, "disorder", None):
        if args.disorder not in DISEASES:
            raise ValueError(f"Unknown --disorder '{args.disorder}'. Choices: {', '.join(DISEASES)}")
        dcfg = DISEASES[args.disorder]
        disease_label = dcfg["disease_label"]
        mouse_phenotypes = dcfg["mouse_phenotypes"]
        if not args.genes and not args.genes_file:
            args.genes_file = dcfg["genes_file"]
        if not args.clinical_keywords:
            args.clinical_keywords = list(dcfg["clinical_keywords"])
        print(f"[INFO] Disorder preset: {args.disorder} -> label='{disease_label}', genes_file={args.genes_file}")

    # Override clinical keywords / retmax if provided
    clinical_kw_used = args.clinical_keywords or CLINICAL_KEYWORDS
    retmax_used = args.retmax
    retmax_override = retmax_used if retmax_used != CLINICAL_RETMAX else None
    apply_search_overrides(args.clinical_keywords, retmax_override)
    if args.clinical_keywords:
        print(f"[INFO] Clinical keywords overridden: {clinical_kw_used}")
    if retmax_override:
        print(f"[INFO] retmax overridden: {retmax_used}")

    # Determine run directory. When a disorder is set, nest it under the parent
    # run name if given (e.g. --run run_003 --disorder asd -> output/run_003/asd),
    # otherwise use output/<disorder> (backward compatible).
    disorder = getattr(args, "disorder", None)
    if disorder:
        sub = os.path.join(args.run, disorder) if args.run else disorder
        run_path = os.path.join(OUTPUT_DIR, sub)
        if os.path.isdir(run_path):
            print(f"[INFO] Resuming run: {run_path}")
        else:
            os.makedirs(os.path.join(run_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(run_path, "reports"), exist_ok=True)
            print(f"[INFO] Created new run: {run_path}")
    elif args.run:
        run_path = os.path.join(OUTPUT_DIR, args.run)
        if os.path.isdir(run_path):
            print(f"[INFO] Resuming run: {run_path}")
        else:
            os.makedirs(os.path.join(run_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(run_path, "reports"), exist_ok=True)
            print(f"[INFO] Created new run: {run_path}")
    elif args.resume:
        run_path = find_latest_run(OUTPUT_DIR)
        if not run_path:
            print("[INFO] No existing run found, creating new one.")
            run_path = find_or_create_run(OUTPUT_DIR)
        else:
            print(f"[INFO] Resuming run: {run_path}")
    else:
        run_path = find_or_create_run(OUTPUT_DIR)
        print(f"[INFO] Created new run: {run_path}")

    csv_path = os.path.join(run_path, "summary.csv")

    # Load gene list
    if args.genes:
        gene_list = args.genes
    elif args.genes_file:
        gene_list = load_genes(args.genes_file)
    else:
        gene_list = load_genes(GENES_FILE)

    if args.num_genes:
        gene_list = gene_list[:args.num_genes]

    # Check which genes are already done
    completed = get_completed_genes(run_path)
    todo = [g for g in gene_list if g not in completed]

    print(f"[INFO] {len(gene_list)} total genes, {len(completed)} already done, {len(todo)} to process.")

    if not todo:
        print("[INFO] All genes already processed.")
        return

    # Save run config
    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "model": args.model,
            "total_genes": len(gene_list),
            "genes": gene_list,
            "retmax": retmax_used,
            "clinical_keywords": clinical_kw_used,
            "disorder": getattr(args, "disorder", None),
            "disease_label": disease_label,
            "mouse_phenotypes": mouse_phenotypes,
        }, f, indent=2)

    model = args.model
    max_workers = max(1, args.n_core)

    pbar = tqdm(total=len(gene_list), desc="Genes", unit="gene")
    pbar.n = len(completed)
    pbar.refresh()

    max_papers = args.max_papers
    verbose = getattr(args, 'verbose', False)
    payloads = [{"gene_name": g, "model": model, "max_papers": max_papers, "verbose": verbose,
                 "mouse_phenotypes": mouse_phenotypes, "disease_label": disease_label,
                 "clinical_keywords": args.clinical_keywords, "retmax": retmax_override} for g in todo]

    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    def _handle(result):
        nonlocal total_input_tokens, total_output_tokens, total_llm_calls
        save_gene_result(run_path, result)
        append_csv_rows(csv_path, result["csv_rows"])
        tu = result.get("token_usage", {})
        total_input_tokens += tu.get("input_tokens", 0)
        total_output_tokens += tu.get("output_tokens", 0)
        total_llm_calls += tu.get("total_calls", 0)
        status = "OK" if result["status"] == "ok" else "ERROR"
        n_rows = len(result["csv_rows"])
        pbar.set_postfix_str(f"{result['gene_name']} [{status}, {n_rows} rows]")
        pbar.update(1)

    try:
        if max_workers == 1:
            for payload in payloads:
                _handle(process_gene(payload))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_gene, p): p["gene_name"] for p in payloads}
                for future in as_completed(futures):
                    _handle(future.result())
    finally:
        pbar.close()

    elapsed = time.time() - start_time

    n_processed = len(todo)
    print(f"\n{'='*60}")
    print(f"[INFO] Done in {elapsed:.1f}s. Results in {run_path}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Reports: {os.path.join(run_path, 'reports')}/")
    print(f"\n[TOKEN USAGE]")
    print(f"  LLM calls: {total_llm_calls}")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Genes processed: {n_processed}")
    if n_processed > 0:
        avg_input = total_input_tokens / n_processed
        avg_output = total_output_tokens / n_processed
        print(f"  Avg per gene: {avg_input:,.0f} input, {avg_output:,.0f} output")
        print(f"  Wall-clock time: {elapsed:.1f}s ({elapsed/n_processed:.1f}s per gene with {max_workers} workers)")
        total_genes = 172
        est_input = avg_input * total_genes
        est_output = avg_output * total_genes
        batches = (total_genes + max_workers - 1) // max_workers
        est_wall = batches * (elapsed / ((n_processed + max_workers - 1) // max_workers))
        print(f"\n[EXTRAPOLATION FOR {total_genes} GENES]")
        print(f"  Est. wall-clock time with {max_workers} workers: {est_wall/60:.0f} min ({est_wall/3600:.1f}h)")
        print(f"  Est. input tokens:  {est_input:,.0f}")
        print(f"  Est. output tokens: {est_output:,.0f}")
        for name, ip, op in [("Haiku 4.5", 1.0, 5.0), ("Sonnet 4.6", 3.0, 15.0), ("Opus 4.6", 5.0, 25.0)]:
            cost = est_input / 1e6 * ip + est_output / 1e6 * op
            print(f"  {name}: ${cost:.2f}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="NDD Gene Evidence Report Pipeline")
    parser.add_argument("--genes", nargs="+", default=None,
                        help="Specific gene(s) to process (e.g., --genes SATB2 TCF4)")
    parser.add_argument("--genes_file", type=str, default=None,
                        help="Path to a gene list file (one gene per line)")
    parser.add_argument("--disorder", type=str, default=None, choices=list(DISEASES) + ["all"],
                        help="Disease preset: selects gene list, clinical keywords, mouse phenotypes, "
                             "and synthesis label (e.g., --disorder asd). Use 'all' to run every disease, "
                             "each in its own subfolder under --run (e.g., output/run_003/asd, .../ndd, ...)")
    parser.add_argument("--num_genes", type=int, default=None,
                        help="Process only the first N genes from the list")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--n_core", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--max_papers", type=int, default=DEFAULT_MAX_PAPERS,
                        help=f"Max papers per category in agent output (default: {DEFAULT_MAX_PAPERS})")
    parser.add_argument("--retmax", type=int, default=CLINICAL_RETMAX,
                        help=f"Max articles to retrieve per category from PubMed/ASHG (default: {CLINICAL_RETMAX})")
    parser.add_argument("--clinical_keywords", nargs="+", default=None,
                        help="Override clinical search keywords (e.g., --clinical_keywords epilepsy seizure)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-gene logs (disables clean progress bar)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the latest run (skip already-processed genes)")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name: resumes if exists, creates if new (e.g., --run cardiac_genes)")
    return parser.parse_args()


def run_all_disorders(args):
    """Run every disease preset sequentially, each in its own subfolder under --run."""
    import copy
    diseases = list(DISEASES.keys())
    print(f"[INFO] Running all disorders ({', '.join(diseases)}) under run='{args.run or '<auto>'}'")
    for d in diseases:
        sub = copy.copy(args)
        sub.disorder = d
        # Let each disease fill its own preset values.
        sub.genes = None
        sub.genes_file = None
        sub.clinical_keywords = None
        print(f"\n{'#'*60}\n[INFO] === Disorder: {d} ===\n{'#'*60}")
        run_pipeline(sub)


if __name__ == "__main__":
    args = parse_args()
    try:
        if getattr(args, "disorder", None) == "all":
            run_all_disorders(args)
        else:
            run_pipeline(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
