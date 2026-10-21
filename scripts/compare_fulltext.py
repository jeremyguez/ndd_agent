#!/usr/bin/env python3
"""
Compare the current abstract-only pipeline with a PESTO-style full-text variant.

abstracts : current pipeline, every retrieved PubMed + ASHG abstract is read.
fulltext  : the same candidates, plus up to 10 articles per category (clinical,
            model organism) that name the gene only in their body, tables or
            supplement (Europe PMC), with passages naming the gene. A fast model
            sorts all candidates into relevance bands, and the reader gets at
            most `budget` search hits plus `extra` full-text hits per category.

Usage:
    python scripts/compare_fulltext.py --pairs asd:DYRK1A ndd:SUPT6H
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL, DISEASES, OUTPUT_DIR
from pipeline import CATEGORIES, apply_search_overrides, gather_articles, run_agent, run_synthesis, _append_references
from services.fulltext_service import add_excerpts, search_body_only
from services.llm_service import get_token_usage, reset_token_usage
from services.pubmed_service import CATEGORY_CONFIG
from services.triage_service import band_articles, select

PRICES = {  # USD per million tokens (input, output)
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
FULLTEXT_CATEGORIES = ("clinical", "model_organism")


def fulltext_inputs(gene, base, cfg, budget, extra):
    arts = copy.deepcopy(base)
    known = {str(a["pmid"]) for v in arts.values() for a in v if a.get("pmid")}
    selected = {}
    for cat in CATEGORIES:
        pool = arts[cat]
        if cat in FULLTEXT_CATEGORIES:
            hits = search_body_only(gene, CATEGORY_CONFIG[cat]["keywords"], exclude_pmids=known)
            add_excerpts(hits, gene)
            hits = [a for a in hits if a.get("n_excerpts")]
            known |= {a["pmid"] for a in hits}
            pool = pool + hits
        band_articles(gene, cat, cfg["disease_label"], pool)
        selected[cat] = select(pool, budget, extra)
        arts[cat] = pool
    return selected, arts


def cost(usage):
    total = 0.0
    for model, u in usage.get("by_model", {}).items():
        pin, pout = PRICES.get(model, (0, 0))
        total += u["input_tokens"] * pin / 1e6 + u["output_tokens"] * pout / 1e6
    return round(total, 4)


def run_agents(gene, arts, model, cfg):
    results = {cat: run_agent(gene, cat, arts[cat], model, mouse_phenotypes=cfg["mouse_phenotypes"])
               for cat in CATEGORIES}
    report = run_synthesis(gene,
                           json.dumps(results["clinical"], indent=2),
                           json.dumps(results["model_organism"], indent=2),
                           json.dumps(results["biological"], indent=2),
                           model, disease_label=cfg["disease_label"])
    return results, _append_references(report, results)


def dump_articles(path, arts, read=None):
    read_ids = {id(a) for v in (read or arts).values() for a in v}
    with open(path, "w") as f:
        json.dump({cat: [{"title": a.get("title"), "pmid": a.get("pmid"), "source": a.get("source"),
                          "found_by": a.get("found_by", "search"), "band": a.get("band"),
                          "read": id(a) in read_ids}
                         for a in v] for cat, v in arts.items()}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="disorder:GENE")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budget", type=int, default=14)
    ap.add_argument("--extra", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "fulltext_comparison"))
    args = ap.parse_args()

    for pair in args.pairs:
        disorder, gene = pair.split(":")
        cfg = DISEASES[disorder]
        apply_search_overrides(cfg["clinical_keywords"], None)
        gene_dir = os.path.join(args.out, disorder, gene)
        os.makedirs(gene_dir, exist_ok=True)
        print(f"[{pair}] fetching articles...", flush=True)
        base = {cat: gather_articles(gene, cat) for cat in CATEGORIES}

        for arm in ("abstracts", "fulltext"):
            reset_token_usage()
            t0 = time.time()
            if arm == "abstracts":
                read, pool = base, base
            else:
                read, pool = fulltext_inputs(gene, base, cfg, args.budget, args.extra)
            dump_articles(os.path.join(gene_dir, f"{arm}_articles.json"), pool, read)
            print(f"[{pair}] {arm}: reading {sum(len(v) for v in read.values())} articles...", flush=True)
            results, report = run_agents(gene, read, args.model, cfg)
            usage = get_token_usage()
            usage.update(seconds=round(time.time() - t0, 1), cost_usd=cost(usage))
            with open(os.path.join(gene_dir, f"{arm}_report.txt"), "w") as f:
                f.write(report)
            with open(os.path.join(gene_dir, f"{arm}_results.json"), "w") as f:
                json.dump({"agent_results": results, "usage": usage}, f, indent=2, default=str)
            print(f"[{pair}] {arm}: done ${usage['cost_usd']} in {usage['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
