#!/usr/bin/env python3
"""
Compare LLM models on identical inputs.

Articles are fetched once per gene, then every model runs the same three
extraction agents and the synthesis agent on them.

Usage:
    python scripts/compare_models.py --disorder asd --genes SHANK2 SOX11 RAB17 \
        --models claude-sonnet-4-6 claude-haiku-5-5
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DISEASES, OUTPUT_DIR
from pipeline import CATEGORIES, apply_search_overrides, gather_articles, run_agent, run_synthesis, _append_references
from services.llm_service import get_token_usage, reset_token_usage

PRICES = {  # USD per million tokens (input, output), prompts <= 100k tokens
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disorder", default="asd", choices=list(DISEASES))
    ap.add_argument("--genes", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "model_comparison"))
    args = ap.parse_args()

    cfg = DISEASES[args.disorder]
    apply_search_overrides(cfg["clinical_keywords"], None)
    out_root = os.path.join(args.out, args.disorder)

    for gene in args.genes:
        gene_dir = os.path.join(out_root, gene)
        os.makedirs(gene_dir, exist_ok=True)
        print(f"[{gene}] fetching articles...", flush=True)
        articles = {cat: gather_articles(gene, cat) for cat in CATEGORIES}
        with open(os.path.join(gene_dir, "articles.json"), "w") as f:
            json.dump({cat: [{"title": a.get("title"), "pmid": a.get("pmid"), "source": a.get("source")}
                             for a in arts] for cat, arts in articles.items()}, f, indent=2)

        for model in args.models:
            reset_token_usage()
            t0 = time.time()
            print(f"[{gene}] {model}: running agents...", flush=True)
            results = {cat: run_agent(gene, cat, articles[cat], model,
                                      mouse_phenotypes=cfg["mouse_phenotypes"])
                       for cat in CATEGORIES}
            report = run_synthesis(gene,
                                   json.dumps(results["clinical"], indent=2),
                                   json.dumps(results["model_organism"], indent=2),
                                   json.dumps(results["biological"], indent=2),
                                   model, disease_label=cfg["disease_label"])
            report = _append_references(report, results)
            tu = get_token_usage()
            pin, pout = PRICES.get(model, (0, 0))
            usage = dict(tu, seconds=round(time.time() - t0, 1),
                         cost_usd=round(tu["input_tokens"] * pin / 1e6 + tu["output_tokens"] * pout / 1e6, 4))
            with open(os.path.join(gene_dir, f"{model}_report.txt"), "w") as f:
                f.write(report)
            with open(os.path.join(gene_dir, f"{model}_results.json"), "w") as f:
                json.dump({"agent_results": results, "usage": usage}, f, indent=2, default=str)
            print(f"[{gene}] {model}: done {usage}", flush=True)


if __name__ == "__main__":
    main()
