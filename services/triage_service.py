"""Relevance triage of candidate articles by a fast model, then a fixed reading budget (PESTO-style)."""
import os

from config import PROMPTS_DIR

TRIAGE_MODEL = "claude-haiku-5-5"
BATCH = 10

CATEGORY_TEXT = {
    "clinical": ("clinical evidence",
                 "Clinical evidence: human individuals or families carrying variants in the gene, with their phenotypes."),
    "model_organism": ("model organism evidence",
                       "Model organism evidence: animals (mouse, rat, zebrafish, fly, Xenopus...) in which the gene is "
                       "knocked out, knocked down or mutated, with their phenotypes."),
    "biological": ("biological evidence",
                   "Biological evidence: experimental data on the gene's function, expression, interactions or pathways."),
}


def band_articles(gene_name, category, disease_label, articles, model=TRIAGE_MODEL):
    """Set a["band"] (0-3) on every article."""
    from pipeline import format_articles_for_prompt, parse_json_response
    from services.llm_service import call_llm

    with open(os.path.join(PROMPTS_DIR, "triage_agent.txt")) as f:
        template = f.read()
    label, definition = CATEGORY_TEXT[category]
    for start in range(0, len(articles), BATCH):
        batch = articles[start:start + BATCH]
        prompt = template.format(gene_name=gene_name, disease_label=disease_label, category_label=label,
                                 category_definition=definition, articles=format_articles_for_prompt(batch))
        bands = {}
        for e in parse_json_response(call_llm(prompt, model=model) or ""):
            try:
                bands[int(e["id"])] = int(e["band"])
            except (KeyError, TypeError, ValueError):
                continue
        for i, a in enumerate(batch, 1):
            a["band"] = bands.get(i, 1)
    return articles


def select(articles, budget=14, extra=6):
    """Strongest search hits up to `budget`, then up to `extra` full-text hits.

    Band 0 is never read. Full-text hits need band 2: they name the gene only in
    their body, so a passing mention (band 1) is usually a gene list or a screen.
    """
    def best(pool, n, min_band):
        ranked = sorted((a for a in pool if a.get("band", 1) >= min_band), key=lambda a: -a.get("band", 1))
        return ranked[:n]
    search = [a for a in articles if a.get("found_by") != "fulltext"]
    fulltext = [a for a in articles if a.get("found_by") == "fulltext"]
    return best(search, budget, 1) + best(fulltext, extra, 2)
