# ndd_agent

Agentic LLM pipeline that mines the biomedical literature to assess the evidence
linking candidate genes to rare developmental disorders.

This tool is the gene-validation framework associated with the manuscript
**"Trio meta-analysis finds disease genes and quantifies diagnostic evidence"**
(Moldovan *et al.*). It was used to build per-gene evidence
assessments that complemented expert manual curation when validating the novel
candidate genes reported in the study.

## What it does

For each candidate gene, the pipeline searches PubMed and a local corpus of
**43,420 ASHG conference abstracts (2009–2023)**, then runs four specialized
LLM (Claude) agents:

1. **Clinical** — patients with heterozygous variants, phenotypes, cohorts.
2. **Model organism** — vertebrate knockout/knockdown phenotypes (mouse, zebrafish, …).
3. **Biological** — molecular function, pathways, expression, cell-based assays.
4. **Synthesis** — integrates the three evidence lines into a structured report
   with an overall evidence level (STRONG / MODERATE / LIMITED / INSUFFICIENT),
   consistent with the tiered validation scheme used in the manuscript.

Searches and prompts are specialized per disorder: each disease preset selects
its own gene list, clinical search keywords, and expected mouse knockout
phenotypes, and labels the synthesis report accordingly.

## Setup

```bash
pip install anthropic pandas requests python-dotenv tqdm
```

Create a `.env` file in this directory:

```
ANTHROPIC_API_KEY=sk-ant-...
NCBI_API_KEY=...          # optional, raises PubMed rate limits
```

The ASHG abstract corpus (`Data/ASHG_2009_2023_v2.csv`) must be present.

## Usage

Run a single disease preset:

```bash
python3 pipeline.py --disorder asd --model claude-sonnet-4-6 --n_core 4
```

Run every disease in one command, each in its own subfolder under a run:

```bash
python3 pipeline.py --run run_003 --disorder all --model claude-sonnet-4-6 --n_core 20
```

Or run an arbitrary gene list:

```bash
python3 pipeline.py --genes SHANK2 SCN2A --clinical_keywords autism epilepsy
```

Useful flags: `--num_genes N`, `--max_papers N`, `--retmax N`, `--resume`,
`--verbose`. Runs are resumable (already-processed genes are skipped).

## Disease presets

| Preset | Disorder | Gene list |
|--------|----------|-----------|
| `asd`  | Autism spectrum disorder | `Data/ASD_genes_union.txt` |
| `chd`  | Congenital heart disease / diaphragmatic hernia | `Data/CHD_genes.txt` |
| `ee`   | Epileptic encephalopathy | `Data/EE_genes.txt` |
| `ndd`  | Neurodevelopmental disorder (generic) | `Data/NDD_genes_04_04.txt` |

Presets are defined in `config.py` (`DISEASES`).

## Output

Each run (per disorder) writes to `output/<run>/<disorder>/`:

```
summary.csv                     # all extracted evidence, one row per finding
reports/<gene>.txt              # synthesis report per gene
results/<gene>.json             # raw agent outputs
results/<gene>_articles.json    # titles + PMIDs submitted to the LLM (traceability)
config.json                     # run parameters
```

## Layout

- `pipeline.py` — orchestration and CLI
- `config.py` — paths, model, keywords, disease presets
- `services/` — PubMed and ASHG search clients, Claude wrapper
- `prompts/` — one prompt template per agent
- `Data/` — gene lists and the ASHG abstract corpus
