"""Summary figure comparing PESTO and the NDD agent on one run.

Usage: python scripts/make_figures.py [output/run_004]

Writes <run>/figures/Fig_PESTO_vs_NDD_agent.{pdf,png} and its source data.
"""
import glob
import os
import re
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "output", "run_004")
OUT = os.path.join(RUN, "figures")
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "legend.frameon": False,
})

PHENOS = ["ASD", "NDD", "EE", "CHD", "CDH"]
PHENO_NAME = {"ASD": "autism spectrum disorder", "NDD": "neurodevelopmental disorder", "EE": "epileptic encephalopathy",
              "CHD": "congenital heart disease", "CDH": "congenital diaphragmatic hernia"}
COHORT = {"ASD": "asd", "NDD": "ndd", "EE": "ee", "CHD": "chd_cdh", "CDH": "chd_cdh"}

PESTO_LEVELS = ["Novel", "Hypothesized", "Existing", "Established"]
NDD_LEVELS = ["Insufficient", "Limited", "Moderate", "Strong"]
LEVEL_COLORS = ["#DEEBF7", "#9ECAE1", "#4292C6", "#08306B"]
CONSENSUS = {"Supported by both": "#08306B", "Reported, limited clinical support": "#6BAED6",
             "Novel by both": "#DEEBF7", "Clinical support only": "#B2182B"}

LEVEL_RE = r"(STRONG|MODERATE|LIMITED|INSUFFICIENT)"


def ndd_verdicts() -> pd.DataFrame:
    """Overall evidence level per gene-phenotype pair, read from the synthesis reports."""
    rows = []
    for ph in PHENOS:
        for f in sorted(glob.glob(os.path.join(RUN, COHORT[ph], "reports", "*.txt"))):
            lines = [l for l in re.sub(r"[*#_]", "", open(f).read()).splitlines()
                     if re.search("overall evidence level", l, re.I) and re.search(LEVEL_RE, l)]
            if COHORT[ph] == "chd_cdh":
                word = "heart" if ph == "CHD" else "diaphragm"
                lines = [l for l in lines if re.search(rf"\b{ph}\b|{word}", l, re.I)]
            rows.append((os.path.basename(f)[:-4], ph, re.search(LEVEL_RE, lines[0]).group(1).capitalize()))
    return pd.DataFrame(rows, columns=["gene", "pheno", "ndd"])


def load_pairs() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(RUN, "pesto", "pesto.tsv"), sep="\t")
    p["pheno"] = p.phenotype.map({v: k for k, v in PHENO_NAME.items()})
    m = p.merge(ndd_verdicts(), on=["gene", "pheno"], validate="one_to_one")
    m["pesto_level"] = m.lit_argmax.map(PESTO_LEVELS.index)
    m["ndd_level"] = m.ndd.map(NDD_LEVELS.index)
    m["pesto_expected"] = (m.p_hypothesized + 2 * m.p_existing + 3 * m.p_established) / 100
    known, clinical = m.pesto_level >= 2, m.ndd_level >= 2
    m["consensus"] = np.select([known & clinical, known, clinical],
                               list(CONSENSUS)[:2] + [list(CONSENSUS)[3]], list(CONSENSUS)[2])
    return m


def weighted_kappa(a, b, k=4):
    o = np.zeros((k, k))
    np.add.at(o, (a, b), 1)
    w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (w * o).sum() / (w * e).sum()


def bootstrap_ci(d, stat, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = [stat(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    return np.nanpercentile(boot, [2.5, 97.5])


def panel_label(ax, letter, x):
    ax.text(x, 1.0, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def stacked(ax, counts: pd.DataFrame, colors: list[str], title: str, legend_cols: int):
    """Horizontal 100% stacked bars, one per phenotype, counts written in the segments."""
    frac = counts.div(counts.sum(axis=1), axis=0)
    y = np.arange(len(PHENOS))[::-1]
    left = np.zeros(len(PHENOS))
    for col, color in zip(counts.columns, colors):
        v = frac[col].values
        ax.barh(y, v, left=left, color=color, edgecolor="white", linewidth=0.4, height=0.72, label=col)
        dark = color in ("#4292C6", "#08306B")
        for yy, l, vv, n in zip(y, left, v, counts[col].values):
            if vv >= 0.07:
                ax.text(l + vv / 2, yy, str(n), ha="center", va="center", fontsize=5,
                        color="white" if dark else "black")
        left += v
    ax.set_yticks(y, [f"{ph} ({counts.loc[ph].sum()})" for ph in PHENOS])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_xlabel("Gene–phenotype pairs (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left")
    ax.legend(ncol=legend_cols, loc="upper left", bbox_to_anchor=(0, -0.3), handlelength=1,
              columnspacing=0.8, handletextpad=0.4)


def joint_heatmap(ax, m):
    t = pd.crosstab(m.lit_argmax, m.ndd).reindex(index=PESTO_LEVELS[::-1], columns=NDD_LEVELS, fill_value=0)
    v = t.values
    ax.imshow(v, cmap="Blues", vmin=0, vmax=v.max() * 1.1, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, v[i, j], ha="center", va="center", fontsize=5.5,
                    color="white" if v[i, j] > v.max() * 0.55 else "black")
    ax.set_xticks(range(4), NDD_LEVELS)
    ax.set_yticks(range(4), t.index)
    ax.set_xlabel("NDD agent")
    ax.set_ylabel("PESTO")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    rho = lambda d: spearmanr(d.pesto_expected, d.ndd_level).statistic
    kappa = lambda d: weighted_kappa(d.pesto_level.values, d.ndd_level.values)
    (r_lo, r_hi), (k_lo, k_hi) = bootstrap_ci(m, rho), bootstrap_ci(m, kappa)
    ax.set_title(f"Agreement between methods (n = {len(m)})", loc="left")
    ax.text(1.0, -0.32, f"Spearman ρ = {rho(m):.2f} [{r_lo:.2f}–{r_hi:.2f}]    "
                        f"weighted κ = {kappa(m):.2f} [{k_lo:.2f}–{k_hi:.2f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="0.25")
    return t


def main():
    m = load_pairs()
    fig = plt.figure(figsize=(183 * MM, 105 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.95)

    ax = fig.add_subplot(gs[0, 0])
    a = pd.crosstab(m.pheno, m.lit_argmax).reindex(index=PHENOS, columns=PESTO_LEVELS, fill_value=0)
    stacked(ax, a, LEVEL_COLORS, "PESTO: is the association already in the literature?", 4)
    panel_label(ax, "a", -0.17)

    ax = fig.add_subplot(gs[0, 1])
    b = pd.crosstab(m.pheno, m.ndd).reindex(index=PHENOS, columns=NDD_LEVELS, fill_value=0)
    stacked(ax, b, LEVEL_COLORS, "NDD agent: strength of clinical evidence", 4)
    panel_label(ax, "b", -0.17)

    ax = fig.add_subplot(gs[1, 0])
    c = joint_heatmap(ax, m)
    panel_label(ax, "c", -0.17)

    ax = fig.add_subplot(gs[1, 1])
    d = pd.crosstab(m.pheno, m.consensus).reindex(index=PHENOS, columns=list(CONSENSUS), fill_value=0)
    d = d.loc[:, d.sum() > 0]
    stacked(ax, d, [CONSENSUS[k] for k in d.columns], "Integrated classification", 3)
    panel_label(ax, "d", -0.17)

    name = "Fig_PESTO_vs_NDD_agent"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    with pd.ExcelWriter(os.path.join(OUT, f"{name}_source_data.xlsx")) as xl:
        for sheet, df in (("a", a), ("b", b), ("c", c), ("d", d)):
            df.to_excel(xl, sheet_name=sheet)
        m[["gene", "pheno", "lit_argmax", "pesto_expected", "ndd", "consensus"]].to_excel(
            xl, sheet_name="pairs", index=False)
    print(f"{len(m)} pairs -> {OUT}/{name}.pdf")


if __name__ == "__main__":
    main()
