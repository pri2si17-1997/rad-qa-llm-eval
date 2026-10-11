"""Figures: zero-shot F1 vs false-answer rate, and the ΔF1 decomposition.

Reads results/ CSVs and writes PNG files (300 dpi) to figures/ (or --out DIR):

  fig1_f1_vs_far          zero-shot F1 vs FAR per model, report-clustered 95% CIs
  fig2_decomposition      pooled ΔF1 decomposition, token vs RadGraph F1

Palette: navy #2b4f8c / brick #b8433a. Identity never relies on colour alone (lightness
difference, marker shape, legend, direct labels), so the figures survive greyscale print.

    radqa_venv/bin/python scripts/make_figures.py [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

COL_W, TEXT_W = 3.03, 6.30            # two-column layout: column / text width, inches
BLUE, ORANGE = "#2b4f8c", "#b8433a"  # navy, brick
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRANS = [("k=0", "+F"), ("+F", "+F+I"), ("+F+I", "+F+I+U"), ("k=0", "+F+I")]
SHORT = {"Gemini 3 Flash": "Gemini 3 Flash", "Gemini 3.1 Pro": "Gemini 3.1 Pro",
         "Claude Sonnet 4.5": "Sonnet 4.5", "Claude Opus 5": "Opus 5",
         "LLaMA 4 Maverick": "Maverick", "Claude Sonnet 5": "Sonnet 5",
         "Claude Haiku 4.5": "Haiku 4.5", "Mistral Large 3": "Mistral Large 3",
         "GPT-OSS 120B": "GPT-OSS 120B", "DeepSeek V3.2": "DeepSeek V3.2",
         "LLaMA 4 Scout": "Scout", "GPT-OSS 20B": "GPT-OSS 20B", "Ministral 3 14B": "Ministral 3 14B"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def cond_tex(c):
    return c.replace("k=0", "$k$=0")


def trans_label(a, b):
    return f"{cond_tex(a)} → {cond_tex(b)}"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def model_order():
    ci = pd.read_csv(RES / "clustered_ci.csv")
    return ci[ci.shot == 0].sort_values("f1", ascending=False).model.tolist()


def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def fig1_f1_vs_far():
    ci = pd.read_csv(RES / "clustered_ci.csv")
    d = ci[ci.shot == 0].set_index("model")
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    # band holding 11 of 13 models
    ax.axvspan(0.669, 0.738, color="#f0efec", zorder=0, lw=0)
    ax.text(0.7035, 0.392, "11 of 13 models", ha="center", va="top", fontsize=6.5, color=INK2)
    pair = {"LLaMA 4 Maverick", "Claude Sonnet 4.5"}
    for m, r in d.iterrows():
        c = ORANGE if m in pair else BLUE
        ax.errorbar(r.f1, r.far, xerr=[[r.f1 - r.f1_ci_lo], [r.f1_ci_hi - r.f1]],
                    yerr=[[r.far - r.far_ci_lo], [r.far_ci_hi - r.far]], fmt="none",
                    ecolor=c, alpha=0.3, elinewidth=0.6, capsize=0, zorder=2)
        ax.plot(r.f1, r.far, marker="s" if m in pair else "o", ms=4.2, color=c,
                mec="white", mew=0.6, zorder=3)
    # Selective direct labels, placed beside their points (no leader lines): the models the
    # text discusses. The unlabelled models are identified in the results table.
    off = {"LLaMA 4 Scout": (-4, 5, "right", "Scout"), "DeepSeek V3.2": (4, 5, "left", "DeepSeek V3.2"),
           "LLaMA 4 Maverick": (5, 0, "left", "Maverick"),
           "Ministral 3 14B": (0, 6, "center", "Ministral 3 14B"),
           "Claude Sonnet 4.5": (-3, -9, "right", "Sonnet 4.5"),
           "Gemini 3 Flash": (4, 7, "left", "Gemini 3 Flash")}
    for m, (dx, dy, ha, lab) in off.items():
        r = d.loc[m]
        ax.annotate(lab, (r.f1, r.far), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.3, ha=ha, va="center", color=INK, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    ax.set_xlabel("Zero-shot F1")
    ax.set_ylabel("False-answer rate (FAR)")
    ax.set_xlim(0.54, 0.81)
    ax.set_ylim(0.04, 0.40)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig1_f1_vs_far")


# ---------------------------------------------------------------------------
def fig2_decomposition():
    dt = pd.read_csv(RES / "delta_decomposition.csv")
    dr = pd.read_csv(RES / "delta_decomposition_radgraph.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
    y = np.arange(len(TRANS))[::-1]
    h = 0.34
    for ax, d, title in [(axes[0], dt, "Token F1"), (axes[1], dr, "RadGraph F1")]:
        p = d[d.model == "POOLED"].set_index(["from", "to"])
        for i, (a, b) in enumerate(TRANS):
            r = p.loc[(a, b)]
            for k, (val, lo, hi, col, lab) in enumerate([
                    (r.contrib_extraction, r.extraction_ci_lo, r.extraction_ci_hi, BLUE, "Extraction"),
                    (r.answerability_shift, r.shift_ci_lo, r.shift_ci_hi, ORANGE, "Answerability shift")]):
                yy = y[i] + (h / 2 + 0.02) * (1 if k == 0 else -1)
                ax.barh(yy, val, height=h, color=col, edgecolor="white", linewidth=0.6,
                        label=lab if i == 0 else None, zorder=2)
                ax.plot([lo, hi], [yy, yy], color=INK, lw=0.7, zorder=3)
            ax.plot(r.total_delta, y[i], marker="D", ms=3.6, color=INK, mec="white", mew=0.5, ls="none",
                    zorder=4, label="Total Δ" if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(title, pad=3)
        grid_x(ax)
        ax.set_xlim(-0.022, 0.036)
        ax.set_xticks([-0.02, 0, 0.02])
        ax.set_xticklabels(["−0.02", "0", "+0.02"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([trans_label(a, b) for a, b in TRANS])
    axes[0].axhline(0.5, color=GRID, lw=0.6)
    axes[1].axhline(0.5, color=GRID, lw=0.6)
    fig.supxlabel("Change in mean score (pooled over 13 models)", fontsize=7, y=-0.07)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.07),
               handlelength=1.2, columnspacing=0.9)
    save(fig, "fig2_decomposition")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default: figures/)")
    OUT = ap.parse_args().out.resolve()
    fig1_f1_vs_far()
    fig2_decomposition()


if __name__ == "__main__":
    main()
