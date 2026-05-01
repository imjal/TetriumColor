#!/usr/bin/env python3
"""
Plot observer genotype distributions following the TetriumColor style guide.

Outputs (saved to output/observer_distributions/):
  1. observer_cdf_all.pdf — CDF of all observers up to pentachromats (male, female, both)
  2. trichromat_pdf.pdf  — PDF bar chart of trichromat genotypes (both sexes)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TetriumColor.Plotting.PlotStyle import apply_style, SINGLE_COL, DOUBLE_COL
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes

apply_style()

TOP_N = 10
OUTPUT_DIR = Path('output/observer_distributions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dimension labels and colors (dimension = total cones including implicit S)
DIM_LABELS = {
    0: 'Monochromat (1)',
    1: 'Dichromat (2)',
    2: 'Trichromat (3)',
    3: 'Tetrachromat (4)',
    4: 'Pentachromat (5)',
}
DIM_COLORS = {
    0: '#d62728',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#1f77b4',
    4: '#9467bd',
}

og = ObserverGenotypes(dimensions=[1, 2, 3, 4, 5])


# ============================================================
# Plot 1: CDF of all observers (male, female, both)
# ============================================================
def plot_cdf_all(log_scale=False):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

    pdf = og.get_pdf('both')
    genotypes = list(pdf.keys())[:TOP_N]
    probs = np.array([pdf[g] for g in genotypes])
    cdf = np.cumsum(probs)

    dims = [len(g) for g in genotypes]
    colors = [DIM_COLORS.get(d, '#7f7f7f') for d in dims]
    labels = [', '.join(str(int(p)) if p == int(p) else str(p) for p in g)
              for g in genotypes]

    x = np.arange(len(labels))
    ax.bar(x, probs, color=colors, edgecolor='black', linewidth=0.3)

    if log_scale:
        ax.set_yscale('log')

    # CDF on secondary axis
    ax_cdf = ax.twinx()
    ax_cdf.plot(x, cdf, color='black', linewidth=1, marker='o',
                markersize=2, alpha=0.7)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_ylabel(r'Cumulative prob.')
    ax_cdf.grid(False)

    # 95% reference line
    ax_cdf.axhline(y=0.95, color='black', linestyle=':', linewidth=1, alpha=0.7)
    ax_cdf.text(len(labels) - 0.5, 0.95, '95%', ha='right', va='bottom',
                fontsize=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(r'M/L cone peaks (nm)')
    ax.set_ylabel(r'Probability')

    # Legend
    present_dims = sorted(set(dims))
    legend_elements = [Patch(facecolor=DIM_COLORS[d], edgecolor='black',
                             linewidth=0.3, label=DIM_LABELS[d])
                       for d in present_dims]
    ax.legend(handles=legend_elements, fontsize=5, loc='center right')

    suffix_tag = '_log' if log_scale else ''
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'observer_cdf_all{suffix_tag}.{suffix}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR}/observer_cdf_all{suffix_tag}.pdf')


# ============================================================
# Plot 2: Trichromat PDF (both sexes)
# ============================================================
def plot_trichromat_pdf(log_scale=False):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

    both_pdf = og.get_pdf('both')
    trichromat_data = [(g, both_pdf[g]) for g in both_pdf if len(g) == 2]
    trichromat_data.sort(key=lambda x: x[1], reverse=True)
    total_trichromat_prob = sum(p for _, p in trichromat_data)
    trichromat_data = trichromat_data[:TOP_N]

    genotypes = [g for g, _ in trichromat_data]
    probs = np.array([p for _, p in trichromat_data])
    probs = probs / total_trichromat_prob  # normalize by ALL trichromats
    labels = [', '.join(str(int(p)) if p == int(p) else str(p) for p in g)
              for g in genotypes]

    x = np.arange(len(labels))
    ax.bar(x, probs, color=DIM_COLORS[2], edgecolor='black', linewidth=0.3)

    if log_scale:
        ax.set_yscale('log')

    # CDF overlay
    ax_cdf = ax.twinx()
    cdf = np.cumsum(probs)
    ax_cdf.plot(x, cdf, color='black', linewidth=1, marker='o',
                markersize=2, alpha=0.7)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_ylabel(r'Cumulative probability')
    ax_cdf.grid(False)

    # 95% reference line
    ax_cdf.axhline(y=0.95, color='black', linestyle=':', linewidth=1, alpha=0.7)
    ax_cdf.text(len(labels) - 0.5, 0.95, '95%', ha='right', va='bottom',
                fontsize=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
    ax.set_xlabel(r'M/L cone peaks (nm)')
    ax.set_ylabel(r'Probability')

    suffix_tag = '_log' if log_scale else ''
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'trichromat_pdf{suffix_tag}.{suffix}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR}/trichromat_pdf{suffix_tag}.pdf')


if __name__ == '__main__':
    plot_cdf_all(log_scale=False)
    plot_cdf_all(log_scale=True)
    plot_trichromat_pdf(log_scale=False)
    plot_trichromat_pdf(log_scale=True)
    print('Done.')
