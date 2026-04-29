#!/usr/bin/env python3
"""
Hyperobserver Projection Residual Analysis (Leave-One-Out)

For each observer type, computes the leave-one-out projection residual:
for each cone in the observer, project it onto the span of the remaining
cones and measure how much is left over.

  - Hyperobserver → 12 residuals
  - Tetrachromat  (S, M, Q547, L) → 4 residuals
  - Trichromat    (S, M, L) → 3 residuals
  - Dichromat     (S, M) → 2 residuals

The residual r = ||φ_perp|| / ||φ|| measures the fraction of each cone's
sensitivity that is NOT explained by the other cones in the same observer.
A small residual means the cone is nearly redundant; a large residual means
it carries unique spectral information.

Usage:
    python hyperobserver_residual_analysis.py
    python hyperobserver_residual_analysis.py --output-dir output/hyperobserver_residual
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TetriumColor.Observer.Observer import Observer
from TetriumColor.Plotting.PlotStyle import (
    apply_style, COLORS as CONE_COLORS, WAVELENGTHS, ALL_PEAKS,
    SINGLE_COL, build_observers,
)

apply_style()


def projection_residual(phi_q: np.ndarray, base_sensor: np.ndarray) -> float:
    """Fraction of φ_q outside span of base_sensor rows.

    Returns ||φ_perp|| / ||φ||.
    """
    Phi = base_sensor  # (k, n_wl)
    G = Phi @ Phi.T    # (k, k)
    coeffs = np.linalg.solve(G, Phi @ phi_q)
    phi_proj = Phi.T @ coeffs
    phi_perp = phi_q - phi_proj
    return np.linalg.norm(phi_perp) / np.linalg.norm(phi_q)


def gram_matrix(sensor_matrix: np.ndarray) -> np.ndarray:
    """Normalized Gram matrix G_ij = ∫φ_i·φ_j / √(∫φ_i²·∫φ_j²)."""
    norms = np.linalg.norm(sensor_matrix, axis=1, keepdims=True)
    normed = sensor_matrix / norms
    return normed @ normed.T


def leave_one_out_residuals(observer: Observer, nominal_peaks: list) -> List[Tuple[float, float]]:
    """For each cone in the observer, compute its residual onto the remaining cones.

    Returns list of (nominal_peak, residual) pairs.
    """
    sensor = observer.get_sensor_matrix(WAVELENGTHS)
    results = []
    for i, peak in enumerate(nominal_peaks):
        phi_i = sensor[i]
        others = np.delete(sensor, i, axis=0)
        r = projection_residual(phi_i, others)
        results.append((peak, r))
    return results




# ============================================================
# Plotting
# ============================================================
def _plot_bars(ax, peaks, vals, annotate_style='plain'):
    """Draw bars with per-cone colors and annotations.

    annotate_style: 'plain' (text above bar), 'angled' (leader lines).
    """
    bar_colors = [CONE_COLORS.get(p, '#95a5a6') for p in peaks]
    bars = ax.bar(range(len(peaks)), vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels([f'{p}' for p in peaks], fontsize=6,
                       rotation=45 if len(peaks) > 4 else 0,
                       ha='right' if len(peaks) > 4 else 'center')

    def _fmt(v):
        if v >= 0.01:
            return f'{v:.4f}'
        elif v >= 0.001:
            return f'{v:.5f}'
        elif v >= 0.0001:
            return f'{v:.6f}'
        else:
            return f'{v:.7f}'

    if annotate_style == 'angled':
        for i, (bar, val) in enumerate(zip(bars, vals)):
            y_off = 14 + (i % 3) * 10
            ax.annotate(
                _fmt(val),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, y_off),
                textcoords='offset points',
                fontsize=5, ha='center',
                va='bottom', rotation=45,
                arrowprops=dict(arrowstyle='->', color='0.35', lw=0.5),
            )
    else:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    _fmt(val), ha='center', va='bottom', fontsize=5,
                    rotation=45)
    return bars


def plot_residuals(all_results: dict, output_dir: Path):
    """Two versions of the residual figure:
      - Log scale: all bars visible via log y on the hyperobserver row
      - Linear scale: hyperobserver bars annotated with arrow leaders
    Both: row 1 = dichromat | trichromat | tetrachromat (shared y),
          row 2 = hyperobserver spanning full width.
    """
    top_keys = ['Dichromat', 'Trichromat',
                'Tetrachromat']
    hyper_key = 'Hyperobserver'

    for variant, suffix in [('log', '_log'), ('linear', '_linear')]:
        fig = plt.figure(figsize=(3.33, 3.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.5, wspace=0.08)

        # --- Row 1: Dichromat / Trichromat / Tetrachromat (shared y) ---
        ax0 = fig.add_subplot(gs[0, 0])
        axes_top = [ax0]
        for col in range(1, 3):
            axes_top.append(fig.add_subplot(gs[0, col], sharey=ax0))

        for col, key in enumerate(top_keys):
            ax = axes_top[col]
            peaks = [r[0] for r in all_results[key]]
            vals = [r[1] for r in all_results[key]]
            _plot_bars(ax, peaks, vals, annotate_style='plain')

            short = key.split('(')[0].strip()
            ax.set_title(short, fontsize=7, fontweight='bold', pad=12)
            ax.tick_params(axis='y', labelsize=6)
            ax.tick_params(axis='x', labelsize=6)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            if col == 0:
                ax.set_ylabel('Residual $r$', fontsize=7, fontweight='bold')
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

        # --- Row 2: Hyperobserver spanning full width ---
        ax_hyp = fig.add_subplot(gs[1, :])
        peaks = [r[0] for r in all_results[hyper_key]]
        vals = [r[1] for r in all_results[hyper_key]]

        if variant == 'log':
            _plot_bars(ax_hyp, peaks, vals, annotate_style='plain')
            ax_hyp.set_yscale('log')
            ax_hyp.set_ylabel('Log Residual $r$', fontsize=7, fontweight='bold')
        else:
            _plot_bars(ax_hyp, peaks, vals, annotate_style='angled')
            ax_hyp.set_ylabel('Residual $r$', fontsize=7, fontweight='bold')

        ax_hyp.set_title('Hyperobserver', fontsize=7, fontweight='bold', pad=8)
        ax_hyp.set_xlabel('Cone Peak Wavelength (nm)', fontsize=7, fontweight='bold')
        ax_hyp.tick_params(axis='y', labelsize=6)
        ax_hyp.grid(axis='y', alpha=0.3, linestyle='--')
        ax_hyp.spines['top'].set_visible(False)

        fig.suptitle('Projection Residual  '
                     '$r_i = \\|\\varphi_{i,\\perp}\\| / \\|\\varphi_i\\|$',
                     fontsize=7, fontweight='bold', y=1.01)
        fig.savefig(output_dir / f'hyperobserver_residuals{suffix}.png',
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(output_dir / f'hyperobserver_residuals{suffix}.pdf',
                    bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved residual bar chart ({variant})")


def plot_gram_matrices(observers: dict, output_dir: Path):
    """Plot Gram matrix heatmap for each observer type."""
    n_obs = len(observers)
    fig, axes = plt.subplots(1, n_obs, figsize=(3.33, 1.6),
                             gridspec_kw={'width_ratios': [o.dimension for o, _ in observers.values()]})

    for idx, (obs_name, (observer, nominal_peaks)) in enumerate(observers.items()):
        ax = axes[idx]
        sensor = observer.get_sensor_matrix(WAVELENGTHS)
        peaks = nominal_peaks
        G = gram_matrix(sensor)

        im = ax.imshow(G, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')

        labels = [f'{p}' for p in peaks]
        ax.set_xticks(range(len(peaks)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_yticks(range(len(peaks)))
        ax.set_yticklabels(labels, fontsize=5)
        # Short label for tight space
        short_name = obs_name.split('(')[0].strip()
        ax.set_title(short_name, fontsize=6, fontweight='bold', pad=2)

        # Annotate cells (skip for hyperobserver — too dense)
        if len(peaks) <= 4:
            for i in range(len(peaks)):
                for j in range(len(peaks)):
                    val = G[i, j]
                    color = 'white' if val > 0.85 else 'black'
                    if i != j:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('$\\rho$', fontsize=6, fontweight='bold')
    fig.suptitle('Gram Matrix (Spectral Correlation)',
                 fontsize=7, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'gram_matrices.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'gram_matrices.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Gram matrix plots")


def plot_comparison(all_results: dict, output_dir: Path):
    """Single plot overlaying residuals for cones shared across observers."""
    fig, ax = plt.subplots(figsize=(3.33, 2.2))

    # For the hyperobserver, plot all 12 as baseline with per-cone colors
    hyper = all_results['Hyperobserver']
    hyper_peaks = [r[0] for r in hyper]
    hyper_vals = [r[1] for r in hyper]

    x = np.arange(len(hyper_peaks))
    _plot_bars(ax, hyper_peaks, hyper_vals, annotate_style='angled')

    # Overlay smaller observers at their matching x positions
    obs_palette = sns.color_palette("crest", n_colors=3)
    marker_styles = [('o', obs_palette[0], 'Dichromat (2)'),
                     ('s', obs_palette[1], 'Trichromat (3)'),
                     ('D', obs_palette[2], 'Tetrachromat (4)')]
    for (obs_name, residuals), (marker, color, label) in zip(
            list(all_results.items())[:3], marker_styles):
        for peak, val in residuals:
            if peak in hyper_peaks:
                xi = hyper_peaks.index(peak)
                ax.scatter(xi, val, marker=marker, color=color, s=25,
                           zorder=5, edgecolors='black', linewidths=0.4,
                           label=label)
                label = None  # only label once

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}' for p in hyper_peaks], fontsize=6, rotation=45, ha='right')
    ax.set_xlabel('Cone Peak Wavelength (nm)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Leave-One-Out Residual $r$', fontsize=8, fontweight='bold')
    ax.set_title('Residual Comparison Across Observer Types',
                 fontsize=8, fontweight='bold', pad=3)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'hyperobserver_residual_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'hyperobserver_residual_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison overlay plot")


def main():
    parser = argparse.ArgumentParser(description='Hyperobserver projection residual analysis')
    parser.add_argument('--output-dir', type=str,
                        default='output/hyperobserver_residual',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hyperobserver Leave-One-Out Residual Analysis")
    print("=" * 60)

    observers = build_observers()

    all_results = {}
    for obs_name, (observer, nominal_peaks) in observers.items():
        print(f"\n{obs_name}: peaks = {nominal_peaks}")
        residuals = leave_one_out_residuals(observer, nominal_peaks)
        all_results[obs_name] = residuals

        for peak, r in residuals:
            print(f"  {peak:>6} nm  →  r = {r:.6f}")

    # Save JSON
    json_data = {}
    for obs_name, residuals in all_results.items():
        json_data[obs_name] = {
            'peaks': [r[0] for r in residuals],
            'residuals': [r[1] for r in residuals],
            'n_cones': len(residuals),
        }
    with open(output_dir / 'residual_summary.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved JSON to {output_dir / 'residual_summary.json'}")

    # Gram matrices
    print("\nGram matrices:")
    for obs_name, (observer, nominal_peaks) in observers.items():
        sensor = observer.get_sensor_matrix(WAVELENGTHS)
        G = gram_matrix(sensor)
        print(f"\n  {obs_name}:")
        header = "        " + "  ".join(f"{p:>6}" for p in nominal_peaks)
        print(header)
        for i, pi in enumerate(nominal_peaks):
            row = f"  {pi:>6}" + "  ".join(f"{G[i,j]:>6.3f}" for j in range(len(nominal_peaks)))
            print(row)

    # Plot
    print("\nGenerating plots...")
    plot_residuals(all_results, output_dir)
    plot_comparison(all_results, output_dir)
    plot_gram_matrices(observers, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
