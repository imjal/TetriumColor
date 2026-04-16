#!/usr/bin/env python3
"""
Compare all trial results from optimize_hyperobserver_display.py.

Reads summary.json files from each trial subdirectory and produces a
comparison table and bar chart.

Usage:
    python compare_trials.py [--output-dir output/display_optimization]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Compare display optimization trials')
    parser.add_argument('--output-dir', type=str, default='output/display_optimization')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trials = []

    for summary_path in sorted(output_dir.glob('*/summary.json')):
        with open(summary_path) as f:
            data = json.load(f)
        d_targets = [c['d_target'] for c in data['per_cone']]
        d_rests = [c['d_rest'] for c in data['per_cone']]
        trials.append({
            'label': data['trial'],
            'observer': data['observer'],
            'n_cones': data['n_cones'],
            'n_primaries': data['n_primaries'],
            'fwhm_range': data['fwhm_range'],
            'cond': data['cond_C'],
            'min_d_target': min(d_targets),
            'median_d_target': float(np.median(d_targets)),
            'max_d_target': max(d_targets),
            'max_d_rest': max(d_rests),
            'd_targets': d_targets,
            'cone_peaks': [c['cone_peak_nm'] for c in data['per_cone']],
        })

    if not trials:
        print(f"No trial results found in {output_dir}")
        return

    # Print comparison table
    print(f"\n{'Trial':<45} {'cones':>5} {'prims':>5} {'FWHM':>10} "
          f"{'cond(C)':>10} {'min_d':>10} {'med_d':>10} {'max_d':>10} {'max_drest':>10}")
    print("-" * 130)
    for t in trials:
        fwhm = f"{t['fwhm_range'][0]:.0f}-{t['fwhm_range'][1]:.0f}"
        print(f"{t['label']:<45} {t['n_cones']:>5} {t['n_primaries']:>5} {fwhm:>10} "
              f"{t['cond']:>10.2e} {t['min_d_target']:>10.4f} "
              f"{t['median_d_target']:>10.4f} {t['max_d_target']:>10.4f} "
              f"{t['max_d_rest']:>10.2e}")

    # Find hyperobserver trials for comparison plot
    hyper_trials = [t for t in trials if t['observer'] == 'hyperobserver']
    if len(hyper_trials) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        n_cones = hyper_trials[0]['n_cones']
        x = np.arange(n_cones)
        width = 0.8 / len(hyper_trials)
        colors = plt.cm.Set2(np.linspace(0, 1, len(hyper_trials)))

        for i, t in enumerate(hyper_trials):
            offset = (i - len(hyper_trials)/2 + 0.5) * width
            bars = ax.bar(x + offset, t['d_targets'], width, label=t['label'],
                          color=colors[i], edgecolor='black', linewidth=0.3)

        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1 JND')
        ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='3 JND')
        ax.set_xticks(x)
        peak_labels = [f"{p:.0f}" if p == int(p) else f"{p}"
                       for p in hyper_trials[0]['cone_peaks']]
        ax.set_xticklabels(peak_labels, rotation=45, ha='right')
        ax.set_xlabel('Cone peak [nm]')
        ax.set_ylabel('d_target (JND)')
        ax.set_title('Cone Isolation Comparison Across Trials')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_yscale('symlog', linthresh=0.01)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(output_dir / 'trial_comparison.png', dpi=150)
        plt.close(fig)
        print(f"\nSaved comparison plot to {output_dir / 'trial_comparison.png'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
