#!/usr/bin/env python3
"""
Analyze primary luminance warm-up drift across validation runs in a day folder.

Finds all validation_* subfolders, extracts per-run primary luminance, and
reports time deltas alongside luminance change percentages to characterize
display thermal stabilization.

Usage:
    python analyze_warmup_drift.py /path/to/measurements/2026-04-30
    python analyze_warmup_drift.py /path/to/measurements/2026-04-30 --primaries-only
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

PRIMARIES = {
    'R': (255, 0, 0, 0),
    'G': (0, 255, 0, 0),
    'B': (0, 0, 255, 0),
    'O': (0, 0, 0, 255),
}


def parse_run_timestamp(folder_name: str) -> datetime:
    """Extract datetime from folder names like 'validation_2026-04-30_11-56-53'."""
    parts = folder_name.split('_')
    # parts: ['validation', '2026-04-30', '11-56-53']
    date_str = parts[1]
    time_str = parts[2].replace('-', ':')
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def find_validation_runs(day_folder: Path) -> list[tuple[datetime, Path]]:
    """Return (timestamp, path) pairs for all validation_* subfolders, sorted by time."""
    runs = []
    for folder in day_folder.iterdir():
        if folder.is_dir() and folder.name.startswith('validation_'):
            try:
                ts = parse_run_timestamp(folder.name)
                runs.append((ts, folder))
            except (ValueError, IndexError):
                print(f"Warning: could not parse timestamp from '{folder.name}', skipping.")
    runs.sort(key=lambda x: x[0])
    return runs


def load_primary_luminance(primaries_dir: Path, rgbo: tuple[int, int, int, int]) -> float | None:
    """Load luminance (cd/m²) for a given RGBO from the most recent matching CSV."""
    r, g, b, o = rgbo
    pattern = f"r{r}g{g}b{b}o{o}"
    matches = sorted(f for f in primaries_dir.iterdir()
                     if f.name.startswith(pattern) and f.name.endswith('.csv'))
    if not matches:
        return None
    with open(matches[-1], newline='') as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 3:
                try:
                    return float(row[2])
                except ValueError:
                    continue
    return None


def load_run_primaries(run_folder: Path) -> dict[str, float | None]:
    """Return {channel: luminance} for a single validation run."""
    primaries_dir = run_folder / 'primaries'
    if not primaries_dir.exists():
        return {ch: None for ch in PRIMARIES}
    return {ch: load_primary_luminance(primaries_dir, rgbo) for ch, rgbo in PRIMARIES.items()}


def format_minutes(delta_seconds: float) -> str:
    m = int(delta_seconds) // 60
    s = int(delta_seconds) % 60
    return f"{m}m {s:02d}s"


def print_table(runs: list[tuple[datetime, Path]], lum: list[dict[str, float | None]]):
    n = len(runs)
    t0 = runs[0][0]

    header = f"{'Run':<6}  {'Folder':<40}  {'Elapsed':>9}  {'ΔPrev':>7}  " + \
             "  ".join(f"{ch:>10}" for ch in PRIMARIES)
    print(header)
    print("-" * len(header))

    for i, ((ts, folder), lums) in enumerate(zip(runs, lum)):
        elapsed = (ts - t0).total_seconds()
        delta_prev = "" if i == 0 else format_minutes((ts - runs[i-1][0]).total_seconds())
        elapsed_str = format_minutes(elapsed)

        row = f"{i+1:<6}  {folder.name:<40}  {elapsed_str:>9}  {delta_prev:>7}  "
        row += "  ".join(
            f"{lums[ch]:>10.5f}" if lums[ch] is not None else f"{'N/A':>10}"
            for ch in PRIMARIES
        )
        print(row)

    print()
    print("=== Luminance change vs Run 1 (cd/m²) ===")
    ref = lum[0]
    header2 = f"{'Run':<6}  {'Elapsed':>9}  " + "  ".join(f"{ch:>10}" for ch in PRIMARIES)
    print(header2)
    print("-" * len(header2))
    for i, ((ts, _), lums) in enumerate(zip(runs, lum)):
        elapsed = format_minutes((ts - t0).total_seconds())
        row = f"{i+1:<6}  {elapsed:>9}  "
        parts = []
        for ch in PRIMARIES:
            if lums[ch] is not None and ref[ch] is not None and ref[ch] != 0:
                pct = 100 * (lums[ch] - ref[ch]) / ref[ch]
                parts.append(f"{pct:>+9.2f}%")
            else:
                parts.append(f"{'N/A':>10}")
        row += "  ".join(parts)
        print(row)

    print()
    print("=== Luminance change vs previous run (cd/m²) ===")
    print(header2)
    print("-" * len(header2))
    for i, ((ts, _), lums) in enumerate(zip(runs, lum)):
        elapsed = format_minutes((ts - t0).total_seconds()) if i > 0 else "    —"
        row = f"{i+1:<6}  {elapsed:>9}  "
        parts = []
        prev = lum[i-1] if i > 0 else None
        for ch in PRIMARIES:
            if prev is None:
                parts.append(f"{'baseline':>10}")
            elif lums[ch] is not None and prev[ch] is not None and prev[ch] != 0:
                pct = 100 * (lums[ch] - prev[ch]) / prev[ch]
                parts.append(f"{pct:>+9.2f}%")
            else:
                parts.append(f"{'N/A':>10}")
        row += "  ".join(parts)
        print(row)


def plot_warmup(runs: list[tuple[datetime, Path]], lum: list[dict[str, float | None]],
                out_path: Path):
    t0 = runs[0][0]
    times = [(ts - t0).total_seconds() / 60 for ts, _ in runs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'R': '#e74c3c', 'G': '#2ecc71', 'B': '#3498db', 'O': '#e67e22'}

    # Left: raw luminance
    ax = axes[0]
    for ch in PRIMARIES:
        vals = [l[ch] for l in lum]
        if any(v is not None for v in vals):
            ax.plot(times, vals, 'o-', color=colors[ch], label=ch, linewidth=2, markersize=7)
    ax.set_xlabel('Time since first run (minutes)')
    ax.set_ylabel('Luminance (cd/m²)')
    ax.set_title('Primary luminance over session')
    ax.legend()

    # Right: % change vs run 1
    ax2 = axes[1]
    ref = lum[0]
    for ch in PRIMARIES:
        pcts = []
        for l in lum:
            if l[ch] is not None and ref[ch] is not None and ref[ch] != 0:
                pcts.append(100 * (l[ch] - ref[ch]) / ref[ch])
            else:
                pcts.append(None)
        if any(v is not None for v in pcts):
            ax2.plot(times, pcts, 'o-', color=colors[ch], label=ch, linewidth=2, markersize=7)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time since first run (minutes)')
    ax2.set_ylabel('Luminance change vs run 1 (%)')
    ax2.set_title('Warm-up drift (% vs baseline)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('day_folder', type=Path,
                        help='Path to a day measurement folder (e.g. measurements/2026-04-30)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating the warm-up plot')
    args = parser.parse_args()

    day_folder = args.day_folder.resolve()
    if not day_folder.exists():
        print(f"Error: folder not found: {day_folder}", file=sys.stderr)
        sys.exit(1)

    runs = find_validation_runs(day_folder)
    if not runs:
        print(f"No validation_* subfolders found in {day_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(runs)} validation run(s) in {day_folder.name}\n")

    lum = [load_run_primaries(folder) for _, folder in runs]
    print_table(runs, lum)

    if not args.no_plot:
        out_path = day_folder / 'warmup_drift.png'
        plot_warmup(runs, lum, out_path)


if __name__ == '__main__':
    main()
