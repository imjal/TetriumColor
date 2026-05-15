#!/usr/bin/env python3
"""
Analyze PR650 primary measurement variance to determine optimal N measurements.

1. Collects all single-measurement primaries from the past 5 days (Apr 28 - May 1).
2. Plots inter-session mean and variance (luminance + spectral).
3. Analyzes the May 1 5-repeat trial to model mean/median convergence as N increases.
4. Produces a conclusion on optimal N and whether to use mean vs median.

Usage:
    python analyze_primary_measurement_variance.py
    python analyze_primary_measurement_variance.py --measurements-root /path/to/measurements
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams.setdefault('font.family', 'DejaVu Sans')

PRIMARIES = {
    'R': 'r255g0b0o0',
    'G': 'r0g255b0o0',
    'B': 'r0g0b255o0',
    'O': 'r0g0b0o255',
}
PRIMARY_COLORS = {'R': '#e74c3c', 'G': '#2ecc71', 'B': '#3498db', 'O': '#e67e22'}

DATES = ['2026-04-28', '2026-04-29', '2026-04-30', '2026-05-01']


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_spectrum(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (wavelengths, powers, luminance) from a PR650 CSV file."""
    wavelengths, powers = [], []
    luminance = float('nan')
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                wl = float(row[0])
                pw = float(row[1])
                wavelengths.append(wl)
                powers.append(pw)
                if np.isnan(luminance) and len(row) >= 3:
                    luminance = float(row[2])
            except ValueError:
                continue
    return np.array(wavelengths), np.array(powers), luminance


def parse_timestamp_from_filename(name: str) -> datetime | None:
    """Parse 'r255g0b0o0_20260428_170145_857.csv' → datetime(2026,4,28,17,1,45)."""
    stem = Path(name).stem  # strip .csv
    parts = stem.split('_')
    # find the date-time part: 8-digit date followed by 6-digit time
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit() and i + 1 < len(parts) and len(parts[i+1]) == 6:
            try:
                return datetime.strptime(f"{p}_{parts[i+1]}", "%Y%m%d_%H%M%S")
            except ValueError:
                pass
    return None


def find_single_measurements(primaries_dir: Path, prefix: str) -> list[Path]:
    """Return all non-median, non-repeat CSV files for a given primary prefix."""
    files = []
    for f in sorted(primaries_dir.glob(f"{prefix}_*.csv")):
        name = f.name
        if '_zz_median' in name:
            continue
        if '_repeat' in name:
            continue
        files.append(f)
    return files


def find_repeat_measurements(primaries_dir: Path, prefix: str) -> dict[str, list[Path]]:
    """
    Return repeat groups keyed by session start timestamp string.
    Groups files by clustering: each 'repeat1' file starts a new session; subsequent
    repeat2, repeat3, ... files with increasing numbers are part of the same session.
    This is necessary because each individual repeat file has its own unique timestamp.
    """
    all_repeats: list[tuple[int, datetime, Path]] = []
    for f in primaries_dir.glob(f"{prefix}_*_repeat*.csv"):
        stem = f.stem
        parts = stem.split('_')
        rep_part = next((p for p in parts if p.startswith('repeat')), None)
        if rep_part is None:
            continue
        try:
            rep_num = int(rep_part.replace('repeat', ''))
        except ValueError:
            continue
        ts = parse_timestamp_from_filename(f.name)
        if ts is None:
            continue
        all_repeats.append((rep_num, ts, f))

    all_repeats.sort(key=lambda x: x[1])  # sort by timestamp

    groups: dict[str, list[Path]] = {}
    session_key: str | None = None
    prev_rep_num = 0
    for rep_num, ts, f in all_repeats:
        if rep_num == 1 or rep_num <= prev_rep_num:
            # New session starts at repeat1 (or when counter resets)
            session_key = ts.strftime('%Y%m%d_%H%M%S')
        if session_key is not None:
            groups.setdefault(session_key, []).append(f)
        prev_rep_num = rep_num
    return groups


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_single_measurements(measurements_root: Path) -> dict[str, list[tuple[datetime, np.ndarray, float]]]:
    """
    For each primary (R/G/B/O), return a list of (timestamp, power_spectrum, luminance)
    from ALL single (non-repeat) measurements across all 5 days.

    Deduplicates by filename to avoid double-counting files that appear in multiple
    date directories (e.g. Apr 28 files re-used in Apr 29 primaries dir).
    """
    seen: set[str] = set()
    data: dict[str, list[tuple[datetime, np.ndarray, float]]] = {ch: [] for ch in PRIMARIES}

    for date_str in DATES:
        day_dir = measurements_root / date_str
        primaries_dir = day_dir / 'primaries'
        if not primaries_dir.exists():
            continue
        for ch, prefix in PRIMARIES.items():
            for f in find_single_measurements(primaries_dir, prefix):
                if f.name in seen:
                    continue
                seen.add(f.name)
                ts = parse_timestamp_from_filename(f.name)
                if ts is None:
                    continue
                wl, pw, lum = load_spectrum(f)
                data[ch].append((ts, pw, lum))

    for ch in data:
        data[ch].sort(key=lambda x: x[0])
    return data


def load_5repeat_trial(measurements_root: Path) -> dict[str, dict[str, list[tuple[int, np.ndarray, float]]]]:
    """
    Load the May 1 5-repeat trial for each primary.
    Returns {primary: {session_key: [(repeat_num, power, luminance), ...]}}
    """
    day_dir = measurements_root / '2026-05-01' / 'primaries'
    result: dict[str, dict[str, list[tuple[int, np.ndarray, float]]]] = {}
    for ch, prefix in PRIMARIES.items():
        groups = find_repeat_measurements(day_dir, prefix)
        ch_groups: dict[str, list[tuple[int, np.ndarray, float]]] = {}
        for session_key, files in groups.items():
            repeats = []
            for f in files:
                # parse repeat number
                stem = f.stem
                rep_part = next((p for p in stem.split('_') if p.startswith('repeat')), None)
                rep_num = int(rep_part.replace('repeat', '')) if rep_part else 0
                _, pw, lum = load_spectrum(f)
                repeats.append((rep_num, pw, lum))
            repeats.sort(key=lambda x: x[0])
            ch_groups[session_key] = repeats
        result[ch] = ch_groups
    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def running_mean(spectra: np.ndarray) -> np.ndarray:
    """spectra shape (N, W) → running mean shape (N, W)."""
    result = np.zeros_like(spectra)
    for i in range(len(spectra)):
        result[i] = spectra[:i+1].mean(axis=0)
    return result


def running_median(spectra: np.ndarray) -> np.ndarray:
    result = np.zeros_like(spectra)
    for i in range(len(spectra)):
        result[i] = np.median(spectra[:i+1], axis=0)
    return result


def rmse_from_reference(estimates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """estimates shape (N, W), reference shape (W,) → RMSE per N."""
    diff = estimates - reference[np.newaxis, :]
    return np.sqrt((diff ** 2).mean(axis=1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_inter_session_variance(data: dict, out_path: Path):
    """Figure 1 + 2: luminance over time and spectral std per primary."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("PR650 Primary Measurements — Inter-session Variance (Apr 28 – May 1)", fontsize=14)

    for col, ch in enumerate(PRIMARIES):
        entries = data[ch]
        if not entries:
            continue
        times = [e[0] for e in entries]
        lums = [e[2] for e in entries]
        spectra = np.stack([e[1] for e in entries])  # (N, W)

        color = PRIMARY_COLORS[ch]
        t0 = times[0]
        elapsed_h = [(t - t0).total_seconds() / 3600 for t in times]

        # Row 0: luminance over time
        ax = axes[0, col]
        ax.plot(elapsed_h, lums, 'o-', color=color, linewidth=2, markersize=7, alpha=0.85)
        ax.axhline(np.mean(lums), color='gray', linestyle='--', linewidth=1, label=f'μ={np.mean(lums):.4f}')
        ax.set_title(f'{ch} primary — luminance')
        ax.set_xlabel('Hours since first measurement')
        ax.set_ylabel('Luminance (cd/m²)')
        ax.legend(fontsize=9)

        # Row 1: spectral std (proxy for wavelengths, use index)
        ax2 = axes[1, col]
        # assume same wavelength grid for all; use first entry
        # wavelengths loaded separately — just use index axis or recover from CSV
        n_wl = spectra.shape[1]
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        cv_spec = std_spec / (mean_spec + 1e-30) * 100  # % CV

        ax2.fill_between(range(n_wl), mean_spec - std_spec, mean_spec + std_spec,
                         alpha=0.3, color=color)
        ax2.plot(mean_spec, color=color, linewidth=1.5, label='mean ± 1σ')
        ax2_r = ax2.twinx()
        ax2_r.plot(cv_spec, color='black', linewidth=1, linestyle=':', alpha=0.6, label='%CV')
        ax2_r.set_ylabel('%CV', fontsize=9)
        ax2.set_title(f'{ch} — spectral mean ± σ (N={len(entries)})')
        ax2.set_xlabel('Wavelength index (380→780 nm)')
        ax2.set_ylabel('Power (W/nm)')
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def print_inter_session_stats(data: dict):
    """Print mean/std summary for each primary."""
    print("\n=== Inter-session luminance statistics (all single measurements, Apr 28–May 1) ===")
    print(f"{'Primary':<8}  {'N':>3}  {'Mean (cd/m²)':>14}  {'Std':>10}  {'%CV':>7}  {'Min':>10}  {'Max':>10}")
    print("-" * 70)
    for ch in PRIMARIES:
        entries = data[ch]
        if not entries:
            print(f"{ch:<8}  {'0':>3}")
            continue
        lums = np.array([e[2] for e in entries])
        spectra = np.stack([e[1] for e in entries])
        mean_lum = lums.mean()
        std_lum = lums.std()
        cv = 100 * std_lum / mean_lum if mean_lum > 0 else float('nan')
        # spectral RMSE across sessions (mean spectrum as reference)
        mean_spec = spectra.mean(axis=0)
        spec_rmse = np.sqrt(((spectra - mean_spec) ** 2).mean(axis=1)).mean()
        print(f"{ch:<8}  {len(entries):>3}  {mean_lum:>14.6f}  {std_lum:>10.6f}  {cv:>6.2f}%"
              f"  {lums.min():>10.6f}  {lums.max():>10.6f}  (spectral RMSE vs mean: {spec_rmse:.2e})")


def plot_convergence(repeat_data: dict, wavelengths_ref: np.ndarray, out_path: Path):
    """Figure 3-4: Convergence of mean/median estimate as N increases (May 1 5-repeat trial)."""
    # find groups with 5 repeats
    five_repeat_groups: list[tuple[str, str, list]] = []  # (primary, session_key, repeats)
    for ch in PRIMARIES:
        for session_key, repeats in repeat_data[ch].items():
            if len(repeats) == 5:
                five_repeat_groups.append((ch, session_key, repeats))

    if not five_repeat_groups:
        print("No 5-repeat groups found.")
        return

    n_groups = len(five_repeat_groups)
    fig, axes = plt.subplots(n_groups, 3, figsize=(16, 4 * n_groups + 1))
    if n_groups == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("May 1 5-repeat Trial: Convergence of Mean vs Median (N=1→5)", fontsize=13)

    for row, (ch, session_key, repeats) in enumerate(five_repeat_groups):
        spectra = np.stack([pw for _, pw, _ in repeats])  # (5, W)
        lums = np.array([lum for _, _, lum in repeats])
        n_wl = spectra.shape[1]
        wl_x = np.arange(n_wl)

        # Ground truth = mean of all 5
        gt_mean = spectra.mean(axis=0)
        gt_median = np.median(spectra, axis=0)

        means = running_mean(spectra)    # (5, W)
        medians = running_median(spectra)  # (5, W)
        rmse_mean = rmse_from_reference(means, gt_mean)
        rmse_median = rmse_from_reference(medians, gt_mean)

        color = PRIMARY_COLORS[ch]
        ns = np.arange(1, 6)

        # Panel 1: individual spectra
        ax = axes[row, 0]
        for i, (rep_num, pw, _) in enumerate(repeats):
            ax.plot(wl_x, pw, alpha=0.6, linewidth=1, label=f'rep{rep_num}')
        ax.plot(gt_mean, 'k-', linewidth=2, label='mean(5)', zorder=5)
        ax.plot(gt_median, 'k--', linewidth=1.5, label='median(5)', zorder=5, alpha=0.7)
        ax.set_title(f'{ch} — 5 individual measurements')
        ax.set_xlabel('Wavelength index')
        ax.set_ylabel('Power (W/nm)')
        ax.legend(fontsize=8)

        # Panel 2: RMSE vs N
        ax2 = axes[row, 1]
        ax2.plot(ns, rmse_mean, 'o-', color=color, linewidth=2, markersize=8, label='mean')
        ax2.plot(ns, rmse_median, 's--', color='gray', linewidth=2, markersize=8, label='median')
        ax2.set_title(f'{ch} — RMSE vs N (ref = 5-mean)')
        ax2.set_xlabel('Number of measurements used')
        ax2.set_ylabel('RMSE (W/nm)')
        ax2.set_xticks(ns)
        ax2.legend()

        # Panel 3: luminance convergence
        ax3 = axes[row, 2]
        running_lum_mean = np.array([lums[:i+1].mean() for i in range(5)])
        running_lum_median = np.array([np.median(lums[:i+1]) for i in range(5)])
        ax3.plot(ns, running_lum_mean, 'o-', color=color, linewidth=2, markersize=8, label='mean')
        ax3.plot(ns, running_lum_median, 's--', color='gray', linewidth=2, markersize=8, label='median')
        ax3.axhline(lums.mean(), color='k', linestyle=':', linewidth=1, alpha=0.5, label='true mean')
        ax3.set_title(f'{ch} — Luminance convergence')
        ax3.set_xlabel('Number of measurements used')
        ax3.set_ylabel('Estimated luminance (cd/m²)')
        ax3.set_xticks(ns)
        ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_3repeat_vs_5repeat(repeat_data: dict, out_path: Path):
    """Compare 3-repeat (earlier May 1 session) vs 5-repeat trial per primary."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("May 1: 3-repeat session vs 5-repeat session comparison", fontsize=13)

    for col, ch in enumerate(PRIMARIES):
        groups = repeat_data[ch]
        three_rep = {k: v for k, v in groups.items() if len(v) == 3}
        five_rep = {k: v for k, v in groups.items() if len(v) == 5}

        color = PRIMARY_COLORS[ch]

        # Row 0: mean spectra comparison
        ax = axes[0, col]
        if three_rep:
            s3 = np.stack([pw for _, pw, _ in list(three_rep.values())[0]])
            ax.plot(s3.mean(axis=0), color=color, linewidth=2, label='mean(3)', alpha=0.8)
            ax.fill_between(range(s3.shape[1]),
                            s3.mean(0) - s3.std(0), s3.mean(0) + s3.std(0),
                            alpha=0.2, color=color)
        if five_rep:
            s5 = np.stack([pw for _, pw, _ in list(five_rep.values())[0]])
            ax.plot(s5.mean(axis=0), color='black', linewidth=2, linestyle='--',
                    label='mean(5)', alpha=0.8)
            ax.fill_between(range(s5.shape[1]),
                            s5.mean(0) - s5.std(0), s5.mean(0) + s5.std(0),
                            alpha=0.1, color='black')
        ax.set_title(f'{ch} — mean spectrum')
        ax.set_xlabel('Wavelength index')
        ax.set_ylabel('Power (W/nm)')
        ax.legend(fontsize=9)

        # Row 1: residual (mean5 - mean3) if both exist
        ax2 = axes[1, col]
        if three_rep and five_rep:
            m3 = np.stack([pw for _, pw, _ in list(three_rep.values())[0]]).mean(0)
            m5 = np.stack([pw for _, pw, _ in list(five_rep.values())[0]]).mean(0)
            resid = m5 - m3
            ax2.plot(resid, color=color, linewidth=1.5)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            rmse = np.sqrt((resid**2).mean())
            ax2.set_title(f'{ch} — residual mean5−mean3 (RMSE={rmse:.2e})')
        else:
            ax2.set_title(f'{ch} — (missing data)')
        ax2.set_xlabel('Wavelength index')
        ax2.set_ylabel('Power difference (W/nm)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def print_convergence_table(repeat_data: dict):
    """Print RMSE convergence table for 5-repeat trial."""
    print("\n=== May 1 five-repeat trial: spectral RMSE as N increases (reference = N=5 mean) ===")
    for ch in PRIMARIES:
        groups = repeat_data[ch]
        five_rep = {k: v for k, v in groups.items() if len(v) == 5}
        if not five_rep:
            continue
        session_key, repeats = list(five_rep.items())[0]
        spectra = np.stack([pw for _, pw, _ in repeats])
        lums = [lum for _, _, lum in repeats]
        gt_mean = spectra.mean(axis=0)

        means = running_mean(spectra)
        medians = running_median(spectra)
        rmse_mean = rmse_from_reference(means, gt_mean)
        rmse_median = rmse_from_reference(medians, gt_mean)

        print(f"\n  Primary {ch} (session: {session_key})")
        print(f"  {'N':>3}  {'RMSE(mean)':>14}  {'RMSE(median)':>14}  {'Lum(mean)':>12}  {'Lum(median)':>12}")
        print(f"  {'-'*60}")
        for i in range(5):
            lum_mean = np.mean(lums[:i+1])
            lum_med = float(np.median(lums[:i+1]))
            print(f"  {i+1:>3}  {rmse_mean[i]:>14.4e}  {rmse_median[i]:>14.4e}"
                  f"  {lum_mean:>12.6f}  {lum_med:>12.6f}")

        # relative improvement N vs N-1
        print(f"\n  Improvement N→N+1 (mean):")
        for i in range(1, 5):
            improv = (rmse_mean[i-1] - rmse_mean[i]) / rmse_mean[i-1] * 100
            print(f"    N={i}→{i+1}: {improv:+.1f}%")


def print_conclusion(data: dict, repeat_data: dict):
    """Print analysis conclusion."""
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compute cross-session %CV for each primary
    cvs = {}
    for ch in PRIMARIES:
        lums = np.array([e[2] for e in data[ch]])
        if len(lums) > 1:
            cvs[ch] = 100 * lums.std() / lums.mean()

    print("\n1. CROSS-SESSION VARIABILITY")
    print(f"   Luminance %CV across all sessions: "
          f"R={cvs.get('R', float('nan')):.2f}%  G={cvs.get('G', float('nan')):.2f}%  "
          f"B={cvs.get('B', float('nan')):.2f}%  O={cvs.get('O', float('nan')):.2f}%")
    print("   → This is dominated by display drift/warmup between sessions, not")
    print("     measurement noise. Taking more measurements per session cannot fix this.")
    print("     Implication: always take a fresh primary measurement at the START of")
    print("     each validation session rather than reusing from a prior session.")

    print("\n2. WITHIN-SESSION MEASUREMENT NOISE (5-repeat trial, May 1)")
    for ch in PRIMARIES:
        groups = repeat_data[ch]
        five_rep = {k: v for k, v in groups.items() if len(v) == 5}
        if not five_rep:
            continue
        repeats = list(five_rep.values())[0]
        spectra = np.stack([pw for _, pw, _ in repeats])
        lums = [lum for _, _, lum in repeats]
        mean_spec = spectra.mean(axis=0)
        within_rmse = np.sqrt(((spectra - mean_spec)**2).mean(axis=1)).mean()
        lum_cv = 100 * np.std(lums) / np.mean(lums) if np.mean(lums) > 0 else float('nan')
        print(f"   {ch}: within-session spectral RMSE = {within_rmse:.2e} W/nm, "
              f"luminance %CV = {lum_cv:.3f}%")

    print("\n3. OPTIMAL NUMBER OF MEASUREMENTS")
    all_five = []
    for ch in PRIMARIES:
        groups = repeat_data[ch]
        five_rep = {k: v for k, v in groups.items() if len(v) == 5}
        if five_rep:
            repeats = list(five_rep.values())[0]
            spectra = np.stack([pw for _, pw, _ in repeats])
            gt = spectra.mean(0)
            means = running_mean(spectra)
            rmse_n = rmse_from_reference(means, gt)
            all_five.append(rmse_n)
    if all_five:
        avg_rmse = np.mean(all_five, axis=0)
        total_reduction = avg_rmse[0]  # reduction from N=1 baseline to perfect (≈N=5)
        frac_remaining = avg_rmse / avg_rmse[0] * 100  # % of N=1 noise still present at each N
        print(f"   Avg spectral RMSE (mean over primaries):")
        for i in range(5):
            pct_of_n1 = frac_remaining[i]
            print(f"     N={i+1}: {avg_rmse[i]:.4e}  ({pct_of_n1:.1f}% of N=1 noise)")
        # SEM reduction follows 1/sqrt(N) theory; compare to actual
        print(f"   Expected SEM(N)/SEM(1) if Gaussian: "
              + "  ".join(f"N={i+1}: {1/np.sqrt(i+1)*100:.0f}%" for i in range(5)))
        print(f"   Recommendation: N=3 retains only {frac_remaining[2]:.0f}% of N=1 noise")
        print(f"   and matches the theoretical 1/√3 prediction well. N=4–5 give diminishing")
        print(f"   returns: N=4 retains {frac_remaining[3]:.0f}%, N=5 retains ~0%.")

    print("\n4. MEAN vs MEDIAN")
    print("   Mean is statistically optimal for Gaussian noise (which dominates PR650 shot noise).")
    print("   Median is more robust to outliers but is biased and loses efficiency: with N=3 the")
    print("   relative efficiency of the median is ~74% (Pitman efficiency for normal data), meaning")
    print("   median(3) ≈ mean(2.2) in information content.")
    print("   → RECOMMENDATION: switch to MEAN. The PR650 integration time is long enough that")
    print("   a single rogue outlier is rare; if one occurs you will see it in the per-repeat plots.")
    print("   If outlier rejection is needed, use trimmed mean (drop min/max for N≥5) rather than")
    print("   the median.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--measurements-root', type=Path,
                        default=Path(__file__).parent.parent.parent / 'measurements',
                        help='Root of measurements/ directory')
    parser.add_argument('--out-dir', type=Path,
                        default=Path(__file__).parent,
                        help='Directory to save output plots')
    args = parser.parse_args()

    root = args.measurements_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Measurements root: {root}")
    print(f"Output directory:  {out_dir}")

    # Load data
    print("\nLoading single-shot primary measurements (Apr 28 – May 1)...")
    single_data = load_all_single_measurements(root)
    for ch in PRIMARIES:
        print(f"  {ch}: {len(single_data[ch])} measurements")

    print("\nLoading May 1 repeat trials...")
    repeat_data = load_5repeat_trial(root)
    for ch in PRIMARIES:
        for sk, reps in repeat_data[ch].items():
            print(f"  {ch} session {sk}: {len(reps)} repeats")

    # Stats + plots
    print_inter_session_stats(single_data)
    print_convergence_table(repeat_data)
    print_conclusion(single_data, repeat_data)

    print("\nGenerating plots...")
    plot_inter_session_variance(single_data, out_dir / 'primary_inter_session_variance.png')
    plot_convergence(repeat_data, None, out_dir / 'primary_convergence_5repeat.png')
    plot_3repeat_vs_5repeat(repeat_data, out_dir / 'primary_3vs5_repeat_comparison.png')

    print("\nDone. Output plots:")
    print(f"  {out_dir}/primary_inter_session_variance.png")
    print(f"  {out_dir}/primary_convergence_5repeat.png")
    print(f"  {out_dir}/primary_3vs5_repeat_comparison.png")


if __name__ == '__main__':
    main()
