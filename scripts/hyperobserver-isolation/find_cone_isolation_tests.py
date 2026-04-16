#!/usr/bin/env python3
"""
12-Primary Display Cone Isolation Analysis
===========================================
Determines the best 12-primary Gaussian-LED display configuration for isolating
each of the 12 hyperobserver cones, sweeping FWHM from narrowband to broadband.

The hyperobserver has 12 cones at:
    S: 420 nm | M: 530, 533, 536 nm | L: 547, 551, 552, 553, 555, 556, 556.5, 559 nm

Two approaches are used for finding stimulus pairs:

  1. ANALYTICAL (exact isolation, d_rest = 0):
     Direction w1-w2 is forced along C^{-1}[:, k].  Only cone k differs.
     Used inside the LED-position optimizer (fast).

  2. DIRECT OPTIMIZATION (relaxed, d_rest <= threshold):
     w1 and w2 are optimized independently in [0, 1]^12 to maximize d_target
     while keeping d_rest below a threshold.  This finds better pairs because
     it can trade a small amount of cross-talk for a much larger target signal.
     Run as a final step once LED positions are determined.

Primary peak positions are optimized per FWHM to maximize worst-case isolation.

Usage:
    python find_cone_isolation_tests.py --fixed-peaks --fwhm-max 20
    python find_cone_isolation_tests.py --fwhm-min 1 --fwhm-max 25 --fwhm-step 2
    python find_cone_isolation_tests.py --d-rest-threshold 1.0
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from TetriumColor.Observer.Observer import Observer

# ============================================================
# Configuration
# ============================================================
WAVELENGTHS = np.arange(400, 701, 1)
HYPEROBSERVER_PEAKS = np.array([420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559])
N_CONES = len(HYPEROBSERVER_PEAKS)

SIGMA_NOISE = np.ones(N_CONES)

PEAK_LABELS = [f"{p:.0f}" if p == int(p) else f"{p}" for p in HYPEROBSERVER_PEAKS]

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247


# ============================================================
# Data structures
# ============================================================
@dataclass
class ConeIsolationTest:
    """A single cone-isolation test: two stimuli that differ primarily in cone k."""
    target_cone: int
    target_peak_nm: float
    w1: np.ndarray          # display LED weights for stimulus 1
    w2: np.ndarray          # display LED weights for stimulus 2
    r1: np.ndarray          # 12D cone responses to stimulus 1
    r2: np.ndarray          # 12D cone responses to stimulus 2
    d_target: float         # Mahalanobis distance for the target cone
    d_rest: float           # Mahalanobis distance for the other 11 cones
    cone_contrast: np.ndarray  # per-cone (r1-r2)/avg


# ============================================================
# Spectral Primitives
# ============================================================
def gaussian_led(wavelengths: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    """Unit-amplitude Gaussian LED SPD."""
    sigma = fwhm * FWHM_TO_SIGMA
    return np.exp(-(wavelengths - peak) ** 2 / (2 * sigma ** 2))


def build_display_matrix(wavelengths: np.ndarray, peaks: np.ndarray, fwhm: float) -> np.ndarray:
    """Build (n_wl, n_primaries) display matrix of Gaussian LED SPDs."""
    return np.column_stack([gaussian_led(wavelengths, p, fwhm) for p in peaks])


# ============================================================
# Transfer Matrix & Cone Distance
# ============================================================
def build_transfer_matrix(sensor_matrix: np.ndarray, display_matrix: np.ndarray) -> np.ndarray:
    """C = sensor_matrix @ D.  Maps display weights to raw cone excitations."""
    return sensor_matrix @ display_matrix


def compute_cone_distance(r1: np.ndarray, r2: np.ndarray,
                          sigma_noise: np.ndarray,
                          target_cone: int) -> tuple:
    """Noise-weighted Mahalanobis distance between two cone response vectors.

    Follows Observer.cone_distance:
        cone_contrast[i] = (r1[i] - r2[i]) / avg[i]
        d_target = |cone_contrast[target]| / sigma[target]
        d_rest   = sqrt( sum( (cone_contrast[j] / sigma[j])^2  for j != target ) )

    Returns (d_target, d_rest, cone_contrast).
    """
    avg = (r1 + r2) / 2.0
    avg = np.maximum(avg, 1e-30)
    cone_contrast = (r1 - r2) / avg

    d_target = abs(cone_contrast[target_cone]) / sigma_noise[target_cone]

    rest_idx = [j for j in range(len(r1)) if j != target_cone]
    d_rest = float(np.sqrt(np.sum((cone_contrast[rest_idx] / sigma_noise[rest_idx]) ** 2)))

    return d_target, d_rest, cone_contrast


# ============================================================
# Approach 1: Analytical exact isolation (fast, for LED-position optimizer)
# ============================================================
def build_isolation_tests_analytical(
    C: np.ndarray,
    sigma_noise: np.ndarray,
    w_max: float = 1.0,
) -> Optional[List[ConeIsolationTest]]:
    """Construct stimulus pairs via C^{-1} with optimal (centered) w_base.

    w_base = w_max/2 * ones maximizes the symmetric perturbation range.
    """
    n = C.shape[0]
    cond = np.linalg.cond(C)
    if cond > 1e14:
        return None

    C_inv = np.linalg.inv(C)
    w_base = np.full(n, w_max / 2.0)
    r_base = C @ w_base
    if np.any(r_base <= 0):
        return None

    tests = []
    for k in range(n):
        iso_dir = C_inv[:, k]
        abs_dir = np.abs(iso_dir)
        ratios = np.where(abs_dir > 1e-15, w_base / abs_dir, np.inf)
        alpha_max = ratios.min()

        w1 = w_base + alpha_max * iso_dir
        w2 = w_base - alpha_max * iso_dir

        r1 = C @ w1
        r2 = C @ w2
        d_target, d_rest, cone_contrast = compute_cone_distance(r1, r2, sigma_noise, k)

        tests.append(ConeIsolationTest(
            target_cone=k, target_peak_nm=HYPEROBSERVER_PEAKS[k],
            w1=w1, w2=w2, r1=r1, r2=r2,
            d_target=d_target, d_rest=d_rest, cone_contrast=cone_contrast,
        ))

    return tests


def evaluate_isolation_fast(
    C: np.ndarray, sigma_noise: np.ndarray, w_max: float = 1.0,
) -> tuple:
    """Returns (d_targets, cond) using the fast analytical approach."""
    n = C.shape[0]
    cond = np.linalg.cond(C)
    tests = build_isolation_tests_analytical(C, sigma_noise, w_max)
    if tests is None:
        return np.zeros(n), cond
    return np.array([t.d_target for t in tests]), cond


# ============================================================
# Approach 2: Direct optimization of (w1, w2) per cone
# ============================================================
def optimize_stimulus_pair(
    C: np.ndarray,
    target_cone: int,
    sigma_noise: np.ndarray,
    d_rest_threshold: float = 0.5,
    w_max: float = 100.0,
    popsize: int = 30,
    maxiter: int = 1000,
    seed: int = 42,
    analytical_warm_start: Optional[ConeIsolationTest] = None,
) -> ConeIsolationTest:
    """Directly optimize w1, w2 in [0, w_max]^n to maximize d_target for cone k.

    Uses differential_evolution (global optimizer) with L-BFGS-B polish.
    Allows cross-talk up to d_rest_threshold in the other cones.
    """
    n = C.shape[0]
    penalty_weight = 1e4
    rest_idx = np.array([j for j in range(n) if j != target_cone])

    def objective(x):
        w1 = x[:n]
        w2 = x[n:]
        r1 = C @ w1
        r2 = C @ w2
        avg = (r1 + r2) / 2.0
        if np.any(avg < 1e-10):
            return 1e6
        cc = (r1 - r2) / avg
        d_tgt = abs(cc[target_cone]) / sigma_noise[target_cone]
        d_rst = np.sqrt(np.sum((cc[rest_idx] / sigma_noise[rest_idx]) ** 2))
        penalty = penalty_weight * max(0, d_rst - d_rest_threshold) ** 2
        return -d_tgt + penalty

    bounds = [(0, w_max)] * (2 * n)

    init_strategy = 'latinhypercube'
    if analytical_warm_start is not None:
        ws = analytical_warm_start
        warm_vec = np.clip(np.concatenate([ws.w1, ws.w2]), 0, w_max)
        rng = np.random.default_rng(seed)
        total_pop = popsize * (2 * n)
        pop = rng.uniform(0, w_max, (total_pop, 2 * n))
        pop[0] = warm_vec
        init_strategy = pop

    result = differential_evolution(
        objective, bounds,
        seed=seed, maxiter=maxiter, tol=1e-12,
        popsize=popsize,
        init=init_strategy,
        polish=True,
        mutation=(0.5, 1.5), recombination=0.9,
    )

    best_x = result.x
    w1 = best_x[:n]
    w2 = best_x[n:]
    r1 = C @ w1
    r2 = C @ w2
    d_target, d_rest, cone_contrast = compute_cone_distance(r1, r2, sigma_noise, target_cone)

    return ConeIsolationTest(
        target_cone=target_cone, target_peak_nm=HYPEROBSERVER_PEAKS[target_cone],
        w1=w1, w2=w2, r1=r1, r2=r2,
        d_target=d_target, d_rest=d_rest, cone_contrast=cone_contrast,
    )


def optimize_all_stimulus_pairs(
    C: np.ndarray,
    sigma_noise: np.ndarray,
    d_rest_threshold: float = 0.5,
    w_max: float = 100.0,
    popsize: int = 30,
    maxiter: int = 1000,
    seed: int = 42,
    analytical_tests: Optional[List[ConeIsolationTest]] = None,
) -> List[ConeIsolationTest]:
    """Optimize stimulus pairs for all 12 cones."""
    n = C.shape[0]
    tests = []
    for k in range(n):
        warm = analytical_tests[k] if analytical_tests is not None else None
        test = optimize_stimulus_pair(
            C, k, sigma_noise,
            d_rest_threshold=d_rest_threshold, w_max=w_max,
            popsize=popsize, maxiter=maxiter,
            seed=seed + k,
            analytical_warm_start=warm,
        )
        tests.append(test)
    return tests


# ============================================================
# LED Position Optimization (outer loop, uses fast analytical eval)
# ============================================================
def compute_min_d_target(peaks: np.ndarray, sensor_matrix: np.ndarray,
                         wavelengths: np.ndarray, fwhm: float,
                         sigma_noise: np.ndarray) -> float:
    """Objective for LED-position optimizer: min d_target across all 12 cones."""
    D = build_display_matrix(wavelengths, peaks, fwhm)
    C = build_transfer_matrix(sensor_matrix, D)
    d_targets, cond = evaluate_isolation_fast(C, sigma_noise)
    if cond > 1e14:
        return 0.0
    return float(d_targets.min())


def optimize_primary_positions(
    sensor_matrix: np.ndarray,
    wavelengths: np.ndarray,
    fwhm: float,
    sigma_noise: np.ndarray,
    n_starts: int = 20,
    seed: int = 42,
) -> tuple:
    """Optimize 12 LED peak positions to maximize min(d_target).

    Returns (best_peaks, best_d_targets, best_cond).
    """
    n = sensor_matrix.shape[0]
    bounds = [(405, 695)] * n

    def objective(x):
        return -compute_min_d_target(np.array(x), sensor_matrix, wavelengths,
                                     fwhm, sigma_noise)

    result = differential_evolution(
        objective, bounds,
        seed=seed, maxiter=300, tol=1e-6,
        popsize=max(15, n_starts),
        init='latinhypercube', polish=True,
    )

    best_peaks = np.sort(result.x)
    D = build_display_matrix(wavelengths, best_peaks, fwhm)
    C = build_transfer_matrix(sensor_matrix, D)
    d_targets, cond = evaluate_isolation_fast(C, sigma_noise)

    return best_peaks, d_targets, cond


# ============================================================
# Visualization
# ============================================================
def make_cone_colors(n_cones: int) -> np.ndarray:
    cmap = plt.cm.turbo
    return cmap(np.linspace(0.05, 0.95, n_cones))


def plot_condition_number(fwhm_values, cond_fixed, cond_opt, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    if cond_fixed is not None:
        ax.semilogy(fwhm_values, cond_fixed, 'o--', color='gray', label='Fixed peaks', alpha=0.7)
    if cond_opt is not None:
        ax.semilogy(fwhm_values, cond_opt, 's-', color='steelblue', label='Optimized peaks')
    ax.set_xlabel('FWHM [nm]')
    ax.set_ylabel('Condition number of C')
    ax.set_title('Transfer Matrix Condition Number vs LED Bandwidth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'condition_number_vs_fwhm.png', dpi=150)
    plt.close(fig)


def plot_d_target_vs_fwhm(fwhm_values, d_target_matrix, title_suffix, fname, output_dir):
    colors = make_cone_colors(N_CONES)
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(N_CONES):
        ax.plot(fwhm_values, d_target_matrix[:, k], '-o', color=colors[k],
                label=f'{PEAK_LABELS[k]} nm', markersize=3)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1 JND')
    ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='3 JND (clearly visible)')
    ax.set_xlabel('FWHM [nm]')
    ax.set_ylabel('Max achievable d_target (JND units)')
    ax.set_title(f'Cone Isolation Capability vs LED Bandwidth ({title_suffix})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)


def plot_optimized_positions(fwhm_values, peaks_matrix, output_dir):
    colors = make_cone_colors(N_CONES)
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(N_CONES):
        ax.plot(fwhm_values, peaks_matrix[:, k], '-o', color=colors[k],
                label=f'LED {k+1} (near {PEAK_LABELS[k]} nm)', markersize=3)
        ax.axhline(HYPEROBSERVER_PEAKS[k], color=colors[k], linestyle=':', alpha=0.3)
    ax.set_xlabel('FWHM [nm]')
    ax.set_ylabel('Optimized LED Peak [nm]')
    ax.set_title('Optimal Primary Positions vs LED Bandwidth\n(dotted lines = hyperobserver cone peaks)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'optimized_peaks_vs_fwhm.png', dpi=150)
    plt.close(fig)


def plot_heatmap(fwhm_values, d_target_matrix, title_suffix, fname, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    data = d_target_matrix.T
    vmin = max(data[data > 0].min(), 0.01) if np.any(data > 0) else 0.01
    im = ax.imshow(data, aspect='auto', origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=data.max()),
                   cmap='viridis',
                   extent=[fwhm_values[0], fwhm_values[-1], -0.5, N_CONES - 0.5])
    ax.set_yticks(range(N_CONES))
    ax.set_yticklabels([f'{PEAK_LABELS[k]} nm' for k in range(N_CONES)])
    ax.set_xlabel('FWHM [nm]')
    ax.set_ylabel('Cone')
    ax.set_title(f'd_target Heatmap ({title_suffix})')
    plt.colorbar(im, ax=ax, label='d_target (JND)')
    fig.tight_layout()
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)


def plot_stimulus_pairs(tests: List[ConeIsolationTest], fwhm: float,
                        led_peaks: np.ndarray, wavelengths: np.ndarray,
                        label: str, output_dir: Path):
    """Plot the actual two-stimulus spectra for each cone isolation test."""
    n = len(tests)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
    axes = axes.flatten()

    D = build_display_matrix(wavelengths, led_peaks, fwhm)
    for k, test in enumerate(tests):
        ax = axes[k]
        spd1 = D @ test.w1
        spd2 = D @ test.w2

        ax.plot(wavelengths, spd1, 'b-', linewidth=1.2, label='Stim 1')
        ax.plot(wavelengths, spd2, 'r-', linewidth=1.2, label='Stim 2')
        ax.fill_between(wavelengths, spd1, spd2, alpha=0.15, color='purple')
        ax.set_title(f'Cone {PEAK_LABELS[k]} nm\n'
                     f'd_tgt={test.d_target:.2e}  d_rst={test.d_rest:.2e}',
                     fontsize=8)
        ax.set_ylim(bottom=0)
        if k == 0:
            ax.legend(fontsize=7)

    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    fig.supxlabel('Wavelength [nm]')
    fig.supylabel('Spectral Power')
    fig.suptitle(f'Stimulus Pairs — {label}  (FWHM = {fwhm} nm)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    safe_label = label.lower().replace(' ', '_').replace('=', '')
    fig.savefig(output_dir / f'stimulus_pairs_{safe_label}_fwhm{fwhm:.0f}.png', dpi=150)
    plt.close(fig)


def plot_test_diagnostics(tests: List[ConeIsolationTest], fwhm: float,
                          led_peaks: np.ndarray, wavelengths: np.ndarray,
                          sensor_matrix: np.ndarray,
                          sigma_noise: np.ndarray,
                          label: str, output_dir: Path):
    """For each cone isolation test, plot three panels:
      1. Spectra: the two stimulus SPDs (D @ w1, D @ w2)
      2. Cone responses: r1 and r2 bar chart (projection onto hyperobserver basis)
      3. Mahalanobis distance: per-cone |cone_contrast| / sigma (JND contribution)
    """
    n = len(tests)
    D = build_display_matrix(wavelengths, led_peaks, fwhm)
    colors = make_cone_colors(n)
    x = np.arange(n)

    fig, axes = plt.subplots(n, 3, figsize=(18, 3.0 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, test in enumerate(tests):
        k = test.target_cone
        spd1 = D @ test.w1
        spd2 = D @ test.w2

        r1 = test.r1
        r2 = test.r2
        raw_diff = np.abs(r1 - r2)
        avg = np.maximum((r1 + r2) / 2.0, 1e-30)
        mahal_per_cone = np.abs((r1 - r2) / avg) / sigma_noise

        # ---- Panel 1: Spectra ----
        ax_sp = axes[row, 0]
        ax_sp.plot(wavelengths, spd1, 'b-', linewidth=1.2, label='Stim 1')
        ax_sp.plot(wavelengths, spd2, 'r-', linewidth=1.2, label='Stim 2')
        ax_sp.fill_between(wavelengths, spd1, spd2, alpha=0.12, color='purple')
        ax_sp.set_ylim(bottom=0)
        ax_sp.set_ylabel(f'Cone {PEAK_LABELS[k]} nm', fontsize=9, fontweight='bold')
        if row == 0:
            ax_sp.set_title('Stimulus Spectra', fontsize=10, fontweight='bold')
            ax_sp.legend(fontsize=7)
        if row == n - 1:
            ax_sp.set_xlabel('Wavelength [nm]')

        # ---- Panel 2: Raw cone distance |r1 - r2| ----
        ax_cr = axes[row, 1]
        bar_cols_cd = ['red' if i == k else colors[i] for i in range(n)]
        ax_cr.bar(x, raw_diff, color=bar_cols_cd, edgecolor='black', linewidth=0.3)
        ax_cr.axvline(k, color='red', linewidth=1.5, alpha=0.4, linestyle='--')
        ax_cr.set_xticks(x)
        ax_cr.set_xticklabels(['' for _ in range(n)])
        if row == 0:
            ax_cr.set_title('Raw Cone Distance |r1 − r2|', fontsize=10, fontweight='bold')
        if row == n - 1:
            ax_cr.set_xticklabels(PEAK_LABELS, rotation=45, ha='right', fontsize=7)
            ax_cr.set_xlabel('Cone')

        # ---- Panel 3: Mahalanobis distance per cone ----
        ax_mh = axes[row, 2]
        bar_cols = ['red' if i == k else colors[i] for i in range(n)]
        ax_mh.bar(x, mahal_per_cone, color=bar_cols, edgecolor='black', linewidth=0.3)
        ax_mh.axhline(1.0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        ax_mh.set_xticks(x)
        ax_mh.set_xticklabels(['' for _ in range(n)])
        d_tgt_str = f'{test.d_target:.2e}' if test.d_target < 1 else f'{test.d_target:.1f}'
        ax_mh.text(0.98, 0.95, f'd_tgt={d_tgt_str}\nd_rst={test.d_rest:.2e}',
                   transform=ax_mh.transAxes, fontsize=7, va='top', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if row == 0:
            ax_mh.set_title('Mahalanobis Distance per Cone', fontsize=10, fontweight='bold')
        if row == n - 1:
            ax_mh.set_xticklabels(PEAK_LABELS, rotation=45, ha='right', fontsize=7)
            ax_mh.set_xlabel('Cone')

    fig.suptitle(f'Cone Isolation Diagnostics — {label}  (FWHM = {fwhm} nm)',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout()
    safe_label = label.lower().replace(' ', '_').replace('=', '')
    fig.savefig(output_dir / f'diagnostics_{safe_label}_fwhm{fwhm:.0f}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(fwhm_values, d_analytical, d_direct, output_dir):
    """Side-by-side comparison of analytical vs direct-optimized d_target."""
    colors = make_cone_colors(N_CONES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for k in range(N_CONES):
        ax1.plot(fwhm_values, d_analytical[:, k], '-o', color=colors[k],
                 label=f'{PEAK_LABELS[k]} nm', markersize=3)
        ax2.plot(fwhm_values, d_direct[:, k], '-o', color=colors[k],
                 label=f'{PEAK_LABELS[k]} nm', markersize=3)

    for ax in (ax1, ax2):
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1 JND')
        ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='3 JND')
        ax.set_xlabel('FWHM [nm]')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('d_target (JND)')
    ax1.set_title('Analytical (exact isolation, d_rest = 0)')
    ax2.set_title('Direct Optimization (relaxed d_rest)')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.suptitle('Analytical vs Direct-Optimized Stimulus Pairs', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'analytical_vs_direct.png', dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='12-primary display cone isolation analysis')
    parser.add_argument('--fwhm-min', type=float, default=1.0)
    parser.add_argument('--fwhm-max', type=float, default=30.0)
    parser.add_argument('--fwhm-step', type=float, default=1.0)
    parser.add_argument('--n-starts', type=int, default=20,
                        help='Population size for LED-position optimizer (default: 20)')
    parser.add_argument('--stim-popsize', type=int, default=30,
                        help='Population size for DE stimulus optimizer (default: 30)')
    parser.add_argument('--stim-maxiter', type=int, default=1000,
                        help='Max iterations for DE stimulus optimizer (default: 1000)')
    parser.add_argument('--d-rest-threshold', type=float, default=0.5,
                        help='Max allowed d_rest for direct optimization (default: 0.5 JND)')
    parser.add_argument('--w-max', type=float, default=100.0,
                        help='Upper bound on LED weights (default: 100.0)')
    parser.add_argument('--output-dir', type=str, default='output/hyperobserver_isolation')
    parser.add_argument('--fixed-peaks', action='store_true',
                        help='Also evaluate with fixed hyperobserver peaks')
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Only run fixed-peaks analysis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fwhm_values = np.arange(args.fwhm_min, args.fwhm_max + args.fwhm_step / 2, args.fwhm_step)
    n_fwhm = len(fwhm_values)

    # Build the 12D hyperobserver
    print("Building 12D hyperobserver...")
    hyperobs = Observer.hyperobserver(wavelengths=WAVELENGTHS)
    sensor_matrix = hyperobs.sensor_matrix
    print(f"  Sensor matrix shape: {sensor_matrix.shape}")
    print(f"  Cone peaks (after filtering): {[f'{s.peak}' for s in hyperobs.sensors]}")
    print(f"  d_rest threshold: {args.d_rest_threshold} JND")
    print(f"  w_max (LED bound): {args.w_max}")
    print()

    # ---- Fixed-peaks evaluation ----
    d_ana_fixed = None
    d_dir_fixed = None
    cond_fixed = None

    if args.fixed_peaks or args.skip_optimization:
        print(f"=== Fixed hyperobserver peaks across {n_fwhm} FWHM values ===")
        d_ana_fixed = np.zeros((n_fwhm, N_CONES))
        d_dir_fixed = np.zeros((n_fwhm, N_CONES))
        cond_fixed = np.zeros(n_fwhm)

        for fi, fwhm in enumerate(fwhm_values):
            D = build_display_matrix(WAVELENGTHS, HYPEROBSERVER_PEAKS, fwhm)
            C = build_transfer_matrix(sensor_matrix, D)
            cond = np.linalg.cond(C)
            cond_fixed[fi] = cond

            # --- Analytical ---
            ana_tests = build_isolation_tests_analytical(C, SIGMA_NOISE)
            if ana_tests is not None:
                d_ana_fixed[fi] = np.array([t.d_target for t in ana_tests])
            else:
                ana_tests = None

            # --- Direct optimization ---
            print(f"  FWHM={fwhm:5.1f} nm  cond={cond:.2e}  ", end="", flush=True)
            dir_tests = optimize_all_stimulus_pairs(
                C, SIGMA_NOISE,
                d_rest_threshold=args.d_rest_threshold,
                w_max=args.w_max,
                popsize=args.stim_popsize, maxiter=args.stim_maxiter,
                seed=args.seed + fi * 100,
                analytical_tests=ana_tests,
            )
            d_dir_fixed[fi] = np.array([t.d_target for t in dir_tests])
            ana_min = d_ana_fixed[fi].min() if ana_tests else 0
            dir_min = d_dir_fixed[fi].min()
            print(f"analytical min={ana_min:.2e}  direct min={dir_min:.2e}")

            if fi == 0:
                _print_test_details(ana_tests, "Analytical (exact)", fwhm)
                _print_test_details(dir_tests, f"Direct (d_rest<={args.d_rest_threshold})", fwhm)
                plot_stimulus_pairs(ana_tests, fwhm, HYPEROBSERVER_PEAKS, WAVELENGTHS,
                                    "analytical_fixed", output_dir)
                plot_stimulus_pairs(dir_tests, fwhm, HYPEROBSERVER_PEAKS, WAVELENGTHS,
                                    f"direct_drest{args.d_rest_threshold}_fixed", output_dir)
                plot_test_diagnostics(dir_tests, fwhm, HYPEROBSERVER_PEAKS, WAVELENGTHS,
                                      sensor_matrix, SIGMA_NOISE,
                                      f"direct_drest{args.d_rest_threshold}_fixed", output_dir)
        print()

    # ---- Optimized-peaks evaluation ----
    d_ana_opt = None
    d_dir_opt = None
    cond_opt = None
    peaks_opt = None

    if not args.skip_optimization:
        print(f"=== Optimizing primary positions across {n_fwhm} FWHM values ===")
        d_ana_opt = np.zeros((n_fwhm, N_CONES))
        d_dir_opt = np.zeros((n_fwhm, N_CONES))
        cond_opt = np.zeros(n_fwhm)
        peaks_opt = np.zeros((n_fwhm, N_CONES))

        for fi, fwhm in enumerate(fwhm_values):
            print(f"  FWHM={fwhm:5.1f} nm  ", end="", flush=True)

            # Step 1: optimize LED positions (uses fast analytical eval)
            best_peaks, ana_d, cond = optimize_primary_positions(
                sensor_matrix, WAVELENGTHS, fwhm, SIGMA_NOISE,
                n_starts=args.n_starts, seed=args.seed + fi)
            d_ana_opt[fi] = ana_d
            cond_opt[fi] = cond
            peaks_opt[fi] = best_peaks

            # Step 2: direct-optimize stimulus pairs at these LED positions
            D = build_display_matrix(WAVELENGTHS, best_peaks, fwhm)
            C = build_transfer_matrix(sensor_matrix, D)
            ana_tests = build_isolation_tests_analytical(C, SIGMA_NOISE)
            dir_tests = optimize_all_stimulus_pairs(
                C, SIGMA_NOISE,
                d_rest_threshold=args.d_rest_threshold,
                w_max=args.w_max,
                popsize=args.stim_popsize, maxiter=args.stim_maxiter,
                seed=args.seed + fi * 100,
                analytical_tests=ana_tests,
            )
            d_dir_opt[fi] = np.array([t.d_target for t in dir_tests])

            print(f"cond={cond:.2e}  ana_min={ana_d.min():.2e}  "
                  f"dir_min={d_dir_opt[fi].min():.2e}")
            print(f"           peaks=[{', '.join(f'{p:.1f}' for p in best_peaks)}]")

            if fi == 0:
                _print_test_details(ana_tests, "Analytical (exact)", fwhm)
                _print_test_details(dir_tests, f"Direct (d_rest<={args.d_rest_threshold})", fwhm)
                plot_stimulus_pairs(dir_tests, fwhm, best_peaks, WAVELENGTHS,
                                    f"direct_drest{args.d_rest_threshold}_opt", output_dir)
                plot_test_diagnostics(dir_tests, fwhm, best_peaks, WAVELENGTHS,
                                      sensor_matrix, SIGMA_NOISE,
                                      f"direct_drest{args.d_rest_threshold}_opt", output_dir)
        print()

    # ---- Save results ----
    save_dict = dict(
        fwhm_values=fwhm_values,
        hyperobserver_peaks=HYPEROBSERVER_PEAKS,
        sigma_noise=SIGMA_NOISE,
        d_rest_threshold=args.d_rest_threshold,
    )
    if d_ana_fixed is not None:
        save_dict.update(d_ana_fixed=d_ana_fixed, d_dir_fixed=d_dir_fixed, cond_fixed=cond_fixed)
    if d_ana_opt is not None:
        save_dict.update(d_ana_opt=d_ana_opt, d_dir_opt=d_dir_opt,
                         cond_opt=cond_opt, peaks_opt=peaks_opt)

    np.savez(output_dir / 'results.npz', **save_dict)
    print(f"Saved results to {output_dir / 'results.npz'}")

    # ---- Visualization ----
    print("Generating plots...")

    if d_ana_fixed is not None:
        plot_d_target_vs_fwhm(fwhm_values, d_ana_fixed,
                              'Fixed Peaks, Analytical', 'd_target_ana_fixed.png', output_dir)
        plot_d_target_vs_fwhm(fwhm_values, d_dir_fixed,
                              f'Fixed Peaks, Direct (d_rest<={args.d_rest_threshold})',
                              'd_target_dir_fixed.png', output_dir)
        plot_comparison(fwhm_values, d_ana_fixed, d_dir_fixed, output_dir)

    if d_ana_opt is not None:
        plot_d_target_vs_fwhm(fwhm_values, d_ana_opt,
                              'Optimized Peaks, Analytical', 'd_target_ana_opt.png', output_dir)
        plot_d_target_vs_fwhm(fwhm_values, d_dir_opt,
                              f'Optimized Peaks, Direct (d_rest<={args.d_rest_threshold})',
                              'd_target_dir_opt.png', output_dir)
        plot_optimized_positions(fwhm_values, peaks_opt, output_dir)
        plot_comparison(fwhm_values, d_ana_opt, d_dir_opt, output_dir)

    cond_f = cond_fixed if d_ana_fixed is not None else None
    cond_o = cond_opt if d_ana_opt is not None else None
    plot_condition_number(fwhm_values, cond_f, cond_o, output_dir)

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    if d_ana_opt is not None:
        _print_summary("Optimized Peaks — Analytical", fwhm_values, d_ana_opt, cond_opt)
        _print_summary(f"Optimized Peaks — Direct (d_rest<={args.d_rest_threshold})",
                       fwhm_values, d_dir_opt, cond_opt)
    if d_ana_fixed is not None:
        _print_summary("Fixed Peaks — Analytical", fwhm_values, d_ana_fixed, cond_fixed)
        _print_summary(f"Fixed Peaks — Direct (d_rest<={args.d_rest_threshold})",
                       fwhm_values, d_dir_fixed, cond_fixed)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done.")


def _print_test_details(tests: Optional[List[ConeIsolationTest]], label: str, fwhm: float):
    if tests is None:
        print(f"\n  {label} (FWHM={fwhm} nm): C too ill-conditioned\n")
        return
    print(f"\n  {label} (FWHM={fwhm} nm):")
    print(f"  {'Cone':>8}  {'d_target':>10}  {'d_rest':>10}  "
          f"{'w1 range':>14}  {'w2 range':>14}  "
          f"{'r1[k]':>8}  {'r2[k]':>8}  {'contrast':>10}")
    for t in tests:
        k = t.target_cone
        print(f"  {PEAK_LABELS[k]+' nm':>8}  {t.d_target:10.2e}  {t.d_rest:10.2e}  "
              f"  [{t.w1.min():.3f},{t.w1.max():.3f}]"
              f"  [{t.w2.min():.3f},{t.w2.max():.3f}]"
              f"  {t.r1[k]:8.4f}  {t.r2[k]:8.4f}"
              f"  {t.cone_contrast[k]:10.2e}")
    print()


def _print_summary(label, fwhm_values, d_target_matrix, cond_values):
    print(f"\n--- {label} ---")
    header = f"{'FWHM':>6}  {'cond(C)':>10}  {'min_d':>10}  {'median_d':>10}  {'max_d':>10}"
    print(header)
    for fi, fwhm in enumerate(fwhm_values):
        d = d_target_matrix[fi]
        print(f"{fwhm:6.1f}  {cond_values[fi]:10.2e}  {d.min():10.2e}  "
              f"{np.median(d):10.2e}  {d.max():10.2e}")

    key_indices = [0, len(fwhm_values) // 4, len(fwhm_values) // 2, -1]
    key_indices = sorted(set(max(0, min(i, len(fwhm_values)-1)) for i in key_indices))
    print(f"\n  Per-cone d_target (JND) at selected FWHM values:")
    hdr = f"  {'Cone':>8}"
    for fi in key_indices:
        hdr += f"  {'FWHM='+str(fwhm_values[fi]):>12}"
    print(hdr)
    for k in range(N_CONES):
        row = f"  {PEAK_LABELS[k]+' nm':>8}"
        for fi in key_indices:
            val = d_target_matrix[fi, k]
            if val >= 1.0:
                row += f"  {val:12.1f}"
            else:
                row += f"  {val:12.2e}"
        isolable = "YES" if d_target_matrix[0, k] >= 1.0 else "no"
        row += f"   [{isolable}]"
        print(row)


if __name__ == "__main__":
    main()
