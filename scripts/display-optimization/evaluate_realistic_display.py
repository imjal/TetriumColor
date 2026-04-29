#!/usr/bin/env python3
"""
Optimize a realistic 4-primary LED display to maximally discriminate
between trichromat observers covering 99% of the population.

For each 3-cone observer on a 4-primary display, there is a 1-D null
space of stimuli that appear identical (metamers).  We construct a
metamer pair along this null direction for each observer, then measure
how every other observer perceives the difference (Mahalanobis distance).

The display (4 LED peaks + 4 FWHMs) is optimized to maximize the
worst-case pairwise discrimination across all observer pairs.

Outputs per observer:
  - Metamer pair spectra (identical for this observer, maximal diff for others)
  - Cone response differences across all observers
  - Mahalanobis distances per observer pair

Additional outputs:
  - Discrimination heatmap (observer x observer)
  - Hyperobserver (12-cone) response to each metamer pair
  - Summary JSON

Usage:
    python evaluate_realistic_display.py
    python evaluate_realistic_display.py --fwhm-min 15 --fwhm-max 30 --n-seeds 5
    python evaluate_realistic_display.py --coverage 0.95
"""

from optimize_hyperobserver_display import (
    WAVELENGTHS, FWHM_TO_SIGMA,
    make_sigma_noise, gaussian_led, build_display_matrix,
    build_normalized_transfer as build_normalized_transfer_square,
    compute_isolation, score_design, IsolationResult,
    make_cone_colors,
)
from TetriumColor.Measurement.TetriumMeasurementRoutines import load_primaries_from_csv
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.Observer import Observer
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from colour import MSDS_CMFS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# Photopic luminous efficiency V(lambda)
# ============================================================
def _get_photopic_vlambda(wavelengths: np.ndarray) -> np.ndarray:
    """CIE 1931 2-degree Y-bar (photopic luminous efficiency), interpolated to wavelengths."""
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    y_bar = cmfs.values[:, 1]  # Y column
    cmf_wl = cmfs.wavelengths
    return np.interp(wavelengths, cmf_wl, y_bar, left=0.0, right=0.0)


# Precompute for WAVELENGTHS (lazy, filled on first use)
_V_LAMBDA_CACHE: Optional[np.ndarray] = None


def _get_vlambda() -> np.ndarray:
    global _V_LAMBDA_CACHE
    if _V_LAMBDA_CACHE is None:
        _V_LAMBDA_CACHE = _get_photopic_vlambda(WAVELENGTHS)
    return _V_LAMBDA_CACHE


# Minimum peak separation (nm) — prevents degenerate solutions
MIN_PEAK_SEPARATION = 30.0


# ============================================================
# Data structures
# ============================================================
@dataclass
class PopulationDisplay:
    peaks: np.ndarray           # (4,) LED peak wavelengths
    fwhms: np.ndarray           # (4,) LED FWHMs
    amplitudes: np.ndarray      # (4,) scaling from reference observer
    D_scaled: np.ndarray        # (n_wl, 4) scaled display matrix
    D_raw: np.ndarray = None    # (n_wl, 4) unscaled display matrix


@dataclass
class MetamerPair:
    observer_idx: int
    genotype: tuple
    w1: np.ndarray              # (4,) primary weights stimulus 1
    w2: np.ndarray              # (4,) primary weights stimulus 2
    null_dir: np.ndarray        # (4,) null space direction


# ============================================================
# Null-space metamer computation
# ============================================================
def compute_normalized_transfer(sensor_matrix: np.ndarray,
                                D_scaled: np.ndarray) -> Optional[np.ndarray]:
    """Normalized transfer matrix: C such that C @ ones ≈ ones."""
    C_raw = sensor_matrix @ D_scaled
    white = C_raw @ np.ones(D_scaled.shape[1])
    if np.any(white <= 0):
        return None
    return C_raw / white[:, np.newaxis]


def compute_metamer_pair(C_norm: np.ndarray, n_primaries: int = 4
                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Metamer pair from null space of C_norm.

    Returns (w1, w2, null_dir) with w1, w2 in [0,1] and C_norm @ w1 ≈ C_norm @ w2.
    """
    ns = null_space(C_norm)
    if ns.shape[1] == 0:
        return None, None, None

    null_dir = ns[:, 0]
    w_base = np.full(n_primaries, 0.5)
    abs_dir = np.abs(null_dir)
    ratios = np.where(abs_dir > 1e-15, 0.5 / abs_dir, np.inf)
    t = ratios.min()

    w1 = np.clip(w_base + t * null_dir, 0, 1)
    w2 = np.clip(w_base - t * null_dir, 0, 1)
    return w1, w2, null_dir


def estimate_fwhm(spectra) -> float:
    """Estimate FWHM of a spectral peak."""
    peak_idx = np.argmax(spectra.data)
    peak_val = spectra.data[peak_idx]
    half_max = peak_val / 2
    above = spectra.data >= half_max
    wl_above = spectra.wavelengths[above]
    if len(wl_above) > 1:
        return float(wl_above[-1] - wl_above[0])
    return 0.0


def build_population_display_from_primaries(primaries, ref_sensor_matrix, wavelengths):
    """Build a PopulationDisplay from real measured LED Spectra.

    Returns PopulationDisplay or None if amplitude scaling fails.
    """
    n_p = len(primaries)
    D = np.zeros((len(wavelengths), n_p))
    for i, p in enumerate(primaries):
        interp = p.interpolate_values(wavelengths)
        D[:, i] = interp.data

    a = compute_amplitude_scaling(ref_sensor_matrix, D)
    if a is None:
        # Fallback: row-normalize
        C_raw = ref_sensor_matrix @ D
        row_sums = C_raw.sum(axis=1)
        a = np.ones(n_p)
        if np.all(row_sums > 0):
            # Scale so white ≈ 1
            a = 1.0 / row_sums.mean() * np.ones(n_p)

    D_scaled = D @ np.diag(a)
    peaks = np.array([p.wavelengths[np.argmax(p.data)] for p in primaries])
    fwhms = np.array([estimate_fwhm(p) for p in primaries])

    return PopulationDisplay(peaks=peaks, fwhms=fwhms, amplitudes=a, D_scaled=D_scaled, D_raw=D)


def compute_amplitude_scaling(ref_sensor_matrix: np.ndarray,
                              D: np.ndarray) -> Optional[np.ndarray]:
    """Amplitude scaling so reference observer's white = ones."""
    C_ref = ref_sensor_matrix @ D
    n_cones, n_p = C_ref.shape
    ones = np.ones(n_cones)
    try:
        if n_p == n_cones:
            a = np.linalg.solve(C_ref, ones)
        else:
            # C_ref is (n_cones, n_p), solve C_ref @ a = ones for a in R^n_p
            a = np.linalg.lstsq(C_ref, ones, rcond=None)[0]
        if np.all(a > 0):
            return a
    except np.linalg.LinAlgError:
        pass
    return None


# ============================================================
# Full discrimination evaluation
# ============================================================
def compute_observer_response(sensor_matrix: np.ndarray,
                              D_scaled: np.ndarray,
                              w1: np.ndarray, w2: np.ndarray) -> dict:
    """How an observer perceives a stimulus pair."""
    spd1 = D_scaled @ w1
    spd2 = D_scaled @ w2
    r1 = sensor_matrix @ spd1
    r2 = sensor_matrix @ spd2
    white_r = sensor_matrix @ (D_scaled @ np.ones(D_scaled.shape[1]))
    r1_n = r1 / np.maximum(white_r, 1e-30)
    r2_n = r2 / np.maximum(white_r, 1e-30)

    raw_diff = np.abs(r1_n - r2_n)
    avg = np.maximum((r1_n + r2_n) / 2.0, 1e-30)
    cone_contrast = (r1_n - r2_n) / avg
    sigma = make_sigma_noise(sensor_matrix.shape[0])
    mahal_per_cone = np.abs(cone_contrast) / sigma
    mahal_total = float(np.sqrt(np.sum((cone_contrast / sigma) ** 2)))

    return {
        'r1': r1_n, 'r2': r2_n,
        'raw_diff': raw_diff,
        'cone_contrast': cone_contrast,
        'mahal_per_cone': mahal_per_cone,
        'mahal_total': mahal_total,
    }


def evaluate_display_on_population(display: PopulationDisplay,
                                   observers: List[Tuple],
                                   hyperobs: Observer) -> dict:
    """Full evaluation of a display on all observers.

    Returns dict with discrimination_matrix, metamer_pairs, responses, hyper_responses.
    """
    n_obs = len(observers)
    D_scaled = display.D_scaled
    n_p = D_scaled.shape[1]

    # Compute metamer pairs and cross-observer responses
    discrim = np.zeros((n_obs, n_obs))
    metamer_pairs = []
    responses = {}  # (i, j) -> response dict

    for i, (gt_i, obs_i, _) in enumerate(observers):
        C_i = compute_normalized_transfer(obs_i.sensor_matrix, D_scaled)
        if C_i is None:
            metamer_pairs.append(None)
            continue

        w1, w2, null_dir = compute_metamer_pair(C_i, n_p)
        if w1 is None:
            metamer_pairs.append(None)
            continue

        metamer_pairs.append(MetamerPair(i, gt_i, w1, w2, null_dir))

        for j, (gt_j, obs_j, _) in enumerate(observers):
            resp = compute_observer_response(obs_j.sensor_matrix, D_scaled, w1, w2)
            discrim[i, j] = resp['mahal_total']
            responses[(i, j)] = resp

    # Hyperobserver responses to each metamer pair
    hyper_responses = {}
    for i, mp in enumerate(metamer_pairs):
        if mp is None:
            continue
        resp = compute_observer_response(hyperobs.sensor_matrix, D_scaled, mp.w1, mp.w2)
        hyper_responses[i] = resp

    return {
        'discrimination_matrix': discrim,
        'metamer_pairs': metamer_pairs,
        'responses': responses,
        'hyper_responses': hyper_responses,
    }


# ============================================================
# Optimization objective
# ============================================================
def _eval_objective(x: np.ndarray,
                    sensor_matrices: List[np.ndarray],
                    ref_sensor_matrix: np.ndarray,
                    wavelengths: np.ndarray,
                    n_primaries: int) -> float:
    """Minimize negative of worst-case pairwise discrimination."""
    peaks = x[:n_primaries]
    fwhms = x[n_primaries:]

    # Penalize primaries that are too close together
    sep_penalty = _peak_separation_penalty(peaks)
    if sep_penalty > 0:
        return 1e6 + sep_penalty

    D = build_display_matrix(wavelengths, peaks, fwhms)
    a = compute_amplitude_scaling(ref_sensor_matrix, D)
    if a is None:
        return 1e6

    D_scaled = D @ np.diag(a)
    n_obs = len(sensor_matrices)
    min_off_diag = np.inf

    for i in range(n_obs):
        C_i = compute_normalized_transfer(sensor_matrices[i], D_scaled)
        if C_i is None:
            return 1e6

        w1, w2, _ = compute_metamer_pair(C_i, n_primaries)
        if w1 is None:
            return 1e6

        spd1 = D_scaled @ w1
        spd2 = D_scaled @ w2

        for j in range(n_obs):
            if i == j:
                continue
            r1 = sensor_matrices[j] @ spd1
            r2 = sensor_matrices[j] @ spd2
            white_r = sensor_matrices[j] @ (D_scaled @ np.ones(n_primaries))
            r1_n = r1 / np.maximum(white_r, 1e-30)
            r2_n = r2 / np.maximum(white_r, 1e-30)
            avg = np.maximum((r1_n + r2_n) / 2.0, 1e-30)
            cc = (r1_n - r2_n) / avg
            sigma = make_sigma_noise(sensor_matrices[j].shape[0])
            mahal = np.sqrt(np.sum((cc / sigma) ** 2))
            min_off_diag = min(min_off_diag, mahal)

    if min_off_diag == np.inf or min_off_diag <= 0:
        return 1e6

    # Soft condition-number penalty on reference transfer matrix
    C_ref = compute_normalized_transfer(ref_sensor_matrix, D_scaled)
    cond_penalty = 0.0
    if C_ref is not None:
        sv = np.linalg.svd(C_ref, compute_uv=False)
        cond = sv[0] / max(sv[-1], 1e-30)
        cond_penalty = max(0, np.log10(max(cond, 1)) - 6) * 0.05

    return -(min_off_diag - cond_penalty)


def optimize_population_display(observers: List[Tuple],
                                wavelengths: np.ndarray,
                                n_primaries: int = 4,
                                fwhm_min: float = 15.0,
                                fwhm_max: float = 30.0,
                                n_seeds: int = 3,
                                maxiter: int = 300,
                                popsize: int = 20) -> Tuple[PopulationDisplay, List[float]]:
    """Optimize LED peaks + FWHMs for population discrimination."""
    sensor_matrices = [obs.sensor_matrix for _, obs, _ in observers]
    ref_sensor_matrix = sensor_matrices[0]  # most common genotype

    bounds = [(405, 695)] * n_primaries + [(fwhm_min, fwhm_max)] * n_primaries

    best_score = -np.inf
    best_result = None
    all_scores = []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds} ... ", end="", flush=True)
        result = differential_evolution(
            _eval_objective, bounds,
            args=(sensor_matrices, ref_sensor_matrix, wavelengths, n_primaries),
            seed=seed * 17 + 42,
            maxiter=maxiter, popsize=popsize, tol=1e-8,
            init='latinhypercube', polish=True,
            mutation=(0.5, 1.5), recombination=0.9,
        )
        score = -result.fun
        all_scores.append(score)
        print(f"score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None or best_score <= -1e5:
        raise RuntimeError(f"No feasible display found after {n_seeds} seeds.")

    x = best_result.x
    sort_idx = np.argsort(x[:n_primaries])
    peaks = x[:n_primaries][sort_idx]
    fwhms = x[n_primaries:][sort_idx]

    D = build_display_matrix(wavelengths, peaks, fwhms)
    a = compute_amplitude_scaling(ref_sensor_matrix, D)
    D_scaled = D @ np.diag(a)

    display = PopulationDisplay(peaks=peaks, fwhms=fwhms,
                                amplitudes=a, D_scaled=D_scaled)
    return display, all_scores


# ============================================================
# Plotting
# ============================================================
def plot_discrimination_heatmap(discrim: np.ndarray,
                                genotype_labels: List[str],
                                output_dir: Path, title: str):
    """Heatmap of pairwise Mahalanobis discrimination."""
    n = discrim.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))

    # Mask diagonal for better color range
    masked = discrim.copy()
    np.fill_diagonal(masked, np.nan)

    im = ax.imshow(masked, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=max(np.nanmax(masked), 3))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(genotype_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(genotype_labels, fontsize=7)
    ax.set_xlabel("Observer j (perceiver)")
    ax.set_ylabel("Observer i (metamer source)")

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, 'self', ha='center', va='center', fontsize=6, color='gray')
            else:
                val = discrim[i, j]
                color = 'white' if val < 1.0 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, label='Mahalanobis distance (JND)')
    ax.axhline(y=-0.5, color='red', linewidth=0.5)  # reference lines
    for threshold in [1.0, 3.0]:
        cbar.ax.axhline(y=threshold, color='gray', linewidth=0.8, linestyle='--')

    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'discrimination_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'discrimination_heatmap.png'}")


def plot_metamer_spectra_and_responses(eval_results: dict,
                                       observers: List[Tuple],
                                       display: PopulationDisplay,
                                       wavelengths: np.ndarray,
                                       output_dir: Path):
    """For each observer's metamer pair: spectra + cross-observer cone diffs + Mahalanobis."""
    metamer_pairs = eval_results['metamer_pairs']
    responses = eval_results['responses']
    D_scaled = display.D_scaled
    n_obs = len(observers)

    for i, mp in enumerate(metamer_pairs):
        if mp is None:
            continue

        gt_i = observers[i][0]
        gt_label = ','.join(str(int(p)) if p == int(p) else str(p) for p in gt_i)
        prob_i = observers[i][2]

        spd1 = D_scaled @ mp.w1
        spd2 = D_scaled @ mp.w2

        fig, axes = plt.subplots(1, 3, figsize=(22, 5))

        # Panel 1: Stimulus spectra
        ax = axes[0]
        ax.plot(wavelengths, spd1, 'b-', linewidth=1.5, label='Stimulus 1')
        ax.plot(wavelengths, spd2, 'r-', linewidth=1.5, label='Stimulus 2')
        ax.fill_between(wavelengths, spd1, spd2, alpha=0.12, color='purple')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Scaled Power')
        ax.set_title('Metamer Pair Spectra', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Raw cone difference per observer (grouped bars)
        ax = axes[1]
        obs_colors = plt.cm.tab20(np.linspace(0, 1, n_obs))
        all_n_cones = [observers[j][1].dimension for j in range(n_obs)]
        max_cones = max(all_n_cones)
        bar_w = 0.8 / n_obs

        for j in range(n_obs):
            resp = responses[(i, j)]
            n_c = len(resp['raw_diff'])
            x_c = np.arange(n_c)
            offset = (j - n_obs / 2 + 0.5) * bar_w
            gt_j = observers[j][0]
            lbl = ','.join(str(int(p)) if p == int(p) else str(p) for p in gt_j)
            marker = ' *' if i == j else ''
            ax.bar(x_c + offset, resp['raw_diff'], bar_w,
                   label=f'{lbl}{marker}', color=obs_colors[j],
                   edgecolor='black', linewidth=0.2)

        ax.set_xlabel('Cone index')
        ax.set_ylabel('|r1 - r2|')
        ax.set_title('Raw Cone Diff per Observer', fontsize=10, fontweight='bold')
        ax.legend(fontsize=5, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3: Mahalanobis total per observer (single bar each)
        ax = axes[2]
        mahals = [responses[(i, j)]['mahal_total'] for j in range(n_obs)]
        bar_colors = ['gray' if j == i else obs_colors[j] for j in range(n_obs)]
        x_obs = np.arange(n_obs)
        ax.bar(x_obs, mahals, color=bar_colors, edgecolor='black', linewidth=0.3)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='1 JND')
        ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, linewidth=0.8, label='3 JND')
        obs_labels = []
        for j in range(n_obs):
            gt_j = observers[j][0]
            lbl = ','.join(str(int(p)) if p == int(p) else str(p) for p in gt_j)
            obs_labels.append(lbl)
        ax.set_xticks(x_obs)
        ax.set_xticklabels(obs_labels, rotation=45, ha='right', fontsize=6)
        ax.set_xlabel('Observer genotype')
        ax.set_ylabel('Mahalanobis distance')
        ax.set_title('Total Mahalanobis per Observer', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            f'Metamer Pair for Observer ({gt_label}) — prob={prob_i:.4f}\n'
            f'Identical for this observer, detected by others',
            fontsize=11, fontweight='bold', y=1.02)
        fig.tight_layout()
        fname = f'metamer_obs{i}_{gt_label.replace(",", "_")}.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_dir / fname}")


def plot_hyperobserver_responses(eval_results: dict,
                                 observers: List[Tuple],
                                 hyperobs: Observer,
                                 display: PopulationDisplay,
                                 wavelengths: np.ndarray,
                                 output_dir: Path):
    """For each metamer pair, show how the 12-cone hyperobserver perceives it."""
    metamer_pairs = eval_results['metamer_pairs']
    hyper_responses = eval_results['hyper_responses']
    D_scaled = display.D_scaled
    n_hyper = hyperobs.dimension
    hyper_peaks = np.array([s.peak for s in hyperobs.sensors], dtype=float)
    hyper_peak_labels = [f"{p:.0f}" if p == int(p) else f"{p}" for p in hyper_peaks]
    hyper_colors = make_cone_colors(n_hyper)
    x_hyper = np.arange(n_hyper)

    valid_pairs = [(i, mp) for i, mp in enumerate(metamer_pairs) if mp is not None]
    n_stim = len(valid_pairs)
    if n_stim == 0:
        return

    fig, axes = plt.subplots(n_stim, 3, figsize=(22, 3.5 * n_stim))
    if n_stim == 1:
        axes = axes[np.newaxis, :]

    for row, (i, mp) in enumerate(valid_pairs):
        gt_i = observers[i][0]
        gt_label = ','.join(str(int(p)) if p == int(p) else str(p) for p in gt_i)
        resp = hyper_responses[i]

        spd1 = D_scaled @ mp.w1
        spd2 = D_scaled @ mp.w2

        # Panel 1: Spectra
        ax = axes[row, 0]
        ax.plot(wavelengths, spd1, 'b-', linewidth=1.2, label='Stimulus 1')
        ax.plot(wavelengths, spd2, 'r-', linewidth=1.2, label='Stimulus 2')
        ax.fill_between(wavelengths, spd1, spd2, alpha=0.12, color='purple')
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f'Obs ({gt_label})', fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title('Stimulus Spectra', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
        if row == n_stim - 1:
            ax.set_xlabel('Wavelength [nm]')

        # Panel 2: Raw cone diff on hyperobserver
        ax = axes[row, 1]
        ax.bar(x_hyper, resp['raw_diff'], color=hyper_colors,
               edgecolor='black', linewidth=0.3)
        ax.set_xticks(x_hyper)
        if row == 0:
            ax.set_title('Hyperobserver Raw |r1-r2|', fontsize=10, fontweight='bold')
        if row == n_stim - 1:
            ax.set_xticklabels(hyper_peak_labels, rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Cone peak [nm]')
        else:
            ax.set_xticklabels([])

        # Panel 3: Mahalanobis on hyperobserver
        ax = axes[row, 2]
        ax.bar(x_hyper, resp['mahal_per_cone'], color=hyper_colors,
               edgecolor='black', linewidth=0.3)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8, label='1 JND')
        ax.set_xticks(x_hyper)
        mahal_total = resp['mahal_total']
        ax.text(0.98, 0.95, f'd_total={mahal_total:.2f}',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if row == 0:
            ax.set_title('Hyperobserver Mahalanobis per Cone', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
        if row == n_stim - 1:
            ax.set_xticklabels(hyper_peak_labels, rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Cone peak [nm]')
        else:
            ax.set_xticklabels([])

    peaks_str = ', '.join(f'{p:.0f}' for p in display.peaks)
    fwhms_str = ', '.join(f'{f:.1f}' for f in display.fwhms)
    fig.suptitle(
        f'Hyperobserver ({n_hyper} cones) Response to Population Metamer Pairs\n'
        f'LED peaks: [{peaks_str}]  FWHMs: [{fwhms_str}]',
        fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / 'hyperobserver_responses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'hyperobserver_responses.png'}")


def plot_display_spectra(display: PopulationDisplay,
                         observers: List[Tuple],
                         wavelengths: np.ndarray,
                         output_dir: Path):
    """Plot LED spectra and all observer sensitivities."""
    n_p = len(display.peaks)
    D = build_display_matrix(wavelengths, display.peaks, display.fwhms)
    D_scaled = D @ np.diag(display.amplitudes)
    led_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_p))

    # Collect all unique cone peaks across observers
    all_sensor_data = []
    for gt, obs, prob in observers:
        for s in obs.sensors:
            all_sensor_data.append((s.peak, s))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for i in range(n_p):
        ax1.plot(wavelengths, D_scaled[:, i], color=led_colors[i], linewidth=1.2,
                 label=f'{display.peaks[i]:.0f}nm (FWHM={display.fwhms[i]:.1f})')
    ax1.set_title('Optimized Display Primaries')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('Scaled LED Power')
    ax1.grid(True, alpha=0.3)

    # Plot unique sensitivities
    seen_peaks = set()
    obs_colors = make_cone_colors(12)
    ci = 0
    for gt, obs, prob in observers:
        for s in obs.sensors:
            if s.peak not in seen_peaks:
                seen_peaks.add(s.peak)
                lbl = f'{s.peak:.0f}nm' if s.peak == int(s.peak) else f'{s.peak}nm'
                ax2.plot(wavelengths, obs.sensor_matrix[list(obs.sensors).index(s)],
                         color=obs_colors[ci % len(obs_colors)], linewidth=0.8, label=lbl)
                ci += 1
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Sensitivity')
    ax2.set_title('Cone Sensitivities (all unique peaks)')
    ax2.legend(fontsize=6, ncol=4, loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'display_and_cones.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'display_and_cones.png'}")


def plot_seed_convergence(all_scores: List[float], output_dir: Path, title: str):
    """Bar chart of optimizer scores across seeds."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(all_scores))
    ax.bar(x, all_scores, color='steelblue', edgecolor='black')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Score (min pairwise Mahalanobis)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / 'seed_convergence.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'seed_convergence.png'}")


# ============================================================
# Pareto: tetrachromat isolation + population discrimination
# ============================================================
def _compute_f_tetra(D: np.ndarray, tetra_sensor_matrix: np.ndarray,
                     tetra_sigma: np.ndarray, tetra_peaks: np.ndarray,
                     n_primaries: int) -> float:
    """Tetrachromat cone-isolation score: min d_target across all cones.

    Normalizes D directly for the tetrachromat (not a trichromat reference).
    """
    C_norm, amps = build_normalized_transfer_square(tetra_sensor_matrix, D)
    if C_norm is None:
        return 0.0
    results = compute_isolation(C_norm, tetra_sigma, tetra_peaks)
    if results is None:
        return 0.0
    return float(min(r.d_target for r in results))


def _compute_f_pop(D_scaled: np.ndarray,
                   sensor_matrices: List[np.ndarray],
                   n_primaries: int) -> float:
    """Population discrimination score: min off-diagonal Mahalanobis."""
    n_obs = len(sensor_matrices)
    min_off_diag = np.inf

    for i in range(n_obs):
        C_i = compute_normalized_transfer(sensor_matrices[i], D_scaled)
        if C_i is None:
            return 0.0
        w1, w2, _ = compute_metamer_pair(C_i, n_primaries)
        if w1 is None:
            return 0.0
        spd1 = D_scaled @ w1
        spd2 = D_scaled @ w2
        for j in range(n_obs):
            if i == j:
                continue
            r1 = sensor_matrices[j] @ spd1
            r2 = sensor_matrices[j] @ spd2
            white_r = sensor_matrices[j] @ (D_scaled @ np.ones(n_primaries))
            r1_n = r1 / np.maximum(white_r, 1e-30)
            r2_n = r2 / np.maximum(white_r, 1e-30)
            avg = np.maximum((r1_n + r2_n) / 2.0, 1e-30)
            cc = (r1_n - r2_n) / avg
            sigma = make_sigma_noise(sensor_matrices[j].shape[0])
            mahal = np.sqrt(np.sum((cc / sigma) ** 2))
            min_off_diag = min(min_off_diag, mahal)

    return float(min_off_diag) if min_off_diag != np.inf else 0.0


def _compute_f_lum(D_scaled: np.ndarray) -> float:
    """Relative luminance of the display white point.

    Returns Y = V(lambda)^T @ (D_scaled @ ones), normalized so that
    a flat-spectrum equal-energy illuminant over [400,700] has Y = 1.
    Higher values mean a brighter display.
    """
    V = _get_vlambda()
    white_spd = D_scaled @ np.ones(D_scaled.shape[1])
    Y = float(V @ white_spd)
    # Normalize: equal-energy white has Y = sum(V)
    Y_ref = float(V.sum())
    return Y / max(Y_ref, 1e-30)


def _peak_separation_penalty(peaks: np.ndarray) -> float:
    """Penalty for primaries that are too close together.

    Returns 0 if all pairs are >= MIN_PEAK_SEPARATION apart,
    otherwise a large positive value proportional to the violation.
    """
    n = len(peaks)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(peaks[i] - peaks[j])
            if gap < MIN_PEAK_SEPARATION:
                penalty += (MIN_PEAK_SEPARATION - gap) / MIN_PEAK_SEPARATION
    return penalty


def _eval_pareto_objective(x: np.ndarray,
                           sensor_matrices: List[np.ndarray],
                           ref_sensor_matrix: np.ndarray,
                           tetra_sensor_matrix: np.ndarray,
                           tetra_sigma: np.ndarray,
                           tetra_peaks: np.ndarray,
                           wavelengths: np.ndarray,
                           n_primaries: int,
                           beta: float,
                           gamma: float,
                           f_tetra_ref: float,
                           f_pop_ref: float,
                           f_lum_ref: float) -> float:
    """Pareto objective blending three normalized sub-objectives.

    score = beta * f_tetra/ref + (1-beta-gamma) * f_pop/ref + gamma * f_lum/ref

    beta controls tetra isolation, gamma controls luminance, and the
    remainder goes to population discrimination.  All three are normalized
    by reference values from single-objective optima.
    """
    peaks = x[:n_primaries]
    fwhms = x[n_primaries:]

    # Penalize primaries that are too close together
    sep_penalty = _peak_separation_penalty(peaks)
    if sep_penalty > 0:
        return 1e6 + sep_penalty

    D = build_display_matrix(wavelengths, peaks, fwhms)
    a = compute_amplitude_scaling(ref_sensor_matrix, D)
    if a is None:
        return 1e6

    D_scaled = D @ np.diag(a)

    f_tetra = _compute_f_tetra(D, tetra_sensor_matrix,
                               tetra_sigma, tetra_peaks, n_primaries)
    f_pop = _compute_f_pop(D_scaled, sensor_matrices, n_primaries)
    f_lum = _compute_f_lum(D_scaled)

    if f_tetra <= 0 and f_pop <= 0:
        return 1e6

    # Normalize by reference values (from single-objective optima)
    f_tetra_norm = f_tetra / max(f_tetra_ref, 1e-30)
    f_pop_norm = f_pop / max(f_pop_ref, 1e-30)
    f_lum_norm = f_lum / max(f_lum_ref, 1e-30)

    pop_weight = max(1.0 - beta - gamma, 0.0)
    score = beta * f_tetra_norm + pop_weight * f_pop_norm + gamma * f_lum_norm
    return -score


def optimize_pareto_point(observers: List[Tuple],
                          tetrachromat: Observer,
                          wavelengths: np.ndarray,
                          beta: float,
                          gamma: float,
                          f_tetra_ref: float,
                          f_pop_ref: float,
                          f_lum_ref: float,
                          n_primaries: int = 4,
                          fwhm_min: float = 15.0,
                          fwhm_max: float = 30.0,
                          n_seeds: int = 3,
                          maxiter: int = 300,
                          popsize: int = 20) -> Tuple[PopulationDisplay, float, float, float]:
    """Optimize one point on the pareto front for given beta and gamma."""
    sensor_matrices = [obs.sensor_matrix for _, obs, _ in observers]
    ref_sensor_matrix = sensor_matrices[0]
    tetra_sensor_matrix = tetrachromat.sensor_matrix
    tetra_peaks = np.array([s.peak for s in tetrachromat.sensors], dtype=float)
    tetra_sigma = make_sigma_noise(tetrachromat.dimension)

    bounds = [(405, 695)] * n_primaries + [(fwhm_min, fwhm_max)] * n_primaries

    best_score = -np.inf
    best_result = None

    for seed in range(n_seeds):
        print(f"    Seed {seed + 1}/{n_seeds} ... ", end="", flush=True)
        result = differential_evolution(
            _eval_pareto_objective, bounds,
            args=(sensor_matrices, ref_sensor_matrix,
                  tetra_sensor_matrix, tetra_sigma, tetra_peaks,
                  wavelengths, n_primaries, beta, gamma,
                  f_tetra_ref, f_pop_ref, f_lum_ref),
            seed=seed * 17 + 42,
            maxiter=maxiter, popsize=popsize, tol=1e-8,
            init='latinhypercube', polish=True,
            mutation=(0.5, 1.5), recombination=0.9,
        )
        score = -result.fun
        print(f"score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_result = result

    x = best_result.x
    sort_idx = np.argsort(x[:n_primaries])
    peaks = x[:n_primaries][sort_idx]
    fwhms = x[n_primaries:][sort_idx]

    D = build_display_matrix(wavelengths, peaks, fwhms)
    a = compute_amplitude_scaling(ref_sensor_matrix, D)
    D_scaled = D @ np.diag(a)

    # Compute actual (un-normalized) scores
    f_tetra = _compute_f_tetra(D, tetra_sensor_matrix,
                               tetra_sigma, tetra_peaks, n_primaries)
    f_pop = _compute_f_pop(D_scaled, sensor_matrices, n_primaries)
    f_lum = _compute_f_lum(D_scaled)

    display = PopulationDisplay(peaks=peaks, fwhms=fwhms,
                                amplitudes=a, D_scaled=D_scaled)
    return display, f_tetra, f_pop, f_lum


def sweep_pareto_front(observers: List[Tuple],
                       tetrachromat: Observer,
                       wavelengths: np.ndarray,
                       betas: List[float],
                       gamma: float = 0.0,
                       n_primaries: int = 4,
                       fwhm_min: float = 15.0,
                       fwhm_max: float = 30.0,
                       n_seeds: int = 3,
                       maxiter: int = 300,
                       popsize: int = 20) -> List[dict]:
    """Sweep beta values to trace the pareto front.

    First runs pure single-objective optima for reference normalization,
    then runs each beta with gamma held fixed.
    """
    sensor_matrices = [obs.sensor_matrix for _, obs, _ in observers]
    ref_sensor_matrix = sensor_matrices[0]
    tetra_sensor_matrix = tetrachromat.sensor_matrix
    tetra_peaks = np.array([s.peak for s in tetrachromat.sensors], dtype=float)
    tetra_sigma = make_sigma_noise(tetrachromat.dimension)

    # Step 1: Get reference values from single-objective optima
    # For reference runs: beta=1,gamma=0 (pure tetra), beta=0,gamma=0 (pure pop),
    # and beta=0,gamma=1 (pure lum) if gamma > 0
    print("\n  Computing reference values...")

    print("  Reference: pure tetrachromat isolation (beta=1, gamma=0):")
    disp_tetra, f_tetra_at_1, f_pop_at_1, f_lum_at_1 = optimize_pareto_point(
        observers, tetrachromat, wavelengths, beta=1.0, gamma=0.0,
        f_tetra_ref=1.0, f_pop_ref=1.0, f_lum_ref=1.0,
        n_primaries=n_primaries, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
        n_seeds=n_seeds, maxiter=maxiter, popsize=popsize)
    f_tetra_ref = f_tetra_at_1
    print(f"    f_tetra_ref = {f_tetra_ref:.4f}, f_pop = {f_pop_at_1:.4f}, f_lum = {f_lum_at_1:.4f}")

    print("  Reference: pure population discrimination (beta=0, gamma=0):")
    disp_pop, f_tetra_at_0, f_pop_at_0, f_lum_at_0 = optimize_pareto_point(
        observers, tetrachromat, wavelengths, beta=0.0, gamma=0.0,
        f_tetra_ref=1.0, f_pop_ref=1.0, f_lum_ref=1.0,
        n_primaries=n_primaries, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
        n_seeds=n_seeds, maxiter=maxiter, popsize=popsize)
    f_pop_ref = f_pop_at_0
    print(f"    f_tetra = {f_tetra_at_0:.4f}, f_pop_ref = {f_pop_ref:.4f}, f_lum = {f_lum_at_0:.4f}")

    # For luminance reference, use the best luminance seen so far (or run pure lum if gamma > 0)
    f_lum_ref = max(f_lum_at_1, f_lum_at_0)
    if gamma > 0:
        print("  Reference: pure luminance (beta=0, gamma=1):")
        disp_lum, _, _, f_lum_pure = optimize_pareto_point(
            observers, tetrachromat, wavelengths, beta=0.0, gamma=1.0,
            f_tetra_ref=1.0, f_pop_ref=1.0, f_lum_ref=1.0,
            n_primaries=n_primaries, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
            n_seeds=n_seeds, maxiter=maxiter, popsize=popsize)
        f_lum_ref = f_lum_pure
        print(f"    f_lum_ref = {f_lum_ref:.4f}")

    results = [
        {'beta': 1.0, 'display': disp_tetra,
         'f_tetra': f_tetra_at_1, 'f_pop': f_pop_at_1, 'f_lum': f_lum_at_1},
        {'beta': 0.0, 'display': disp_pop,
         'f_tetra': f_tetra_at_0, 'f_pop': f_pop_at_0, 'f_lum': f_lum_at_0},
    ]

    # Step 2: Sweep intermediate betas with normalized objectives
    intermediate_betas = [b for b in betas if b not in (0.0, 1.0)]
    for beta in sorted(intermediate_betas, reverse=True):
        print(f"\n  beta={beta:.2f}, gamma={gamma:.2f}:")
        disp, f_t, f_p, f_l = optimize_pareto_point(
            observers, tetrachromat, wavelengths, beta=beta, gamma=gamma,
            f_tetra_ref=f_tetra_ref, f_pop_ref=f_pop_ref, f_lum_ref=f_lum_ref,
            n_primaries=n_primaries, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
            n_seeds=n_seeds, maxiter=maxiter, popsize=popsize)
        print(f"    f_tetra = {f_t:.4f}, f_pop = {f_p:.4f}, f_lum = {f_l:.4f}")
        print(f"    Peaks: [{', '.join(f'{p:.0f}' for p in disp.peaks)}]  "
              f"FWHMs: [{', '.join(f'{f:.1f}' for f in disp.fwhms)}]")
        results.append({'beta': beta, 'display': disp,
                        'f_tetra': f_t, 'f_pop': f_p, 'f_lum': f_l})

    # Sort by beta descending
    results.sort(key=lambda r: r['beta'], reverse=True)
    return results


def plot_pareto_front(pareto_results: List[dict], output_dir: Path, gamma: float = 0.0,
                      real_display_points: Optional[List[dict]] = None):
    """Plot the pareto front: f_tetra vs f_pop, with point size proportional to f_lum.

    real_display_points: list of dicts with keys 'label', 'f_tetra', 'f_pop', 'led_peaks'.
    """
    from TetriumColor.Plotting.PlotStyle import apply_style, SINGLE_COL, DOUBLE_COL
    apply_style()

    betas = [r['beta'] for r in pareto_results]
    f_tetras = [r['f_tetra'] for r in pareto_results]
    f_pops = [r['f_pop'] for r in pareto_results]
    f_lums = [r.get('f_lum', 0.0) for r in pareto_results]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85))

    # Point size scales with relative luminance (min 20, max 80)
    if max(f_lums) > 0:
        lum_arr = np.array(f_lums)
        sizes = 20 + 60 * (lum_arr / max(lum_arr.max(), 1e-30))
    else:
        sizes = np.full(len(f_lums), 35)

    # Scatter with color = beta, size = luminance
    sc = ax.scatter(f_pops, f_tetras, c=betas, cmap='coolwarm', s=sizes,
                    edgecolors='black', linewidths=0.4, zorder=3)
    # Connect with line
    sorted_idx = np.argsort(f_pops)
    ax.plot(np.array(f_pops)[sorted_idx], np.array(f_tetras)[sorted_idx],
            'k--', alpha=0.4, linewidth=0.8, zorder=2)

    # Callout for beta=0.5 showing LED peaks
    for r in pareto_results:
        plot_beta = 0.5
        if abs(r['beta'] - plot_beta) < 1e-6:
            peaks = r.get('led_peaks', getattr(r.get('display', None), 'peaks', []))
            peaks_str = ', '.join(f'{p:.0f}' for p in sorted(peaks))
            ax.annotate(
                rf"$\beta={plot_beta}$" + "\n" + rf"[{peaks_str}]\,nm",
                (r['f_pop'], r['f_tetra']),
                textcoords='offset points', xytext=(10, -30), fontsize=5,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=0.5))
            break

    # Plot real display points
    markers = ['*', 'D', 's', 'P', 'X', 'v']
    if real_display_points:
        for i, rdp in enumerate(real_display_points):
            marker = markers[i % len(markers)]
            ax.scatter(rdp['f_pop'], rdp['f_tetra'], marker=marker, s=60,
                       c='gold', edgecolors='black', linewidths=0.5, zorder=5)
            peaks = rdp.get('led_peaks', [])
            peaks_str = ', '.join(f'{p:.0f}' for p in sorted(peaks))
            ax.annotate(
                rdp.get('label', 'Real') + f"\n[{peaks_str}]\\,nm",
                (rdp['f_pop'], rdp['f_tetra']),
                textcoords='offset points', xytext=(10, -10), fontsize=5,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='black', linewidth=0.5))

    # Auto-scale axes to data with a small margin
    all_x = np.array(f_pops + [rdp['f_pop'] for rdp in (real_display_points or [])])
    all_y = np.array(f_tetras + [rdp['f_tetra'] for rdp in (real_display_points or [])])
    x_margin = max((all_x.max() - all_x.min()) * 0.15, all_x.max() * 0.1)
    y_margin = max((all_y.max() - all_y.min()) * 0.15, all_y.max() * 0.1)
    ax.set_xlim(max(0, all_x.min() - x_margin), all_x.max() + x_margin)
    ax.set_ylim(max(0, all_y.min() - y_margin), all_y.max() + y_margin)

    ax.set_xlabel(r'$f_{\mathrm{pop}}$')
    ax.set_ylabel(r'$f_{\mathrm{tetra}}$')
    cbar = fig.colorbar(sc, ax=ax, label=r'$\beta$')

    fig.tight_layout()
    fig.savefig(output_dir / 'pareto_front.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pareto_front.pdf'}")
    print(f"  Saved: {output_dir / 'pareto_front.png'}")


def plot_combined_summary(ideal_display: PopulationDisplay,
                          real_display: PopulationDisplay,
                          pareto_results: List[dict],
                          wavelengths: np.ndarray,
                          output_dir: Path,
                          gamma: float = 0.0,
                          real_display_points: Optional[List[dict]] = None):
    """Combined figure: ideal display spectra (top-left), real display spectra (bottom-left),
    pareto front (right column spanning both rows)."""
    from TetriumColor.Plotting.PlotStyle import apply_style, DOUBLE_COL
    apply_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.55, wspace=0.55)

    ax_pareto = fig.add_subplot(gs[:, 0])
    ax_ideal = fig.add_subplot(gs[0, 1])
    ax_real = fig.add_subplot(gs[1, 1], sharex=ax_ideal, sharey=ax_ideal)

    # --- Left: Pareto front ---
    _plot_pareto_on_ax(ax_pareto, pareto_results, gamma, real_display_points)

    # --- Top-right: Ideal (optimized) display spectra ---
    _plot_display_spectra_on_ax(ax_ideal, ideal_display, wavelengths,
                                title='Ideal Display', show_xlabel=False,
                                show_ylabel=True)

    # --- Bottom-right: Real (measured) display spectra ---
    _plot_display_spectra_on_ax(ax_real, real_display, wavelengths,
                                title='Our Display', show_ylabel=True)

    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(output_dir / f'combined_summary.{suffix}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'combined_summary.pdf'}")
    print(f"  Saved: {output_dir / 'combined_summary.png'}")


def _wavelength_to_color(peak_nm: float) -> str:
    """Map a LED peak wavelength to an intuitive color."""
    if peak_nm < 470:
        return '#2166ac'   # blue
    elif peak_nm < 510:
        return '#1b7837'   # green-ish
    elif peak_nm < 545:
        return '#4d9221'   # green
    elif peak_nm < 575:
        return '#d4b800'   # yellow
    elif peak_nm < 610:
        return '#e85a00'   # orange
    else:
        return '#b2182b'   # red


def _plot_display_spectra_on_ax(ax, display: PopulationDisplay,
                                wavelengths: np.ndarray, title: str,
                                show_xlabel: bool = True,
                                show_ylabel: bool = True):
    """Plot LED spectra on a given axes."""
    n_p = len(display.peaks)
    # Use raw spectra if available, else reconstruct from peaks/fwhms
    if display.D_raw is not None:
        D = display.D_raw
    else:
        D = build_display_matrix(wavelengths, display.peaks, display.fwhms)

    # Sort by peak wavelength for consistent legend order
    sort_idx = np.argsort(display.peaks)

    for i in sort_idx:
        col = D[:, i]
        peak_val = col.max()
        if peak_val > 0:
            col = col / peak_val
        c = _wavelength_to_color(display.peaks[i])
        ax.plot(wavelengths, col, color=c, linewidth=1.0,
                label=f'{display.peaks[i]:.0f}\\,nm')
        ax.fill_between(wavelengths, col, alpha=0.15, color=c)

    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(bottom=0)
    if show_xlabel:
        ax.set_xlabel('Wavelength (nm)')
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel('Normalized Power')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')


def _plot_pareto_on_ax(ax, pareto_results: List[dict], gamma: float = 0.0,
                       real_display_points: Optional[List[dict]] = None):
    """Plot pareto front on a given axes."""
    betas = [r['beta'] for r in pareto_results]
    f_tetras = [r['f_tetra'] for r in pareto_results]
    f_pops = [r['f_pop'] for r in pareto_results]
    f_lums = [r.get('f_lum', 0.0) for r in pareto_results]

    if max(f_lums) > 0:
        lum_arr = np.array(f_lums)
        sizes = 20 + 60 * (lum_arr / max(lum_arr.max(), 1e-30))
    else:
        sizes = np.full(len(f_lums), 35)

    sc = ax.scatter(f_pops, f_tetras, c=betas, cmap='coolwarm', s=sizes,
                    edgecolors='black', linewidths=0.4, zorder=3)
    sorted_idx = np.argsort(f_pops)
    ax.plot(np.array(f_pops)[sorted_idx], np.array(f_tetras)[sorted_idx],
            'k--', alpha=0.4, linewidth=0.8, zorder=2)

    # Callout for beta=0.5
    for r in pareto_results:
        if abs(r['beta'] - 0.5) < 1e-6:
            peaks = r.get('led_peaks', getattr(r.get('display', None), 'peaks', []))
            peaks_str = ', '.join(f'{p:.0f}' for p in sorted(peaks))
            ax.annotate(
                rf"$\beta=0.5$" + "\n" + rf"[{peaks_str}]\,nm",
                (r['f_pop'], r['f_tetra']),
                textcoords='offset points', xytext=(10, -30), fontsize=5,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=0.5))
            break

    markers = ['*', 'D', 's', 'P', 'X', 'v']
    if real_display_points:
        for i, rdp in enumerate(real_display_points):
            marker = markers[i % len(markers)]
            ax.scatter(rdp['f_pop'], rdp['f_tetra'], marker=marker, s=60,
                       c='gold', edgecolors='black', linewidths=0.5, zorder=5)
            peaks = rdp.get('led_peaks', [])
            peaks_str = ', '.join(f'{p:.0f}' for p in sorted(peaks))
            ax.annotate(
                rdp.get('label', 'Real') + f"\n[{peaks_str}]\\,nm",
                (rdp['f_pop'], rdp['f_tetra']),
                textcoords='offset points', xytext=(10, -10), fontsize=5,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='black', linewidth=0.5))

    all_x = np.array(f_pops + [rdp['f_pop'] for rdp in (real_display_points or [])])
    all_y = np.array(f_tetras + [rdp['f_tetra'] for rdp in (real_display_points or [])])
    x_margin = max((all_x.max() - all_x.min()) * 0.15, all_x.max() * 0.1)
    y_margin = max((all_y.max() - all_y.min()) * 0.15, all_y.max() * 0.1)
    ax.set_xlim(max(0, all_x.min() - x_margin), all_x.max() + x_margin)
    ax.set_ylim(max(0, all_y.min() - y_margin), all_y.max() + y_margin)

    ax.set_xlabel(r'$f_{\mathrm{pop}}$')
    ax.set_ylabel(r'$f_{\mathrm{tetra}}$')
    plt.colorbar(sc, ax=ax, label=r'$\beta$')


def run_evaluation(display: PopulationDisplay,
                   observers: List[Tuple],
                   hyperobs: Observer,
                   wavelengths: np.ndarray,
                   output_dir: Path,
                   label: str,
                   coverage: float,
                   extra_summary: Optional[dict] = None):
    """Run full evaluation, plots, and summary for a given display."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {label} on all observers...")
    eval_results = evaluate_display_on_population(display, observers, hyperobs)

    discrim = eval_results['discrimination_matrix']
    metamer_pairs = eval_results['metamer_pairs']

    genotype_labels = []
    for gt, _, _ in observers:
        genotype_labels.append(','.join(str(int(p)) if p == int(p) else str(p) for p in gt))

    print(f"\n{'Source obs':<20} {'min off-diag':>14} {'max off-diag':>14} {'diagonal':>10}")
    print("-" * 62)
    for i in range(len(observers)):
        off_diag = [discrim[i, j] for j in range(len(observers)) if j != i]
        if off_diag:
            print(f"  {genotype_labels[i]:<18} {min(off_diag):>14.4f} {max(off_diag):>14.4f} "
                  f"{discrim[i, i]:>10.4f}")

    worst_pair_val = np.inf
    worst_i, worst_j = 0, 0
    for i in range(len(observers)):
        for j in range(len(observers)):
            if i != j and discrim[i, j] < worst_pair_val:
                worst_pair_val = discrim[i, j]
                worst_i, worst_j = i, j
    print(f"\n  Worst-case pair: ({genotype_labels[worst_i]}) -> ({genotype_labels[worst_j]}) "
          f"= {worst_pair_val:.4f} JND")

    # Plots
    print("\nGenerating plots...")

    plot_display_spectra(display, observers, wavelengths, output_dir)

    # plot_discrimination_heatmap(
    #     discrim, genotype_labels, output_dir,
    #     f'Pairwise Mahalanobis Discrimination — {label}\n'
    #     f'LEDs: [{", ".join(f"{p:.0f}" for p in display.peaks)}]  '
    #     f'FWHMs: [{", ".join(f"{f:.1f}" for f in display.fwhms)}]')

    plot_metamer_spectra_and_responses(
        eval_results, observers, display, wavelengths, output_dir)

    plot_hyperobserver_responses(
        eval_results, observers, hyperobs, display, wavelengths, output_dir)

    # Summary
    summary = {
        'label': label,
        'coverage': coverage,
        'n_observers': len(observers),
        'n_primaries': len(display.peaks),
        'led_peaks': [float(p) for p in display.peaks],
        'led_fwhms': [float(f) for f in display.fwhms],
        'observers': [{
            'genotype': list(gt),
            'probability': float(prob),
            'dimension': obs.dimension,
            'cone_peaks': [float(s.peak) for s in obs.sensors],
        } for gt, obs, prob in observers],
        'discrimination_matrix': discrim.tolist(),
        'worst_case': {
            'source': genotype_labels[worst_i],
            'perceiver': genotype_labels[worst_j],
            'mahalanobis': float(worst_pair_val),
        },
        'metamer_pairs': [{
            'observer_idx': mp.observer_idx,
            'genotype': list(mp.genotype),
            'w1': mp.w1.tolist(),
            'w2': mp.w2.tolist(),
        } if mp is not None else None for mp in metamer_pairs],
    }
    if extra_summary:
        summary.update(extra_summary)

    summary_path = output_dir / 'population_discrimination_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")
    print(f"  All results saved to {output_dir}/")

    return eval_results, worst_pair_val


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Optimize 4-primary display for population discrimination')
    parser.add_argument('--coverage', type=float, default=0.99,
                        help='Population coverage fraction (default: 0.99)')
    parser.add_argument('--primaries-dir', type=str,
                        default='measurements/2026-03-04/validation_2026-03-04_13-44-34/primaries',
                        help='Path to measured display primaries (RGBO order)')
    parser.add_argument('--fwhm-min', type=float, default=5.0,
                        help='Minimum LED FWHM in nm (default: 5.0)')
    parser.add_argument('--fwhm-max', type=float, default=40.0,
                        help='Maximum LED FWHM in nm (default: 40.0)')
    parser.add_argument('--n-primaries', type=int, default=4,
                        help='Number of display primaries (default: 4)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Random seeds for optimizer (default: 3)')
    parser.add_argument('--maxiter', type=int, default=300,
                        help='Max iterations per seed (default: 300)')
    parser.add_argument('--popsize', type=int, default=20,
                        help='Population size for DE (default: 20)')
    parser.add_argument('--output-dir', type=str, default='output/display_optimization')
    parser.add_argument('--skip-optimize', action='store_true',
                        help='Skip optimization, only run real-display evaluation')
    parser.add_argument('--beta', type=str, default=None,
                        help='Pareto sweep: comma-separated beta values (e.g. "0,0.2,0.4,0.6,0.8,1.0"). '
                             'beta=1 pure tetrachromat isolation, beta=0 pure population discrimination.')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Luminance weight in pareto objective (default: 0.0). '
                             'Score = beta*f_tetra + (1-beta-gamma)*f_pop + gamma*f_lum. '
                             'Higher gamma favors brighter displays (V(lambda)-weighted white point).')
    parser.add_argument('--replot', type=str, default=None,
                        help='Path to pareto_summary.json — regenerate the pareto front plot '
                             'from existing data without re-running the optimization.')
    args = parser.parse_args()

    # --replot: regenerate plot from saved JSON and exit
    if args.replot is not None:
        summary_path = Path(args.replot)
        with open(summary_path) as f:
            summary = json.load(f)
        pareto_results = summary['points']
        output_dir = summary_path.parent
        real_pts = summary.get('real_display_points', None)
        plot_pareto_front(pareto_results, output_dir, gamma=summary.get('gamma', 0.0),
                          real_display_points=real_pts)

        # Also generate combined summary figure
        # Reconstruct ideal display (beta=1.0) from saved peaks/fwhms
        ideal_pt = next(p for p in pareto_results if abs(p['beta'] - 1.0) < 1e-6)
        ideal_peaks = np.array(ideal_pt['led_peaks'])
        ideal_fwhms = np.array(ideal_pt['led_fwhms'])
        ideal_D = build_display_matrix(WAVELENGTHS, ideal_peaks, ideal_fwhms)
        ideal_display = PopulationDisplay(peaks=ideal_peaks, fwhms=ideal_fwhms,
                                          amplitudes=np.ones(len(ideal_peaks)),
                                          D_scaled=ideal_D, D_raw=ideal_D)

        # Load real measured primaries
        primaries = load_primaries_from_csv(args.primaries_dir, extract_zero=False,
                                            primary_order='RGBO')
        og = ObserverGenotypes(wavelengths=WAVELENGTHS, dimensions=[3], seed=42)
        genotypes = og.get_genotypes_covering_probability(args.coverage, sex='both')
        ref_obs = og.get_observer_for_peaks(genotypes[0])
        real_display = build_population_display_from_primaries(
            primaries, ref_obs.sensor_matrix, WAVELENGTHS)

        plot_combined_summary(ideal_display, real_display, pareto_results,
                              WAVELENGTHS, output_dir, gamma=summary.get('gamma', 0.0),
                              real_display_points=real_pts)
        return

    base_output = Path(args.output_dir)

    # ================================================================
    # Load observers covering target % of population
    # ================================================================
    print(f"\nLoading trichromat observers covering {args.coverage*100:.0f}% of population...")
    og = ObserverGenotypes(wavelengths=WAVELENGTHS, dimensions=[3], seed=42)
    genotypes = og.get_genotypes_covering_probability(args.coverage, sex='both')

    observers = []  # list of (genotype, Observer, probability)
    for gt in genotypes:
        obs = og.get_observer_for_peaks(gt)
        prob = og.get_probability_for_genotype(gt, sex='both')
        observers.append((gt, obs, prob))

    print(f"  {len(observers)} genotypes needed for {args.coverage*100:.0f}% coverage:")
    cumulative = 0.0
    for gt, obs, prob in observers:
        cumulative += prob
        peaks_str = ', '.join(str(int(p)) if p == int(p) else str(p) for p in gt)
        print(f"    ({peaks_str})  dim={obs.dimension}  prob={prob:.4f}  cum={cumulative:.4f}")

    hyperobs = Observer.hyperobserver(wavelengths=WAVELENGTHS)

    # ================================================================
    # ANALYSIS 1: Real measured display
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Real measured 4-primary display")
    print("=" * 70)

    primaries = load_primaries_from_csv(args.primaries_dir, extract_zero=False,
                                        primary_order='RGBO')
    led_names = ['R', 'G', 'B', 'O']
    for name, p in zip(led_names, primaries):
        peak_wl = p.wavelengths[np.argmax(p.data)]
        fwhm = estimate_fwhm(p)
        print(f"  {name}: peak={peak_wl:.0f}nm, FWHM={fwhm:.0f}nm")

    ref_sensor_matrix = observers[0][1].sensor_matrix  # most common genotype
    real_display = build_population_display_from_primaries(
        primaries, ref_sensor_matrix, WAVELENGTHS)
    print(f"  LED peaks: [{', '.join(f'{p:.0f}' for p in real_display.peaks)}]")
    print(f"  LED FWHMs: [{', '.join(f'{f:.0f}' for f in real_display.fwhms)}]")

    real_output = base_output / 'real_display'
    run_evaluation(real_display, observers, hyperobs, WAVELENGTHS,
                   real_output, label='Real Measured Display',
                   coverage=args.coverage,
                   extra_summary={'primaries_dir': args.primaries_dir})

    # ================================================================
    # ANALYSIS 2: Optimized display
    # ================================================================
    if not args.skip_optimize:
        print("\n" + "=" * 70)
        print("  ANALYSIS 2: Optimized 4-primary display")
        print("=" * 70)

        print(f"  FWHM range: [{args.fwhm_min}, {args.fwhm_max}] nm")
        print(f"  Seeds: {args.n_seeds}, maxiter: {args.maxiter}, popsize: {args.popsize}")

        opt_display, all_scores = optimize_population_display(
            observers, WAVELENGTHS,
            n_primaries=args.n_primaries,
            fwhm_min=args.fwhm_min, fwhm_max=args.fwhm_max,
            n_seeds=args.n_seeds, maxiter=args.maxiter, popsize=args.popsize,
        )

        print(f"\n  Best display:")
        print(f"    Peaks: [{', '.join(f'{p:.1f}' for p in opt_display.peaks)}]")
        print(f"    FWHMs: [{', '.join(f'{f:.1f}' for f in opt_display.fwhms)}]")

        opt_output = base_output / 'optimized_display'
        run_evaluation(opt_display, observers, hyperobs, WAVELENGTHS,
                       opt_output, label='Optimized Display',
                       coverage=args.coverage,
                       extra_summary={
                           'fwhm_range': [args.fwhm_min, args.fwhm_max],
                           'all_scores': all_scores,
                       })

        plot_seed_convergence(all_scores, opt_output,
                              'Population Discrimination Optimizer')

    # ================================================================
    # ANALYSIS 3: Pareto front sweep (if --beta provided)
    # ================================================================
    if args.beta is not None:
        betas = [float(b.strip()) for b in args.beta.split(',')]
        gamma = args.gamma
        print("\n" + "=" * 70)
        print(f"  ANALYSIS 3: Pareto front sweep (betas={betas}, gamma={gamma:.2f})")
        print("=" * 70)

        tetrachromat = Observer.tetrachromat(wavelengths=WAVELENGTHS)
        print(f"  Tetrachromat: {tetrachromat.dimension} cones, "
              f"peaks={[s.peak for s in tetrachromat.sensors]}")

        pareto_output = base_output / 'pareto'
        pareto_output.mkdir(parents=True, exist_ok=True)

        pareto_results = sweep_pareto_front(
            observers, tetrachromat, WAVELENGTHS, betas,
            gamma=gamma,
            n_primaries=args.n_primaries,
            fwhm_min=args.fwhm_min, fwhm_max=args.fwhm_max,
            n_seeds=args.n_seeds, maxiter=args.maxiter, popsize=args.popsize)

        # Print summary table
        print(f"\n  {'beta':>5}  {'f_tetra':>10}  {'f_pop':>10}  {'f_lum':>8}  {'Peaks':<40}  {'FWHMs':<40}")
        print("  " + "-" * 120)
        for r in pareto_results:
            peaks_str = ', '.join(f'{p:.0f}' for p in r['display'].peaks)
            fwhms_str = ', '.join(f'{f:.1f}' for f in r['display'].fwhms)
            print(f"  {r['beta']:>5.2f}  {r['f_tetra']:>10.4f}  {r['f_pop']:>10.4f}  "
                  f"{r.get('f_lum', 0):>8.4f}  [{peaks_str}]  [{fwhms_str}]")

        # Compute f_tetra/f_pop for the real measured display
        tetra_sensor_matrix = tetrachromat.sensor_matrix
        tetra_sigma = make_sigma_noise(tetrachromat.dimension)
        tetra_peaks = np.array([s.peak for s in tetrachromat.sensors])
        sensor_matrices = [obs.sensor_matrix for _, obs, _ in observers]
        n_p = args.n_primaries

        real_f_tetra = _compute_f_tetra(real_display.D_raw, tetra_sensor_matrix,
                                        tetra_sigma, tetra_peaks, n_p)
        real_f_pop = _compute_f_pop(real_display.D_scaled, sensor_matrices, n_p)
        real_display_point = {
            'label': 'Measured RGBO',
            'f_tetra': real_f_tetra,
            'f_pop': real_f_pop,
            'led_peaks': [float(p) for p in real_display.peaks],
        }
        print(f"\n  Real display: f_tetra={real_f_tetra:.4f}, f_pop={real_f_pop:.4f}")

        plot_pareto_front(pareto_results, pareto_output, gamma=gamma,
                          real_display_points=[real_display_point])

        # Run full evaluation for each pareto point
        for r in pareto_results:
            beta_label = f"beta{r['beta']:.2f}"
            point_dir = pareto_output / beta_label
            run_evaluation(r['display'], observers, hyperobs, WAVELENGTHS,
                           point_dir, label=f"Pareto β={r['beta']:.2f}",
                           coverage=args.coverage,
                           extra_summary={
                               'beta': r['beta'],
                               'gamma': gamma,
                               'f_tetra': r['f_tetra'],
                               'f_pop': r['f_pop'],
                               'f_lum': r.get('f_lum', 0.0),
            })

        # Save pareto summary
        pareto_summary = {
            'betas': betas,
            'gamma': gamma,
            'fwhm_range': [args.fwhm_min, args.fwhm_max],
            'n_primaries': args.n_primaries,
            'points': [{
                'beta': r['beta'],
                'f_tetra': r['f_tetra'],
                'f_pop': r['f_pop'],
                'f_lum': r.get('f_lum', 0.0),
                'led_peaks': [float(p) for p in r['display'].peaks],
                'led_fwhms': [float(f) for f in r['display'].fwhms],
            } for r in pareto_results],
            'real_display_points': [real_display_point],
        }
        with open(pareto_output / 'pareto_summary.json', 'w') as f:
            json.dump(pareto_summary, f, indent=2)
        print(f"\n  Saved: {pareto_output / 'pareto_summary.json'}")

        # Combined summary figure: ideal display, real display, pareto front
        # Ideal = beta=1.0 (pure tetrachromat isolation optimum)
        ideal_display = next(r['display'] for r in pareto_results if abs(r['beta'] - 1.0) < 1e-6)
        plot_combined_summary(ideal_display, real_display, pareto_results,
                              WAVELENGTHS, pareto_output, gamma=gamma,
                              real_display_points=[real_display_point])

    print("\nDone.")


if __name__ == '__main__':
    main()
