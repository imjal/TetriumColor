#!/usr/bin/env python3
"""
Joint optimization of an N-primary Gaussian-LED display for cone isolation
of every photoreceptor in an observer.

Supports any observer (trichromat, tetrachromat, hyperobserver) with any
number of primaries >= number of cones.  When n_primaries > n_cones the
transfer matrix is rectangular and we use least-squares isolation.

Decision variables (2 * n_primaries):
    - n_primaries LED peak wavelengths  (380-780 nm)
    - n_primaries LED bandwidths / FWHM (fwhm_min - fwhm_max nm)

White-point normalization:
    LED amplitudes are scaled so that all-LEDs-at-max produces the ones
    vector [1, ..., 1] in cone space.  Cone responses lie in [0, 1].

Cone isolation:
    Square case (n_primaries == n_cones):  C^{-1}[:, k] gives exact isolation.
    Overdetermined case (n_primaries > n_cones):  C^+ (pseudoinverse) gives
        minimum-norm isolation direction.

Two blendable objectives:
    f_raw   = min_k  |r1[k] - r2[k]|           (raw cone difference)
    f_mahal = min_k  d_target_k                 (noise-weighted Mahalanobis)
    score   = alpha * f_raw + (1 - alpha) * f_mahal

Multiple seeds verify that the global optimum is robust.

Usage:
    # Validation runs
    python optimize_hyperobserver_display.py --observer trichromat
    python optimize_hyperobserver_display.py --observer tetrachromat

    # Hyperobserver trials
    python optimize_hyperobserver_display.py --observer hyperobserver --n-primaries 12 --fwhm-min 1 --fwhm-max 40
    python optimize_hyperobserver_display.py --observer hyperobserver --n-primaries 36 --fwhm-min 5 --fwhm-max 40
    python optimize_hyperobserver_display.py --observer hyperobserver --n-primaries 36 --fwhm-min 1 --fwhm-max 40
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from TetriumColor.Observer.Observer import Observer
from TetriumColor.Observer.Spectra import Spectra
from TetriumColor.Plotting.PlotStyle import (
    apply_style, COLORS, WAVELENGTHS as STYLE_WL,
    SINGLE_COL, DOUBLE_COL, build_observers,
)

# ============================================================
# Constants
# ============================================================
WAVELENGTHS = np.arange(400, 701, 1)
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def make_sigma_noise(n_cones: int) -> np.ndarray:
    """Default sigma (all ones) — unweighted cone contrast distance.

    Returns an array of ones so that dividing by sigma is a no-op.
    To get a noise-weighted (Mahalanobis / d') metric, pass in
    empirical per-cone noise standard deviations instead.
    """
    return np.ones(n_cones)


# ============================================================
# Data structures
# ============================================================
@dataclass
class DisplayDesign:
    peaks: np.ndarray           # (n_primaries,)
    fwhms: np.ndarray           # (n_primaries,)
    amplitudes: np.ndarray      # (n_primaries,) LED scaling factors
    C: np.ndarray               # (n_cones, n_primaries) normalized transfer matrix
    cond: float


@dataclass
class IsolationResult:
    target_cone: int
    target_peak_nm: float
    w1: np.ndarray              # (n_primaries,)
    w2: np.ndarray              # (n_primaries,)
    r1: np.ndarray              # (n_cones,) cone responses
    r2: np.ndarray
    raw_diff: np.ndarray        # (n_cones,) |r1 - r2|
    cone_contrast: np.ndarray   # (n_cones,) (r1-r2)/avg
    d_target: float
    d_rest: float


# ============================================================
# Spectral primitives
# ============================================================
def gaussian_led(wavelengths: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    sigma = fwhm * FWHM_TO_SIGMA
    return np.exp(-(wavelengths - peak) ** 2 / (2 * sigma ** 2))


def build_display_matrix(wavelengths: np.ndarray, peaks: np.ndarray,
                         fwhms: np.ndarray) -> np.ndarray:
    """(n_wl, n_primaries) display matrix."""
    return np.column_stack([gaussian_led(wavelengths, p, f)
                            for p, f in zip(peaks, fwhms)])


# ============================================================
# Transfer matrix & white-point normalization
# ============================================================
def build_normalized_transfer(sensor_matrix: np.ndarray,
                              display_matrix: np.ndarray) -> tuple:
    """Build C such that C @ 1 = 1 and cone responses in [0,1].

    For square C: amplitude scaling via C^{-1} @ 1 (preferred) or row norm (fallback).
    For rectangular C (n_primaries > n_cones): least-squares amplitude scaling.

    Returns (C_normalized, amplitudes) or (None, None) if infeasible.
    """
    C_raw = sensor_matrix @ display_matrix  # (n_cones, n_primaries)
    n_cones, n_primaries = C_raw.shape

    if np.any(np.isnan(C_raw)) or np.any(np.isinf(C_raw)):
        return None, None

    ones_cones = np.ones(n_cones)

    # Try exact amplitude scaling: C_raw @ a = 1 with a > 0
    if n_primaries == n_cones:
        try:
            cond = np.linalg.cond(C_raw)
            if cond < 1e14:
                a = np.linalg.solve(C_raw, ones_cones)
                if np.all(a > 0):
                    C_norm = C_raw @ np.diag(a)
                    return C_norm, a
        except np.linalg.LinAlgError:
            pass
    else:
        # Rectangular: minimum-norm solution via SVD-based pseudoinverse
        try:
            C_pinv = _regularized_right_inverse(C_raw, rcond=1e-10)
            if C_pinv is not None:
                a = C_pinv @ ones_cones
                if np.all(a > 0):
                    C_norm = C_raw @ np.diag(a)
                    return C_norm, a
        except Exception:
            pass

    # Fallback: row normalization (always valid for non-negative C)
    row_sums = C_raw.sum(axis=1)
    if np.any(row_sums <= 0):
        return None, None
    C_norm = C_raw / row_sums[:, np.newaxis]
    return C_norm, np.ones(n_primaries)


# ============================================================
# Isolation computation
# ============================================================
def _regularized_right_inverse(C: np.ndarray, rcond: float = 1e-10) -> Optional[np.ndarray]:
    """Compute a regularized right-inverse of C using SVD.

    For a fat matrix C (n_cones x n_primaries, n_primaries >= n_cones):
        C_inv = C^T (C C^T + lambda * I)^{-1}

    This always produces a usable inverse, even for ill-conditioned C.
    Singular values below rcond * max(sv) are treated as zero.

    Returns (n_primaries, n_cones) matrix, or None on failure.
    """
    try:
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        threshold = rcond * s[0]
        # Regularized inverse of singular values
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        # Right inverse: V @ diag(1/s) @ U^T
        C_inv = Vt.T @ np.diag(s_inv) @ U.T  # (n_primaries, n_cones)
        return C_inv
    except np.linalg.LinAlgError:
        return None


def compute_isolation(C: np.ndarray, sigma_noise: np.ndarray,
                      cone_peaks: np.ndarray) -> Optional[List[IsolationResult]]:
    """For each cone k, find the maximal-contrast pair.

    Square: uses C^{-1}[:, k].
    Rectangular: uses pseudoinverse C^+[:, k] (minimum-norm direction that
    changes only cone k).

    Uses Tikhonov regularization when the matrix is ill-conditioned to
    produce a usable (if imperfect) isolation direction.
    """
    n_cones, n_primaries = C.shape

    if n_primaries == n_cones:
        cond = np.linalg.cond(C)
        if cond < 1e13:
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = None
        else:
            C_inv = None
        # Fallback: regularized pseudoinverse via SVD
        if C_inv is None:
            C_inv = _regularized_right_inverse(C)
            if C_inv is None:
                return None
    else:
        C_inv = _regularized_right_inverse(C)
        if C_inv is None:
            return None

    w_base = np.full(n_primaries, 0.5)
    results = []
    for k in range(n_cones):
        iso_dir = C_inv[:, k]  # (n_primaries,)
        abs_dir = np.abs(iso_dir)
        ratios = np.where(abs_dir > 1e-15, 0.5 / abs_dir, np.inf)
        alpha = ratios.min()

        w1 = np.clip(w_base + alpha * iso_dir, 0, 1)
        w2 = np.clip(w_base - alpha * iso_dir, 0, 1)

        r1 = C @ w1
        r2 = C @ w2
        raw_diff = np.abs(r1 - r2)
        avg = np.maximum((r1 + r2) / 2.0, 1e-30)
        cone_contrast = (r1 - r2) / avg

        d_target = abs(cone_contrast[k]) / sigma_noise[k]
        rest_idx = [j for j in range(n_cones) if j != k]
        d_rest = float(np.sqrt(np.sum((cone_contrast[rest_idx] / sigma_noise[rest_idx]) ** 2)))

        results.append(IsolationResult(
            target_cone=k, target_peak_nm=cone_peaks[k],
            w1=w1, w2=w2, r1=r1, r2=r2,
            raw_diff=raw_diff, cone_contrast=cone_contrast,
            d_target=d_target, d_rest=d_rest,
        ))

    return results


# ============================================================
# Scoring
# ============================================================
def score_design(results: List[IsolationResult], alpha: float) -> float:
    """Higher = better. Maximizes worst-case across cones.
    alpha=1.0: pure raw difference; alpha=0.0: pure Mahalanobis.
    """
    raw_scores = np.array([r.raw_diff[r.target_cone] for r in results])
    mahal_scores = np.array([r.d_target for r in results])
    return alpha * raw_scores.min() + (1 - alpha) * mahal_scores.min()


# ============================================================
# Joint optimizer
# ============================================================
def evaluate_design(x: np.ndarray, sensor_matrix: np.ndarray,
                    wavelengths: np.ndarray, sigma_noise: np.ndarray,
                    cone_peaks: np.ndarray, n_primaries: int,
                    alpha: float) -> float:
    peaks = x[:n_primaries]
    fwhms = x[n_primaries:]
    D = build_display_matrix(wavelengths, peaks, fwhms)
    C_norm, amps = build_normalized_transfer(sensor_matrix, D)
    if C_norm is None:
        return 1e6
    results = compute_isolation(C_norm, sigma_noise, cone_peaks)
    if results is None:
        return 1e6
    score = score_design(results, alpha)
    # Soft penalty for high condition number (via SVD)
    sv = np.linalg.svd(C_norm, compute_uv=False)
    cond = sv[0] / max(sv[-1], 1e-30)
    cond_penalty = max(0, np.log10(max(cond, 1)) - 6) * 0.1
    return -(score - cond_penalty)


def optimize_display(sensor_matrix: np.ndarray, wavelengths: np.ndarray,
                     sigma_noise: np.ndarray, cone_peaks: np.ndarray,
                     n_primaries: int,
                     fwhm_min: float = 5.0, fwhm_max: float = 40.0,
                     alpha: float = 0.0, n_seeds: int = 5,
                     maxiter: int = 500, popsize: int = 30) -> tuple:
    bounds = [(405, 695)] * n_primaries + [(fwhm_min, fwhm_max)] * n_primaries

    best_score = -np.inf
    best_result = None
    all_scores = []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds} ... ", end="", flush=True)
        result = differential_evolution(
            evaluate_design, bounds,
            args=(sensor_matrix, wavelengths, sigma_noise, cone_peaks,
                  n_primaries, alpha),
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
        raise RuntimeError(
            f"No feasible display found after {n_seeds} seeds.")

    x = best_result.x
    sort_idx = np.argsort(x[:n_primaries])
    peaks = x[:n_primaries][sort_idx]
    fwhms = x[n_primaries:][sort_idx]

    D = build_display_matrix(wavelengths, peaks, fwhms)
    C_norm, amps = build_normalized_transfer(sensor_matrix, D)
    sv = np.linalg.svd(C_norm, compute_uv=False)
    cond = sv[0] / max(sv[-1], 1e-30)

    design = DisplayDesign(peaks=peaks, fwhms=fwhms, amplitudes=amps,
                           C=C_norm, cond=cond)
    return design, all_scores


# ============================================================
# DeltaE validation (trichromat / tetrachromat only)
# ============================================================
def compute_delta_e(observer: Observer, design: DisplayDesign,
                    results: List[IsolationResult],
                    wavelengths: np.ndarray) -> List[float]:
    D = build_display_matrix(wavelengths, design.peaks, design.fwhms)
    D_scaled = D @ np.diag(design.amplitudes)
    delta_es = []
    for res in results:
        spd1 = D_scaled @ res.w1
        spd2 = D_scaled @ res.w2
        s1 = Spectra(wavelengths=wavelengths, data=spd1, normalized=False)
        s2 = Spectra(wavelengths=wavelengths, data=spd2, normalized=False)
        de = observer.delta_E(s1, s2)
        delta_es.append(de)
    return delta_es


# ============================================================
# Visualization
# ============================================================
def make_cone_colors(n: int, cone_peaks: np.ndarray = None) -> list:
    """Return per-cone colors using PlotStyle.COLORS when peaks are known."""
    if cone_peaks is not None:
        out = []
        for p in cone_peaks:
            # Find closest matching key in COLORS
            best = min(COLORS.keys(), key=lambda k: abs(k - p))
            out.append(COLORS[best])
        return out
    return list(plt.cm.turbo(np.linspace(0.05, 0.95, n)))


def plot_isolation_diagnostics(results: List[IsolationResult],
                               design: DisplayDesign,
                               wavelengths: np.ndarray,
                               sigma_noise: np.ndarray,
                               cone_peaks: np.ndarray,
                               output_dir: Path,
                               trial_label: str,
                               delta_es: Optional[List[float]] = None):
    apply_style()
    n_cones = len(results)
    n_primaries = len(design.peaks)
    D = build_display_matrix(wavelengths, design.peaks, design.fwhms)
    D_scaled = D @ np.diag(design.amplitudes)
    colors = make_cone_colors(n_cones, cone_peaks)
    x_pos = np.arange(n_cones)
    peak_labels = [f"{p:.0f}" if p == int(p) else f"{p}" for p in cone_peaks]

    fig, axes = plt.subplots(n_cones, 3,
                              figsize=(DOUBLE_COL, 1.8 * n_cones))
    if n_cones == 1:
        axes = axes[np.newaxis, :]

    for row, res in enumerate(results):
        k = res.target_cone
        spd1 = D_scaled @ res.w1
        spd2 = D_scaled @ res.w2
        raw_diff = res.raw_diff
        avg = np.maximum((res.r1 + res.r2) / 2.0, 1e-30)
        contrast_per_cone = np.abs((res.r1 - res.r2) / avg) / sigma_noise

        # Panel 1: Spectra
        ax = axes[row, 0]
        ax.plot(wavelengths, spd1, color='#2166ac', linewidth=0.9, label='Stimulus 1')
        ax.plot(wavelengths, spd2, color='#b2182b', linewidth=0.9, label='Stimulus 2')
        ax.fill_between(wavelengths, spd1, spd2, alpha=0.10, color='#7570b3')
        ax.set_ylim(bottom=0)
        label_str = f'{peak_labels[k]}\\,nm'
        if delta_es is not None:
            label_str += f'  ($\\Delta E$={delta_es[row]:.2f})'
        ax.set_ylabel(label_str)
        if row == 0:
            ax.set_title('Stimulus spectra')
            ax.legend(fontsize=6)
        if row == n_cones - 1:
            ax.set_xlabel('Wavelength (nm)')

        # Panel 2: Raw cone difference
        ax = axes[row, 1]
        bar_colors = [colors[i] for i in range(n_cones)]
        ax.bar(x_pos, raw_diff, color=bar_colors, edgecolor='black', linewidth=0.3)
        ax.set_xticks(x_pos)
        if row == 0:
            ax.set_title('Raw cone difference $|r_1 - r_2|$')
        if row == n_cones - 1:
            ax.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=6)
            ax.set_xlabel('Cone peak (nm)')
        else:
            ax.set_xticklabels([])

        # Panel 3: Cone contrast distance per cone
        ax = axes[row, 2]
        ax.bar(x_pos, contrast_per_cone, color=bar_colors,
               edgecolor='black', linewidth=0.3)
        ax.set_xticks(x_pos)
        d_tgt = f'{res.d_target:.1f}' if res.d_target >= 1 else f'{res.d_target:.2e}'
        ax.text(0.98, 0.95, f'$d_{{\\mathrm{{tgt}}}}$={d_tgt}\n$d_{{\\mathrm{{rest}}}}$={res.d_rest:.2e}',
                transform=ax.transAxes, fontsize=6, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if row == 0:
            ax.set_title('Cone contrast distance')
        if row == n_cones - 1:
            ax.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=6)
            ax.set_xlabel('Cone peak (nm)')
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(output_dir / 'cone_isolation_diagnostics.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'cone_isolation_diagnostics.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'cone_isolation_diagnostics.pdf'}")


def plot_display_spectra(design: DisplayDesign, wavelengths: np.ndarray,
                         cone_peaks: np.ndarray, sensor_matrix: np.ndarray,
                         output_dir: Path, trial_label: str):
    apply_style()
    n_cones = len(cone_peaks)
    n_primaries = len(design.peaks)
    D = build_display_matrix(wavelengths, design.peaks, design.fwhms)
    D_scaled = D @ np.diag(design.amplitudes)
    colors_cones = make_cone_colors(n_cones, cone_peaks)
    colors_leds = plt.cm.Greys(np.linspace(0.35, 0.75, n_primaries))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6),
                                    sharex=True)

    # LED spectra
    for i in range(n_primaries):
        label = (f'{design.peaks[i]:.0f}\\,nm'
                 if n_primaries <= 12 else None)
        ax1.fill_between(wavelengths, D_scaled[:, i], alpha=0.25,
                         color=colors_leds[i])
        ax1.plot(wavelengths, D_scaled[:, i], color=colors_leds[i],
                 linewidth=0.8, label=label)
    if n_primaries <= 12:
        ax1.legend(fontsize=6, ncol=min(4, n_primaries), loc='upper right',
                   framealpha=0.8)
    ax1.set_ylabel('Scaled LED power')
    ax1.set_xlim(wavelengths[0], wavelengths[-1])

    # Cone sensitivities
    for i in range(n_cones):
        peak = cone_peaks[i]
        lbl = f'{peak:.0f}\\,nm' if peak == int(peak) else f'{peak}\\,nm'
        ax2.plot(wavelengths, sensor_matrix[i], color=colors_cones[i],
                 linewidth=1.0, label=lbl)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Sensitivity')
    ax2.legend(fontsize=6, ncol=min(4, n_cones), loc='upper right',
               framealpha=0.8)
    ax2.set_xlim(wavelengths[0], wavelengths[-1])

    fig.tight_layout()
    fig.savefig(output_dir / 'display_and_cones.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'display_and_cones.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'display_and_cones.pdf'}")


def plot_seed_convergence(all_scores: List[float], output_dir: Path, trial_label: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(all_scores))
    ax.bar(x, all_scores, color='steelblue', edgecolor='black')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Score (higher = better)')
    ax.set_title(f'Optimizer Score — {trial_label}')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / 'seed_convergence.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'seed_convergence.png'}")


# ============================================================
# Run one trial
# ============================================================
def run_trial(observer: Observer, observer_name: str,
              n_primaries: int, fwhm_min: float, fwhm_max: float,
              wavelengths: np.ndarray, alpha: float,
              n_seeds: int, maxiter: int, popsize: int,
              output_dir: Path, use_delta_e: bool = False):
    """Full pipeline: optimize display, compute isolation, plot, validate."""
    sensor_matrix = observer.sensor_matrix
    cone_peaks = np.array([s.peak for s in observer.sensors], dtype=float)
    n_cones = observer.dimension
    sigma_noise = make_sigma_noise(n_cones)

    trial_label = f'{observer_name}_{n_primaries}p_fwhm{fwhm_min:.0f}-{fwhm_max:.0f}'
    trial_dir = output_dir / trial_label
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Trial: {trial_label}")
    print(f"  Observer: {observer_name} ({n_cones} cones)")
    print(f"  Primaries: {n_primaries},  FWHM: [{fwhm_min}, {fwhm_max}] nm")
    print(f"  Cone peaks: {cone_peaks}")
    print(f"  Alpha: {alpha},  Seeds: {n_seeds}")
    print(f"{'='*70}")

    # ---- Optimize ----
    print("Optimizing LED peaks + FWHMs...")
    design, all_scores = optimize_display(
        sensor_matrix, wavelengths, sigma_noise, cone_peaks,
        n_primaries=n_primaries, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
        alpha=alpha, n_seeds=n_seeds, maxiter=maxiter, popsize=popsize,
    )

    print(f"\nBest design:")
    if n_primaries <= 12:
        print(f"  Peaks:  [{', '.join(f'{p:.1f}' for p in design.peaks)}]")
        print(f"  FWHMs:  [{', '.join(f'{f:.1f}' for f in design.fwhms)}]")
    else:
        print(f"  Peaks:  {design.peaks.min():.1f} - {design.peaks.max():.1f} nm "
              f"({n_primaries} LEDs)")
        print(f"  FWHMs:  {design.fwhms.min():.1f} - {design.fwhms.max():.1f} nm")
    print(f"  cond(C): {design.cond:.2e}")

    # ---- Isolation ----
    results = compute_isolation(design.C, sigma_noise, cone_peaks)

    peak_labels = [f"{p:.0f}" if p == int(p) else f"{p}" for p in cone_peaks]
    print(f"\n{'Cone':>8}  {'d_target':>10}  {'d_rest':>12}  "
          f"{'raw_diff[k]':>12}  {'contrast[k]':>12}")
    print("-" * 65)
    for r in results:
        k = r.target_cone
        print(f"{peak_labels[k]+'nm':>8}  {r.d_target:10.4f}  {r.d_rest:12.2e}  "
              f"{r.raw_diff[k]:12.6f}  {r.cone_contrast[k]:12.6f}")

    # White point check
    white = design.C @ np.ones(n_primaries)
    print(f"\nWhite point (C @ 1): min={white.min():.8f}  max={white.max():.8f}")

    # ---- DeltaE validation ----
    delta_es = None
    if use_delta_e:
        print("\nDeltaE 2000 validation:")
        delta_es = compute_delta_e(observer, design, results, wavelengths)
        for i, (r, de) in enumerate(zip(results, delta_es)):
            k = r.target_cone
            print(f"  Cone {peak_labels[k]}nm:  DeltaE = {de:.4f}")

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_display_spectra(design, wavelengths, cone_peaks, sensor_matrix,
                         trial_dir, trial_label)
    plot_isolation_diagnostics(results, design, wavelengths, sigma_noise,
                               cone_peaks, trial_dir, trial_label,
                               delta_es=delta_es)
    plot_seed_convergence(all_scores, trial_dir, trial_label)

    # ---- Save ----
    np.savez(trial_dir / 'results.npz',
             peaks=design.peaks, fwhms=design.fwhms, amplitudes=design.amplitudes,
             C=design.C, cond=design.cond,
             cone_peaks=cone_peaks, sigma_noise=sigma_noise,
             all_scores=np.array(all_scores), alpha=alpha)

    # Save human-readable summary
    summary = {
        'trial': trial_label,
        'observer': observer_name,
        'n_cones': int(n_cones),
        'n_primaries': int(n_primaries),
        'fwhm_range': [float(fwhm_min), float(fwhm_max)],
        'alpha': float(alpha),
        'n_seeds': n_seeds,
        'cond_C': float(design.cond),
        'led_peaks': [float(p) for p in design.peaks],
        'led_fwhms': [float(f) for f in design.fwhms],
        'per_cone': [{
            'cone_peak_nm': float(cone_peaks[r.target_cone]),
            'd_target': float(r.d_target),
            'd_rest': float(r.d_rest),
            'raw_diff': float(r.raw_diff[r.target_cone]),
            'cone_contrast': float(r.cone_contrast[r.target_cone]),
            'delta_e': float(delta_es[i]) if delta_es else None,
        } for i, r in enumerate(results)],
        'white_point_min': float(white.min()),
        'white_point_max': float(white.max()),
    }
    with open(trial_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {trial_dir / 'summary.json'}")

    return design, results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Optimize N-primary display for cone isolation')
    parser.add_argument('--observer', type=str, default='hyperobserver',
                        choices=['dichromat', 'trichromat', 'tetrachromat', 'hyperobserver', 'all'],
                        help='Observer type (default: hyperobserver)')
    parser.add_argument('--n-primaries', type=int, default=None,
                        help='Number of LED primaries (default: match cone count)')
    parser.add_argument('--fwhm-min', type=float, default=5.0,
                        help='Minimum LED FWHM in nm (default: 5.0)')
    parser.add_argument('--fwhm-max', type=float, default=40.0,
                        help='Maximum LED FWHM in nm (default: 40.0)')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Blend: 0.0=Mahalanobis, 1.0=raw difference (default: 0.0)')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Random seeds for global optimizer (default: 5)')
    parser.add_argument('--maxiter', type=int, default=500,
                        help='Max iterations per seed (default: 500)')
    parser.add_argument('--popsize', type=int, default=30,
                        help='Population size for DE (default: 30)')
    parser.add_argument('--output-dir', type=str, default='output/display_optimization')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build observers
    obs_configs = []
    if args.observer in ('dichromat', 'all'):
        obs = Observer.dichromat(wavelengths=WAVELENGTHS)
        np_ = args.n_primaries or obs.dimension
        obs_configs.append(('dichromat', obs, np_, True))
    if args.observer in ('trichromat', 'all'):
        obs = Observer.trichromat(wavelengths=WAVELENGTHS)
        np_ = args.n_primaries or obs.dimension
        obs_configs.append(('trichromat', obs, np_, True))
    if args.observer in ('tetrachromat', 'all'):
        obs = Observer.tetrachromat(wavelengths=WAVELENGTHS)
        np_ = args.n_primaries or obs.dimension
        obs_configs.append(('tetrachromat', obs, np_, True))
    if args.observer in ('hyperobserver', 'all'):
        obs = Observer.hyperobserver(wavelengths=WAVELENGTHS)
        np_ = args.n_primaries or obs.dimension
        obs_configs.append(('hyperobserver', obs, np_, False))

    for name, obs, n_p, use_de in obs_configs:
        run_trial(obs, name, n_p,
                  fwhm_min=args.fwhm_min, fwhm_max=args.fwhm_max,
                  wavelengths=WAVELENGTHS, alpha=args.alpha,
                  n_seeds=args.n_seeds, maxiter=args.maxiter,
                  popsize=args.popsize, output_dir=output_dir,
                  use_delta_e=use_de)

    print(f"\nAll results saved to {output_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
