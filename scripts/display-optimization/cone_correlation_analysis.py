#!/usr/bin/env python3
"""
Cone Correlation Analysis: Fundamental Limits on Discriminating Close Cone Variants

Studies the tetrachromat isolation problem: given a standard trichromat
(S, M, L) on a 4-primary display, how well can we detect a 4th cone Q
as its peak wavelength varies from M (530nm) to L (559nm)?

The setup mirrors real observer discrimination:
- Trichromat: S(420), M(530), L(559) — using Neitz template
- Tetrachromat: S, M, Q, L — where Q varies from 530nm to 559nm
- Display: 4 Gaussian LEDs optimally placed
- For each Q position, find the trichromat's null-space metamer pair,
  then measure the Mahalanobis d' the Q cone achieves on that pair

This captures the real constraint: the display must create a stimulus pair
that looks identical to the trichromat (3 cones matched) but maximally
different to the 4th cone. The difficulty grows as Q approaches M or L
because the null-space direction that is metameric for 3 cones barely
excites a 4th cone that is spectrally similar to one of them.

Additional analyses:
1. Hyperobserver 12x12 Gram matrix (spectral correlations)
2. Spectral difference visualization (Δφ shape and amplitude)

Usage:
    python cone_correlation_analysis.py
    python cone_correlation_analysis.py --q-range 530 559 --q-step 1.0
    python cone_correlation_analysis.py --threshold 1.0 --n-seeds 5
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from TetriumColor.Observer.Observer import Observer, Cone

# ============================================================
# Constants
# ============================================================
WAVELENGTHS = np.arange(400, 701, 1)
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Standard trichromat cone peaks (Neitz template)
S_PEAK = 420
M_PEAK = 530
L_PEAK = 559

# Minimum separation between LED primaries
MIN_PEAK_SEPARATION = 30.0


def make_sigma_noise(n_cones: int) -> np.ndarray:
    """Default sigma (all ones) — unweighted cone contrast distance."""
    return np.ones(n_cones)


# ============================================================
# Cone construction
# ============================================================
def make_cone_sensitivity(peak: float, wavelengths: np.ndarray = WAVELENGTHS,
                          template: str = 'neitz', od: float = 0.5) -> np.ndarray:
    """Single cone sensitivity curve as (n_wl,) array."""
    cone = Cone.cone(peak, template=template, od=od, wavelengths=wavelengths)
    return cone.data


def make_trichromat_sensor_matrix(wavelengths: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Standard S, M, L trichromat sensor matrix (3, n_wl)."""
    phi_s = make_cone_sensitivity(S_PEAK, wavelengths, od=0.4)
    phi_m = make_cone_sensitivity(M_PEAK, wavelengths)
    phi_l = make_cone_sensitivity(L_PEAK, wavelengths)
    return np.vstack([phi_s, phi_m, phi_l])


def make_tetrachromat_sensor_matrix(q_peak: float,
                                    wavelengths: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """S, M, Q, L tetrachromat sensor matrix (4, n_wl)."""
    phi_s = make_cone_sensitivity(S_PEAK, wavelengths, od=0.4)
    phi_m = make_cone_sensitivity(M_PEAK, wavelengths)
    phi_q = make_cone_sensitivity(q_peak, wavelengths)
    phi_l = make_cone_sensitivity(L_PEAK, wavelengths)
    return np.vstack([phi_s, phi_m, phi_q, phi_l])


# ============================================================
# Spectral correlation helpers
# ============================================================
def spectral_correlation(phi1: np.ndarray, phi2: np.ndarray) -> float:
    """Normalized inner product ρ = ∫φ₁φ₂ / √(∫φ₁²·∫φ₂²)."""
    return np.dot(phi1, phi2) / (np.linalg.norm(phi1) * np.linalg.norm(phi2))


def spectral_angle(phi1: np.ndarray, phi2: np.ndarray) -> float:
    """Angle between two spectral sensitivity curves in degrees."""
    rho = np.clip(spectral_correlation(phi1, phi2), -1, 1)
    return np.degrees(np.arccos(rho))


def gram_matrix(sensor_matrix: np.ndarray) -> np.ndarray:
    """Normalized Gram matrix G_ij = ∫φ_i·φ_j / √(∫φ_i²·∫φ_j²)."""
    norms = np.linalg.norm(sensor_matrix, axis=1, keepdims=True)
    normed = sensor_matrix / norms
    return normed @ normed.T


def projection_residual(phi_q: np.ndarray, tri_sensor: np.ndarray) -> float:
    """Fraction of φ_Q outside span{φ_S, φ_M, φ_L}.

    Projects φ_Q onto the row space of tri_sensor and returns
    ||φ_Q_perp|| / ||φ_Q||.  This is the display-free ceiling on
    tetrachromat isolation: no display can exploit more signal than this.
    """
    # tri_sensor is (3, n_wl).  Project φ_Q onto row space.
    # φ_proj = Φᵀ (Φ Φᵀ)⁻¹ Φ · φ_Q
    Phi = tri_sensor  # (3, n_wl)
    G = Phi @ Phi.T   # (3, 3)
    coeffs = np.linalg.solve(G, Phi @ phi_q)  # (3,)
    phi_proj = Phi.T @ coeffs  # (n_wl,)
    phi_perp = phi_q - phi_proj
    return np.linalg.norm(phi_perp) / np.linalg.norm(phi_q)


def ideal_display_dprime(phi_q: np.ndarray, tri_sensor: np.ndarray,
                         sigma_q: float = 1.0,
                         n_primaries: int = 100) -> float:
    """Upper-bound d' from an ideal (many-primary) display.

    Uses a dense set of narrow-band primaries spanning 400-700nm,
    matching the physical setup of the display-mediated case but
    with far more degrees of freedom.  This gives a tight upper
    bound that a real display approaches.

    The computation: build a (3, N) trichromat transfer matrix for
    N narrow-band primaries, find its null space, and maximize
    Q-cone contrast along null directions.
    """
    Phi = tri_sensor  # (3, n_wl)

    # Residual check — if Q is in span{S,M,L}, no display helps
    G = Phi @ Phi.T
    coeffs = np.linalg.solve(G, Phi @ phi_q)
    phi_perp = phi_q - Phi.T @ coeffs
    if np.linalg.norm(phi_perp) < 1e-30:
        return 0.0

    # Build ideal display: N evenly-spaced narrow-band primaries
    # Use same FWHM as the real display (15nm) for fair comparison
    wl = WAVELENGTHS
    led_peaks = np.linspace(405, 695, n_primaries)
    fwhm = 15.0
    D = build_display_matrix(wl, led_peaks, np.full(n_primaries, fwhm))

    # Trichromat transfer
    C_tri_raw = Phi @ D  # (3, N)
    white_tri = C_tri_raw @ np.ones(n_primaries)
    if np.any(white_tri <= 0):
        return 0.0
    C_tri = C_tri_raw / white_tri[:, np.newaxis]

    # Q-cone transfer (full 4-cone)
    phi_q_row = phi_q[np.newaxis, :]  # (1, n_wl)
    tetra = np.vstack([Phi, phi_q_row])  # (4, n_wl)
    C_tetra_raw = tetra @ D
    white_tetra = C_tetra_raw @ np.ones(n_primaries)
    if np.any(white_tetra <= 0):
        return 0.0
    C_tetra = C_tetra_raw / white_tetra[:, np.newaxis]

    # Null space of trichromat transfer
    ns = null_space(C_tri)  # (N, k) where k = N - 3
    if ns.shape[1] == 0:
        return 0.0

    # The Q-cone transfer row
    c_q = C_tetra[3, :]  # (N,)

    # Project c_q onto the null space to find the optimal metameric
    # direction that maximizes Q-cone contrast.
    # c_q_null = ns @ (ns^T @ c_q) is the component of c_q in the null space
    coeffs_null = ns.T @ c_q  # (k,)
    optimal_dir = ns @ coeffs_null  # (N,) — optimal direction in primary space
    dir_norm = np.linalg.norm(optimal_dir)
    if dir_norm < 1e-30:
        return 0.0
    optimal_dir /= dir_norm

    # Max excursion in [0,1]^N
    w_base = np.full(n_primaries, 0.5)
    abs_dir = np.abs(optimal_dir)
    ratios = np.where(abs_dir > 1e-15, 0.5 / abs_dir, np.inf)
    t = ratios.min()

    w1 = np.clip(w_base + t * optimal_dir, 0, 1)
    w2 = np.clip(w_base - t * optimal_dir, 0, 1)

    r1 = C_tetra @ w1
    r2 = C_tetra @ w2
    avg = np.maximum((r1 + r2) / 2.0, 1e-30)
    cone_contrast = (r1 - r2) / avg

    sigma = make_sigma_noise(4)
    return abs(cone_contrast[3]) / sigma[3]


# ============================================================
# Display primitives
# ============================================================
def gaussian_led(wavelengths: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    sigma = fwhm * FWHM_TO_SIGMA
    return np.exp(-(wavelengths - peak) ** 2 / (2 * sigma ** 2))


def build_display_matrix(wavelengths: np.ndarray, peaks: np.ndarray,
                         fwhms: np.ndarray) -> np.ndarray:
    """(n_wl, n_primaries) display matrix."""
    return np.column_stack([gaussian_led(wavelengths, p, f)
                            for p, f in zip(peaks, fwhms)])


def _peak_separation_penalty(peaks: np.ndarray) -> float:
    """Hard penalty if any two LED peaks are closer than MIN_PEAK_SEPARATION."""
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            gap = abs(peaks[i] - peaks[j])
            if gap < MIN_PEAK_SEPARATION:
                return 1e6 + (MIN_PEAK_SEPARATION - gap) * 1e4
    return 0.0


# ============================================================
# Core: tetrachromat isolation d' for a given display
# ============================================================
def compute_tetrachromat_isolation(tri_sensor: np.ndarray,
                                   tetra_sensor: np.ndarray,
                                   D: np.ndarray,
                                   sigma: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute Q-cone isolation d' for a tetrachromat on display D.

    1. Build trichromat transfer matrix C_tri = tri_sensor @ D  (3 x 4)
    2. White-point normalize: C_tri @ 1 = 1
    3. Find null space of C_tri (1-D for 3 cones, 4 primaries)
    4. Build metamer pair along null direction, constrained to [0,1]^4
    5. Evaluate Q-cone contrast on this metamer pair
    6. Return Mahalanobis d' = |cone_contrast_Q| / sigma_Q

    Returns (d_prime, w1, w2).
    """
    n_primaries = D.shape[1]

    # Trichromat transfer matrix
    C_tri_raw = tri_sensor @ D  # (3, 4)
    white_tri = C_tri_raw @ np.ones(n_primaries)
    if np.any(white_tri <= 0):
        return 0.0, None, None
    C_tri = C_tri_raw / white_tri[:, np.newaxis]

    # Tetrachromat transfer matrix (same white-point normalization)
    C_tetra_raw = tetra_sensor @ D  # (4, 4)
    white_tetra = C_tetra_raw @ np.ones(n_primaries)
    if np.any(white_tetra <= 0):
        return 0.0, None, None
    C_tetra = C_tetra_raw / white_tetra[:, np.newaxis]

    # Null space of trichromat (metameric direction)
    ns = null_space(C_tri)
    if ns.shape[1] == 0:
        return 0.0, None, None

    # Find maximal metamer pair along null direction
    best_d = 0.0
    best_w1, best_w2 = None, None
    w_base = np.full(n_primaries, 0.5)

    for col in range(ns.shape[1]):
        null_dir = ns[:, col]
        abs_dir = np.abs(null_dir)
        ratios = np.where(abs_dir > 1e-15, 0.5 / abs_dir, np.inf)
        t = ratios.min()

        w1 = np.clip(w_base + t * null_dir, 0, 1)
        w2 = np.clip(w_base - t * null_dir, 0, 1)

        # Q-cone response (index 2 in SMQL order)
        r1 = C_tetra @ w1
        r2 = C_tetra @ w2

        # Mahalanobis across all 4 cones (but Q is what matters)
        avg = np.maximum((r1 + r2) / 2.0, 1e-30)
        cone_contrast = (r1 - r2) / avg

        # d' for Q cone specifically (index 2)
        d_q = abs(cone_contrast[2]) / sigma[2]

        if d_q > best_d:
            best_d = d_q
            best_w1 = w1
            best_w2 = w2

    return best_d, best_w1, best_w2


def optimize_display_for_tetrachromat(q_peak: float,
                                      n_primaries: int = 4,
                                      fwhm_min: float = 15.0,
                                      fwhm_max: float = 30.0,
                                      n_seeds: int = 3,
                                      maxiter: int = 200,
                                      popsize: int = 15) -> Tuple[float, np.ndarray, np.ndarray]:
    """Optimize display to maximize Q-cone isolation d'.

    Returns (best_d_prime, best_led_peaks, best_led_fwhms).
    """
    tri_sensor = make_trichromat_sensor_matrix()
    tetra_sensor = make_tetrachromat_sensor_matrix(q_peak)
    sigma = make_sigma_noise(4)

    bounds = ([(400.0, 700.0)] * n_primaries +
              [(fwhm_min, fwhm_max)] * n_primaries)

    def objective(x):
        peaks = x[:n_primaries]
        fwhms = x[n_primaries:]
        penalty = _peak_separation_penalty(peaks)
        if penalty > 0:
            return penalty
        D = build_display_matrix(WAVELENGTHS, peaks, fwhms)
        d, _, _ = compute_tetrachromat_isolation(tri_sensor, tetra_sensor, D, sigma)
        return -d  # minimize negative = maximize

    best_score = 0.0
    best_peaks = None
    best_fwhms = None

    for seed in range(n_seeds):
        result = differential_evolution(objective, bounds, seed=seed + 42,
                                        maxiter=maxiter, popsize=popsize,
                                        tol=1e-6, mutation=(0.5, 1.5),
                                        recombination=0.9)
        score = -result.fun
        if score > best_score:
            best_score = score
            best_peaks = result.x[:n_primaries]
            best_fwhms = result.x[n_primaries:]

    return best_score, np.sort(best_peaks), best_fwhms


# ============================================================
# Sweep Q peak across M-L range
# ============================================================
def sweep_q_peak(q_peaks: np.ndarray,
                 n_primaries: int = 4,
                 fwhm_min: float = 15.0,
                 fwhm_max: float = 30.0,
                 n_seeds: int = 3,
                 maxiter: int = 200,
                 popsize: int = 15) -> dict:
    """For each Q peak position, optimize display and compute d'.

    Returns dict with arrays for each Q position.
    """
    tri_sensor = make_trichromat_sensor_matrix()

    # Precompute correlations and display-free quantities
    phi_m = tri_sensor[1]  # M cone
    phi_l = tri_sensor[2]  # L cone
    rho_qm = np.zeros(len(q_peaks))
    rho_ql = np.zeros(len(q_peaks))
    residual_frac = np.zeros(len(q_peaks))
    d_prime_ideal = np.zeros(len(q_peaks))
    d_primes = np.zeros(len(q_peaks))
    all_led_peaks = []
    all_led_fwhms = []

    print("\n  Display-free analysis (projection residual):")
    for i, qp in enumerate(q_peaks):
        phi_q = make_cone_sensitivity(qp)
        rho_qm[i] = spectral_correlation(phi_q, phi_m)
        rho_ql[i] = spectral_correlation(phi_q, phi_l)
        residual_frac[i] = projection_residual(phi_q, tri_sensor)
        d_prime_ideal[i] = ideal_display_dprime(phi_q, tri_sensor)

    # Print display-free summary
    for i, qp in enumerate(q_peaks):
        if i % max(1, len(q_peaks) // 15) == 0 or qp in (q_peaks[0], q_peaks[-1]):
            print(f"    Q={qp:>5.1f}nm  residual={residual_frac[i]:.6f}  "
                  f"d'_ideal={d_prime_ideal[i]:.2f}")

    print(f"\n  Display-mediated optimization ({n_primaries} primaries):")
    for i, qp in enumerate(q_peaks):
        d, peaks, fwhms = optimize_display_for_tetrachromat(
            qp, n_primaries=n_primaries,
            fwhm_min=fwhm_min, fwhm_max=fwhm_max,
            n_seeds=n_seeds, maxiter=maxiter, popsize=popsize)
        d_primes[i] = d
        all_led_peaks.append(peaks.tolist() if peaks is not None else [])
        all_led_fwhms.append(fwhms.tolist() if fwhms is not None else [])

        gap_from_m = qp - M_PEAK
        gap_from_l = L_PEAK - qp
        print(f"    Q={qp:>5.1f}nm  (M+{gap_from_m:.1f}, L-{gap_from_l:.1f})  "
              f"residual={residual_frac[i]:.6f}  d'={d:.2f}  "
              f"peaks=[{', '.join(f'{p:.0f}' for p in peaks)}]")

    # Min correlation with either M or L (the bottleneck)
    rho_min_gap = np.maximum(rho_qm, rho_ql)

    return {
        'q_peaks': q_peaks,
        'rho_qm': rho_qm,
        'rho_ql': rho_ql,
        'rho_closest': rho_min_gap,
        'residual_frac': residual_frac,
        'd_prime_ideal': d_prime_ideal,
        'd_primes': d_primes,
        'led_peaks': all_led_peaks,
        'led_fwhms': all_led_fwhms,
        'n_primaries': n_primaries,
    }


# ============================================================
# Hyperobserver Gram matrix
# ============================================================
def analyze_hyperobserver_gram():
    """Compute the 12x12 Gram matrix of the hyperobserver."""
    hyper = Observer.hyperobserver(wavelengths=WAVELENGTHS)
    G = gram_matrix(hyper.sensor_matrix)
    peaks = [s.peak for s in hyper.sensors]
    return G, peaks


# ============================================================
# Plotting
# ============================================================
def plot_gram_matrix(G: np.ndarray, peaks: list, output_dir: Path):
    """Heatmap of hyperobserver Gram matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(G, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')

    labels = [f'{p}' for p in peaks]
    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(peaks)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Cone peak (nm)', fontsize=12)
    ax.set_ylabel('Cone peak (nm)', fontsize=12)
    ax.set_title('Hyperobserver Gram Matrix\n(spectral correlation $\\rho_{ij}$)', fontsize=13)

    for i in range(len(peaks)):
        for j in range(len(peaks)):
            val = G[i, j]
            color = 'white' if val > 0.8 else 'black'
            if i != j:
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Spectral correlation $\\rho$', shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'hyperobserver_gram_matrix.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'hyperobserver_gram_matrix.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'hyperobserver_gram_matrix.png'}")


def plot_cone_difference(output_dir: Path):
    """Plot cone sensitivities and Δφ for Q positions spanning M to L."""
    tri_sensor = make_trichromat_sensor_matrix()
    q_examples = [532, 536, 540, 545, 550, 555, 557]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all sensitivities
    ax = axes[0]
    ax.plot(WAVELENGTHS, tri_sensor[0], 'b-', lw=2, label=f'S ({S_PEAK}nm)')
    ax.plot(WAVELENGTHS, tri_sensor[1], 'g-', lw=2, label=f'M ({M_PEAK}nm)')
    ax.plot(WAVELENGTHS, tri_sensor[2], 'r-', lw=2, label=f'L ({L_PEAK}nm)')
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(q_examples)))
    for qp, color in zip(q_examples, colors):
        phi_q = make_cone_sensitivity(qp)
        ax.plot(WAVELENGTHS, phi_q, '--', color=color, lw=1.2, alpha=0.8,
                label=f'Q ({qp}nm)')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('Trichromat S,M,L + candidate Q cones', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: difference Q - nearest(M or L)
    ax = axes[1]
    phi_m = tri_sensor[1]
    phi_l = tri_sensor[2]
    for qp, color in zip(q_examples, colors):
        phi_q = make_cone_sensitivity(qp)
        # Show difference from nearest of M, L
        diff_m = phi_q - phi_m
        diff_l = phi_q - phi_l
        # Use whichever has smaller max (the harder-to-distinguish one)
        if np.max(np.abs(diff_m)) < np.max(np.abs(diff_l)):
            diff = diff_m
            ref_label = 'M'
        else:
            diff = diff_l
            ref_label = 'L'
        ax.plot(WAVELENGTHS, diff, '-', color=color, lw=1.5,
                label=f'Q({qp}) - {ref_label}: max|Δφ|={np.max(np.abs(diff)):.4f}')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Δφ = φ_Q − φ_nearest', fontsize=12)
    ax.set_title('Spectral difference from nearest cone', fontsize=13)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'cone_spectral_differences.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'cone_spectral_differences.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'cone_spectral_differences.png'}")


def plot_tetrachromat_isolation(sweep: dict, output_dir: Path):
    """Plot d' vs Q peak position: ideal ceiling, display-mediated, and residual."""
    q_peaks = sweep['q_peaks']
    d_primes = sweep['d_primes']
    d_prime_ideal = sweep['d_prime_ideal']
    residual_frac = sweep['residual_frac']
    rho_qm = sweep['rho_qm']
    rho_ql = sweep['rho_ql']
    n_prim = sweep['n_primaries']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left panel: d' (display-mediated) ----
    ax = axes[0]
    ax.plot(q_peaks, d_primes, 'o-', color='tab:blue', lw=2.5, markersize=5,
            label=f"d' optimized ({n_prim} primaries)")

    # Annotate peak
    best_idx = np.argmax(d_primes)
    ax.annotate(f"Peak: Q={q_peaks[best_idx]:.0f}nm, d'={d_primes[best_idx]:.1f}",
                xy=(q_peaks[best_idx], d_primes[best_idx]),
                xytext=(q_peaks[best_idx] + 3, d_primes[best_idx] + 0.5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1))

    # Mark known genotype positions
    known_q = [533, 536, 547, 551, 552, 553, 555, 556, 556.5]
    for qp in known_q:
        if q_peaks[0] <= qp <= q_peaks[-1]:
            idx = np.argmin(np.abs(q_peaks - qp))
            ax.annotate(f'{qp}', xy=(q_peaks[idx], d_primes[idx]),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=7, ha='center', color='gray',
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label="d' = 1 threshold")
    ax.axvline(M_PEAK, color='green', ls=':', alpha=0.4, label=f'M ({M_PEAK}nm)')
    ax.axvline(L_PEAK, color='red', ls=':', alpha=0.4, label=f'L ({L_PEAK}nm)')

    ax.set_xlabel('Q cone peak wavelength (nm)', fontsize=13)
    ax.set_ylabel("Mahalanobis d'", fontsize=13)
    ax.set_title(f'Tetrachromat Isolation: Ideal Ceiling vs {n_prim}-Primary Display',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # ---- Right panel: projection residual and correlation ----
    ax = axes[1]

    color_res = 'tab:purple'
    ax.plot(q_peaks, residual_frac * 100, 'o-', color=color_res, lw=2.5,
            markersize=4, label='Projection residual $\\|\\phi_{Q\\perp}\\| / \\|\\phi_Q\\|$')
    ax.set_xlabel('Q cone peak wavelength (nm)', fontsize=13)
    ax.set_ylabel('Residual outside span{S,M,L} (%)', fontsize=13, color=color_res)
    ax.tick_params(axis='y', labelcolor=color_res)

    ax2 = ax.twinx()
    ax2.plot(q_peaks, rho_qm, '--', color='green', lw=1.5, alpha=0.7,
             label='$\\rho$(Q, M)')
    ax2.plot(q_peaks, rho_ql, '--', color='red', lw=1.5, alpha=0.7,
             label='$\\rho$(Q, L)')
    ax2.set_ylabel('Spectral correlation $\\rho$', fontsize=13)
    ax2.set_ylim(0.90, 1.005)

    ax.axvline(M_PEAK, color='green', ls=':', alpha=0.4)
    ax.axvline(L_PEAK, color='red', ls=':', alpha=0.4)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    ax.set_title('Display-Free Analysis:\nProjection Residual & Spectral Correlation',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(output_dir / 'tetrachromat_isolation_vs_q.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'tetrachromat_isolation_vs_q.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'tetrachromat_isolation_vs_q.png'}")


def plot_optimal_led_positions(sweep: dict, output_dir: Path):
    """Show how optimal LED peaks change as Q varies."""
    q_peaks = sweep['q_peaks']
    led_peaks = np.array(sweep['led_peaks'])
    n_prim = sweep['n_primaries']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_prim))
    for j in range(n_prim):
        ax.plot(q_peaks, led_peaks[:, j], 'o-', color=colors[j], lw=1.5,
                markersize=4, label=f'LED {j+1}')

    # Show Q peak as reference
    ax.plot(q_peaks, q_peaks, 'k--', lw=1, alpha=0.4, label='Q = LED peak')

    ax.axhline(M_PEAK, color='green', ls=':', alpha=0.3)
    ax.axhline(L_PEAK, color='red', ls=':', alpha=0.3)
    ax.set_xlabel('Q cone peak wavelength (nm)', fontsize=13)
    ax.set_ylabel('Optimal LED peak wavelength (nm)', fontsize=13)
    ax.set_title(f'Optimal {n_prim}-LED Positions vs Q Cone', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'optimal_led_positions.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'optimal_led_positions.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'optimal_led_positions.png'}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Analyze tetrachromat isolation limits as Q cone varies from M to L')
    parser.add_argument('--q-range', type=float, nargs=2, default=[530.0, 559.0],
                        help='Range of Q cone peak positions in nm (default: 530 559)')
    parser.add_argument('--q-step', type=float, default=1.0,
                        help='Step size for Q peak sweep in nm (default: 1.0)')
    parser.add_argument('--n-primaries', type=int, default=4,
                        help='Number of display primaries (default: 4)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Detection threshold d\' (default: 1.0)')
    parser.add_argument('--fwhm-min', type=float, default=15.0)
    parser.add_argument('--fwhm-max', type=float, default=30.0)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--popsize', type=int, default=15)
    parser.add_argument('--output-dir', type=str,
                        default='output/cone_correlation_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 1. Hyperobserver Gram matrix
    # ================================================================
    print("=" * 60)
    print("  1. Hyperobserver Gram Matrix")
    print("=" * 60)
    G, peaks = analyze_hyperobserver_gram()
    plot_gram_matrix(G, peaks, output_dir)

    print("\n  Most correlated cone pairs:")
    pairs = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            pairs.append((G[i, j], peaks[i], peaks[j]))
    pairs.sort(reverse=True)
    for rho, p1, p2 in pairs[:15]:
        gap = abs(p2 - p1)
        print(f"    {p1:>5} vs {p2:>5}  Δλ={gap:5.1f}nm  "
              f"ρ={rho:.6f}  angle={np.degrees(np.arccos(np.clip(rho, -1, 1))):.3f}°")

    # ================================================================
    # 2. Spectral difference visualization
    # ================================================================
    print("\n" + "=" * 60)
    print("  2. Spectral Difference Visualization")
    print("=" * 60)
    plot_cone_difference(output_dir)

    # ================================================================
    # 3. Tetrachromat isolation sweep
    # ================================================================
    print("\n" + "=" * 60)
    print(f"  3. Tetrachromat Isolation Sweep")
    print(f"     S={S_PEAK}, M={M_PEAK}, L={L_PEAK}")
    print(f"     Q: {args.q_range[0]:.0f} → {args.q_range[1]:.0f} nm "
          f"(step={args.q_step})")
    print(f"     Display: {args.n_primaries} primaries, "
          f"FWHM=[{args.fwhm_min}, {args.fwhm_max}]")
    print("=" * 60)

    q_peaks = np.arange(args.q_range[0], args.q_range[1] + args.q_step / 2,
                        args.q_step)

    sweep = sweep_q_peak(
        q_peaks, n_primaries=args.n_primaries,
        fwhm_min=args.fwhm_min, fwhm_max=args.fwhm_max,
        n_seeds=args.n_seeds, maxiter=args.maxiter, popsize=args.popsize)

    plot_tetrachromat_isolation(sweep, output_dir)
    plot_optimal_led_positions(sweep, output_dir)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print(f"  SUMMARY (d' threshold = {args.threshold})")
    print("=" * 60)

    # Find Q positions above/below threshold
    above = sweep['d_primes'] >= args.threshold
    if np.all(above):
        print(f"  All Q positions achieve d' >= {args.threshold}")
    elif np.any(above):
        # Find boundaries
        transitions = np.diff(above.astype(int))
        for idx in np.where(transitions == 1)[0]:
            print(f"  d' crosses {args.threshold} upward near Q = {q_peaks[idx+1]:.1f} nm")
        for idx in np.where(transitions == -1)[0]:
            print(f"  d' crosses {args.threshold} downward near Q = {q_peaks[idx]:.1f} nm")
    else:
        print(f"  NO Q position achieves d' >= {args.threshold} in this range")

    # Peak d'
    best_idx = np.argmax(sweep['d_primes'])
    best_idx_res = np.argmax(sweep['residual_frac'])
    print(f"\n  Best display-mediated: Q = {q_peaks[best_idx]:.1f} nm, "
          f"d' = {sweep['d_primes'][best_idx]:.2f}")
    print(f"    residual = {sweep['residual_frac'][best_idx]:.6f}")
    print(f"    ρ(Q,M) = {sweep['rho_qm'][best_idx]:.5f}, "
          f"ρ(Q,L) = {sweep['rho_ql'][best_idx]:.5f}")
    print(f"    LEDs: [{', '.join(f'{p:.0f}' for p in sweep['led_peaks'][best_idx])}]")

    print(f"\n  Max residual (display-free peak): Q = {q_peaks[best_idx_res]:.1f} nm, "
          f"residual = {sweep['residual_frac'][best_idx_res]:.6f}")

    # Known genotype positions
    print(f"\n  Known genotype Q-cone peaks:")
    known_q = {
        'M variants': [530, 533, 536],
        'L variants': [547, 551, 552, 553, 555, 556, 556.5, 559],
    }
    for group, peaks_list in known_q.items():
        print(f"    {group}:")
        for qp in peaks_list:
            idx = np.argmin(np.abs(q_peaks - qp))
            if abs(q_peaks[idx] - qp) < args.q_step:
                d = sweep['d_primes'][idx]
                res = sweep['residual_frac'][idx]
                marker = ' *' if d < args.threshold else ''
                print(f"      Q={qp:>5.1f}nm  d'={d:>6.2f}  "
                      f"residual={res:.6f}  "
                      f"ρ(Q,M)={sweep['rho_qm'][idx]:.5f}  "
                      f"ρ(Q,L)={sweep['rho_ql'][idx]:.5f}{marker}")

    # Save JSON
    summary = {
        'trichromat_peaks': {'S': S_PEAK, 'M': M_PEAK, 'L': L_PEAK},
        'n_primaries': args.n_primaries,
        'fwhm_range': [args.fwhm_min, args.fwhm_max],
        'threshold': args.threshold,
        'q_peaks': q_peaks.tolist(),
        'rho_qm': sweep['rho_qm'].tolist(),
        'rho_ql': sweep['rho_ql'].tolist(),
        'residual_frac': sweep['residual_frac'].tolist(),
        'd_prime_ideal': sweep['d_prime_ideal'].tolist(),
        'd_primes': sweep['d_primes'].tolist(),
        'led_peaks': sweep['led_peaks'],
        'led_fwhms': sweep['led_fwhms'],
    }
    with open(output_dir / 'tetrachromat_isolation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {output_dir / 'tetrachromat_isolation_summary.json'}")
    print("\nDone.")


if __name__ == '__main__':
    main()
