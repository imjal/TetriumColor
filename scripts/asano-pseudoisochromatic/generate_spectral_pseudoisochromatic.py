#!/usr/bin/env python3
"""
Spectral Pseudoisochromatic Image Generation
=============================================
Implements the procedure from Yuta Asano's PhD Dissertation (RIT, 2015), Section 4.2.

Creates spectral pseudoisochromatic images to classify color-normal observers
into categorical groups. Each image targets one categorical observer: the target
can distinguish the hidden number while other observers cannot.

Procedure (per the dissertation):
  1. Define 10 categorical observers with different LMS cone fundamentals
  2. Generate Gaussian SPDs for number layer and 3 background layers
  3. Enumerate valid 3-primary combinations for the number layer (Eq 4.5-4.6)
  4. For each combination, optimize 18 background parameters via nonlinear optimization
  5. Select parameters that maximize ΔE₀₀(target) subject to max(ΔE₀₀(rest)) ≤ 5
  6. Visualize as pseudo-sRGB for each observer's CMFs

Uses:
  - TetriumColor: Observer cone fundamentals (Govardovskii nomogram + pre-receptoral filtering)
  - colour-science: CIE 1931 CMFs, XYZ↔Lab, sRGB conversions

Usage:
    python generate_spectral_pseudoisochromatic.py --target 5
    python generate_spectral_pseudoisochromatic.py --target all --n-starts 50
"""

import sys
import os
import io
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import colour
from colour import XYZ_to_Lab, XYZ_to_sRGB, XYZ_to_xy, MSDS_CMFS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from TetriumColor.Observer.Observer import Observer, Cone, GovardovskiiNomogram
from TetriumColor.Observer.Spectra import Spectra, Illuminant

# ============================================================
# Configuration
# ============================================================
WAVELENGTHS = np.arange(400, 701, 1)
SIGMA = 15.0                    # Gaussian σ [nm]; HWHM ≈ σ√(2 ln2) ≈ 17.7 nm
CANDIDATE_PEAKS = np.array([400, 440, 480, 520, 560, 600, 640, 680])
L_TARGET = 50                   # CIELAB L* for the number layer
DE_THRESHOLD = 5.0              # max ΔE₀₀ for non-target observers (Sec 4.2.2)
KL = 2                          # lightness parametric factor in CIEDE2000

# 10 categorical observer definitions
# (s_peak, m_peak, l_peak, od_s, od_lm, lens, macular)
# Modeled after the CIEPO06 individual observer model parameters:
#   - Cone λmax shifts (genetic L/M polymorphism, S cone variation)
#   - Photopigment optical density (field-size / eccentricity dependent)
#   - Lens pigment density (age dependent)
#   - Macular pigment density (individual variation)
CATEGORICAL_OBSERVER_PARAMS = [
    # Cat 1: Standard (CIEPO06 age 38, 2° baseline)
    {"s_peak": 420, "m_peak": 530, "l_peak": 559, "od_s": 0.40, "od_lm": 0.50, "lens": 1.0, "macular": 1.0},
    # Cat 2: L-cone λmax shifted −3 nm (Ser180 → Ala180 polymorphism)
    {"s_peak": 420, "m_peak": 530, "l_peak": 556, "od_s": 0.40, "od_lm": 0.50, "lens": 1.0, "macular": 1.0},
    # Cat 3: M-cone λmax shifted +3 nm
    {"s_peak": 420, "m_peak": 533, "l_peak": 559, "od_s": 0.40, "od_lm": 0.50, "lens": 1.0, "macular": 1.0},
    # Cat 4: Both M and L shifted
    {"s_peak": 420, "m_peak": 533, "l_peak": 556, "od_s": 0.40, "od_lm": 0.50, "lens": 1.0, "macular": 1.0},
    # Cat 5: Higher lens density (≈ age 55)
    {"s_peak": 420, "m_peak": 530, "l_peak": 559, "od_s": 0.40, "od_lm": 0.50, "lens": 1.30, "macular": 1.0},
    # Cat 6: Larger effective field / lower OD + macular
    {"s_peak": 420, "m_peak": 530, "l_peak": 559, "od_s": 0.30, "od_lm": 0.38, "lens": 1.0, "macular": 0.30},
    # Cat 7: L-cone shifted −6 nm + slight lens increase
    {"s_peak": 420, "m_peak": 530, "l_peak": 553, "od_s": 0.40, "od_lm": 0.48, "lens": 1.15, "macular": 1.0},
    # Cat 8: M-cone shifted +6 nm
    {"s_peak": 420, "m_peak": 536, "l_peak": 559, "od_s": 0.40, "od_lm": 0.50, "lens": 1.0, "macular": 1.0},
    # Cat 9: Younger observer (lower lens + OD)
    {"s_peak": 418, "m_peak": 530, "l_peak": 559, "od_s": 0.35, "od_lm": 0.45, "lens": 0.85, "macular": 0.80},
    # Cat 10: Combined physiological shifts
    {"s_peak": 420, "m_peak": 533, "l_peak": 553, "od_s": 0.38, "od_lm": 0.48, "lens": 1.10, "macular": 0.90},
]


# ============================================================
# Observer Construction
# ============================================================
def _make_cone(peak: float, od: float, lens: float, macular: float) -> Cone:
    """Create a cone with explicit physiological parameters via the Govardovskii nomogram."""
    raw = GovardovskiiNomogram(WAVELENGTHS, peak)
    return raw.with_preceptoral(od=od, macular=macular, lens=lens)


def create_categorical_observers() -> List[Observer]:
    """Instantiate the 10 categorical observers."""
    observers = []
    for p in CATEGORICAL_OBSERVER_PARAMS:
        s = _make_cone(p["s_peak"], p["od_s"], p["lens"], p["macular"])
        m = _make_cone(p["m_peak"], p["od_lm"], p["lens"], p["macular"])
        l = _make_cone(p["l_peak"], p["od_lm"], p["lens"], p["macular"])
        observers.append(Observer([s, m, l], illuminant=Illuminant.get('D65')))
    return observers


# ============================================================
# Spectral Primitives
# ============================================================
def gaussian_spd(peaks: npt.NDArray, scalars: npt.NDArray,
                 sigma: float = SIGMA) -> npt.NDArray:
    """Sum-of-Gaussians SPD (Eq 4.2).

    S(λ) = Σᵢ cᵢ exp(−(λ − pᵢ)² / (2σ²))
    """
    spd = np.zeros(len(WAVELENGTHS), dtype=float)
    for p, c in zip(peaks, scalars):
        spd += c * np.exp(-(WAVELENGTHS - p) ** 2 / (2 * sigma ** 2))
    return spd


def precompute_basis_spectra() -> npt.NDArray:
    """Unit-amplitude Gaussian at each candidate peak.  Shape: (n_peaks, n_wl)."""
    return np.array([np.exp(-(WAVELENGTHS - pk) ** 2 / (2 * SIGMA ** 2))
                     for pk in CANDIDATE_PEAKS])


# ============================================================
# Observer-Specific XYZ CMFs  (Eq 4.3–4.4)
# ============================================================
_CIE1931_CACHE = None

def _cie1931_on_grid() -> npt.NDArray:
    """CIE 1931 2° standard observer, interpolated to WAVELENGTHS. Shape: (3, n_wl)."""
    global _CIE1931_CACHE
    if _CIE1931_CACHE is not None:
        return _CIE1931_CACHE
    cie = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    T = np.zeros((3, len(WAVELENGTHS)))
    for j, wl in enumerate(WAVELENGTHS):
        T[:, j] = cie[wl]
    _CIE1931_CACHE = T
    return T


def observer_xyz_cmfs(obs: Observer) -> npt.NDArray:
    """Compute observer-specific XYZ CMFs via Eq 4.3–4.4.

    Txyz_i = M_i · Tlms_i,  where  M_i = T1931 · pinv(Tlms_i)

    Returns: (3, n_wl) array.
    """
    T_lms = obs.sensor_matrix                     # (3, n_wl)
    T_1931 = _cie1931_on_grid()                    # (3, n_wl)
    M = T_1931 @ np.linalg.pinv(T_lms)            # (3, 3)
    return M @ T_lms                               # (3, n_wl)


# ============================================================
# Reference White
# ============================================================
def reference_white_spd() -> npt.NDArray:
    """Phosphor-converted white LED (Figure 4.8 analogue), peak-normalized to 1."""
    sd = colour.SDS_LIGHT_SOURCES['Phosphor LED YAG']
    spd = np.array([sd[wl] if sd.wavelengths[0] <= wl <= sd.wavelengths[-1] else 0.0
                    for wl in WAVELENGTHS])
    return spd / spd.max()


# ============================================================
# Vectorized CIEDE2000 with Parametric kL  (CIE 142:2001)
# ============================================================
def ciede2000_batch(Lab1: npt.NDArray, Lab2: npt.NDArray, kL: float = 1.0) -> npt.NDArray:
    """Vectorized CIEDE2000 over arrays of Lab pairs.

    Lab1, Lab2: (..., 3) arrays.  Returns: (...) array of ΔE values.
    """
    Lab1 = np.asarray(Lab1, dtype=float)
    Lab2 = np.asarray(Lab2, dtype=float)
    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0.0,
                   np.where(np.abs(dh) <= 180, dh,
                            np.where(dh > 180, dh - 360, dh + 360)))

    dHp = 2 * np.sqrt(np.maximum(C1p * C2p, 0)) * np.sin(np.radians(dhp / 2))

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    # Average hue
    abs_dh = np.abs(h1p - h2p)
    sum_h = h1p + h2p
    hp_bar = np.where(C1p * C2p == 0, sum_h,
                      np.where(abs_dh <= 180, sum_h / 2.0,
                               np.where(sum_h < 360, (sum_h + 360) / 2.0,
                                        (sum_h - 360) / 2.0)))

    T = (1
         - 0.17 * np.cos(np.radians(hp_bar - 30))
         + 0.24 * np.cos(np.radians(2 * hp_bar))
         + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
         - 0.20 * np.cos(np.radians(4 * hp_bar - 63)))

    SL = 1 + 0.015 * (Lp_bar - 50) ** 2 / np.sqrt(20 + (Lp_bar - 50) ** 2)
    SC = 1 + 0.045 * Cp_bar
    SH = 1 + 0.015 * Cp_bar * T

    d_theta = 30 * np.exp(-((hp_bar - 275) / 25) ** 2)
    Cp_bar7 = Cp_bar ** 7
    RC = 2 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    term_L = dLp / (kL * SL)
    term_C = dCp / SC
    term_H = dHp / SH

    return np.sqrt(np.maximum(0, term_L ** 2 + term_C ** 2 + term_H ** 2
                              + RT * term_C * term_H))


# ============================================================
# Vectorized XYZ → Lab  (avoids looping over colour.XYZ_to_Lab)
# ============================================================
_LAB_EPSILON = (6.0 / 29.0) ** 3
_LAB_KAPPA = (29.0 / 6.0) ** 2 / 3.0
_LAB_OFFSET = 4.0 / 29.0


def _f_lab(t: npt.NDArray) -> npt.NDArray:
    """CIE Lab nonlinearity, vectorized."""
    out = np.empty_like(t)
    mask = t > _LAB_EPSILON
    out[mask] = np.cbrt(t[mask])
    out[~mask] = _LAB_KAPPA * t[~mask] + _LAB_OFFSET
    return out


def _xyz_to_lab_batch(xyz: npt.NDArray, xyz_n: npt.NDArray) -> npt.NDArray:
    """Vectorized XYZ → Lab conversion.

    xyz:   (..., 3)   tristimulus values
    xyz_n: (..., 3)   reference white XYZ (broadcastable)
    Returns: (..., 3)  Lab values
    """
    ratio = xyz / xyz_n
    f = _f_lab(ratio)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ============================================================
# Color Difference Computation  (Figure 4.7)
# ============================================================
def _precompute_observer_data(
    all_xyz_cmfs: List[npt.NDArray],
    white_spd: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Pre-compute stacked CMFs matrix and per-observer white-point XYZ.

    Returns: (cmfs_stack (n_obs, 3, n_wl), xyz_whites (n_obs, 3))
    """
    step = float(WAVELENGTHS[1] - WAVELENGTHS[0])
    cmfs_stack = np.array(all_xyz_cmfs)            # (n_obs, 3, n_wl)
    xyz_whites = cmfs_stack @ white_spd * step     # (n_obs, 3)
    return cmfs_stack, xyz_whites


def compute_delta_e_table(
    spd_number: npt.NDArray,
    spd_backgrounds: List[npt.NDArray],
    all_xyz_cmfs: List[npt.NDArray],
    white_spd: npt.NDArray,
    kL: float = KL,
    _cache: Optional[Tuple] = None,
) -> npt.NDArray:
    """ΔE₀₀ between number and each background, for every observer.

    Returns: (n_observers, n_backgrounds)
    """
    step = float(WAVELENGTHS[1] - WAVELENGTHS[0])

    if _cache is not None:
        cmfs_stack, xyz_whites = _cache
    else:
        cmfs_stack, xyz_whites = _precompute_observer_data(all_xyz_cmfs, white_spd)

    bg_stack = np.array(spd_backgrounds)           # (n_bg, n_wl)
    xyz_num = cmfs_stack @ spd_number * step       # (n_obs, 3)
    xyz_bgs = np.einsum('oij,bj->obi', cmfs_stack, bg_stack) * step  # (n_obs, n_bg, 3)

    # Vectorized XYZ → Lab (no Python loops)
    xyz_w = xyz_whites[:, np.newaxis, :]           # (n_obs, 1, 3)
    lab_num = _xyz_to_lab_batch(xyz_num, xyz_whites)               # (n_obs, 3)
    lab_bgs = _xyz_to_lab_batch(xyz_bgs, xyz_w)                    # (n_obs, n_bg, 3)

    # Vectorized CIEDE2000
    lab_num_exp = lab_num[:, np.newaxis, :]        # (n_obs, 1, 3)
    table = ciede2000_batch(
        np.broadcast_to(lab_num_exp, lab_bgs.shape), lab_bgs, kL=kL)
    return table


# ============================================================
# Number Layer Scalar Computation  (Eq 4.5–4.6)
# ============================================================
def number_layer_scalars(
    basis_indices: npt.NDArray,
    basis: npt.NDArray,
    cat1_cmfs: npt.NDArray,
    white_spd: npt.NDArray,
) -> Optional[npt.NDArray]:
    """Compute mixture scalars so the number layer matches the reference white at L*=L_TARGET.

    C = (T·S)⁻¹ · T·S_w · r,  r = ((L_target + 16)/116)³
    Returns None if the combination is not realizable (negative scalar or singular).
    """
    r = ((L_TARGET + 16) / 116.0) ** 3
    step = float(WAVELENGTHS[1] - WAVELENGTHS[0])

    S = basis[basis_indices].T                       # (n_wl, 3)
    TS = cat1_cmfs @ S * step                        # (3, 3)
    Tw = cat1_cmfs @ white_spd * step                # (3,)

    if np.linalg.matrix_rank(TS) < 3:
        return None
    try:
        C = np.linalg.solve(TS, Tw * r)
    except np.linalg.LinAlgError:
        return None
    if np.any(C < 0):
        return None
    return C


# ============================================================
# Background Layer Optimization  (Sec 4.2.2, step 3)
# ============================================================
def optimize_backgrounds(
    spd_number: npt.NDArray,
    all_xyz_cmfs: List[npt.NDArray],
    white_spd: npt.NDArray,
    target_idx: int,
    n_backgrounds: int = 3,
    n_starts: int = 25,
) -> Tuple[float, Optional[List[npt.NDArray]]]:
    """Find background SPDs that maximize ΔE_min for the target observer
    while keeping ΔE_min for all other observers below DE_THRESHOLD.

    Each background layer: 3 Gaussian peaks + 3 scalars → 6 params × 3 backgrounds = 18.
    """
    n_params = 6 * n_backgrounds
    rest_mask = np.array([i != target_idx for i in range(len(all_xyz_cmfs))])
    rng = np.random.default_rng(42)

    # Pre-compute observer white-point data (avoids recomputing every objective call)
    cache = _precompute_observer_data(all_xyz_cmfs, white_spd)

    best_score = -np.inf
    best_spds: Optional[List[npt.NDArray]] = None

    def _decode(x: npt.NDArray) -> List[npt.NDArray]:
        spds = []
        for b in range(n_backgrounds):
            off = b * 6
            peaks = x[off:off + 3]
            scalars = np.abs(x[off + 3:off + 6])
            spds.append(gaussian_spd(peaks, scalars))
        return spds

    def _objective(x: npt.NDArray) -> float:
        bg_spds = _decode(x)
        dE = compute_delta_e_table(spd_number, bg_spds, all_xyz_cmfs, white_spd,
                                   _cache=cache)
        dE_min = dE.min(axis=1)
        target_val = dE_min[target_idx]
        rest_max = dE_min[rest_mask].max()
        penalty = 100.0 * max(0, rest_max - DE_THRESHOLD) ** 2
        return -target_val + penalty

    for _ in range(n_starts):
        x0 = np.zeros(n_params)
        for b in range(n_backgrounds):
            off = b * 6
            x0[off:off + 3] = rng.uniform(400, 680, 3)
            x0[off + 3:off + 6] = rng.uniform(0.01, 1.0, 3)

        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 0.5, 'fatol': 0.1})
        except Exception:
            continue

        bg_spds = _decode(res.x)
        dE = compute_delta_e_table(spd_number, bg_spds, all_xyz_cmfs, white_spd,
                                   _cache=cache)
        dE_min = dE.min(axis=1)
        target_val = float(dE_min[target_idx])
        rest_max = float(dE_min[rest_mask].max())

        score = target_val if rest_max <= DE_THRESHOLD * 1.5 else -np.inf
        if score > best_score:
            best_score = score
            best_spds = bg_spds

    return best_score, best_spds


# ============================================================
# Visualization  (Sec 4.2.2 "Visualization")
# ============================================================
def render_plate_for_observer(
    spd_number: npt.NDArray,
    spd_backgrounds: List[npt.NDArray],
    xyz_cmfs: npt.NDArray,
    white_spd: npt.NDArray,
    image_size: int = 300,
    seed: int = 0,
) -> npt.NDArray:
    """Render a simplified pseudoisochromatic patch as an RGB image array.

    Layout: 3 horizontal background bands with a central number circle.
    Colors are pseudo-sRGB derived from the observer's xyz-CMFs.
    """
    step = float(WAVELENGTHS[1] - WAVELENGTHS[0])
    xyz_w = xyz_cmfs @ white_spd * step
    xy_w = XYZ_to_xy(xyz_w)

    def _to_srgb(spd):
        xyz = xyz_cmfs @ spd * step
        return np.clip(XYZ_to_sRGB(xyz / xyz_w[1], illuminant=xy_w), 0, 1)

    rgb_num = _to_srgb(spd_number)
    rgb_bgs = [_to_srgb(bg) for bg in spd_backgrounds]

    img = np.zeros((image_size, image_size, 3))
    band_h = image_size // len(rgb_bgs)
    for k, rgb in enumerate(rgb_bgs):
        y0 = k * band_h
        y1 = (k + 1) * band_h if k < len(rgb_bgs) - 1 else image_size
        img[y0:y1, :] = rgb

    # Luminance noise mask
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.03, (image_size, image_size, 1))
    img = np.clip(img + noise, 0, 1)

    cy, cx, r = image_size // 2, image_size // 2, image_size // 5
    yy, xx = np.ogrid[:image_size, :image_size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    img[mask] = rgb_num + rng.normal(0, 0.03, (mask.sum(), 3))
    img = np.clip(img, 0, 1)

    return img


def visualize_results(
    spd_number: npt.NDArray,
    spd_backgrounds: List[npt.NDArray],
    all_xyz_cmfs: List[npt.NDArray],
    white_spd: npt.NDArray,
    target_idx: int,
    dE_table: npt.NDArray,
    output_path: Optional[str] = None,
):
    """2×5 grid showing pseudo-sRGB appearance for each categorical observer."""
    n_obs = len(all_xyz_cmfs)
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[4, 4, 2], hspace=0.3, wspace=0.15)
    fig.suptitle(f'Spectral Pseudoisochromatic Image — Target: Cat {target_idx + 1}',
                 fontsize=16, fontweight='bold')

    for i in range(n_obs):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, col])
        img = render_plate_for_observer(spd_number, spd_backgrounds,
                                        all_xyz_cmfs[i], white_spd, seed=42)
        ax.imshow(img)
        is_target = (i == target_idx)
        title = f'Cat {i + 1}'
        if is_target:
            title += ' ★'
        ax.set_title(title, fontsize=11, fontweight='bold' if is_target else 'normal',
                     color='red' if is_target else 'black')
        ax.axis('off')

    # ΔE table at the bottom
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    col_labels = [f'Cat {i+1}' for i in range(n_obs)]
    row_labels = [f'Bkgr {j+1}' for j in range(dE_table.shape[1])]
    cell_text = [[f'{dE_table[i, j]:.1f}' for i in range(n_obs)] for j in range(dE_table.shape[1])]

    tbl = ax_table.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    # Highlight target column
    for key, cell in tbl.get_celld().items():
        row, col = key
        if col == target_idx:
            cell.set_facecolor('#ffe0e0')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_spds(
    spd_number: npt.NDArray,
    spd_backgrounds: List[npt.NDArray],
    white_spd: npt.NDArray,
    target_idx: int,
    output_path: Optional[str] = None,
):
    """Plot the optimized SPDs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(WAVELENGTHS, spd_number, 'k-', linewidth=2, label='Number')
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for j, bg in enumerate(spd_backgrounds):
        ax.plot(WAVELENGTHS, bg, color=colors[j], linewidth=1.5, label=f'Background {j+1}')
    ax.plot(WAVELENGTHS, white_spd * 0.3, 'k--', alpha=0.4, linewidth=1, label='White LED (scaled)')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Relative Intensity')
    ax.set_title(f'Optimized SPDs — Target Cat {target_idx + 1}')
    ax.legend()
    ax.set_xlim(400, 700)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate spectral pseudoisochromatic images (Asano Diss. §4.2)')
    parser.add_argument('--target', type=str, default='5',
                        help='Target observer index (1-10) or "all"')
    parser.add_argument('--n-starts', type=int, default=25,
                        help='Random restarts per optimization (dissertation: 125)')
    parser.add_argument('--output-dir', type=str, default='output/asano_pseudoisochromatic',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build observers ----
    print("Creating 10 categorical observers...")
    observers = create_categorical_observers()
    for i, obs in enumerate(observers):
        p = CATEGORICAL_OBSERVER_PARAMS[i]
        print(f"  Cat {i+1}: S={p['s_peak']} M={p['m_peak']} L={p['l_peak']}  "
              f"OD_s={p['od_s']} OD_lm={p['od_lm']}  lens={p['lens']} mac={p['macular']}")

    print("\nComputing observer-specific XYZ CMFs (Eq 4.3–4.4)...")
    all_cmfs = [observer_xyz_cmfs(obs) for obs in observers]

    # ---- Reference white + basis ----
    white_spd = reference_white_spd()
    basis = precompute_basis_spectra()
    cat1_cmfs = all_cmfs[0]

    # ---- Enumerate valid number-layer combinations (Eq 4.5–4.6) ----
    print("\nEnumerating number-layer primary combinations...")
    all_combos = list(combinations(range(len(CANDIDATE_PEAKS)), 3))
    valid_combos: List[Tuple[npt.NDArray, npt.NDArray]] = []
    for combo in all_combos:
        idx = np.array(combo)
        C = number_layer_scalars(idx, basis, cat1_cmfs, white_spd)
        if C is not None:
            valid_combos.append((idx, C))
    print(f"  {len(valid_combos)} realizable out of {len(all_combos)} "
          f"(dissertation found 18)")

    # ---- Targets ----
    if args.target.lower() == 'all':
        targets = list(range(len(observers)))
    else:
        targets = [int(args.target) - 1]

    # ---- Run optimization for each target ----
    for t_idx in targets:
        print(f"\n{'=' * 65}")
        print(f"  Target observer: Cat {t_idx + 1}")
        print(f"{'=' * 65}")

        overall_best = -np.inf
        overall_spd_num: Optional[npt.NDArray] = None
        overall_spd_bgs: Optional[List[npt.NDArray]] = None

        for ci, (combo_idx, scalars) in enumerate(valid_combos):
            peaks = CANDIDATE_PEAKS[combo_idx]
            spd_num = gaussian_spd(peaks, scalars)
            print(f"  [{ci+1:2d}/{len(valid_combos)}] peaks={peaks}  "
                  f"scalars=[{', '.join(f'{s:.3f}' for s in scalars)}]", end="")

            score, bg_spds = optimize_backgrounds(
                spd_num, all_cmfs, white_spd, t_idx, n_starts=args.n_starts)

            print(f"  → ΔE_min(target)={score:.1f}")

            if score > overall_best:
                overall_best = score
                overall_spd_num = spd_num
                overall_spd_bgs = bg_spds

        if overall_spd_bgs is None:
            print(f"  ✗ No valid solution found for Cat {t_idx + 1}")
            continue

        # ---- Results ----
        dE = compute_delta_e_table(overall_spd_num, overall_spd_bgs,
                                   all_cmfs, white_spd)
        dE_min = dE.min(axis=1)

        print(f"\n  Best ΔE_min(target=Cat {t_idx+1}) = {dE_min[t_idx]:.1f}")
        print(f"\n  ΔE₀₀ Table:")
        header = "  Bkgr " + "".join(f"  Cat{i+1:02d}" for i in range(len(observers)))
        print(header)
        for j in range(dE.shape[1]):
            row = f"    {j+1}  "
            for i in range(dE.shape[0]):
                row += f"  {dE[i, j]:5.1f}"
            print(row)
        print(f"  min   " + "".join(f"  {dE_min[i]:5.1f}" for i in range(len(observers))))

        # ---- Save & visualize ----
        np.savez(output_dir / f"spds_cat{t_idx+1}.npz",
                 wavelengths=WAVELENGTHS,
                 spd_number=overall_spd_num,
                 spd_backgrounds=np.array(overall_spd_bgs),
                 white_spd=white_spd,
                 delta_e_table=dE)

        visualize_results(overall_spd_num, overall_spd_bgs, all_cmfs, white_spd,
                          t_idx, dE, str(output_dir / f"plate_cat{t_idx+1}.png"))
        plot_spds(overall_spd_num, overall_spd_bgs, white_spd, t_idx,
                  str(output_dir / f"spds_cat{t_idx+1}.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
