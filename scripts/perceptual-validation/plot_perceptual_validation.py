#!/usr/bin/env python3
"""
Generate the perceptual-validation main-results figure.

Layout: 5 genotype conditions (columns) x 2 rows:
  Row A — Simulation (observer sRGB): pseudoisochromatic plate generated
          via TetraPlate's PseudoIsochromaticPlateGenerator, output as sRGB.
          The plate's metamer pair is computed for the condition genotype
          (2-cone ML observer with 3 primaries R,G,O), so it is invisible
          to an observer matching that genotype and visible to others.
  Row B — Hyper-observer bars: cone response for M1 and M2 across the
          12-cone hyperobserver, following the paired-bar style of
          validate_display_measurements.py.

Conditions (columns):
  Cond. 1  (530, 559)   — the target observer's own genotype (matching)
  Cond. 2  (530, 555)
  Cond. 3  (533, 559)
  Cond. 4  (533, 555)
  Cond. 5  (530, 551)

Usage:
    python plot_perceptual_validation.py --primaries-dir <path>
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.linalg import null_space
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TetriumColor.Plotting.PlotStyle import apply_style, DOUBLE_COL
from TetriumColor.Observer.Observer import Observer
from TetriumColor.Observer import Cone, Spectra
from TetriumColor import ColorSpace, ColorSpaceType
from TetriumColor.PsychoPhys.IshiharaPlate import generate_ishihara_plate
from TetriumColor.Measurement.TetriumMeasurementRoutines import load_primaries_from_csv


# ============================================================
# Constants
# ============================================================
TARGET_GENOTYPE = (530, 559)
CONDITIONS = [
    (530, 559),   # matching condition
    (530, 555),
    (533, 559),
    (533, 555),
    (530, 551),
]
CONDITION_LABELS = [
    r'$(530, 559)$',
    r'$(530, 555)$',
    r'$(533, 559)$',
    r'$(533, 555)$',
    r'$(530, 551)$',
]

LANDOLT_DIRS = [
    'landolt_right', 'landolt_up', 'landolt_left', 'landolt_down', 'landolt_up-right',
]

# Hyperobserver cone peaks (ascending order, matching validate_display_measurements.py)
HYPER_PEAKS = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
HYPER_CONE_LABELS = [
    'S\n420', 'M\n530', 'M\n533', 'M\n536',
    'L\n547', 'L\n551', 'L\n552', 'L\n553',
    'L\n555', 'L\n556', 'L\n556.5', 'L\n559',
]

OUTPUT_DIR = Path('output/perceptual_validation')


# ============================================================
# ML null-direction computation (from genetic_test_app)
# ============================================================
def ml_null_direction(genotype_peaks, wavelengths, primaries_rgo):
    """Compute ML null-space direction and gamut midpoint for a genotype.

    With 3 primaries (R,G,O) and 2 ML constraints, there is exactly 1-D
    of freedom — the direction in display space where ML response is constant.
    """
    peaks_sorted = sorted(genotype_peaks)
    m_peak, l_peak = peaks_sorted[0], peaks_sorted[1]

    m_cone = Cone.cone(m_peak, wavelengths=wavelengths, template="neitz", od=0.5)
    l_cone = Cone.cone(l_peak, wavelengths=wavelengths, template="neitz", od=0.5)
    ml_obs = Observer([m_cone, l_cone])

    resp = ml_obs.observe_spectras(primaries_rgo)  # (3, 2)
    ML_mat = resp.T  # (2, 3)

    ns = null_space(ML_mat)
    if ns.shape[1] == 0:
        raise ValueError(f"No null space for {genotype_peaks}")
    null_dir = ns[:, 0]
    null_dir /= np.linalg.norm(null_dir)

    gamut_center = np.array([0.5, 0.5, 0.5])
    ml_target = ML_mat @ gamut_center
    p0 = np.linalg.lstsq(ML_mat, ml_target, rcond=None)[0]

    t_lo, t_hi = -np.inf, np.inf
    for i in range(3):
        if abs(null_dir[i]) > 1e-12:
            t_for_0 = -p0[i] / null_dir[i]
            t_for_1 = (1.0 - p0[i]) / null_dir[i]
            t_lo = max(t_lo, min(t_for_0, t_for_1))
            t_hi = min(t_hi, max(t_for_0, t_for_1))

    t_mid = (t_lo + t_hi) / 2.0
    center = np.clip(p0 + t_mid * null_dir, 0, 1)
    t_max = (t_hi - t_lo) / 2.0
    return null_dir, center, t_max


def compute_plate_colors(null_dir, center, t_max, q_value=1.0):
    """Compute inside/outside DISP colours along the ML null direction."""
    t = q_value * t_max
    color_a = np.clip(center + t * null_dir, 0, 1)
    color_b = np.clip(center - t * null_dir, 0, 1)
    if color_a[2] >= color_b[2]:
        return color_a, color_b
    else:
        return color_b, color_a


# ============================================================
# Plate generation
# ============================================================
def generate_plate_for_condition(
    condition_genotype, render_cs, primaries_rgo, wavelengths,
    secret='landolt_right', image_size=512, seed=42,
):
    """Generate an Ishihara plate whose metamer pair is tuned to condition_genotype.

    Returns (plate_srgb_image, inside_disp, outside_disp).
    """
    null_dir, center, t_max = ml_null_direction(
        condition_genotype, wavelengths, primaries_rgo)
    inside_disp, outside_disp = compute_plate_colors(null_dir, center, t_max, q_value=1.0)

    # Convert DISP → CONE for plate generation (matching genetic_test_app)
    inside_cone = render_cs.convert(inside_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)
    outside_cone = render_cs.convert(outside_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)

    images = generate_ishihara_plate(
        inside_cone=inside_cone,
        outside_cone=outside_cone,
        color_space=render_cs,
        secret=secret,
        image_size=image_size,
        seed=seed,
        lum_noise=0.0,
        s_cone_noise=0.1,
        output_space=ColorSpaceType.SRGB,
        blur_radius=1.0,
        background_color=np.array([0, 0, 0], dtype=int),
    )
    return images[0], inside_disp, outside_disp


# ============================================================
# Hyperobserver responses
# ============================================================
def compute_hyperobserver_responses(
    inside_disp, outside_disp, render_cs, hyperobs,
):
    """Project the two metamer stimuli through the 12-cone hyperobserver.

    Uses the ColorSpace's internal disp_to_cone matrix (which includes
    the white-point scaling) to reconstruct the actual spectra, then
    projects through the hyperobserver.

    inside_disp / outside_disp are (R,G,O) display weights.
    Returns (hyper_m1, hyper_m2).
    """
    # Go DISP → CONE in the render observer's space.  The cone_to_disp
    # matrix encodes the scaled primaries, so its inverse (disp_to_cone)
    # gives us the correct cone values that account for the scaling.
    cone_to_disp = render_cs._get_cone_to_disp()  # (n_cones, n_primaries)
    disp_to_cone = np.linalg.inv(cone_to_disp)     # (n_primaries, n_cones)

    # Cone responses in the *render* observer's space
    cone_inside = disp_to_cone @ inside_disp
    cone_outside = disp_to_cone @ outside_disp

    # Now we need the actual spectra.  The render observer's sensor_matrix
    # is (n_cones, n_wl).  We can recover the spectrum by using the scaled
    # display primary matrix.  The relationship is:
    #   cone = (sensor_matrix @ scaled_primary_matrix.T) @ disp_weights
    # So: scaled_primary_matrix.T = sensor_matrix^+ @ (sensor -> cone mapping)
    #
    # But more directly: the ColorSpace stores display_primaries, and the
    # scaling_factor is embedded.  We can reconstruct the scaled primaries
    # from: disp_to_cone = (sensor_matrix @ (scaling * primary_matrix).T)^-1
    # So:  sensor_matrix @ (scaling * primary_matrix).T = inv(disp_to_cone) = cone_to_disp^-1... no.
    #
    # Actually: disp_to_cone = inv(cone_to_disp), and
    #   cone_to_disp = inv(sensor_matrix @ scaled_primary_matrix.T)
    # Therefore: sensor_matrix @ scaled_primary_matrix.T = inv(cone_to_disp)
    #   scaled_primary_matrix.T = sensor_matrix^{-1} @ inv(cone_to_disp) ... only square case
    #
    # Simpler approach: we have the render observer and its scaling.  Just use
    # observe_spectras on the *raw* primaries, then apply the same scaling
    # factor to get the effective primary spectra.

    # The scaling factor is stored in the metadata when the cone_to_disp was computed.
    # Let's reconstruct the scaled disp-to-cone matrix from the sensor matrix and raw primaries.
    primaries = render_cs.display_primaries
    sensor_matrix = render_cs.observer.sensor_matrix  # (n_cones, n_wl)
    raw_primary_matrix = np.array([p.data for p in primaries])  # (n_primaries, n_wl)

    # The intensities matrix is sensor_matrix @ raw_primary_matrix.T, possibly scaled.
    # cone = intensities_scaled @ disp_weights
    # We know cone values, we know disp weights.  The actual spectrum is:
    #   spectrum = disp_weights @ raw_primary_matrix  (unscaled)
    # The scaling only affects the cone-to-disp mapping, not the physical spectrum.
    # The issue is that raw primary spectra are in absolute radiometric units,
    # and the DISP weights are in the *scaled* system.
    #
    # Let's just invert: find the actual scaling by comparing.
    raw_intensities = sensor_matrix @ raw_primary_matrix.T  # (n_cones, n_primaries)
    # cone_to_disp_unscaled = inv(raw_intensities) (for square case)
    # The actual cone_to_disp has scaling baked in.
    # So: cone_to_disp = inv(raw_intensities * s) = inv(raw_intensities) / s
    # Therefore: s = inv(raw_intensities) / cone_to_disp ... element-wise doesn't work.
    #
    # Better: cone_to_disp = inv(s * raw_intensities)
    # So: s * raw_intensities = inv(cone_to_disp)
    # s * raw_intensities = disp_to_cone^{-1}  no... cone_to_disp maps cone→disp.
    # disp_to_cone = inv(cone_to_disp)
    # And: disp_to_cone = scaled_intensities = s * raw_intensities
    # So: scaled_primary_matrix = s * raw_primary_matrix where s makes
    #     sensor_matrix @ scaled_primary_matrix.T = disp_to_cone^{-1} ... hmm
    #
    # Let me just compute it directly:
    # disp_to_cone @ disp_weights = cone_responses (in the render observer)
    # spectrum = disp_weights @ (some_scaled_primary_matrix)
    # where sensor_matrix @ spectrum = cone_responses
    #
    # Actually the simplest correct approach: the disp_to_cone matrix IS
    # the scaled intensities matrix.  So the "effective" primary-to-cone
    # matrix is inv(cone_to_disp).  And spectrum = disp_weights @ scaled_primaries
    # where scaled_primaries = raw_primaries * column_scales such that
    # sensor_matrix @ scaled_primaries.T = inv(cone_to_disp).
    #
    # scaled_primaries.T = sensor_matrix^{-1} @ inv(cone_to_disp)  (square case)
    # = sensor_matrix^{-1} @ disp_to_cone^{-1}  ... no.
    #
    # OK let me just be pragmatic.  We have:
    #   cone_to_disp (n_cones x n_primaries)
    #   inv(cone_to_disp) = disp_to_cone = the effective "how much cone per unit disp weight"
    #
    # The actual spectrum for display weight w is:
    #   spectrum = w @ (diag(scale_factors) @ raw_primary_matrix)
    # And the cone response is:
    #   sensor_matrix @ spectrum = sensor_matrix @ raw_primary_matrix.T @ diag(scale_factors) @ w
    # This should equal disp_to_cone @ w for the render observer.
    # So: sensor_matrix @ raw_primary_matrix.T @ diag(scale_factors) = disp_to_cone^T
    # Wait no: = (the matrix that maps disp weights to cones)
    # That matrix is inv(cone_to_disp).
    # sensor_matrix @ raw_primary_matrix.T @ diag(s) = inv(cone_to_disp)
    # diag(s) = (sensor_matrix @ raw_primary_matrix.T)^{-1} @ inv(cone_to_disp)
    # = inv(raw_intensities) @ inv(cone_to_disp)

    inv_raw = np.linalg.inv(raw_intensities)
    inv_ctd = np.linalg.inv(cone_to_disp)
    scale_diag = inv_raw @ inv_ctd  # should be ~diagonal

    # The scale factors are the diagonal of scale_diag
    scale_factors = np.diag(scale_diag)

    # Reconstruct actual spectra
    scaled_primary_matrix = np.diag(scale_factors) @ raw_primary_matrix
    spd_inside = inside_disp @ scaled_primary_matrix
    spd_outside = outside_disp @ scaled_primary_matrix

    wl = primaries[0].wavelengths
    spec_inside = Spectra(wavelengths=wl, data=spd_inside)
    spec_outside = Spectra(wavelengths=wl, data=spd_outside)

    hyper_m1 = hyperobs.observe_spectras([spec_inside])[0]
    hyper_m2 = hyperobs.observe_spectras([spec_outside])[0]
    return hyper_m1, hyper_m2


def plot_hyperobserver_bars_on_ax(
    ax, hyper_m1, hyper_m2, condition_genotype, is_first=False,
):
    """Plot M1/M2 hyperobserver cone responses as paired bars,
    following validate_display_measurements.py style."""
    x = np.arange(len(HYPER_PEAKS))
    w = 0.35

    ax.bar(x - w/2, hyper_m1, w, color='steelblue', alpha=0.85,
           edgecolor='black', linewidth=0.3, label='M1')
    ax.bar(x + w/2, hyper_m2, w, color='indianred', alpha=0.85,
           edgecolor='black', linewidth=0.3, label='M2')

    # Shade S / M / L regions
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='blue')
    ax.axvspan(0.5, 3.5, alpha=0.06, color='green')
    ax.axvspan(3.5, 11.5, alpha=0.06, color='red')

    # Bold labels for designed cones
    designed_set = set(condition_genotype) | {420}
    ax.set_xticks(x)
    ax.set_xticklabels(HYPER_CONE_LABELS, fontsize=3.5)
    for tick, peak in zip(ax.get_xticklabels(), HYPER_PEAKS):
        if peak in designed_set:
            tick.set_fontweight('bold')

    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelsize=5)

    if is_first:
        ax.legend(fontsize=5, loc='upper left')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Generate perceptual validation figure')
    parser.add_argument('--primaries-dir', type=str, required=True,
                        help='Path to measured primaries directory (RGBO order)')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    apply_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load display primaries (RGBO order)
    print(f"Loading primaries from: {args.primaries_dir}")
    primaries_rgbo = load_primaries_from_csv(
        args.primaries_dir, extract_zero=False, primary_order='RGBO')
    print(f"  Loaded {len(primaries_rgbo)} primaries")

    # Use R, G, O for 3-primary plate generation (indices 0, 1, 3)
    primaries_rgo = [primaries_rgbo[0], primaries_rgbo[1], primaries_rgbo[3]]

    wavelengths = primaries_rgbo[0].wavelengths

    # Build standard trichromat + 3 primaries ColorSpace for plate rendering
    render_observer = Observer.trichromat(wavelengths=wavelengths)
    render_cs = ColorSpace(render_observer, display_primaries=primaries_rgo)

    # Build hyperobserver
    hyperobs = Observer.hyperobserver(wavelengths=wavelengths)

    n_cond = len(CONDITIONS)

    # Generate plates and compute hyperobserver responses
    print("\nGenerating plates and computing responses...")
    plate_images = []
    hyper_data = []  # list of (hyper_m1, hyper_m2)

    for ci, (cond, secret) in enumerate(zip(CONDITIONS, LANDOLT_DIRS)):
        print(f"  Condition {ci+1}: {cond} ({secret})")

        plate_img, inside_disp, outside_disp = generate_plate_for_condition(
            cond, render_cs, primaries_rgo, wavelengths,
            secret=secret, image_size=args.image_size, seed=42 + ci)
        plate_images.append(plate_img)

        # Save individual plate
        plate_path = output_dir / f'plate_cond{ci+1}.png'
        plate_img.save(str(plate_path))

        # Hyperobserver responses
        hyper_m1, hyper_m2 = compute_hyperobserver_responses(
            inside_disp, outside_disp, render_cs, hyperobs)
        hyper_data.append((hyper_m1, hyper_m2))

        abs_diff = np.abs(hyper_m1 - hyper_m2)
        print(f"    DISP inside:  [{', '.join(f'{v:.3f}' for v in inside_disp)}]")
        print(f"    DISP outside: [{', '.join(f'{v:.3f}' for v in outside_disp)}]")
        print(f"    Hyper |M1-M2| max: {abs_diff.max():.6f}")

    # ============================================================
    # Build figure: 2 rows x 5 columns
    # ============================================================
    print("\nBuilding figure...")
    n_rows = 2
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))
    gs = GridSpec(n_rows, n_cond, figure=fig,
                  height_ratios=[1, 0.7],
                  hspace=0.3, wspace=0.08,
                  left=0.06, right=0.98, top=0.88, bottom=0.08)

    # Global y-max for bar charts
    global_ymax = 0
    for hm1, hm2 in hyper_data:
        global_ymax = max(global_ymax, hm1.max(), hm2.max())
    global_ymax *= 1.1

    for ci in range(n_cond):
        # --- Row A: Plate sRGB ---
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(np.array(plate_images[ci]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Cond.~{ci+1}\n{CONDITION_LABELS[ci]}', fontsize=7, pad=3)
        if ci == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('#d62728')
                spine.set_linewidth(2)

        # --- Row B: Hyperobserver bars ---
        ax = fig.add_subplot(gs[1, ci])
        hyper_m1, hyper_m2 = hyper_data[ci]
        plot_hyperobserver_bars_on_ax(
            ax, hyper_m1, hyper_m2, CONDITIONS[ci], is_first=(ci == 0))
        ax.set_ylim(0, global_ymax)
        if ci > 0:
            ax.set_yticks([])

    # Row labels
    for ri, label in enumerate(['Simulation', 'Hyper-obs.']):
        ax = fig.axes[ri * n_cond]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, fontsize=7, rotation=90,
                 ha='center', va='center', fontweight='bold')

    for suffix in ['pdf', 'png']:
        fig.savefig(output_dir / f'perceptual_validation.{suffix}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'perceptual_validation.pdf'}")
    print(f"Saved: {output_dir / 'perceptual_validation.png'}")


if __name__ == '__main__':
    main()
