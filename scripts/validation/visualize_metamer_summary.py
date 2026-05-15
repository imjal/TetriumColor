#!/usr/bin/env python3
"""
Visualize TetraColorPicker metamer pairs as paper figures.

The paper-facing hyperobserver figure shows one row per top genotype:
spectra on the left and hyperobserver difference bars on the right.

Uses the top trichromat genotypes by probability. For observers present in
the metamer config, uses the stored TetraColorPicker pair. For others, computes
a metamer pair on the fly via null-space of the observer's transfer matrix.

Usage:
    python visualize_metamer_summary.py \
        --primaries measurements/2026-03-04/validation_2026-03-04_13-44-34/primaries \
        --plots-dir output/metamer_summary

    python visualize_metamer_summary.py \
        --primaries measurements/2026-03-04/validation_2026-03-04_16-42-45/primaries \
        --measurements measurements/2026-03-04/validation_2026-03-04_16-42-45/ \
        --plots-dir output/metamer_summary --mode gaussian
"""

import argparse
import csv
import json
import numpy as np
from scipy.linalg import null_space
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TetriumColor.Measurement import load_primaries_from_csv, get_spectras_from_rgbo_list
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.Plotting.PlotStyle import (
    apply_style,
    DOUBLE_COL,
    PAPER_BLUE,
    PAPER_RED,
    PAPER_NEUTRAL,
)

VALIDATION_OBSERVER_DEGREE = 2.0
SUMMARY_FIG_HEIGHT = 5.0

# Hyperobserver cone info
HYPER_PEAKS = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
HYPER_CONE_LABELS = [
    '420', '530', '533', '536',
    '547', '551', '552', '553',
    '555', '556', '556.5', '559',
]


def _resolve_metameric_axis(genotype, requested_axis: int, peak_to_test: float = 547) -> int:
    peaks_with_s = sorted(set([420] + list(genotype)))
    if peak_to_test in peaks_with_s:
        return peaks_with_s.index(peak_to_test)
    return requested_axis


def _compute_metamer_pair(observer, primaries, color_space, metameric_axis=2):
    """Compute a metamer pair via null-space of the trichromatic transfer matrix.

    Uses only the non-metameric cones (drops the Q cone at metameric_axis) so that
    a 4-primary display has a 1D null space for a 3-cone trichromat.

    Returns (cone_1, cone_2, bgor_1, bgor_2) or None if no null space exists.
    """
    wavelengths = primaries[0].wavelengths
    n_p = len(primaries)

    # Build display matrix
    D = np.zeros((len(wavelengths), n_p))
    for i, p in enumerate(primaries):
        D[:, i] = p.interpolate_values(wavelengths).data

    # Use only the trichromatic cones (drop metameric axis)
    sensor_3 = np.delete(observer.sensor_matrix, metameric_axis, axis=0)

    # Normalized transfer: C such that C @ ones ≈ ones
    C_raw = sensor_3 @ D
    white = C_raw @ np.ones(n_p)
    if np.any(white <= 0):
        return None
    C_norm = C_raw / white[:, np.newaxis]

    ns = null_space(C_norm)
    if ns.shape[1] == 0:
        return None

    null_dir = ns[:, 0]
    w_base = np.full(n_p, 0.5)
    abs_dir = np.abs(null_dir)
    ratios = np.where(abs_dir > 1e-15, 0.5 / abs_dir, np.inf)
    t = ratios.min()

    bgor_1 = np.clip(w_base + t * null_dir, 0, 1)
    bgor_2 = np.clip(w_base - t * null_dir, 0, 1)

    # Convert to cone space (full 4D)
    cone_1 = color_space.convert(bgor_1.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
    cone_2 = color_space.convert(bgor_2.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

    return cone_1, cone_2, bgor_1, bgor_2


def _render_plate_srgb(trichromat_cone_1, trichromat_cone_2, trichromat_cs,
                       hidden_symbol=42, seed=42, image_size=1024):
    """Render an Ishihara plate in sRGB from trichromat cone values (3D)."""
    generator = IshiharaPlateGenerator()
    plates = generator.GeneratePlate(
        inside_cone=trichromat_cone_1,
        outside_cone=trichromat_cone_2,
        color_space=trichromat_cs,
        hidden_symbol=hidden_symbol,
        output_space=ColorSpaceType.SRGB,
        lum_noise=0,
        s_cone_noise=0,
        dot_size=1.0,
        seed=seed,
        image_size=image_size,
    )
    return plates[0]


def _render_gaussian_blob_srgb(trichromat_cone_1, trichromat_cone_2, trichromat_cs,
                               direction='right', image_size=512):
    """Render a Gaussian blob stimulus in sRGB: background=cone_2, blob=cone_1."""
    size = image_size
    n_cones = len(trichromat_cone_2)
    center = size / 2.0
    radius = size * 0.475

    Y, X = np.ogrid[:size, :size]
    dist_sq = (X - center) ** 2 + (Y - center) ** 2
    circle_mask = dist_sq <= radius ** 2

    cone_img = np.full((size, size, n_cones), np.nan, dtype=np.float64)
    for c in range(n_cones):
        cone_img[:, :, c] = np.where(circle_mask, trichromat_cone_2[c], np.nan)

    gap_px = radius * 2.0 * 0.2
    blob_sigma = max(gap_px / 4.0, 1.0)
    offset = radius * 0.5
    positions = {
        'up':    (center, center - offset),
        'down':  (center, center + offset),
        'left':  (center - offset, center),
        'right': (center + offset, center),
    }
    bx, by = positions[direction]

    hw = int(np.ceil(4 * blob_sigma))
    x0, x1 = max(int(bx) - hw, 0), min(int(bx) + hw + 1, size)
    y0, y1 = max(int(by) - hw, 0), min(int(by) + hw + 1, size)

    xs = np.arange(x0, x1, dtype=np.float64) - bx
    ys = np.arange(y0, y1, dtype=np.float64) - by
    alpha = (np.exp(-ys[:, None] ** 2 / (2 * blob_sigma ** 2))
             * np.exp(-xs[None, :] ** 2 / (2 * blob_sigma ** 2)))
    alpha *= circle_mask[y0:y1, x0:x1]

    for c in range(n_cones):
        patch = cone_img[y0:y1, x0:x1, c]
        cone_img[y0:y1, x0:x1, c] = patch * (1.0 - alpha) + trichromat_cone_1[c] * alpha

    np.clip(cone_img, 0, None, out=cone_img)

    flat = cone_img.reshape(-1, n_cones)
    mask_flat = circle_mask.ravel()
    srgb_flat = np.zeros((size * size, 3), dtype=np.float64)
    srgb_flat[mask_flat] = trichromat_cs.convert(
        flat[mask_flat], ColorSpaceType.CONE, ColorSpaceType.SRGB)
    srgb_img = np.clip(srgb_flat.reshape(size, size, 3) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(srgb_img, 'RGB')


def _draw_hyperobserver_diff_bars(ax, diffs, designed_genotype, show_ylabel=False,
                                  show_legend=False, summary_values=None,
                                  ylabel=r'Cone excitation difference, $|\Delta c|$'):
    """Draw hyperobserver |M1-M2| bars: pred, pred-8bit, measured (matching validate script style).

    diffs: dict with keys 'pred', 'pred_8bit', optionally 'meas' — each np.ndarray(12,).
    """
    designed_set = (set(designed_genotype) | {420}) - {547}
    designed_mask = np.array([p in designed_set for p in HYPER_PEAKS])

    x = np.arange(len(HYPER_CONE_LABELS))
    w = 0.22

    has_meas = 'meas' in diffs and diffs['meas'] is not None

    ax.bar(x - w, diffs['pred'], w, color=PAPER_BLUE, alpha=0.35,
           edgecolor='black', linewidth=0.3, label=r'Pred $|\Delta c|$')
    ax.bar(x, diffs['pred_8bit'], w, color=PAPER_BLUE, alpha=0.60,
           edgecolor=PAPER_BLUE, linewidth=0.5, label=r'8-bit $|\Delta c|$')
    if has_meas:
        ax.bar(x + w, diffs['meas'], w, color=PAPER_BLUE, alpha=0.85,
               edgecolor=PAPER_BLUE, linewidth=0.5, label=r'Meas $|\Delta c|$')

    # Shade S / M / L regions
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='blue')
    ax.axvspan(0.5, 3.5, alpha=0.06, color='green')
    ax.axvspan(3.5, 11.5, alpha=0.06, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(HYPER_CONE_LABELS, rotation=45, ha='right',
                       rotation_mode='anchor')
    for tick, is_des in zip(ax.get_xticklabels(), designed_mask):
        if is_des:
            tick.set_fontweight('black')
            tick.set_color('black')
            tick.set_fontsize(6)
            tick.set_path_effects([pe.withStroke(linewidth=0.005, foreground='black')])
        else:
            tick.set_fontweight('normal')
            tick.set_color(PAPER_NEUTRAL)
            tick.set_path_effects([])

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='x', which='major', labelsize=6, length=1.0,
                   width=0.3, pad=1)
    ax.tick_params(axis='y', which='major', labelsize=6, length=1.0,
                   width=0.3, pad=1)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=6, labelpad=1)
    if show_legend:
        ax.legend(fontsize=6, loc='upper right')
    if summary_values is not None:
        if summary_values['source'] == 'meas':
            mse_text = (
                f"Mean Meas Null {summary_values['nulled_mean']:.2e}\n"
                f"Mean Meas Other {summary_values['non_nulled_mean']:.2e}"
            )
        else:
            mse_text = (
                f"Mean 8-bit Null {summary_values['nulled_mean']:.2e}\n"
                f"Mean 8-bit Other {summary_values['non_nulled_mean']:.2e}"
            )
        ax.text(0.995, 0.96, mse_text, transform=ax.transAxes,
                fontsize=6, ha='right', va='top',
                bbox=dict(boxstyle='square,pad=0.18', fc='white', alpha=0.82,
                          ec='none'))


def _display_values_for_metamer(metamer, color_space, cone_1, cone_2, primary_order):
    """Return display weights in the same order as the loaded primaries."""
    if metamer.get('raw_cone_1') is not None and metamer.get('raw_cone_2') is not None:
        raw_to_cone = color_space.get_raw_display_to_cone_matrix()
        return (
            np.linalg.solve(raw_to_cone, np.asarray(metamer['raw_cone_1'], dtype=float)),
            np.linalg.solve(raw_to_cone, np.asarray(metamer['raw_cone_2'], dtype=float)),
            primary_order,
        )
    if metamer.get('rgbo_1') is not None and metamer.get('rgbo_2') is not None:
        return np.array(metamer['rgbo_1']), np.array(metamer['rgbo_2']), 'RGBO'
    if metamer.get('bgor_1') is not None and metamer.get('bgor_2') is not None:
        return np.array(metamer['bgor_1']), np.array(metamer['bgor_2']), 'BGOR'
    return (
        color_space.convert(cone_1.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0],
        color_space.convert(cone_2.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0],
        'DISP',
    )


def _rgbo_from_display_values(values, value_order):
    if value_order == 'RGBO':
        return np.array(values)
    if value_order == 'BGOR':
        return np.array([values[3], values[1], values[0], values[2]])
    return np.array(values)


def _projection_mse(projections):
    """MSE of raw hyperobserver projections: predicted endpoint minus measured endpoint."""
    if projections.get('meas_1') is None:
        return None
    pred_err = np.concatenate([
        projections['pred_1'] - projections['meas_1'],
        projections['pred_2'] - projections['meas_2'],
    ])
    pred_8bit_err = np.concatenate([
        projections['pred_1_8bit'] - projections['meas_1'],
        projections['pred_2_8bit'] - projections['meas_2'],
    ])
    return {
        'pred_meas': float(np.mean(np.square(pred_err))),
        'pred_8bit_meas': float(np.mean(np.square(pred_8bit_err))),
    }


def _diff_mse(diffs):
    """MSE between predicted and measured hyperobserver difference bars."""
    if diffs.get('meas') is None:
        return None
    return {
        'pred_meas_diff': float(np.mean(np.square(diffs['pred'] - diffs['meas']))),
        'pred_8bit_meas_diff': float(np.mean(np.square(diffs['pred_8bit'] - diffs['meas']))),
    }


def _null_summary_from_diffs(diffs, designed_genotype, target_peak=547):
    designed_set = set(designed_genotype) | {420}
    nulled_mask = np.array([
        peak in designed_set and peak != target_peak
        for peak in HYPER_PEAKS
    ])
    non_nulled_mask = ~nulled_mask
    source = 'meas' if diffs.get('meas') is not None else '8-bit'
    values = diffs['meas'] if source == 'meas' else diffs['pred_8bit']
    return {
        'source': source,
        'nulled_mean': float(np.mean(values[nulled_mask])),
        'non_nulled_mean': float(np.mean(values[non_nulled_mask])),
        'nulled_count': int(np.sum(nulled_mask)),
        'non_nulled_count': int(np.sum(non_nulled_mask)),
    }


def generate_summary(
    metamers_config_path: str,
    primaries_path: str,
    measurements_dir: str | None,
    plots_dir: str,
    n_observers: int = 8,
    synthetic_epsilon: float = 0.01,
    mode: str = 'plate',
    figure_version: str = 'both',
    primary_order: str | None = None,
):
    """Generate single compilation figure: columns = top N observers."""
    # --- Load inputs ---
    print(f"Loading metamer config from: {metamers_config_path}")
    with open(metamers_config_path, 'r') as f:
        config = json.load(f)

    config_primary_order = config['metadata'].get('primaries_order', 'RGBO')
    primary_order = primary_order or config_primary_order
    print(f"Loading display primaries from: {primaries_path} ({primary_order})")
    primaries = load_primaries_from_csv(
        primaries_path, extract_zero=False, primary_order=primary_order)
    assert len(primaries) >= 4, f"Expected 4 primaries, got {len(primaries)}"

    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata'].get('seed', 42),
        template='baylor',
    )
    metameric_axis = config['metadata'].get('metameric_axis', 2)

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Build hyperobserver and trichromat for sRGB rendering
    print("Building hyperobserver (12D)...")
    hyperobs = Observer.hyperobserver(
        wavelengths=wavelengths,
        degree=VALIDATION_OBSERVER_DEGREE,
        illuminant='raw',
        template='baylor')
    trichromat = Observer.trichromat(
        wavelengths=wavelengths,
        illuminant='raw',
        template='baylor')
    trichromat_cs = ColorSpace(trichromat, display_primaries=primaries, metameric_axis=metameric_axis)

    # Index config observers by genotype for fast lookup
    grid_size = config['metadata']['grid_size']
    center = [grid_size // 2, grid_size // 2]
    config_by_genotype = {}
    for obs_data in config['observers']:
        g = tuple(sorted(obs_data['genotype']))
        config_by_genotype[g] = obs_data

    # Get top N trichromat genotypes
    pdf = observer_genotypes.get_pdf('both')
    trichromats = [(g, pdf[g]) for g in pdf if len(g) == 2]
    trichromats.sort(key=lambda x: x[1], reverse=True)
    top_genotypes = trichromats[:n_observers]

    print(f"Top {n_observers} trichromat genotypes:")
    observer_data = []

    for i, (ml_peaks, prob) in enumerate(top_genotypes):
        # ml_peaks is (M, L) — add Q=547 for tetrachromat-style genotype
        genotype_with_q = tuple(sorted(ml_peaks + (547,)))
        genotype_ml = tuple(sorted(ml_peaks))

        observer = observer_genotypes.get_observer_for_peaks(
            genotype_with_q,
            degree=VALIDATION_OBSERVER_DEGREE,
            illuminant='raw',
            template='baylor')

        gt_str = ', '.join(str(int(p)) if p == int(p) else str(p) for p in genotype_with_q)
        in_config = genotype_with_q in config_by_genotype
        obs_cfg = config_by_genotype.get(genotype_with_q) if in_config else None
        observer_metameric_axis = (
            obs_cfg.get('metameric_axis')
            if obs_cfg is not None and 'metameric_axis' in obs_cfg
            else _resolve_metameric_axis(genotype_with_q, metameric_axis)
        )
        color_space = ColorSpace(
            observer,
            display_primaries=primaries,
            metameric_axis=observer_metameric_axis)
        scaling_factor = color_space._disp_metadata['scaling_factor']

        # Try to get the stored metamer from config. New TetraColorPicker configs
        # have grid_size=1, so center is [0, 0].
        pred_1 = pred_2 = pred_1_rounded = pred_2_rounded = None
        meas_1 = meas_2 = None
        cone_1 = cone_2 = None

        if in_config:
            metamer = None
            for m in obs_cfg['metamers']:
                if m['grid_position'] == center:
                    metamer = m
                    break
            if metamer is not None:
                cone_1 = np.array(metamer['cone_1'])
                cone_2 = np.array(metamer['cone_2'])
                disp_1, disp_2, disp_value_order = _display_values_for_metamer(
                    metamer, color_space, cone_1, cone_2, primary_order)

                # Predicted spectra (continuous)
                pred_1_data = sum(w * p.data for w, p in zip(disp_1, primaries))
                pred_2_data = sum(w * p.data for w, p in zip(disp_2, primaries))
                pred_1 = Spectra(wavelengths=wavelengths,
                                 data=pred_1_data * scaling_factor, normalized=False)
                pred_2 = Spectra(wavelengths=wavelengths,
                                 data=pred_2_data * scaling_factor, normalized=False)

                # Predicted spectra after 8-bit rounding
                disp_1_8bit = np.clip(np.round(disp_1 * 255), 0, 255).astype(int)
                disp_2_8bit = np.clip(np.round(disp_2 * 255), 0, 255).astype(int)
                disp_1_r = disp_1_8bit / 255.0
                disp_2_r = disp_2_8bit / 255.0
                pred_1_rounded = Spectra(
                    wavelengths=wavelengths,
                    data=scaling_factor * sum(w * p.data for w, p in zip(disp_1_r, primaries)),
                    normalized=False)
                pred_2_rounded = Spectra(
                    wavelengths=wavelengths,
                    data=scaling_factor * sum(w * p.data for w, p in zip(disp_2_r, primaries)),
                    normalized=False)

                # Measured spectra if available
                if measurements_dir is not None:
                    rgbo_1_8bit = _rgbo_from_display_values(disp_1_8bit, disp_value_order)
                    rgbo_2_8bit = _rgbo_from_display_values(disp_2_8bit, disp_value_order)
                    rgbo_1 = tuple(int(v) for v in rgbo_1_8bit)
                    rgbo_2 = tuple(int(v) for v in rgbo_2_8bit)
                    try:
                        measured_list = get_spectras_from_rgbo_list(
                            measurements_dir, [rgbo_1, rgbo_2], smooth_method='gaussian')
                        meas_1 = Spectra(
                            wavelengths=wavelengths,
                            data=scaling_factor * measured_list[0].interpolate_values(wavelengths).data,
                            normalized=False)
                        meas_2 = Spectra(
                            wavelengths=wavelengths,
                            data=scaling_factor * measured_list[1].interpolate_values(wavelengths).data,
                            normalized=False)
                    except Exception:
                        pass

        # Fall back to computing metamer pair on the fly
        if pred_1 is None:
            result = _compute_metamer_pair(
                observer, primaries, color_space,
                metameric_axis=observer_metameric_axis)
            if result is None:
                print(f"  {i}: ({gt_str}) prob={prob:.4f} — no null space, skipping")
                continue
            cone_1, cone_2, disp_1, disp_2 = result
            disp_value_order = primary_order

            pred_1_data = sum(w * p.data for w, p in zip(disp_1, primaries))
            pred_2_data = sum(w * p.data for w, p in zip(disp_2, primaries))
            pred_1 = Spectra(wavelengths=wavelengths,
                             data=pred_1_data * scaling_factor, normalized=False)
            pred_2 = Spectra(wavelengths=wavelengths,
                             data=pred_2_data * scaling_factor, normalized=False)

            disp_1_8bit = np.clip(np.round(disp_1 * 255), 0, 255).astype(int)
            disp_2_8bit = np.clip(np.round(disp_2 * 255), 0, 255).astype(int)
            disp_1_r = disp_1_8bit / 255.0
            disp_2_r = disp_2_8bit / 255.0
            pred_1_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(disp_1_r, primaries)),
                normalized=False)
            pred_2_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(disp_2_r, primaries)),
                normalized=False)

        src = 'config' if in_config else 'computed'
        has_meas = meas_1 is not None
        print(f"  {i}: ({gt_str}) prob={prob:.4f} [{src}]"
              f"{' +meas' if has_meas else ''}")

        observer_data.append({
            'idx': i,
            'genotype': genotype_with_q,
            'ml_peaks': genotype_ml,
            'prob': prob,
            'cone_1': cone_1,
            'cone_2': cone_2,
            'predicted_1': pred_1,
            'predicted_2': pred_2,
            'predicted_1_rounded': pred_1_rounded,
            'predicted_2_rounded': pred_2_rounded,
            'measured_1': meas_1,
            'measured_2': meas_2,
            'disp_1_8bit': disp_1_8bit,
            'disp_2_8bit': disp_2_8bit,
            'disp_value_order': disp_value_order,
            'metameric_axis': observer_metameric_axis,
        })

    # --- Apply style ---
    apply_style()

    # --- Build per-observer trichromat color spaces for cross-simulation ---
    # Each observer j's trichromat is built from just its (M, L) peaks; S=420 is added automatically.
    print("Building per-observer trichromat color spaces...")
    observer_trichromats = []
    for od in observer_data:
        obs_tri = observer_genotypes.get_observer_for_peaks(
            od['ml_peaks'],
            degree=VALIDATION_OBSERVER_DEGREE,
            illuminant='raw',
            template='baylor')
        cs_tri = ColorSpace(obs_tri, display_primaries=primaries, metameric_axis=metameric_axis)
        observer_trichromats.append((obs_tri, cs_tri))

    def _genotype_label(genotype):
        return ', '.join(str(int(p)) if p == int(p) else str(p)
                         for p in genotype if p != 547)

    def _raw_hyperobserver_response(spectrum):
        spd = spectrum.interpolate_values(hyperobs.wavelengths).data
        return hyperobs.sensor_matrix @ spd

    def _hyperobserver_projections(od_i):
        projections = {
            'pred_1': _raw_hyperobserver_response(od_i['predicted_1']),
            'pred_2': _raw_hyperobserver_response(od_i['predicted_2']),
            'pred_1_8bit': _raw_hyperobserver_response(od_i['predicted_1_rounded']),
            'pred_2_8bit': _raw_hyperobserver_response(od_i['predicted_2_rounded']),
            'meas_1': None,
            'meas_2': None,
        }
        if od_i['measured_1'] is not None:
            projections['meas_1'] = _raw_hyperobserver_response(od_i['measured_1'])
            projections['meas_2'] = _raw_hyperobserver_response(od_i['measured_2'])
        return projections

    def _hyperobserver_diffs(projections):
        diffs = {
            'pred': np.abs(projections['pred_1'] - projections['pred_2']),
            'pred_8bit': np.abs(projections['pred_1_8bit'] - projections['pred_2_8bit']),
        }
        if projections['meas_1'] is not None:
            diffs['meas'] = np.abs(projections['meas_1'] - projections['meas_2'])
        return diffs

    def _draw_rendering(ax, row, col, od_i, obs_j, cs_j, n_plate_cols):
        tri_cone_1 = obs_j.observe_spectras([od_i['predicted_1']])[0]
        tri_cone_2 = obs_j.observe_spectras([od_i['predicted_2']])[0]
        if mode == 'gaussian':
            directions = ['up', 'down', 'left', 'right']
            stim_img = _render_gaussian_blob_srgb(
                tri_cone_1, tri_cone_2, cs_j,
                direction=directions[col % 4], image_size=512)
        else:
            stim_img = _render_plate_srgb(
                tri_cone_1, tri_cone_2, cs_j,
                hidden_symbol=42, seed=row * n_plate_cols + col, image_size=1024)
        ax.imshow(np.array(stim_img))
        ax.set_xticks([])
        ax.set_yticks([])

        if row == col:
            for spine in ax.spines.values():
                spine.set_edgecolor('#e05c00')
                spine.set_linewidth(1.5)

    def _draw_spectra(ax_spec, od_i, gt_label, show_xlabel=False,
                      show_legend=False):
        wl = od_i['predicted_1'].wavelengths
        ax_spec.plot(wl, od_i['predicted_1'].data, color=PAPER_BLUE,
                     linewidth=0.8, alpha=0.45, label='Pred M1')
        ax_spec.plot(wl, od_i['predicted_2'].data, color=PAPER_RED,
                     linewidth=0.8, alpha=0.45, label='Pred M2')
        if od_i['measured_1'] is not None and od_i['measured_2'] is not None:
            ax_spec.plot(wl, od_i['measured_1'].data, color=PAPER_BLUE,
                         linewidth=0.9, linestyle='--', alpha=0.95, label='Meas M1')
            ax_spec.plot(wl, od_i['measured_2'].data, color=PAPER_RED,
                         linewidth=0.9, linestyle='--', alpha=0.95, label='Meas M2')
        ax_spec.set_xlim(400, 700)
        ax_spec.set_ylim(bottom=0)
        ax_spec.tick_params(labelsize=6, pad=2)
        ax_spec.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
        fmt = plt.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        ax_spec.yaxis.set_major_formatter(fmt)
        ax_spec.grid(True, alpha=0.2, linewidth=0.4)
        ax_spec.set_title(f'({gt_label})', fontsize=8, fontweight='bold', pad=2)
        ax_spec.set_ylabel('Power', fontsize=6, labelpad=1)

        if show_legend:
            ax_spec.legend(fontsize=6, loc='lower right', handlelength=1.0,
                           borderpad=0.3, labelspacing=0.2)

        disp_1_text = ','.join(str(v) for v in od_i['disp_1_8bit'])
        disp_2_text = ','.join(str(v) for v in od_i['disp_2_8bit'])
        disp_text = f'M1 [{disp_1_text}]\nM2 [{disp_2_text}]'
        ax_spec.text(0.03, 0.97, disp_text, transform=ax_spec.transAxes,
                     fontsize=6, va='top', ha='left',
                     bbox=dict(boxstyle='square,pad=0.2', fc='white', alpha=0.85,
                               ec='#888888', lw=0.6))

        if show_xlabel:
            ax_spec.set_xlabel('Wavelength (nm)', fontsize=8)
        else:
            ax_spec.tick_params(labelbottom=False)

    def _save_figure(fig, stem):
        fname = plots_path / f'{stem}.png'
        fig.savefig(fname, dpi=600)
        fname_pdf = plots_path / f'{stem}.pdf'
        fig.savefig(fname_pdf, dpi=300)
        plt.close(fig)
        print(f"\n  Saved: {fname}")
        print(f"  Saved: {fname_pdf}")

    def _save_mse_csv(stem, rows):
        fname = plots_path / f'{stem}_mse.csv'
        with open(fname, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'observer_index',
                    'genotype',
                    'probability',
                    'pred_meas_projection_mse',
                    'pred_8bit_meas_projection_mse',
                    'pred_meas_diff_mse',
                    'pred_8bit_meas_diff_mse',
                    'summary_source',
                    'nulled_mean',
                    'non_nulled_mean',
                    'nulled_count',
                    'non_nulled_count',
                ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {fname}")

    def _add_hyperobserver_legend(fig, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', ncol=len(handles),
                       bbox_to_anchor=(0.99, 1.0), fontsize=6, frameon=False,
                       handlelength=1.0, handletextpad=0.4, columnspacing=0.9)

    def _add_spectra_legend(fig, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper left', ncol=len(handles),
                       bbox_to_anchor=(0.01, 1.01), fontsize=6, frameon=False,
                       handlelength=1.0, handletextpad=0.4, columnspacing=0.9)

    # --- Generate compilation figure ---
    # Layout for "both": rows = observer i's metamer pair
    #                    col 0   = spectra overlay (M1 vs M2)
    #                    col 1   = hyperobserver bar chart
    #                    cols 2..N+1 = sRGB simulation through observer j's eyes
    n_rows = len(observer_data)
    n_plate_cols = n_rows  # square grid

    if figure_version == 'hyperobserver':
        n_observer_cols = 2
        n_grid_rows = int(np.ceil(n_rows / n_observer_cols))
        fig = plt.figure(figsize=(DOUBLE_COL, SUMMARY_FIG_HEIGHT), layout='constrained')
        fig.get_layout_engine().set(h_pad=0.015, w_pad=0.02, hspace=0.035, wspace=0.04,
                                    rect=(0, 0, 1, 0.975))
        gs = GridSpec(n_grid_rows, n_observer_cols * 2, figure=fig,
                      width_ratios=[1.35, 3.2, 1.35, 3.2])
        diff_axes = []
        spec_axes = []
        mse_rows = []
        nulled_means = []
        non_nulled_means = []
        pred_meas_diff_mses = []
        pred_8bit_meas_diff_mses = []

        for idx, od_i in enumerate(observer_data):
            row = idx // n_observer_cols
            col = idx % n_observer_cols
            spec_col = col * 2
            diff_col = spec_col + 1
            gt_label = _genotype_label(od_i['genotype'])

            if idx == 0:
                ax_spec = fig.add_subplot(gs[row, spec_col])
            else:
                ax_spec = fig.add_subplot(gs[row, spec_col], sharey=spec_axes[0])
            spec_axes.append(ax_spec)
            _draw_spectra(
                ax_spec, od_i, gt_label,
                show_xlabel=(row == n_grid_rows - 1),
                show_legend=False)

            if idx == 0:
                ax_diff = fig.add_subplot(gs[row, diff_col])
            else:
                ax_diff = fig.add_subplot(gs[row, diff_col])
            diff_axes.append(ax_diff)

            projections = _hyperobserver_projections(od_i)
            diffs = _hyperobserver_diffs(projections)
            mse_values = _projection_mse(projections)
            diff_mse_values = _diff_mse(diffs)
            if diff_mse_values is not None:
                pred_meas_diff_mses.append(diff_mse_values['pred_meas_diff'])
                pred_8bit_meas_diff_mses.append(diff_mse_values['pred_8bit_meas_diff'])
            null_summary = _null_summary_from_diffs(diffs, od_i['genotype'])
            nulled_means.append(null_summary['nulled_mean'])
            non_nulled_means.append(null_summary['non_nulled_mean'])
            mse_rows.append({
                'observer_index': od_i['idx'],
                'genotype': _genotype_label(od_i['genotype']),
                'probability': f"{od_i['prob']:.8g}",
                'pred_meas_projection_mse': (
                    f"{mse_values['pred_meas']:.8e}"
                    if mse_values is not None else ''
                ),
                'pred_8bit_meas_projection_mse': (
                    f"{mse_values['pred_8bit_meas']:.8e}"
                    if mse_values is not None else ''
                ),
                'pred_meas_diff_mse': (
                    f"{diff_mse_values['pred_meas_diff']:.8e}"
                    if diff_mse_values is not None else ''
                ),
                'pred_8bit_meas_diff_mse': (
                    f"{diff_mse_values['pred_8bit_meas_diff']:.8e}"
                    if diff_mse_values is not None else ''
                ),
                'summary_source': null_summary['source'],
                'nulled_mean': f"{null_summary['nulled_mean']:.8e}",
                'non_nulled_mean': f"{null_summary['non_nulled_mean']:.8e}",
                'nulled_count': null_summary['nulled_count'],
                'non_nulled_count': null_summary['non_nulled_count'],
            })
            _draw_hyperobserver_diff_bars(
                ax_diff, diffs, od_i['genotype'],
                show_ylabel=(col == 0 and row == n_grid_rows // 2),
                show_legend=False,
                summary_values=null_summary)
            if row == n_grid_rows - 1:
                ax_diff.set_xlabel('Cone Peak (nm)', fontsize=8)

        bottom_axes = [
            ax for ax in spec_axes + diff_axes
            if ax.get_subplotspec().rowspan.stop == n_grid_rows
        ]
        fig.align_xlabels(bottom_axes)
        _add_spectra_legend(fig, spec_axes[0])
        _add_hyperobserver_legend(fig, diff_axes[0])
        _save_figure(fig, f'metamer_compilation_{mode}_hyperobserver')
        _save_mse_csv(f'metamer_compilation_{mode}_hyperobserver', mse_rows)
        print(
            "Total mean cone excitation difference "
            f"({mse_rows[0]['summary_source']}): "
            f"nulled={np.mean(nulled_means):.6e}, "
            f"other={np.mean(non_nulled_means):.6e}"
        )
        if pred_meas_diff_mses:
            print(
                "Total hyperobserver diff MSE (predicted - measured): "
                f"pred={np.mean(pred_meas_diff_mses):.6e}, "
                f"8-bit={np.mean(pred_8bit_meas_diff_mses):.6e}"
            )
        print("Done.")
        return

    if figure_version == 'renderings':
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL), layout='constrained')
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.03, wspace=0.03,
                                    rect=(0, 0, 1, 0.93))
        gs = GridSpec(n_rows, n_plate_cols, figure=fig)

        for row, od_i in enumerate(observer_data):
            ax_row_label = None
            for col, (obs_j, cs_j) in enumerate(observer_trichromats):
                ax = fig.add_subplot(gs[row, col])
                if col == 0:
                    ax_row_label = ax
                _draw_rendering(ax, row, col, od_i, obs_j, cs_j, n_plate_cols)
            if ax_row_label is not None:
                ax_row_label.set_ylabel(f'({_genotype_label(od_i["genotype"])})',
                                        fontsize=4, fontweight='bold',
                                        rotation=90, labelpad=2)

        _save_figure(fig, f'metamer_compilation_{mode}_renderings')
        print("Done.")
        return

    n_cols = n_plate_cols + 2

    # Spectra narrow, bar chart ≈ single SIGGRAPH col, plate cols equal.
    # Last row gets extra height to accommodate x-tick labels; others are square-image height.
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.03, wspace=0.03,
                                rect=(0, 0, 1, 0.90))
    height_ratios = [1] * (n_rows)
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  width_ratios=[1.5, 4] + [1] * n_plate_cols,
                  height_ratios=height_ratios)

    diff_axes = []
    spec_axes = []
    plate_axes_row0 = []

    for row, od_i in enumerate(observer_data):
        genotype_i = od_i['genotype']
        # Drop the dummy Q=547 cone from display labels
        gt_label_i = _genotype_label(genotype_i)

        # Col 0: spectra overlay
        if row == 0:
            ax_spec = fig.add_subplot(gs[row, 0])
        else:
            ax_spec = fig.add_subplot(gs[row, 0], sharey=spec_axes[0])
        spec_axes.append(ax_spec)
        _draw_spectra(
            ax_spec, od_i, gt_label_i,
            show_xlabel=(row == n_rows - 1),
            show_legend=False)

        # Col 1: hyperobserver diff bars
        if row == 0:
            ax_diff = fig.add_subplot(gs[row, 1])
        else:
            ax_diff = fig.add_subplot(gs[row, 1], sharey=diff_axes[0])
        diff_axes.append(ax_diff)

        _draw_hyperobserver_diff_bars(
            ax_diff, _hyperobserver_diffs(_hyperobserver_projections(od_i)), genotype_i,
            show_ylabel=False,
            show_legend=False)

        if row == n_rows - 1:
            ax_diff.set_xlabel('Cone Peak (nm)', fontsize=7)

        # Cols 2..N+1: how observer j sees observer i's metamer pair
        for col, (obs_j, cs_j) in enumerate(observer_trichromats):
            ax = fig.add_subplot(gs[row, col + 2])
            _draw_rendering(ax, row, col, od_i, obs_j, cs_j, n_plate_cols)

            if row == 0:
                plate_axes_row0.append(ax)

    # Align x-axis labels of the bottom spectra and diff axes
    fig.align_xlabels([spec_axes[-1], diff_axes[-1]])
    _add_spectra_legend(fig, spec_axes[0])
    _add_hyperobserver_legend(fig, diff_axes[0])

    # Draw once so constrained_layout finalises axis positions
    fig.canvas.draw()

    _save_figure(fig, f'metamer_compilation_{mode}')
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize metamer pairs as Ishihara plates + hyperobserver diffs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_metamer_summary.py \\
    --primaries measurements/2026-03-04/validation_2026-03-04_13-44-34/primaries \\
    --plots-dir output/metamer_summary
        """
    )

    parser.add_argument('--primaries', type=str, required=True,
                        help='Path to directory containing display primary measurements (BGOR order)')
    parser.add_argument('--measurements', type=str, default=None,
                        help='Path to directory containing measured spectra CSV files (optional)')
    parser.add_argument('--synthetic-epsilon', type=float, default=0.01,
                        help='Noise level for synthetic BGOR weights when measurements are not provided')
    parser.add_argument('--metamers', type=str,
                        default='config/display_validation_metamers.json',
                        help='Path to BGYR metamer configuration JSON')
    parser.add_argument('--plots-dir', type=str, required=True,
                        help='Directory to save output figures')
    parser.add_argument('--n-observers', type=int, default=10,
                        help='Number of top observers to show (default: 10)')
    parser.add_argument('--mode', type=str, default='plate',
                        choices=['plate', 'gaussian'],
                        help='Row A stimulus type: "plate" (Ishihara) or "gaussian" (Gaussian blob)')
    parser.add_argument('--figure-version', type=str, default='both',
                        choices=['hyperobserver', 'renderings', 'both'],
                        help='Figure contents: hyperobserver bars only, renderings only, or both (default)')
    parser.add_argument('--primary-order', type=str, default=None,
                        choices=['RGBO', 'BGOR'],
                        help='Display primary order. Defaults to metadata primaries_order.')

    args = parser.parse_args()

    generate_summary(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        measurements_dir=args.measurements,
        plots_dir=args.plots_dir,
        n_observers=args.n_observers,
        synthetic_epsilon=args.synthetic_epsilon,
        mode=args.mode,
        figure_version=args.figure_version,
        primary_order=args.primary_order,
    )


if __name__ == '__main__':
    main()
