#!/usr/bin/env python3
"""
Visualize metamer pairs as a single compilation figure (8 columns = 8 observers).

Row A: Ishihara plates (or Gaussian blobs) rendered in sRGB through each observer
Row B: Hyperobserver difference plots with pred, pred-8bit, and measured bars

Uses the top 8 trichromat genotypes by probability. For observers present in
the metamer config, uses the stored center-point pair. For others, computes
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

from TetriumColor.Measurement import load_primaries_from_csv, get_spectras_from_rgbo_list
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.Visualization.PlotStyle import apply_style, FULL_PAGE

import argparse
import json
import numpy as np
from scipy.linalg import null_space
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Hyperobserver cone info
HYPER_PEAKS = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
HYPER_CONE_LABELS = [
    '420', '530', '533', '536',
    '547', '551', '552', '553',
    '555', '556', '556.5', '559',
]


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
                                  show_legend=False):
    """Draw hyperobserver |M1-M2| bars: pred, pred-8bit, measured (matching validate script style).

    diffs: dict with keys 'pred', 'pred_8bit', optionally 'meas' — each np.ndarray(12,).
    """
    designed_set = set(designed_genotype) | {420}
    designed_mask = np.array([p in designed_set for p in HYPER_PEAKS])

    x = np.arange(len(HYPER_CONE_LABELS))
    w = 0.22

    has_meas = 'meas' in diffs and diffs['meas'] is not None

    ax.bar(x - w, diffs['pred'], w, color='steelblue', alpha=0.85,
           edgecolor='black', linewidth=0.3, label=r'Pred $|$M1$-$M2$|$')
    ax.bar(x, diffs['pred_8bit'], w, color='steelblue', alpha=0.55,
           edgecolor='steelblue', linewidth=0.5, label=r'8-bit $|$M1$-$M2$|$')
    if has_meas:
        ax.bar(x + w, diffs['meas'], w, color='steelblue', alpha=0.3,
               edgecolor='steelblue', linewidth=0.5, label=r'Meas $|$M1$-$M2$|$')

    # Shade S / M / L regions
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='blue')
    ax.axvspan(0.5, 3.5, alpha=0.06, color='green')
    ax.axvspan(3.5, 11.5, alpha=0.06, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(HYPER_CONE_LABELS, rotation=45, ha='right')
    for tick, is_des in zip(ax.get_xticklabels(), designed_mask):
        if is_des:
            tick.set_fontweight('bold')

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', which='major', pad=-2)

    if show_ylabel:
        ax.set_ylabel(r'$|\mathrm{M1} - \mathrm{M2}|$')
    if show_legend:
        ax.legend(fontsize=7, loc='upper right')


def generate_summary(
    metamers_config_path: str,
    primaries_path: str,
    measurements_dir: str | None,
    plots_dir: str,
    n_observers: int = 8,
    synthetic_epsilon: float = 0.01,
    mode: str = 'plate',
):
    """Generate single compilation figure: columns = top N observers."""
    # --- Load inputs ---
    print(f"Loading metamer config from: {metamers_config_path}")
    with open(metamers_config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading display primaries from: {primaries_path}")
    primaries = load_primaries_from_csv(primaries_path, extract_zero=False, primary_order='BGOR')
    assert len(primaries) >= 4, f"Expected 4 primaries (BGOR), got {len(primaries)}"

    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata'].get('seed', 42)
    )
    metameric_axis = config['metadata'].get('metameric_axis', 2)

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Build hyperobserver and trichromat for sRGB rendering
    print("Building hyperobserver (12D)...")
    hyperobs = Observer.hyperobserver(wavelengths=wavelengths)
    trichromat = Observer.trichromat(wavelengths=wavelengths)
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

        observer = observer_genotypes.get_observer_for_peaks(genotype_with_q)
        color_space = ColorSpace(observer, display_primaries=primaries, metameric_axis=metameric_axis)
        scaling_factor = color_space._disp_metadata['scaling_factor']

        gt_str = ', '.join(str(int(p)) if p == int(p) else str(p) for p in genotype_with_q)
        in_config = genotype_with_q in config_by_genotype

        # Try to get center-point metamer from config
        pred_1 = pred_2 = pred_1_rounded = pred_2_rounded = None
        meas_1 = meas_2 = None
        cone_1 = cone_2 = None

        if in_config:
            obs_cfg = config_by_genotype[genotype_with_q]
            metamer = None
            for m in obs_cfg['metamers']:
                if m['grid_position'] == center:
                    metamer = m
                    break
            if metamer is not None:
                cone_1 = np.array(metamer['cone_1'])
                cone_2 = np.array(metamer['cone_2'])
                bgor_1 = color_space.convert(cone_1.reshape(1, -1),
                                             ColorSpaceType.CONE, ColorSpaceType.DISP)[0]
                bgor_2 = color_space.convert(cone_2.reshape(1, -1),
                                             ColorSpaceType.CONE, ColorSpaceType.DISP)[0]

                # Predicted spectra (continuous)
                pred_1_data = sum(w * p.data for w, p in zip(bgor_1, primaries))
                pred_2_data = sum(w * p.data for w, p in zip(bgor_2, primaries))
                pred_1 = Spectra(wavelengths=wavelengths,
                                 data=pred_1_data * scaling_factor, normalized=False)
                pred_2 = Spectra(wavelengths=wavelengths,
                                 data=pred_2_data * scaling_factor, normalized=False)

                # Predicted spectra after 8-bit rounding
                bgor_1_8bit = np.clip(np.round(bgor_1 * 255), 0, 255).astype(int)
                bgor_2_8bit = np.clip(np.round(bgor_2 * 255), 0, 255).astype(int)
                bgor_1_r = bgor_1_8bit / 255.0
                bgor_2_r = bgor_2_8bit / 255.0
                pred_1_rounded = Spectra(
                    wavelengths=wavelengths,
                    data=scaling_factor * sum(w * p.data for w, p in zip(bgor_1_r, primaries)),
                    normalized=False)
                pred_2_rounded = Spectra(
                    wavelengths=wavelengths,
                    data=scaling_factor * sum(w * p.data for w, p in zip(bgor_2_r, primaries)),
                    normalized=False)

                # Measured spectra if available
                if measurements_dir is not None:
                    rgbo_1 = (int(bgor_1_8bit[3]), int(bgor_1_8bit[1]),
                              int(bgor_1_8bit[0]), int(bgor_1_8bit[2]))
                    rgbo_2 = (int(bgor_2_8bit[3]), int(bgor_2_8bit[1]),
                              int(bgor_2_8bit[0]), int(bgor_2_8bit[2]))
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
            result = _compute_metamer_pair(observer, primaries, color_space)
            if result is None:
                print(f"  {i}: ({gt_str}) prob={prob:.4f} — no null space, skipping")
                continue
            cone_1, cone_2, bgor_1, bgor_2 = result

            pred_1_data = sum(w * p.data for w, p in zip(bgor_1, primaries))
            pred_2_data = sum(w * p.data for w, p in zip(bgor_2, primaries))
            pred_1 = Spectra(wavelengths=wavelengths,
                             data=pred_1_data * scaling_factor, normalized=False)
            pred_2 = Spectra(wavelengths=wavelengths,
                             data=pred_2_data * scaling_factor, normalized=False)

            bgor_1_8bit = np.clip(np.round(bgor_1 * 255), 0, 255).astype(int)
            bgor_2_8bit = np.clip(np.round(bgor_2 * 255), 0, 255).astype(int)
            bgor_1_r = bgor_1_8bit / 255.0
            bgor_2_r = bgor_2_8bit / 255.0
            pred_1_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(bgor_1_r, primaries)),
                normalized=False)
            pred_2_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(bgor_2_r, primaries)),
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
            'bgor_1_8bit': bgor_1_8bit,
            'bgor_2_8bit': bgor_2_8bit,
        })

    # --- Apply style ---
    apply_style()
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.labelpad': 2,
        'legend.fontsize': 7,
    })

    # --- Build per-observer trichromat color spaces for cross-simulation ---
    # Each observer j's trichromat is built from just its (M, L) peaks; S=420 is added automatically.
    print("Building per-observer trichromat color spaces...")
    observer_trichromats = []
    for od in observer_data:
        obs_tri = observer_genotypes.get_observer_for_peaks(od['ml_peaks'])
        cs_tri = ColorSpace(obs_tri, display_primaries=primaries, metameric_axis=metameric_axis)
        observer_trichromats.append((obs_tri, cs_tri))

    # --- Generate compilation figure ---
    # Layout: rows = observer i's metamer pair
    #         col 0   = spectra overlay (M1 vs M2)
    #         col 1   = hyperobserver bar chart
    #         cols 2..N+1 = sRGB simulation through observer j's eyes
    n_rows = len(observer_data)
    n_plate_cols = n_rows  # square grid
    n_cols = n_plate_cols + 2

    # Spectra narrow, bar chart ≈ single SIGGRAPH col, plate cols equal.
    # Last row gets extra height to accommodate x-tick labels; others are square-image height.
    fig = plt.figure(figsize=(FULL_PAGE, FULL_PAGE * 0.58), layout='constrained')
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
        gt_label_i = ', '.join(str(int(p)) if p == int(p) else str(p)
                               for p in genotype_i if p != 547)

        # Col 0: spectra overlay
        if row == 0:
            ax_spec = fig.add_subplot(gs[row, 0])
        else:
            ax_spec = fig.add_subplot(gs[row, 0], sharey=spec_axes[0])
        spec_axes.append(ax_spec)

        wl = od_i['predicted_1'].wavelengths
        ax_spec.plot(wl, od_i['predicted_1'].data, color='steelblue',
                     linewidth=0.8, label='M1')
        ax_spec.plot(wl, od_i['predicted_2'].data, color='firebrick',
                     linewidth=0.8, linestyle='--', label='M2')
        ax_spec.set_xlim(400, 700)
        ax_spec.set_ylim(bottom=0)
        ax_spec.tick_params(labelsize=7, pad=2)
        ax_spec.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
        fmt = plt.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        ax_spec.yaxis.set_major_formatter(fmt)
        ax_spec.grid(True, alpha=0.2, linewidth=0.4)
        ax_spec.set_ylabel(f'({gt_label_i})', fontsize=9, fontweight='bold',
                           rotation=90, labelpad=8)

        if row == 0:
            ax_spec.set_title('A. Spectra', fontsize=7, fontweight='bold')
            ax_spec.legend(fontsize=6, loc='lower right', handlelength=1.2,
                           borderpad=0.3, labelspacing=0.2)
        # BGOR annotation in upper-left corner
        b1, g1, o1, r1 = od_i['bgor_1_8bit']
        b2, g2, o2, r2 = od_i['bgor_2_8bit']
        bgor_text = (f'M1 [{b1},{g1},{o1},{r1}]\n'
                     f'M2 [{b2},{g2},{o2},{r2}]')
        ax_spec.text(0.03, 0.97, bgor_text, transform=ax_spec.transAxes,
                     fontsize=5.5, va='top', ha='left',
                     bbox=dict(boxstyle='square,pad=0.3', fc='white', alpha=0.85,
                               ec='#888888', lw=0.6))

        if row < n_rows - 1:
            ax_spec.tick_params(labelbottom=False)
        else:
            ax_spec.set_xlabel('Wavelength (nm)', fontsize=7)

        # Col 1: hyperobserver diff bars
        if row == 0:
            ax_diff = fig.add_subplot(gs[row, 1])
        else:
            ax_diff = fig.add_subplot(gs[row, 1], sharey=diff_axes[0])
        diff_axes.append(ax_diff)

        hp1 = hyperobs.observe_spectras([od_i['predicted_1']])[0]
        hp2 = hyperobs.observe_spectras([od_i['predicted_2']])[0]
        hp1r = hyperobs.observe_spectras([od_i['predicted_1_rounded']])[0]
        hp2r = hyperobs.observe_spectras([od_i['predicted_2_rounded']])[0]

        diffs = {
            'pred': np.abs(hp1 - hp2),
            'pred_8bit': np.abs(hp1r - hp2r),
        }
        if od_i['measured_1'] is not None:
            hm1 = hyperobs.observe_spectras([od_i['measured_1']])[0]
            hm2 = hyperobs.observe_spectras([od_i['measured_2']])[0]
            diffs['meas'] = np.abs(hm1 - hm2)

        _draw_hyperobserver_diff_bars(
            ax_diff, diffs, genotype_i,
            show_ylabel=False,
            show_legend=(row == 0))

        if row == 0:
            ax_diff.set_title(r'B. Projected Spectra $|$M1$-$M2$|$ onto the Hyper-observer',
                              fontsize=7, fontweight='bold')

        if row < n_rows - 1:
            ax_diff.tick_params(labelbottom=False)
        else:
            ax_diff.set_xlabel('Cone Peak (nm)', fontsize=7)

        # Cols 2..N+1: how observer j sees observer i's metamer pair
        for col, (obs_j, cs_j) in enumerate(observer_trichromats):
            ax = fig.add_subplot(gs[row, col + 2])
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

            # Column headers on first row: observer j's genotype (drop dummy Q=547)
            if row == 0:
                plate_axes_row0.append(ax)
                gj = observer_data[col]['genotype']
                gj_label = ', '.join(str(int(p)) if p == int(p) else str(p)
                                     for p in gj if p != 547)
                ax.set_title(f'({gj_label})', fontsize=7, fontweight='bold')

            # Highlight diagonal (observer sees their own metamer — should look uniform)
            if row == col:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e05c00')
                    spine.set_linewidth(1.5)

    # Align x-axis labels of the bottom spectra and diff axes
    fig.align_xlabels([spec_axes[-1], diff_axes[-1]])

    # Draw once so constrained_layout finalises axis positions
    fig.canvas.draw()

    # Place group headers just above the per-column axis titles.
    x0 = spec_axes[0].get_position().x0
    x1 = diff_axes[0].get_position().x1
    x0p = plate_axes_row0[0].get_position().x0
    x1p = plate_axes_row0[-1].get_position().x1
    # Sit ~2% above the actual drawn top of the first-row axes
    y_group = spec_axes[0].get_position().y1 + 0.06

    fig.text((x0 + x1) / 2, y_group,
             'Spectral Validation of Metamers',
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             transform=fig.transFigure)
    fig.text((x0p + x1p) / 2, y_group,
             'Simulation in sRGB of Metamer Constructed for Row Observer\n'
             'As Seen by Column Observer',
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             transform=fig.transFigure)

    stem = f'metamer_compilation_{mode}'
    fname = plots_path / f'{stem}.png'
    fig.savefig(fname, dpi=600)
    fname_pdf = plots_path / f'{stem}.pdf'
    fig.savefig(fname_pdf, dpi=300)
    plt.close(fig)
    print(f"\n  Saved: {fname}")
    print(f"  Saved: {fname_pdf}")
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
    parser.add_argument('--n-observers', type=int, default=5,
                        help='Number of top observers to show (default: 5)')
    parser.add_argument('--mode', type=str, default='plate',
                        choices=['plate', 'gaussian'],
                        help='Row A stimulus type: "plate" (Ishihara) or "gaussian" (Gaussian blob)')

    args = parser.parse_args()

    generate_summary(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        measurements_dir=args.measurements,
        plots_dir=args.plots_dir,
        n_observers=args.n_observers,
        synthetic_epsilon=args.synthetic_epsilon,
        mode=args.mode,
    )


if __name__ == '__main__':
    main()
