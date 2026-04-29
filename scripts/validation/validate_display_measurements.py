#!/usr/bin/env python3
"""
Validate display measurements against expected LMSQ responses.

For each metamer pair (BGYR), this script:
1. Converts BGYR -> BGOR display primary weights, reconstructs the predicted spectrum,
   and plots it against the measured spectrum.
2. Projects both predicted and measured spectra into BGYR and computes RMSE.
3. Projects both spectra into LMSQ for each observer and computes RMSE.
   (LMS difference should be ~0, Q difference should be large for valid metamers.)

All three panels are output as a single figure per metamer pair.

Summary plots additionally include per-wavelength spectral residual / RMSE figures
(`*_spectral_rmse.png`) comparing measured SPD to the linear primary-mix prediction.
"""

from TetriumColor.Measurement import load_primaries_from_csv, get_spectras_from_rgbo_list
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, convert_spectrum_to_bgyr
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra

import argparse
import csv
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

VALIDATION_OBSERVER_DEGREE = 2.0

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Linux Biolinum'


def validate_measurements(
    metamers_config_path: str,
    primaries_path: str,
    measurements_dir: str | None,
    plots_dir: str,
    synthetic_epsilon: float = 0.01,
    pred_only: bool = False,
):
    """
    Validate measured spectra against predicted spectra from BGYR metamer pairs.

    Args:
        metamers_config_path: Path to BGYR metamer configuration JSON
        primaries_path: Path to directory with display primaries (BGOR order)
        measurements_dir: Directory containing measured spectra CSV files
            (if None, synthetic spectra are generated from primaries)
        plots_dir: Directory to save validation plots
        synthetic_epsilon: Noise level for synthetic BGOR weights (only used when
            measurements_dir is None)
        pred_only: If True, skip all measured-spectra computation and show only
            predicted / 8-bit-predicted in plots.
    """
    # --- Load inputs ---
    print(f"Loading metamer config from: {metamers_config_path}")
    with open(metamers_config_path, 'r') as f:
        config = json.load(f)

    print(f"  Observers: {len(config['observers'])}")
    print(f"  Total metamer pairs: {config['metadata']['total_metamer_pairs']}")

    print(f"Loading display primaries from: {primaries_path}")
    primaries = load_primaries_from_csv(primaries_path, extract_zero=False, primary_order='BGOR')
    assert len(primaries) >= 4, f"Expected 4 primaries (BGOR), got {len(primaries)}"
    print(f"  Loaded {len(primaries)} primaries (BGOR order)")

    # Set up observers
    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata'].get('seed', 42)
    )
    metameric_axis = config['metadata'].get('metameric_axis', 2)

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    if measurements_dir is None:
        print(f"Using SYNTHETIC spectra (epsilon={synthetic_epsilon} BGOR units)")
    else:
        print(f"Loading measured spectra from: {measurements_dir}")

    # Storage for end-of-validation summary plots
    all_pair_data = []  # list of dicts: obs_idx, pair_idx, genotype, predicted_1, predicted_2

    # --- Process each observer ---
    for obs_data in config['observers']:
        genotype = tuple(sorted(obs_data['genotype']))
        obs_idx = obs_data['observer_index']
        observer = observer_genotypes.get_observer_for_peaks(
            genotype, degree=VALIDATION_OBSERVER_DEGREE)

        # Create observer-specific ColorSpace with display primaries
        color_space = ColorSpace(observer, display_primaries=primaries, metameric_axis=metameric_axis)

        # Determine Q index in the sorted LMSQ ordering
        sorted_with_s = tuple(sorted((420,) + genotype))
        q_index = sorted_with_s.index(547)
        lms_indices = [i for i in range(len(sorted_with_s)) if i != q_index]

        print(f"\nObserver {obs_idx}: genotype={genotype}, Q at index {q_index}")
        print(f"  Using observer-specific ColorSpace for BGYR→BGOR conversion")

        for metamer in obs_data['metamers']:
            pair_idx = metamer['pair_index']
            scaling_factor = color_space._disp_metadata['scaling_factor']

            # Use stored cone excitations directly — CONE → DISP is one stable matrix
            # multiply. CONE → BGYR → CONE loses precision because inv(L) is
            # ill-conditioned when M/Q cones are closely spaced (e.g. 530/547nm),
            # corrupting the stored BGYR values at generation time.
            cone_1 = np.array(metamer['cone_1'])
            cone_2 = np.array(metamer['cone_2'])
            bgor_1 = color_space.convert(cone_1.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0]
            bgor_2 = color_space.convert(cone_2.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0]
            # Clip only for 8-bit file lookup: BGOR=[B,G,O,R] -> RGBO=[R,G,B,O]
            bgor_1_8bit = np.clip(np.round(bgor_1 * 255), 0, 255).astype(int)
            bgor_2_8bit = np.clip(np.round(bgor_2 * 255), 0, 255).astype(int)
            rgbo_1 = (int(bgor_1_8bit[3]), int(bgor_1_8bit[1]),
                      int(bgor_1_8bit[0]), int(bgor_1_8bit[2]))
            rgbo_2 = (int(bgor_2_8bit[3]), int(bgor_2_8bit[1]),
                      int(bgor_2_8bit[0]), int(bgor_2_8bit[2]))

            print(f"  Pair {pair_idx}: RGBO1={rgbo_1}, RGBO2={rgbo_2}")
            print(f"    BGOR1 (normalized): {bgor_1}, BGOR2 (normalized): {bgor_2}")

            # --- Predicted spectra: scale BGOR primaries by BGOR weights ---
            predicted_1_data = sum(w * p.data for w, p in zip(bgor_1, primaries))
            predicted_2_data = sum(w * p.data for w, p in zip(bgor_2, primaries))
            predicted_1 = Spectra(wavelengths=wavelengths, data=predicted_1_data * scaling_factor, normalized=False)
            predicted_2 = Spectra(wavelengths=wavelengths, data=predicted_2_data * scaling_factor, normalized=False)

            # --- Predicted spectra after 8-bit rounding ---
            bgor_1_rounded = bgor_1_8bit / 255.0
            bgor_2_rounded = bgor_2_8bit / 255.0
            predicted_1_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(bgor_1_rounded, primaries)), normalized=False)
            predicted_2_rounded = Spectra(
                wavelengths=wavelengths,
                data=scaling_factor * sum(w * p.data for w, p in zip(bgor_2_rounded, primaries)), normalized=False)

            # --- Measured (or synthetic) spectra ---
            measured_1 = None
            measured_2 = None
            if not pred_only:
                if measurements_dir is None:
                    # Synthetic case: perturb BGOR weights by a small epsilon and
                    # reconstruct spectra from primaries.
                    noise_1 = np.random.uniform(-synthetic_epsilon,
                                                synthetic_epsilon,
                                                size=4)
                    noise_2 = np.random.uniform(-synthetic_epsilon,
                                                synthetic_epsilon,
                                                size=4)
                    bgor_1_noisy = np.clip(bgor_1 + noise_1, 0, None)
                    bgor_2_noisy = np.clip(bgor_2 + noise_2, 0, None)

                    measured_1_data = sum(w * p.data
                                          for w, p in zip(bgor_1_noisy, primaries))
                    measured_2_data = sum(w * p.data
                                          for w, p in zip(bgor_2_noisy, primaries))
                    measured_1 = Spectra(wavelengths=wavelengths,
                                         data=scaling_factor * measured_1_data, normalized=False)
                    measured_2 = Spectra(wavelengths=wavelengths,
                                         data=scaling_factor * measured_2_data, normalized=False)
                else:
                    measured_list = get_spectras_from_rgbo_list(
                        measurements_dir, [rgbo_1, rgbo_2], smooth_method='gaussian')
                    # Interpolate to exactly the same wavelength grid as the predicted
                    # spectra so that all downstream observer projections use identical
                    # wavelength assumptions with no further re-interpolation.
                    measured_1 = Spectra(wavelengths=wavelengths,
                                         data=scaling_factor * measured_list[0].interpolate_values(wavelengths).data,
                                         normalized=False)
                    measured_2 = Spectra(wavelengths=wavelengths,
                                         data=scaling_factor * measured_list[1].interpolate_values(wavelengths).data,
                                         normalized=False)

                    if measured_1 is None or measured_2 is None:
                        print(
                            f"    WARNING: Missing measurements for pair {pair_idx}, skipping"
                        )
                        continue

            # --- Project to LMSQ ---
            pred_lmsq_1 = observer.observe_spectras([predicted_1])[0]
            pred_lmsq_2 = observer.observe_spectras([predicted_2])[0]

            # LMS RMSE between the two metamers (should be ~0)
            pred_lms_diff = pred_lmsq_1[lms_indices] - pred_lmsq_2[lms_indices]
            pred_lms_rmse = np.sqrt(np.mean(pred_lms_diff ** 2))

            # Q difference between the two metamers (should be large)
            pred_q_diff = abs(pred_lmsq_1[q_index] - pred_lmsq_2[q_index])

            print(f"    LMS metamer RMSE: pred={pred_lms_rmse:.6f}")
            print(f"    Q metamer diff:   pred={pred_q_diff:.6f}")

            if measured_1 is not None:
                meas_lmsq_1 = observer.observe_spectras([measured_1])[0]
                meas_lmsq_2 = observer.observe_spectras([measured_2])[0]
                meas_lms_diff = meas_lmsq_1[lms_indices] - meas_lmsq_2[lms_indices]
                meas_lms_rmse = np.sqrt(np.mean(meas_lms_diff ** 2))
                meas_q_diff = abs(meas_lmsq_1[q_index] - meas_lmsq_2[q_index])
                lmsq_rmse_1 = np.sqrt(np.mean((pred_lmsq_1 - meas_lmsq_1) ** 2))
                lmsq_rmse_2 = np.sqrt(np.mean((pred_lmsq_2 - meas_lmsq_2) ** 2))
                spec_rmse_1 = float(np.sqrt(np.mean((predicted_1.data - measured_1.data) ** 2)))
                spec_rmse_2 = float(np.sqrt(np.mean((predicted_2.data - measured_2.data) ** 2)))
                print(f"    LMSQ RMSE: m1={lmsq_rmse_1:.4f}, m2={lmsq_rmse_2:.4f}")
                print(f"    LMS metamer RMSE (meas): {meas_lms_rmse:.6f}")
                print(f"    Q metamer diff   (meas): {meas_q_diff:.6f}")
                print(f"    Spectral RMSE (pred vs meas SPD): m1={spec_rmse_1:.4g}, m2={spec_rmse_2:.4g}")

            # Store for end-of-validation summary plots
            all_pair_data.append({
                'obs_idx': obs_idx,
                'pair_idx': pair_idx,
                'genotype': genotype,
                'scaling_factor': scaling_factor,
                'gt_cone_1': cone_1,
                'gt_cone_2': cone_2,
                'rgbo_1': rgbo_1,
                'rgbo_2': rgbo_2,
                'predicted_1': predicted_1,
                'predicted_2': predicted_2,
                'predicted_1_rounded': predicted_1_rounded,
                'predicted_2_rounded': predicted_2_rounded,
                'measured_1': measured_1,
                'measured_2': measured_2,
            })

            # ===== PLOT: 2-panel figure per metamer pair =====
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # --- Panel 1: Predicted vs Measured Spectra ---
            ax = axes[0]
            ax.plot(wavelengths, predicted_1.data, 'b-', lw=2, label='Predicted M1', alpha=0.8)
            ax.plot(wavelengths, predicted_2.data, 'r-', lw=2, label='Predicted M2', alpha=0.8)
            if measured_1 is not None:
                ax.plot(measured_1.wavelengths, measured_1.data, 'b--', lw=2, label='Measured M1', alpha=0.8)
                ax.plot(measured_2.wavelengths, measured_2.data, 'r--', lw=2, label='Measured M2', alpha=0.8)
            ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Power', fontsize=14, fontweight='bold')
            ax.set_title('Predicted vs Measured Spectra', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', labelsize=10)

            # --- Panel 2: LMSQ comparison ---
            ax = axes[1]
            cone_labels = list(sorted_with_s)
            cone_labels_str = [f'{wl}nm' for wl in cone_labels]
            # Mark Q cone
            cone_labels_str[q_index] = f'{cone_labels[q_index]}nm (Q)'

            x = np.arange(len(sorted_with_s))
            w = 0.15
            ax.bar(x - 2.5*w, cone_1, w, label='GT M1', color='forestgreen',
                   alpha=0.9, edgecolor='black', linewidth=0.5)
            ax.bar(x - 1.5*w, cone_2, w, label='GT M2', color='darkorange',
                   alpha=0.9, edgecolor='black', linewidth=0.5)
            ax.bar(x - 0.5*w, pred_lmsq_1, w, label='Pred M1', color='steelblue',
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.bar(x + 0.5*w, pred_lmsq_2, w, label='Pred M2', color='indianred',
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            if measured_1 is not None:
                ax.bar(x + 1.5*w, meas_lmsq_1, w, label='Meas M1', color='steelblue',
                       alpha=0.4, edgecolor='steelblue', linewidth=1.5)
                ax.bar(x + 2.5*w, meas_lmsq_2, w, label='Meas M2', color='indianred',
                       alpha=0.4, edgecolor='indianred', linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cone_labels_str, fontsize=8)
            ax.set_ylabel('Cone Response', fontsize=14, fontweight='bold')
            _meas_title = (f' meas={meas_lms_rmse:.4f}\nQ diff: pred={pred_q_diff:.4f} meas={meas_q_diff:.4f}'
                           if measured_1 is not None else '')
            ax.set_title(
                f'LMSQ Projection\n'
                f'LMS RMSE: pred={pred_lms_rmse:.4f}{_meas_title}',
                fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', labelsize=10)

            fig.suptitle(
                f'Observer {obs_idx} (genotype {genotype}) — Metamer Pair {pair_idx}\n'
                f'RGBO1={rgbo_1}  RGBO2={rgbo_2}',
                fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            fname = plots_path / f'obs{obs_idx}_pair{pair_idx}.png'
            # plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {fname}")

    # ===== END-OF-VALIDATION: multi-observer and hyperobserver summary plots =====
    if not all_pair_data:
        return

    print("\nGenerating end-of-validation summary plots...")

    # Build one Observer per config observer (reuse across all pairs)
    config_observer_list = []  # list of (obs_idx, peaks_tuple, Observer)
    for obs_data in config['observers']:
        g = tuple(sorted(obs_data['genotype']))
        obs = observer_genotypes.get_observer_for_peaks(
            g, degree=VALIDATION_OBSERVER_DEGREE)
        config_observer_list.append((obs_data['observer_index'], g, obs))

    # Build the 12D hyperobserver once
    print("  Building hyperobserver (12D)...")
    hyperobs = Observer.hyperobserver(
        wavelengths=wavelengths, degree=VALIDATION_OBSERVER_DEGREE)

    for pair_data in all_pair_data:
        fname_a = _plot_all_observers_bars(pair_data, config_observer_list, plots_path)
        fname_b = _plot_hyperobserver_bars(pair_data, hyperobs, plots_path)
        fname_c = _plot_hyperobserver_diff(pair_data, hyperobs, plots_path)
        fname_d = _plot_cone_distance_bars(pair_data, config_observer_list, plots_path)
        fname_e = _plot_spectral_additive_residual(pair_data, plots_path)
        print(
            f"  Saved: {fname_a.name}  |  {fname_b.name}  |  {fname_c.name}  |  "
            f"{fname_d.name}  |  {fname_e.name}"
        )

    _generate_rmse_summary(all_pair_data, hyperobs, config_observer_list, plots_path)
    print(f"Summary plots saved to {plots_path}")


def _write_csv(path, fieldnames, rows):
    """Write a list-of-dicts to a CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_spectral_additive_residual(pair_data: dict, plots_path: Path) -> Path:
    """Per-wavelength deviation from linear additive primary model.

    Scalar spectral RMSE per metamer:
        RMSE = sqrt(mean_λ (pred(λ) − meas(λ))²)
    in the same power units as the predicted/measured SPD curves.
    """
    obs_idx = pair_data['obs_idx']
    pair_idx = pair_data['pair_idx']
    genotype = pair_data['genotype']
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_spectral_rmse.png'
    if meas_1 is None:
        return fname

    wl = np.asarray(pred_1.wavelengths, dtype=float)
    d1 = np.asarray(pred_1.data, dtype=float) - np.asarray(meas_1.data, dtype=float)
    d2 = np.asarray(pred_2.data, dtype=float) - np.asarray(meas_2.data, dtype=float)
    rmse1 = float(np.sqrt(np.mean(d1 ** 2)))
    rmse2 = float(np.sqrt(np.mean(d2 ** 2)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(wl, d1, color='steelblue', lw=1.6, alpha=0.9, label='M1 pred − meas')
    ax0.plot(wl, d2, color='indianred', lw=1.6, alpha=0.9, label='M2 pred − meas')
    ax0.axhline(0.0, color='black', lw=0.7, alpha=0.45)
    ax0.set_ylabel('Δ power', fontsize=12, fontweight='bold')
    ax0.set_title('Signed residual (pred − meas)', fontsize=11, fontweight='bold')
    ax0.legend(loc='best', fontsize=9)
    ax0.grid(True, alpha=0.3, linestyle='--')
    ax0.spines['top'].set_visible(False)

    ax1.plot(wl, np.abs(d1), color='steelblue', lw=1.6, alpha=0.9, label='|M1|')
    ax1.plot(wl, np.abs(d2), color='indianred', lw=1.6, alpha=0.9, label='|M2|')
    ax1.fill_between(wl, 0.0, np.abs(d1), color='steelblue', alpha=0.12)
    ax1.fill_between(wl, 0.0, np.abs(d2), color='indianred', alpha=0.12)
    ax1.set_ylabel('|pred − meas|', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute residual vs wavelength', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)

    ax2.plot(wl, d1 ** 2, color='steelblue', lw=1.4, alpha=0.85,
             label=f'M1 (mean = MSE = {float(np.mean(d1**2)):.4g})')
    ax2.plot(wl, d2 ** 2, color='indianred', lw=1.4, alpha=0.85,
             label=f'M2 (mean = MSE = {float(np.mean(d2**2)):.4g})')
    ax2.set_ylabel('(pred − meas)²', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Squared residual (integrand of spectral MSE; RMSE = √mean)',
        fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)

    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx} — Spectral deviation from additive primaries\n'
        f'Genotype {genotype}  |  RMSE(M1)={rmse1:.4g}  RMSE(M2)={rmse2:.4g}\n'
        'Predicted SPD = Σ BGOR float weights × measured primaries (validation scaling).',
        fontsize=9, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.93))
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname


def _plot_all_observers_bars(
    pair_data: dict,
    config_observers: list,
    plots_path,
):
    """One figure per metamer pair: a column per config observer showing cone-response bars.

    Args:
        pair_data: dict with keys obs_idx, pair_idx, genotype, predicted_1, predicted_2
        config_observers: list of (obs_idx, peaks_tuple, Observer) for every config observer
        plots_path: pathlib.Path to save directory
    """
    obs_idx = pair_data['obs_idx']
    pair_idx = pair_data['pair_idx']
    designed_genotype = pair_data['genotype']
    gt_cone_1 = pair_data['gt_cone_1']
    gt_cone_2 = pair_data['gt_cone_2']
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']
    pred_1_rounded = pair_data['predicted_1_rounded']
    pred_2_rounded = pair_data['predicted_2_rounded']
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    n_obs = len(config_observers)
    fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 5))
    if n_obs == 1:
        axes = [axes]

    for ax, (c_obs_idx, c_peaks, c_observer) in zip(axes, config_observers):
        # Sorted peaks including S cone
        sorted_peaks = sorted(c_peaks) if 420 in c_peaks else sorted((420,) + c_peaks)
        cone_labels = [f'{p}nm' for p in sorted_peaks]

        # Mark Q cone (first peak in the L-opsin range that isn't 559 standard)
        if c_observer.dimension == 4 and 547 in sorted_peaks:
            q_idx = sorted_peaks.index(547)
            cone_labels[q_idx] += '\n(Q)'

        lmsq_1 = c_observer.observe_spectras([pred_1])[0]
        lmsq_2 = c_observer.observe_spectras([pred_2])[0]
        lmsq_1_rounded = c_observer.observe_spectras([pred_1_rounded])[0]
        lmsq_2_rounded = c_observer.observe_spectras([pred_2_rounded])[0]
        if meas_1 is not None:
            lmsq_meas_1 = c_observer.observe_spectras([meas_1])[0]
            lmsq_meas_2 = c_observer.observe_spectras([meas_2])[0]

        is_designed = (c_peaks == designed_genotype)
        x = np.arange(len(sorted_peaks))

        if is_designed:
            # GT | Pred | Pred 8-bit | [Meas] bars for the designed observer
            w = 0.11 if meas_1 is not None else 0.14
            ax.bar(x - 3.5*w, gt_cone_1,      w, color='forestgreen', alpha=0.9,
                   edgecolor='black', linewidth=0.5, label='GT M1')
            ax.bar(x - 2.5*w, gt_cone_2,      w, color='darkorange',  alpha=0.9,
                   edgecolor='black', linewidth=0.5, label='GT M2')
            ax.bar(x - 1.5*w, lmsq_1,         w, color='steelblue',   alpha=0.85,
                   edgecolor='black', linewidth=0.5, label='Pred M1')
            ax.bar(x - 0.5*w, lmsq_2,         w, color='indianred',   alpha=0.85,
                   edgecolor='black', linewidth=0.5, label='Pred M2')
            ax.bar(x + 0.5*w, lmsq_1_rounded, w, color='steelblue',   alpha=0.45,
                   edgecolor='steelblue', linewidth=1.2, linestyle=':', label='Pred M1 (8-bit)')
            ax.bar(x + 1.5*w, lmsq_2_rounded, w, color='indianred',   alpha=0.45,
                   edgecolor='indianred', linewidth=1.2, linestyle=':', label='Pred M2 (8-bit)')
            if meas_1 is not None:
                ax.bar(x + 2.5*w, lmsq_meas_1, w, color='steelblue', alpha=0.25,
                       edgecolor='steelblue', linewidth=1.5, label='Meas M1')
                ax.bar(x + 3.5*w, lmsq_meas_2, w, color='indianred', alpha=0.25,
                       edgecolor='indianred', linewidth=1.5, label='Meas M2')
        else:
            # Pred | Pred 8-bit | [Meas] bars for other observers
            w = 0.14 if meas_1 is not None else 0.2
            ax.bar(x - 2.5*w, lmsq_1,         w, color='steelblue', alpha=0.85,
                   edgecolor='black', linewidth=0.5, label='Pred M1')
            ax.bar(x - 1.5*w, lmsq_2,         w, color='indianred', alpha=0.85,
                   edgecolor='black', linewidth=0.5, label='Pred M2')
            ax.bar(x - 0.5*w, lmsq_1_rounded, w, color='steelblue', alpha=0.45,
                   edgecolor='steelblue', linewidth=1.2, linestyle=':', label='Pred M1 (8-bit)')
            ax.bar(x + 0.5*w, lmsq_2_rounded, w, color='indianred', alpha=0.45,
                   edgecolor='indianred', linewidth=1.2, linestyle=':', label='Pred M2 (8-bit)')
            if meas_1 is not None:
                ax.bar(x + 1.5*w, lmsq_meas_1, w, color='steelblue', alpha=0.25,
                       edgecolor='steelblue', linewidth=1.5, label='Meas M1')
                ax.bar(x + 2.5*w, lmsq_meas_2, w, color='indianred', alpha=0.25,
                       edgecolor='indianred', linewidth=1.5, label='Meas M2')
        ax.set_xticks(x)
        ax.set_xticklabels(cone_labels, fontsize=7)
        ax.set_ylabel('Cone Response', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)

        title = f'Obs {c_obs_idx}\n{c_peaks}'
        if c_peaks == designed_genotype:
            title += '\n★ designed for'
        ax.set_title(title, fontsize=8, fontweight='bold')

        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx}: Cone Responses Across All Observers',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_all_observers.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # --- CSV export ---
    csv_rows = []
    for c_obs_idx, c_peaks, c_observer in config_observers:
        sorted_peaks = sorted(c_peaks) if 420 in c_peaks else sorted((420,) + c_peaks)
        cone_labels_csv = [f'{p}nm' for p in sorted_peaks]
        lmsq_1 = c_observer.observe_spectras([pred_1])[0]
        lmsq_2 = c_observer.observe_spectras([pred_2])[0]
        lmsq_1r = c_observer.observe_spectras([pred_1_rounded])[0]
        lmsq_2r = c_observer.observe_spectras([pred_2_rounded])[0]
        if meas_1 is not None:
            lmsq_m1 = c_observer.observe_spectras([meas_1])[0]
            lmsq_m2 = c_observer.observe_spectras([meas_2])[0]
        is_des = (c_peaks == designed_genotype)
        for ci, (label, peak) in enumerate(zip(cone_labels_csv, sorted_peaks)):
            row = {
                'obs_idx': obs_idx, 'pair_idx': pair_idx,
                'designed_genotype': str(designed_genotype),
                'observer_col_idx': c_obs_idx,
                'observer_col_genotype': str(c_peaks),
                'is_designed_observer': is_des,
                'cone_label': label, 'cone_peak_nm': peak,
                'pred_m1': lmsq_1[ci], 'pred_m2': lmsq_2[ci],
                'pred_8bit_m1': lmsq_1r[ci], 'pred_8bit_m2': lmsq_2r[ci],
                'meas_m1': lmsq_m1[ci] if meas_1 is not None else None,
                'meas_m2': lmsq_m2[ci] if meas_1 is not None else None,
            }
            if is_des:
                row['gt_m1'] = gt_cone_1[ci]
                row['gt_m2'] = gt_cone_2[ci]
            else:
                row['gt_m1'] = ''
                row['gt_m2'] = ''
            csv_rows.append(row)
    _write_csv(
        plots_path / f'obs{obs_idx}_pair{pair_idx}_all_observers.csv',
        ['obs_idx', 'pair_idx', 'designed_genotype', 'observer_col_idx',
         'observer_col_genotype', 'is_designed_observer', 'cone_label', 'cone_peak_nm',
         'gt_m1', 'gt_m2', 'pred_m1', 'pred_m2',
         'pred_8bit_m1', 'pred_8bit_m2', 'meas_m1', 'meas_m2'],
        csv_rows)
    return fname


def _draw_rmse_panel(ax, pred_1, pred_2, pred_1r, pred_2r, meas_1, meas_2, ref_ax=None):
    """Draw pairwise RMSE bars (hyperobserver space) into an existing Axes.

    Three groups on the x-axis: Pred vs 8-bit, Pred vs Meas, 8-bit vs Meas.
    Each group has two bars: M1 (steelblue) and M2 (indianred).
    If ref_ax is provided, the y-axis is locked to the same scale.
    """
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    groups = [('Pred\nvs 8-bit', rmse(pred_1, pred_1r), rmse(pred_2, pred_2r))]
    if meas_1 is not None:
        groups += [
            ('Pred\nvs Meas',  rmse(pred_1,  meas_1), rmse(pred_2,  meas_2)),
            ('8-bit\nvs Meas', rmse(pred_1r, meas_1), rmse(pred_2r, meas_2)),
        ]
    labels, m1_vals, m2_vals = zip(*groups)
    xg = np.arange(len(groups))
    w = 0.3
    ax.bar(xg - w/2, m1_vals, w, color='steelblue', alpha=0.8,
           edgecolor='black', linewidth=0.5, label='M1')
    ax.bar(xg + w/2, m2_vals, w, color='indianred', alpha=0.8,
           edgecolor='black', linewidth=0.5, label='M2')
    ax.set_xticks(xg)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('RMSE (hyperobserver)', fontsize=12, fontweight='bold')
    ax.set_title('Pairwise\nRMSE', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    if ref_ax is not None:
        ax.set_ylim(ref_ax.get_ylim())


def _plot_hyperobserver_bars(
    pair_data: dict,
    hyperobserver,
    plots_path,
):
    """One figure per metamer pair showing cone responses in the 12D hyperobserver.

    Args:
        pair_data: dict with keys obs_idx, pair_idx, genotype, predicted_1, predicted_2
        hyperobserver: 12D Observer covering all human opsin variants
        plots_path: pathlib.Path to save directory
    """
    obs_idx = pair_data['obs_idx']
    pair_idx = pair_data['pair_idx']
    designed_genotype = pair_data['genotype']
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']
    pred_1_rounded = pair_data['predicted_1_rounded']
    pred_2_rounded = pair_data['predicted_2_rounded']
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    hyper_pred_1 = hyperobserver.observe_spectras([pred_1])[0]
    hyper_pred_2 = hyperobserver.observe_spectras([pred_2])[0]
    hyper_pred_1_rounded = hyperobserver.observe_spectras([pred_1_rounded])[0]
    hyper_pred_2_rounded = hyperobserver.observe_spectras([pred_2_rounded])[0]
    if meas_1 is not None:
        hyper_meas_1 = hyperobserver.observe_spectras([meas_1])[0]
        hyper_meas_2 = hyperobserver.observe_spectras([meas_2])[0]

    # Hyperobserver peaks in ascending order
    hyper_peaks = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
    designed_set = set(designed_genotype) | {420}
    designed_mask = [p in designed_set for p in hyper_peaks]

    # Labels match hyperobserver peak order: S, M×3, L×8
    cone_labels = [
        'S\n420', 'M\n530', 'M\n533', 'M\n536',
        'L\n547', 'L\n551', 'L\n552', 'L\n553',
        'L\n555', 'L\n556', 'L\n556.5', 'L\n559',
    ]
    x = np.arange(len(cone_labels))
    w = 0.13

    fig, (ax_spec, ax, ax_rmse) = plt.subplots(1, 3, figsize=(28, 5),
                                               gridspec_kw={'width_ratios': [1, 2.5, 0.7]})

    # --- Left panel: spectra ---
    ax_spec.plot(pred_1.wavelengths, pred_1.data, color='steelblue', lw=2, alpha=0.6, linestyle='--', label='Pred M1')
    ax_spec.plot(pred_2.wavelengths, pred_2.data, color='indianred', lw=2, alpha=0.6, linestyle='--', label='Pred M2')
    ax_spec.plot(pred_1_rounded.wavelengths, pred_1_rounded.data, color='steelblue', lw=1.5,
                 alpha=0.6, linestyle=':', label='Pred M1 (8-bit)')
    ax_spec.plot(pred_2_rounded.wavelengths, pred_2_rounded.data, color='indianred', lw=1.5,
                 alpha=0.6, linestyle=':', label='Pred M2 (8-bit)')
    if meas_1 is not None:
        ax_spec.plot(meas_1.wavelengths, meas_1.data, color='steelblue', lw=2,
                     alpha=0.8, label='Meas M1')
        ax_spec.plot(meas_2.wavelengths, meas_2.data, color='indianred', lw=2,
                     alpha=0.8, label='Meas M2')
    ax_spec.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax_spec.set_ylabel('Power', fontsize=14, fontweight='bold')
    ax_spec.set_title('Spectra', fontsize=11, fontweight='bold')
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, linestyle='--')
    ax_spec.spines['top'].set_visible(False)
    ax_spec.tick_params(axis='y', labelsize=10)

    # --- Middle panel: hyperobserver bars (4 or 6 bars per cone) ---
    ax.bar(x - 2.5*w, hyper_pred_1,         w, color='steelblue', alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Pred M1')
    ax.bar(x - 1.5*w, hyper_pred_2,         w, color='indianred', alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Pred M2')
    ax.bar(x - 0.5*w, hyper_pred_1_rounded, w, color='steelblue', alpha=0.55,
           edgecolor='steelblue', linewidth=1.2, linestyle=':', label='Pred M1 (8-bit)')
    ax.bar(x + 0.5*w, hyper_pred_2_rounded, w, color='indianred', alpha=0.55,
           edgecolor='indianred', linewidth=1.2, linestyle=':', label='Pred M2 (8-bit)')
    if meas_1 is not None:
        ax.bar(x + 1.5*w, hyper_meas_1,     w, color='steelblue', alpha=0.3,
               edgecolor='steelblue', linewidth=1.5, label='Meas M1')
        ax.bar(x + 2.5*w, hyper_meas_2,     w, color='indianred', alpha=0.3,
               edgecolor='indianred', linewidth=1.5, label='Meas M2')

    # Shade S / M / L regions
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='blue')
    ax.axvspan(0.5, 3.5, alpha=0.06, color='green')
    ax.axvspan(3.5, 11.5, alpha=0.06, color='red')

    ax.set_xticks(x)
    # Bold tick labels for designed cones
    ax.set_xticklabels(cone_labels, fontsize=8)
    for tick, is_designed in zip(ax.get_xticklabels(), designed_mask):
        if is_designed:
            tick.set_fontweight('bold')
            tick.set_fontsize(9)

    ax.set_ylabel('Cone Response', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Hyperobserver (12D) — Observer {obs_idx} · Pair {pair_idx}\n'
        f'Designed for genotype {designed_genotype}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)

    # --- Right panel: pairwise RMSE in hyperobserver space ---
    _draw_rmse_panel(ax_rmse, hyper_pred_1, hyper_pred_2,
                     hyper_pred_1_rounded, hyper_pred_2_rounded,
                     hyper_meas_1 if meas_1 is not None else None,
                     hyper_meas_2 if meas_1 is not None else None,
                     ref_ax=ax)

    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx} — Designed for genotype {designed_genotype}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # --- CSV export ---
    cone_labels_flat = ['S_420', 'M_530', 'M_533', 'M_536',
                        'L_547', 'L_551', 'L_552', 'L_553',
                        'L_555', 'L_556', 'L_556.5', 'L_559']
    csv_rows = []
    for ci, (label, peak, is_des) in enumerate(
            zip(cone_labels_flat, hyper_peaks, designed_mask)):
        csv_rows.append({
            'obs_idx': obs_idx, 'pair_idx': pair_idx,
            'designed_genotype': str(designed_genotype),
            'cone_label': label, 'cone_peak_nm': peak, 'is_designed': is_des,
            'pred_m1':      hyper_pred_1[ci],
            'pred_m2':      hyper_pred_2[ci],
            'pred_8bit_m1': hyper_pred_1_rounded[ci],
            'pred_8bit_m2': hyper_pred_2_rounded[ci],
            'meas_m1':      float(hyper_meas_1[ci]) if meas_1 is not None else None,
            'meas_m2':      float(hyper_meas_2[ci]) if meas_1 is not None else None,
        })
    _write_csv(
        plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver.csv',
        ['obs_idx', 'pair_idx', 'designed_genotype', 'cone_label', 'cone_peak_nm',
         'is_designed', 'pred_m1', 'pred_m2', 'pred_8bit_m1', 'pred_8bit_m2',
         'meas_m1', 'meas_m2'],
        csv_rows)

    # --- Spectra CSV export ---
    spec_rows = []
    for i, wl in enumerate(pred_1.wavelengths):
        spec_rows.append({
            'obs_idx': obs_idx, 'pair_idx': pair_idx,
            'designed_genotype': str(designed_genotype),
            'wavelength': float(wl),
            'pred_m1':      float(pred_1.data[i]),
            'pred_m2':      float(pred_2.data[i]),
            'pred_8bit_m1': float(pred_1_rounded.data[i]),
            'pred_8bit_m2': float(pred_2_rounded.data[i]),
            'meas_m1':      float(meas_1.data[i]) if meas_1 is not None else None,
            'meas_m2':      float(meas_2.data[i]) if meas_1 is not None else None,
        })
    _write_csv(
        plots_path / f'obs{obs_idx}_pair{pair_idx}_spectra.csv',
        ['obs_idx', 'pair_idx', 'designed_genotype', 'wavelength',
         'pred_m1', 'pred_m2', 'pred_8bit_m1', 'pred_8bit_m2', 'meas_m1', 'meas_m2'],
        spec_rows)
    return fname


def _plot_hyperobserver_diff(
    pair_data: dict,
    hyperobserver,
    plots_path,
):
    """Difference plot (M1 - M2) in the 12D hyperobserver for predicted and measured.

    Two bars per cone: predicted difference and measured difference.
    LMS cones should be ~0; the targeted Q cone should be large.
    """
    obs_idx = pair_data['obs_idx']
    pair_idx = pair_data['pair_idx']
    designed_genotype = pair_data['genotype']
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']
    pred_1_rounded = pair_data['predicted_1_rounded']
    pred_2_rounded = pair_data['predicted_2_rounded']
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']
    rgbo_1 = pair_data.get('rgbo_1')
    rgbo_2 = pair_data.get('rgbo_2')

    hyper_pred_1 = hyperobserver.observe_spectras([pred_1])[0]
    hyper_pred_2 = hyperobserver.observe_spectras([pred_2])[0]
    hyper_pred_1_rounded = hyperobserver.observe_spectras([pred_1_rounded])[0]
    hyper_pred_2_rounded = hyperobserver.observe_spectras([pred_2_rounded])[0]
    if meas_1 is not None:
        hyper_meas_1 = hyperobserver.observe_spectras([meas_1])[0]
        hyper_meas_2 = hyperobserver.observe_spectras([meas_2])[0]
        meas_diff = np.abs(hyper_meas_1 - hyper_meas_2)
    else:
        hyper_meas_1 = hyper_meas_2 = meas_diff = None

    pred_diff = np.abs(hyper_pred_1 - hyper_pred_2)
    pred_rounded_diff = np.abs(hyper_pred_1_rounded - hyper_pred_2_rounded)

    hyper_peaks = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
    designed_set = set(designed_genotype) | {420}
    designed_mask = np.array([p in designed_set for p in hyper_peaks])

    cone_labels = [
        'S\n420', 'M\n530', 'M\n533', 'M\n536',
        'L\n547', 'L\n551', 'L\n552', 'L\n553',
        'L\n555', 'L\n556', 'L\n556.5', 'L\n559',
    ]
    x = np.arange(len(cone_labels))
    w = 0.22

    fig, (ax_spec, ax, ax_rmse) = plt.subplots(1, 3, figsize=(28, 5),
                                               gridspec_kw={'width_ratios': [1, 2.5, 0.7]})

    # --- Left panel: spectra ---
    ax_spec.plot(pred_1.wavelengths, pred_1.data, color='steelblue', lw=2, alpha=0.6, linestyle='--', label='Pred M1')
    ax_spec.plot(pred_2.wavelengths, pred_2.data, color='indianred', lw=2, alpha=0.6, linestyle='--', label='Pred M2')
    ax_spec.plot(pred_1_rounded.wavelengths, pred_1_rounded.data, color='steelblue', lw=1.5,
                 alpha=0.6, linestyle=':', label='Pred M1 (8-bit)')
    ax_spec.plot(pred_2_rounded.wavelengths, pred_2_rounded.data, color='indianred', lw=1.5,
                 alpha=0.6, linestyle=':', label='Pred M2 (8-bit)')
    if meas_1 is not None:
        ax_spec.plot(meas_1.wavelengths, meas_1.data, color='steelblue', lw=2,
                     alpha=0.8, label='Meas M1')
        ax_spec.plot(meas_2.wavelengths, meas_2.data, color='indianred', lw=2,
                     alpha=0.8, label='Meas M2')
    ax_spec.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax_spec.set_ylabel('Power', fontsize=14, fontweight='bold')
    ax_spec.set_title('Spectra', fontsize=11, fontweight='bold')
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, linestyle='--')
    ax_spec.spines['top'].set_visible(False)
    ax_spec.tick_params(axis='y', labelsize=10)

    # --- Middle panel: absolute difference bars (Pred, Pred 8-bit, Meas) ---
    ax.bar(x - w,     pred_diff,         w, color='steelblue', alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Pred |M1−M2|')
    ax.bar(x,         pred_rounded_diff, w, color='steelblue', alpha=0.55,
           edgecolor='steelblue', linewidth=1.2, linestyle=':', label='Pred 8-bit |M1−M2|')
    if meas_diff is not None:
        ax.bar(x + w, meas_diff, w, color='steelblue', alpha=0.3,
               edgecolor='steelblue', linewidth=1.5, label='Meas |M1−M2|')
        # Dashed horizontal line at the max non-Q designed cone difference (measured)
        non_q_mask = designed_mask & np.array([p != 547 and p != 420 for p in hyper_peaks])
        non_q_max = meas_diff[non_q_mask].max()
        ax.axhline(non_q_max, color='black', linewidth=1.2, linestyle='--',
                   label=f'Non-Q target max (meas) ({non_q_max:.4f})')

    # Shade S / M / L regions
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='blue')
    ax.axvspan(0.5, 3.5, alpha=0.06, color='green')
    ax.axvspan(3.5, 11.5, alpha=0.06, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(cone_labels, fontsize=8)
    for tick, is_designed in zip(ax.get_xticklabels(), designed_mask):
        if is_designed:
            tick.set_fontweight('bold')
            tick.set_fontsize(9)

    ax.set_ylabel('Absolute Cone Response Difference |M1 − M2|', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Hyperobserver |Difference| (M1−M2) — Observer {obs_idx} · Pair {pair_idx}\n'
        f'Designed for genotype {designed_genotype}\n'
        + (f'RGBO1={rgbo_1}  RGBO2={rgbo_2}' if rgbo_1 is not None else ''),
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)

    # --- Right panel: pairwise RMSE ---
    _draw_rmse_panel(ax_rmse, hyper_pred_1, hyper_pred_2,
                     hyper_pred_1_rounded, hyper_pred_2_rounded,
                     hyper_meas_1, hyper_meas_2, ref_ax=ax)
    # (meas_1/2 may be None; _draw_rmse_panel handles None gracefully)

    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver_diff.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # --- CSV export ---
    cone_labels_flat = ['S_420', 'M_530', 'M_533', 'M_536',
                        'L_547', 'L_551', 'L_552', 'L_553',
                        'L_555', 'L_556', 'L_556.5', 'L_559']
    csv_rows = []
    for ci, (label, peak, is_des) in enumerate(
            zip(cone_labels_flat, hyper_peaks, designed_mask)):
        csv_rows.append({
            'obs_idx': obs_idx, 'pair_idx': pair_idx,
            'designed_genotype': str(designed_genotype),
            'cone_label': label, 'cone_peak_nm': peak, 'is_designed': bool(is_des),
            'pred_diff':      float(pred_diff[ci]),
            'pred_8bit_diff': float(pred_rounded_diff[ci]),
            'meas_diff':      float(meas_diff[ci]) if meas_diff is not None else None,
        })
    _write_csv(
        plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver_diff.csv',
        ['obs_idx', 'pair_idx', 'designed_genotype', 'cone_label', 'cone_peak_nm',
         'is_designed', 'pred_diff', 'pred_8bit_diff', 'meas_diff'],
        csv_rows)
    return fname


def _cone_distance(observer, s1, s2, metameric_axis=2):
    """Thin wrapper around Observer.cone_distance for use in plotting helpers."""
    return observer.cone_distance(s1, s2, metameric_axis=metameric_axis)


def _plot_cone_distance_bars(
    pair_data: dict,
    config_observers: list,
    plots_path,
):
    """Per-observer noise-weighted cone distance plot.

    Each column corresponds to one observer and shows the Mahalanobis distance
    in normalized cone space between M1 and M2 for pred, 8-bit, and measured.
    d ≈ 1 JND; d > 3 is clearly distinguishable.
    """
    obs_idx = pair_data['obs_idx']
    pair_idx = pair_data['pair_idx']
    designed_genotype = pair_data['genotype']
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']
    pred_1_rounded = pair_data['predicted_1_rounded']
    pred_2_rounded = pair_data['predicted_2_rounded']
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    n_obs = len(config_observers)
    fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 5), sharey=True)
    if n_obs == 1:
        axes = [axes]

    series = [
        # (label,  spectra pair,              lms_alpha, q_alpha, edge)
        ('Pred',   pred_1,        pred_2,        0.85,      0.85,   None),
        ('8-bit',  pred_1_rounded, pred_2_rounded, 0.45,    0.45,  'steelblue'),
    ]
    if meas_1 is not None:
        series.append(('Meas', meas_1, meas_2, 0.25, 0.25, 'steelblue'))

    for ax, (c_obs_idx, c_peaks, c_observer) in zip(axes, config_observers):
        has_q = (c_observer.dimension == 4)
        # Determine Q (547nm) index in this observer's sorted cone order
        sorted_peaks = sorted((420,) + c_peaks) if 420 not in c_peaks else sorted(c_peaks)
        q_axis = sorted_peaks.index(547) if (has_q and 547 in sorted_peaks) else 2

        lms_vals, q_vals = [], []
        for _, s1, s2, _, _, _ in series:
            d_lms, d_q = _cone_distance(c_observer, s1, s2, metameric_axis=q_axis)
            lms_vals.append(d_lms)
            q_vals.append(d_q)

        n = len(series)
        x = np.arange(n)
        w = 0.35

        for i, (lbl, _, _, alms, aq, ec) in enumerate(series):
            kw_lms = dict(color='steelblue', alpha=alms, edgecolor='black', linewidth=0.5)
            kw_q = dict(color='indianred',  alpha=aq,   edgecolor='black', linewidth=0.5)
            if ec:
                kw_lms['edgecolor'] = kw_q['edgecolor'] = ec
                kw_lms['linewidth'] = kw_q['linewidth'] = 1.5

            ax.bar(i - w/2, lms_vals[i], w, label='LMS dist' if i == 0 else '', **kw_lms)
            if has_q:
                ax.bar(i + w/2, q_vals[i], w, label='Q dist' if i == 0 else '', **kw_q)

            for j, (val, offset) in enumerate([(lms_vals[i], -w/2),
                                               (q_vals[i] if has_q else None, w/2)]):
                if val is None:
                    continue
                ax.text(i + offset, val + 0.01, f'{val:.2f}',
                        ha='center', va='bottom', fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in series], fontsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(3.0, color='gray', linewidth=0.8, linestyle=':',  alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)

        title = f'Obs {c_obs_idx}\n{c_peaks}'
        if c_peaks == designed_genotype:
            title += '\n★ designed for'
        ax.set_title(title, fontsize=8, fontweight='bold')
        if ax is axes[0]:
            ax.legend(fontsize=7)

    axes[0].set_ylabel('Noise-weighted cone distance d (JND units)\nBlue=LMS  Red=Q',
                       fontsize=14, fontweight='bold')
    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx}: Cone Distance M1 vs M2\n'
        f'(LMS should be ~0; Q should be large for designed observer)\n'
        f'Designed for genotype {designed_genotype}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_cone_dist.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # --- CSV export ---
    csv_rows = []
    for c_obs_idx, c_peaks, c_observer in config_observers:
        sorted_peaks = sorted((420,) + c_peaks) if 420 not in c_peaks else sorted(c_peaks)
        q_ax = sorted_peaks.index(547) if (c_observer.dimension == 4 and 547 in sorted_peaks) else 2
        for lbl, s1, s2, _, _, _ in series:
            d_lms, d_q = _cone_distance(c_observer, s1, s2, metameric_axis=q_ax)
            csv_rows.append({
                'obs_idx': obs_idx, 'pair_idx': pair_idx,
                'designed_genotype': str(designed_genotype),
                'observer_col_idx': c_obs_idx,
                'observer_col_genotype': str(c_peaks),
                'is_designed_observer': (c_peaks == designed_genotype),
                'q_axis': q_ax, 'series': lbl,
                'd_lms': d_lms, 'd_q': d_q,
            })
    _write_csv(
        plots_path / f'obs{obs_idx}_pair{pair_idx}_cone_dist.csv',
        ['obs_idx', 'pair_idx', 'designed_genotype', 'observer_col_idx',
         'observer_col_genotype', 'is_designed_observer', 'q_axis', 'series',
         'd_lms', 'd_q'],
        csv_rows)
    return fname


def _generate_rmse_summary(
    all_pair_data: list,
    hyperobserver,
    config_observers: list,
    plots_path,
):
    """Aggregate hyperobserver RMSE across all pairs and generate summary plot + CSV.

    For each pair computes RMSE(pred, 8-bit), RMSE(pred, meas), RMSE(8-bit, meas)
    averaged over M1 and M2.  Results are grouped by observer and overall.

    Saves:
        rmse_summary.png  — grouped bar chart
        rmse_summary.csv  — tabular data
    """
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    comparisons = ['Pred vs 8-bit', 'Pred vs Meas', '8-bit vs Meas']

    # Collect per-pair RMSE values keyed by obs_idx
    # Structure: {obs_idx: {comparison: [values]}}
    obs_records = {}  # obs_idx -> list of (comparison, value) for averaging

    rows = []  # for CSV: one row per pair × comparison

    for pair_data in all_pair_data:
        obs_idx = pair_data['obs_idx']
        pair_idx = pair_data['pair_idx']
        genotype = pair_data['genotype']

        hp1 = hyperobserver.observe_spectras([pair_data['predicted_1']])[0]
        hp2 = hyperobserver.observe_spectras([pair_data['predicted_2']])[0]
        hp1r = hyperobserver.observe_spectras([pair_data['predicted_1_rounded']])[0]
        hp2r = hyperobserver.observe_spectras([pair_data['predicted_2_rounded']])[0]
        m1 = pair_data['measured_1']
        m2 = pair_data['measured_2']
        vals = {'Pred vs 8-bit': (rmse(hp1, hp1r) + rmse(hp2, hp2r)) / 2}
        if m1 is not None:
            hm1 = hyperobserver.observe_spectras([m1])[0]
            hm2 = hyperobserver.observe_spectras([m2])[0]
            vals['Pred vs Meas']  = (rmse(hp1, hm1) + rmse(hp2, hm2)) / 2
            vals['8-bit vs Meas'] = (rmse(hp1r, hm1) + rmse(hp2r, hm2)) / 2

        if obs_idx not in obs_records:
            obs_records[obs_idx] = {c: [] for c in comparisons}
        for c, v in vals.items():
            obs_records[obs_idx][c].append(v)

        for c, v in vals.items():
            rows.append({
                'obs_idx': obs_idx,
                'genotype': str(genotype),
                'pair_idx': pair_idx,
                'comparison': c,
                'rmse': v,
            })

    # --- Compute means per observer and overall ---
    obs_order = sorted(obs_records.keys())
    # Find genotype label for each observer
    genotype_label = {}
    for obs_idx, _, _ in config_observers:
        for pd in all_pair_data:
            if pd['obs_idx'] == obs_idx:
                genotype_label[obs_idx] = str(pd['genotype'])
                break

    summary_rows = []  # for CSV summary
    means = {}  # obs_idx -> {comparison: mean}
    for obs_idx in obs_order:
        means[obs_idx] = {}
        for c in comparisons:
            m = float(np.mean(obs_records[obs_idx][c]))
            means[obs_idx][c] = m
            summary_rows.append({
                'obs_idx': obs_idx,
                'genotype': genotype_label.get(obs_idx, ''),
                'comparison': c,
                'mean_rmse': m,
                'n_pairs': len(obs_records[obs_idx][c]),
            })

    # Overall mean across all observers
    all_vals = {c: [] for c in comparisons}
    for obs_idx in obs_order:
        for c in comparisons:
            all_vals[c].extend(obs_records[obs_idx][c])
    overall_means = {c: float(np.mean(all_vals[c])) for c in comparisons}
    for c in comparisons:
        summary_rows.append({
            'obs_idx': 'all',
            'genotype': 'all',
            'comparison': c,
            'mean_rmse': overall_means[c],
            'n_pairs': sum(len(obs_records[o][c]) for o in obs_order),
        })

    # --- Write CSV ---
    csv_path = plots_path / 'rmse_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['obs_idx', 'genotype', 'comparison',
                                               'mean_rmse', 'n_pairs'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved CSV: {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate display measurements against expected LMSQ responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_display_measurements.py \\
    --primaries measurements/2026-02-02/primaries/ \\
    --measurements measurements/2026-02-02/validation/ \\
    --metamers config/display_validation_metamers.json \\
    --plots-dir measurements/2026-02-02/validation_plots/
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
                        help='Directory to save validation plots')
    parser.add_argument('--pred-only', action='store_true', default=False,
                        help='Skip all measured-spectra computation; show only predicted '
                             'and 8-bit-predicted in plots (useful for verifying metamer '
                             'generation without running the PR-650)')

    args = parser.parse_args()

    validate_measurements(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        measurements_dir=args.measurements,
        plots_dir=args.plots_dir,
        synthetic_epsilon=args.synthetic_epsilon,
        pred_only=args.pred_only,
    )


if __name__ == '__main__':
    main()
