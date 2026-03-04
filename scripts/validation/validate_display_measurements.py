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
"""

from TetriumColor.Measurement import load_primaries_from_csv, get_spectras_from_rgbo_list
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, convert_spectrum_to_bgyr
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def validate_measurements(
    metamers_config_path: str,
    primaries_path: str,
    measurements_dir: str | None,
    plots_dir: str,
    synthetic_epsilon: float = 0.01,
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
        observer = observer_genotypes.get_observer_for_peaks(genotype)

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
            predicted_1 = Spectra(wavelengths=wavelengths, data=predicted_1_data)
            predicted_2 = Spectra(wavelengths=wavelengths, data=predicted_2_data)

            # --- Measured (or synthetic) spectra ---
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
                                     data=measured_1_data)
                measured_2 = Spectra(wavelengths=wavelengths,
                                     data=measured_2_data)
            else:
                measured_list = get_spectras_from_rgbo_list(
                    measurements_dir, [rgbo_1, rgbo_2])
                measured_1, measured_2 = measured_list[0], measured_list[1]

                if measured_1 is None or measured_2 is None:
                    print(
                        f"    WARNING: Missing measurements for pair {pair_idx}, skipping"
                    )
                    continue

            # --- Project to LMSQ ---
            pred_lmsq_1 = observer.observe_spectras([predicted_1])[0]
            pred_lmsq_2 = observer.observe_spectras([predicted_2])[0]
            meas_lmsq_1 = observer.observe_spectras([measured_1])[0]
            meas_lmsq_2 = observer.observe_spectras([measured_2])[0]

            # LMS RMSE between the two metamers (should be ~0)
            pred_lms_diff = pred_lmsq_1[lms_indices] - pred_lmsq_2[lms_indices]
            meas_lms_diff = meas_lmsq_1[lms_indices] - meas_lmsq_2[lms_indices]
            pred_lms_rmse = np.sqrt(np.mean(pred_lms_diff ** 2))
            meas_lms_rmse = np.sqrt(np.mean(meas_lms_diff ** 2))

            # Q difference between the two metamers (should be large)
            pred_q_diff = abs(pred_lmsq_1[q_index] - pred_lmsq_2[q_index])
            meas_q_diff = abs(meas_lmsq_1[q_index] - meas_lmsq_2[q_index])

            # LMSQ RMSE between predicted and measured for each metamer
            lmsq_rmse_1 = np.sqrt(np.mean((pred_lmsq_1 - meas_lmsq_1) ** 2))
            lmsq_rmse_2 = np.sqrt(np.mean((pred_lmsq_2 - meas_lmsq_2) ** 2))

            print(f"    LMSQ RMSE: m1={lmsq_rmse_1:.4f}, m2={lmsq_rmse_2:.4f}")
            print(f"    LMS metamer RMSE: pred={pred_lms_rmse:.6f}, meas={meas_lms_rmse:.6f}")
            print(f"    Q metamer diff:   pred={pred_q_diff:.6f}, meas={meas_q_diff:.6f}")

            # Store for end-of-validation summary plots
            all_pair_data.append({
                'obs_idx': obs_idx,
                'pair_idx': pair_idx,
                'genotype': genotype,
                'predicted_1': predicted_1,
                'predicted_2': predicted_2,
                'measured_1': measured_1,
                'measured_2': measured_2,
            })

            # ===== PLOT: 2-panel figure per metamer pair =====
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # --- Panel 1: Predicted vs Measured Spectra ---
            ax = axes[0]
            ax.plot(wavelengths, predicted_1.data, 'b-', lw=2, label='Predicted M1', alpha=0.8)
            ax.plot(measured_1.wavelengths, measured_1.data, 'b--', lw=2, label='Measured M1', alpha=0.8)
            ax.plot(wavelengths, predicted_2.data, 'r-', lw=2, label='Predicted M2', alpha=0.8)
            ax.plot(measured_2.wavelengths, measured_2.data, 'r--', lw=2, label='Measured M2', alpha=0.8)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Power')
            ax.set_title('Predicted vs Measured Spectra')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')

            # --- Panel 2: LMSQ comparison ---
            ax = axes[1]
            cone_labels = list(sorted_with_s)
            cone_labels_str = [f'{wl}nm' for wl in cone_labels]
            # Mark Q cone
            cone_labels_str[q_index] = f'{cone_labels[q_index]}nm (Q)'

            x = np.arange(len(sorted_with_s))
            w = 0.18
            ax.bar(x - 1.5*w, pred_lmsq_1, w, label='Pred M1', color='steelblue', alpha=0.8)
            ax.bar(x - 0.5*w, pred_lmsq_2, w, label='Pred M2', color='indianred', alpha=0.8)
            ax.bar(x + 0.5*w, meas_lmsq_1, w, label='Meas M1', color='steelblue',
                   alpha=0.4, edgecolor='steelblue', linewidth=1.5)
            ax.bar(x + 1.5*w, meas_lmsq_2, w, label='Meas M2', color='indianred',
                   alpha=0.4, edgecolor='indianred', linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cone_labels_str, fontsize=8)
            ax.set_ylabel('Cone Response')
            ax.set_title(
                f'LMSQ Projection\n'
                f'LMS RMSE: pred={pred_lms_rmse:.4f} meas={meas_lms_rmse:.4f}\n'
                f'Q diff: pred={pred_q_diff:.4f} meas={meas_q_diff:.4f}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y', linestyle=':')

            fig.suptitle(
                f'Observer {obs_idx} (genotype {genotype}) — Metamer Pair {pair_idx}\n'
                f'RGBO1={rgbo_1}  RGBO2={rgbo_2}',
                fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            fname = plots_path / f'obs{obs_idx}_pair{pair_idx}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
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
        obs = observer_genotypes.get_observer_for_peaks(g)
        config_observer_list.append((obs_data['observer_index'], g, obs))

    # Build the 12D hyperobserver once
    print("  Building hyperobserver (12D)...")
    hyperobs = Observer.hyperobserver(wavelengths=wavelengths)

    for pair_data in all_pair_data:
        fname_a = _plot_all_observers_bars(pair_data, config_observer_list, plots_path)
        fname_b = _plot_hyperobserver_bars(pair_data, hyperobs, plots_path)
        fname_c = _plot_hyperobserver_diff(pair_data, hyperobs, plots_path)
        print(f"  Saved: {fname_a.name}  |  {fname_b.name}  |  {fname_c.name}")

    print(f"Summary plots saved to {plots_path}")


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
    pred_1 = pair_data['predicted_1']
    pred_2 = pair_data['predicted_2']

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

        x = np.arange(len(sorted_peaks))
        w = 0.3
        ax.bar(x - w / 2, lmsq_1, w, color='steelblue', alpha=0.85, label='M1')
        ax.bar(x + w / 2, lmsq_2, w, color='indianred', alpha=0.85, label='M2')
        ax.set_xticks(x)
        ax.set_xticklabels(cone_labels, fontsize=7)
        ax.set_ylabel('Cone Response')
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')

        title = f'Obs {c_obs_idx}\n{c_peaks}'
        if c_peaks == designed_genotype:
            title += '\n★ designed for'
        ax.set_title(title, fontsize=8)

        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx}: Cone Responses Across All Observers',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_all_observers.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname


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
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    hyper_pred_1 = hyperobserver.observe_spectras([pred_1])[0]
    hyper_pred_2 = hyperobserver.observe_spectras([pred_2])[0]
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
    w = 0.18

    fig, (ax_spec, ax) = plt.subplots(1, 2, figsize=(22, 5),
                                      gridspec_kw={'width_ratios': [1, 2.5]})

    # --- Left panel: spectra ---
    ax_spec.plot(pred_1.wavelengths, pred_1.data, color='steelblue', lw=2, label='Pred M1')
    ax_spec.plot(pred_2.wavelengths, pred_2.data, color='indianred', lw=2, label='Pred M2')
    ax_spec.plot(meas_1.wavelengths, meas_1.data, color='steelblue', lw=2,
                 alpha=0.4, linestyle='--', label='Meas M1')
    ax_spec.plot(meas_2.wavelengths, meas_2.data, color='indianred', lw=2,
                 alpha=0.4, linestyle='--', label='Meas M2')
    ax_spec.set_xlabel('Wavelength (nm)')
    ax_spec.set_ylabel('Power')
    ax_spec.set_title('Spectra')
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, linestyle=':')

    # --- Right panel: hyperobserver bars ---
    ax.bar(x - 1.5*w, hyper_pred_1, w, color='steelblue', alpha=0.85, label='Pred M1')
    ax.bar(x - 0.5*w, hyper_pred_2, w, color='indianred', alpha=0.85, label='Pred M2')
    ax.bar(x + 0.5*w, hyper_meas_1, w, color='steelblue', alpha=0.4,
           edgecolor='steelblue', linewidth=1.5, label='Meas M1')
    ax.bar(x + 1.5*w, hyper_meas_2, w, color='indianred', alpha=0.4,
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

    ax.set_ylabel('Cone Response')
    ax.set_title(
        f'Hyperobserver (12D) — Observer {obs_idx} · Pair {pair_idx}\n'
        f'Designed for genotype {designed_genotype}',
        fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    fig.suptitle(
        f'Observer {obs_idx} · Pair {pair_idx} — Designed for genotype {designed_genotype}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
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
    meas_1 = pair_data['measured_1']
    meas_2 = pair_data['measured_2']

    hyper_pred_1 = hyperobserver.observe_spectras([pred_1])[0]
    hyper_pred_2 = hyperobserver.observe_spectras([pred_2])[0]
    hyper_meas_1 = hyperobserver.observe_spectras([meas_1])[0]
    hyper_meas_2 = hyperobserver.observe_spectras([meas_2])[0]

    pred_diff = np.abs(hyper_pred_1 - hyper_pred_2)
    meas_diff = np.abs(hyper_meas_1 - hyper_meas_2)

    hyper_peaks = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
    designed_set = set(designed_genotype) | {420}
    designed_mask = np.array([p in designed_set for p in hyper_peaks])

    cone_labels = [
        'S\n420', 'M\n530', 'M\n533', 'M\n536',
        'L\n547', 'L\n551', 'L\n552', 'L\n553',
        'L\n555', 'L\n556', 'L\n556.5', 'L\n559',
    ]
    x = np.arange(len(cone_labels))
    w = 0.3

    fig, (ax_spec, ax) = plt.subplots(1, 2, figsize=(22, 5),
                                      gridspec_kw={'width_ratios': [1, 2.5]})

    # --- Left panel: spectra ---
    ax_spec.plot(pred_1.wavelengths, pred_1.data, color='steelblue', lw=2, label='Pred M1')
    ax_spec.plot(pred_2.wavelengths, pred_2.data, color='indianred', lw=2, label='Pred M2')
    ax_spec.plot(meas_1.wavelengths, meas_1.data, color='steelblue', lw=2,
                 alpha=0.4, linestyle='--', label='Meas M1')
    ax_spec.plot(meas_2.wavelengths, meas_2.data, color='indianred', lw=2,
                 alpha=0.4, linestyle='--', label='Meas M2')
    ax_spec.set_xlabel('Wavelength (nm)')
    ax_spec.set_ylabel('Power')
    ax_spec.set_title('Spectra')
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3, linestyle=':')

    # --- Right panel: absolute difference bars ---
    ax.bar(x - 0.5*w, pred_diff, w, color='steelblue', alpha=0.85, label='Pred |M1−M2|')
    ax.bar(x + 0.5*w, meas_diff, w, color='steelblue', alpha=0.4,
           edgecolor='steelblue', linewidth=1.5, label='Meas |M1−M2|')

    # Dashed horizontal line at the max non-Q designed cone difference (measured)
    non_q_mask = designed_mask & np.array([p != 547 for p in hyper_peaks])
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

    ax.set_ylabel('Absolute Cone Response Difference |M1 − M2|')
    ax.set_title(
        f'Hyperobserver |Difference| (M1−M2) — Observer {obs_idx} · Pair {pair_idx}\n'
        f'Designed for genotype {designed_genotype}',
        fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    plt.tight_layout()
    fname = plots_path / f'obs{obs_idx}_pair{pair_idx}_hyperobserver_diff.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname


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

    args = parser.parse_args()

    validate_measurements(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        measurements_dir=args.measurements,
        plots_dir=args.plots_dir,
        synthetic_epsilon=args.synthetic_epsilon,
    )


if __name__ == '__main__':
    main()
