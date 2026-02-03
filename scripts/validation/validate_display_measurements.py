#!/usr/bin/env python3
"""
Validate display measurements against expected LMSQ responses.

This script loads measured spectra and validates that they produce the expected
LMSQ (cone response) values for each observer's metamer pairs.
"""

from TetriumColor.Measurement import load_primaries_from_csv, get_spectras_from_rgbo_list
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, RYGB_CUTPOINTS
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra
import argparse
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_rygb_to_rgbo_matrix(primaries):
    """
    Compute the transformation matrix from RYGB to RGBO using spectral conversion.
    Same logic as in convert_rygb_to_rgbo.py
    """
    wavelengths = primaries[0].wavelengths

    # Create RYGB basis spectra (step functions)
    rygb_basis = []
    # Red: wavelengths >= 608nm
    red_data = (wavelengths >= RYGB_CUTPOINTS[2]).astype(float)
    rygb_basis.append(Spectra(wavelengths=wavelengths, data=red_data))
    # Yellow: 563nm <= wavelengths < 608nm
    yellow_data = ((wavelengths >= RYGB_CUTPOINTS[1]) & (wavelengths < RYGB_CUTPOINTS[2])).astype(float)
    rygb_basis.append(Spectra(wavelengths=wavelengths, data=yellow_data))
    # Green: 493nm <= wavelengths < 563nm
    green_data = ((wavelengths >= RYGB_CUTPOINTS[0]) & (wavelengths < RYGB_CUTPOINTS[1])).astype(float)
    rygb_basis.append(Spectra(wavelengths=wavelengths, data=green_data))
    # Blue: wavelengths < 493nm
    blue_data = (wavelengths < RYGB_CUTPOINTS[0]).astype(float)
    rygb_basis.append(Spectra(wavelengths=wavelengths, data=blue_data))

    # Build primary matrix
    primary_matrix = np.array([p.data for p in primaries]).T

    # Solve for transformation matrix
    transform_matrix = np.zeros((4, 4))
    for i, rygb_spectrum in enumerate(rygb_basis):
        weights = np.linalg.lstsq(primary_matrix, rygb_spectrum.data, rcond=None)[0]
        transform_matrix[:, i] = weights

    return transform_matrix


def validate_measurements(
    metamers_config_path: str,
    primaries_path: str,
    measurements_dir: str,
    output_report_path: str = None,
    plots_dir: str = None
):
    """
    Validate measured spectra against expected LMSQ responses.

    Args:
        metamers_config_path: Path to RYGB metamer configuration JSON
        primaries_path: Path to directory with display primaries
        measurements_dir: Directory containing measured spectra CSV files
        output_report_path: Path to output JSON report (optional)
        plots_dir: Directory to save validation plots (optional)

    Returns:
        Dictionary with validation results
    """
    # Load metamer configuration
    print(f"Loading metamer configuration from: {metamers_config_path}")
    with open(metamers_config_path, 'r') as f:
        config = json.load(f)

    print(f"  Observers: {len(config['observers'])}")
    print(f"  Total metamer pairs: {config['metadata']['total_metamer_pairs']}")
    print()

    # Load display primaries
    print(f"Loading display primaries from: {primaries_path}")
    primaries = load_primaries_from_csv(primaries_path)

    if len(primaries) < 4:
        raise ValueError(f"Expected 4 primaries (RGBO), but got {len(primaries)}")

    print(f"  Loaded {len(primaries)} primaries")
    print()

    # Initialize ObserverGenotypes
    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata']['seed']
    )

    # White spectrum for Delta-E calculations
    white_spectrum_data = np.sum([primary.data for primary in primaries], axis=0)
    white_spectrum = Spectra(wavelengths=wavelengths, data=white_spectrum_data)

    # Process each observer
    validation_results = {
        'date': datetime.now().isoformat(),
        'primaries_path': str(primaries_path),
        'measurements_dir': str(measurements_dir),
        'observers': []
    }

    all_metamers_valid = True
    total_lms_rmse = []
    total_q_diff = []
    total_rygb_rmse = []

    for obs_data in config['observers']:
        genotype = tuple(sorted(tuple(obs_data['genotype'])))  # sort genotype (already includes Q at 547nm)

        # Calculate Q index: Observer will add S cone at 420nm and sort all peaks
        sorted_with_s = tuple(sorted((420,) + genotype))
        q_index = sorted_with_s.index(547)
        lms_indices = [i for i in range(len(sorted_with_s)) if i != q_index]

        print(f"Processing observer {obs_data['observer_index']}: {genotype} (Q at index {q_index})")

        # Create observer (ColorSpace not needed for validation)
        observer = observer_genotypes.get_observer_for_peaks(genotype)

        # Compute RYGB to RGBO transformation matrix (spectral conversion)
        rygb_to_rgbo = compute_rygb_to_rgbo_matrix(primaries)

        # Prepare to collect RGBO values for this observer
        observer_rgbo_list = []
        observer_expected_rygb = []
        observer_metamer_pairs = []

        # Convert all RYGB metamers to RGBO using spectral transformation
        for metamer in obs_data['metamers']:
            rygb_1 = np.array(metamer['rygb_1'])
            rygb_2 = np.array(metamer['rygb_2'])

            # Convert RYGB to RGBO using matrix multiplication
            rgbo_1 = rygb_to_rgbo @ rygb_1
            rgbo_2 = rygb_to_rgbo @ rygb_2

            # Convert to 8-bit
            rgbo_1_8bit = tuple(np.clip(np.round(rgbo_1 * 255), 0, 255).astype(int))
            rgbo_2_8bit = tuple(np.clip(np.round(rgbo_2 * 255), 0, 255).astype(int))

            observer_rgbo_list.extend([rgbo_1_8bit, rgbo_2_8bit])
            observer_expected_rygb.extend([rygb_1, rygb_2])
            observer_metamer_pairs.append({
                'pair_index': metamer['pair_index'],
                'rgbo_1': rgbo_1_8bit,
                'rgbo_2': rgbo_2_8bit,
                'rygb_1': rygb_1,
                'rygb_2': rygb_2
            })

        # Load measured spectra for this observer's RGBO values
        print(f"  Loading {len(observer_rgbo_list)} measured spectra...")
        measured_spectra = get_spectras_from_rgbo_list(measurements_dir, observer_rgbo_list)

        # Filter out None values
        valid_indices = [i for i, spec in enumerate(measured_spectra) if spec is not None]
        if len(valid_indices) == 0:
            print(f"  Warning: No valid spectra found for observer {genotype}")
            all_metamers_valid = False
            continue

        valid_spectra = [measured_spectra[i] for i in valid_indices]
        valid_rgbo = [observer_rgbo_list[i] for i in valid_indices]
        valid_expected_rygb = [observer_expected_rygb[i] for i in valid_indices]

        print(f"  Found {len(valid_spectra)}/{len(observer_rgbo_list)} measured spectra")

        # Ensure all spectra have same wavelengths as primaries
        primaries_wavelengths = primaries[0].wavelengths
        valid_spectra_interp = []
        for spectrum in valid_spectra:
            if not np.array_equal(spectrum.wavelengths, primaries_wavelengths):
                # Interpolate to match primaries wavelengths
                spectrum_interp = spectrum.interpolate_values(primaries_wavelengths)
                valid_spectra_interp.append(spectrum_interp)
            else:
                valid_spectra_interp.append(spectrum)

        # Project measured spectra to LMSQ and RYGB
        measured_lmsq = observer.observe_spectras(valid_spectra_interp)
        measured_rygb_list = []

        # Compute RGBO to RYGB matrix (inverse of RYGB to RGBO)
        rgbo_to_rygb = np.linalg.inv(rygb_to_rgbo)

        for spectrum in valid_spectra_interp:
            # Convert spectrum to DISP, then to RYGB
            # First, compute DISP values by solving primaries
            disp_vals = np.linalg.lstsq(
                np.array([p.data for p in primaries]).T,
                spectrum.data,
                rcond=None
            )[0]

            # Convert DISP to RYGB using inverse matrix
            rygb_measured = rgbo_to_rygb @ disp_vals

            measured_rygb_list.append(rygb_measured)

        measured_rygb = np.array(measured_rygb_list)

        # Validate metamer pairs
        observer_result = {
            'observer_index': obs_data['observer_index'],
            'genotype': list(genotype),
            'probability': obs_data['probability'],
            'metamer_pairs': [],
            'cross_validation': []
        }

        pair_idx = 0
        for i in range(0, len(valid_indices), 2):
            if i + 1 >= len(valid_indices):
                break

            # Get the pair
            lmsq_1 = measured_lmsq[i]
            lmsq_2 = measured_lmsq[i+1]
            rygb_measured_1 = measured_rygb[i]
            rygb_measured_2 = measured_rygb[i+1]
            rygb_expected_1 = valid_expected_rygb[i]
            rygb_expected_2 = valid_expected_rygb[i+1]

            # Calculate LMS difference (should be small for metamers)
            lms_1 = lmsq_1[lms_indices]  # All cones except Q
            lms_2 = lmsq_2[lms_indices]
            lms_diff = lms_1 - lms_2
            lms_rmse = np.sqrt(np.mean(lms_diff ** 2))

            # Calculate Q difference (should be large for metamers)
            q_diff = abs(lmsq_1[q_index] - lmsq_2[q_index])

            # Calculate Delta-E
            try:
                delta_e = Spectra.delta_e(
                    valid_spectra_interp[i],
                    valid_spectra_interp[i+1],
                    white_spectrum
                )
            except:
                delta_e = None

            # Calculate RYGB accuracy
            rygb_error_1 = rygb_measured_1 - rygb_expected_1
            rygb_error_2 = rygb_measured_2 - rygb_expected_2
            rygb_rmse = np.sqrt(np.mean([
                np.mean(rygb_error_1 ** 2),
                np.mean(rygb_error_2 ** 2)
            ]))

            # Check if valid metamer (LMS similar, Q different)
            is_metamer = (lms_rmse < 0.05 and q_diff > 0.005)
            if not is_metamer:
                all_metamers_valid = False

            observer_result['metamer_pairs'].append({
                'pair_index': pair_idx,
                'rgbo_1': [int(x) for x in valid_rgbo[i]],
                'rgbo_2': [int(x) for x in valid_rgbo[i+1]],
                'lmsq_1': [float(x) for x in lmsq_1],
                'lmsq_2': [float(x) for x in lmsq_2],
                'lms_rmse': float(lms_rmse),
                'q_difference': float(q_diff),
                'delta_e': float(delta_e) if delta_e is not None else None,
                'is_metamer': bool(is_metamer),
                'rygb_measured_1': [float(x) for x in rygb_measured_1],
                'rygb_measured_2': [float(x) for x in rygb_measured_2],
                'rygb_expected_1': [float(x) for x in rygb_expected_1],
                'rygb_expected_2': [float(x) for x in rygb_expected_2],
                'rygb_rmse': float(rygb_rmse)
            })

            total_lms_rmse.append(lms_rmse)
            total_q_diff.append(q_diff)
            total_rygb_rmse.append(rygb_rmse)

            print(f"  Pair {pair_idx}: LMS_RMSE={lms_rmse:.4f}, Q_diff={q_diff:.4f}, "
                  f"RYGB_RMSE={rygb_rmse:.4f}, Metamer={is_metamer}")

            pair_idx += 1

        # Cross-validation: check that these spectra look different to other observers
        for other_obs_data in config['observers']:
            if other_obs_data['observer_index'] == obs_data['observer_index']:
                continue

            other_genotype = tuple(other_obs_data['genotype'])
            other_observer = observer_genotypes.get_observer_for_peaks(other_genotype)

            # Project to other observer's LMSQ
            other_lmsq = other_observer.observe_spectras(valid_spectra_interp)

            # Calculate differences for metamer pairs
            other_lms_diffs = []
            other_q_diffs = []
            other_delta_es = []

            for i in range(0, len(valid_indices), 2):
                if i + 1 >= len(valid_indices):
                    break

                other_lmsq_1 = other_lmsq[i]
                other_lmsq_2 = other_lmsq[i+1]

                other_lms_1 = other_lmsq_1[[0, 1, 3]]
                other_lms_2 = other_lmsq_2[[0, 1, 3]]
                other_lms_diff = np.sqrt(np.mean((other_lms_1 - other_lms_2) ** 2))
                # For other observers, Q might be at different index - calculate it
                other_genotype = tuple(sorted(tuple(other_obs_data['genotype'])))
                other_sorted_with_s = tuple(sorted((420,) + other_genotype))
                other_q_index = other_sorted_with_s.index(547)

                other_q_diff = abs(other_lmsq_1[other_q_index] - other_lmsq_2[other_q_index])

                other_lms_diffs.append(other_lms_diff)
                other_q_diffs.append(other_q_diff)

                try:
                    delta_e = Spectra.delta_e(
                        valid_spectra_interp[i],
                        valid_spectra_interp[i+1],
                        white_spectrum
                    )
                    other_delta_es.append(delta_e)
                except:
                    pass

            if len(other_lms_diffs) > 0:
                observer_result['cross_validation'].append({
                    'other_observer_genotype': list(other_genotype),
                    'mean_lms_difference': float(np.mean(other_lms_diffs)),
                    'mean_q_difference': float(np.mean(other_q_diffs)),
                    'mean_delta_e': float(np.mean(other_delta_es)) if len(other_delta_es) > 0 else None
                })

        validation_results['observers'].append(observer_result)
        print()

    # Summary statistics
    validation_results['summary'] = {
        'all_metamers_valid': all_metamers_valid,
        'mean_lms_rmse': float(np.mean(total_lms_rmse)) if len(total_lms_rmse) > 0 else None,
        'mean_q_difference': float(np.mean(total_q_diff)) if len(total_q_diff) > 0 else None,
        'mean_rygb_rmse': float(np.mean(total_rygb_rmse)) if len(total_rygb_rmse) > 0 else None,
        'total_pairs_tested': len(total_lms_rmse)
    }

    # Save report
    if output_report_path:
        output_path = Path(output_report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

        print(f"Saved validation report to: {output_path}")

    # Generate plots
    if plots_dir:
        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Plot 1: LMS RMSE vs Q difference for all pairs
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(validation_results['observers'])))

        for obs_idx, obs_result in enumerate(validation_results['observers']):
            lms_rmses = [p['lms_rmse'] for p in obs_result['metamer_pairs']]
            q_diffs = [p['q_difference'] for p in obs_result['metamer_pairs']]
            is_metamers = [p['is_metamer'] for p in obs_result['metamer_pairs']]

            # Plot valid metamers as circles, invalid as X
            valid_lms = [lms_rmses[i] for i in range(len(lms_rmses)) if is_metamers[i]]
            valid_q = [q_diffs[i] for i in range(len(q_diffs)) if is_metamers[i]]
            invalid_lms = [lms_rmses[i] for i in range(len(lms_rmses)) if not is_metamers[i]]
            invalid_q = [q_diffs[i] for i in range(len(q_diffs)) if not is_metamers[i]]

            if valid_lms:
                ax.scatter(valid_lms, valid_q, c=[colors[obs_idx]], marker='o',
                           label=f"Observer {obs_result['observer_index']} (valid)", s=100, alpha=0.7)
            if invalid_lms:
                ax.scatter(invalid_lms, invalid_q, c=[colors[obs_idx]], marker='x',
                           label=f"Observer {obs_result['observer_index']} (invalid)", s=100)

        ax.set_xlabel('LMS RMSE (should be small)')
        ax.set_ylabel('Q Difference (should be large)')
        ax.set_title('Metamer Validation: LMS Similarity vs Q Difference')
        ax.axhline(y=0.005, color='r', linestyle='--', alpha=0.5, label='Q diff threshold')
        ax.axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='LMS RMSE threshold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_path / 'metamer_validation.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: RYGB accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        for obs_idx, obs_result in enumerate(validation_results['observers']):
            rygb_rmses = [p['rygb_rmse'] for p in obs_result['metamer_pairs']]
            pair_indices = [p['pair_index'] for p in obs_result['metamer_pairs']]

            ax.bar(
                [p + obs_idx * 0.15 for p in pair_indices],
                rygb_rmses,
                width=0.15,
                label=f"Observer {obs_result['observer_index']}",
                color=colors[obs_idx],
                alpha=0.7
            )

        ax.set_xlabel('Metamer Pair Index')
        ax.set_ylabel('RYGB RMSE')
        ax.set_title('RYGB Reconstruction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_path / 'rygb_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved validation plots to: {plots_path}")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"All metamers valid: {validation_results['summary']['all_metamers_valid']}")
    print(f"Total pairs tested: {validation_results['summary']['total_pairs_tested']}")
    lms_rmse = validation_results['summary']['mean_lms_rmse']
    q_diff = validation_results['summary']['mean_q_difference']
    rygb_rmse = validation_results['summary']['mean_rygb_rmse']

    print(f"Mean LMS RMSE: {lms_rmse:.4f}" if lms_rmse is not None else "Mean LMS RMSE: N/A")
    print(f"Mean Q difference: {q_diff:.4f}" if q_diff is not None else "Mean Q difference: N/A")
    print(f"Mean RYGB RMSE: {rygb_rmse:.4f}" if rygb_rmse is not None else "Mean RYGB RMSE: N/A")
    print("="*80)

    return validation_results


def main():
    parser = argparse.ArgumentParser(
        description='Validate display measurements against expected LMSQ responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python validate_display_measurements.py \\
    --primaries measurements/2026-02-02/primaries/ \\
    --measurements measurements/2026-02-02/validation/ \\
    --metamers config/display_validation_metamers.json \\
    --output measurements/2026-02-02/validation_report.json \\
    --plots-dir measurements/2026-02-02/validation_plots/
        """
    )

    parser.add_argument(
        '--primaries',
        type=str,
        required=True,
        help='Path to directory containing display primary measurements'
    )
    parser.add_argument(
        '--measurements',
        type=str,
        required=True,
        help='Path to directory containing measured spectra'
    )
    parser.add_argument(
        '--metamers',
        type=str,
        default='config/display_validation_metamers.json',
        help='Path to RYGB metamer configuration JSON (default: config/display_validation_metamers.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON report file path (optional)'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        help='Directory to save validation plots (optional)'
    )

    args = parser.parse_args()

    # Validate measurements
    validate_measurements(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        measurements_dir=args.measurements,
        output_report_path=args.output,
        plots_dir=args.plots_dir
    )


if __name__ == '__main__':
    main()
