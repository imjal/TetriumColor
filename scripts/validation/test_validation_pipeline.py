#!/usr/bin/env python3
"""
Test the full display validation pipeline.

This script demonstrates the complete workflow:
1. Generate RYGB metamers (already done, loads from config)
2. Convert RYGB to RGBO using primaries
3. Simulate measurements (for testing, uses primaries to generate expected spectra)
4. Validate measurements
"""

from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer import Spectra
import numpy as np
from pathlib import Path
import sys
import json
import csv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def simulate_measurements(rgbo_csv_path: str, primaries_path: str, output_dir: str):
    """
    Simulate measurements by generating spectra from RGBO values and primaries.

    In a real scenario, these would come from actual PR650 measurements.
    For testing, we generate them synthetically.
    """
    print("Simulating measurements...")
    print(f"Reading RGBO targets from: {rgbo_csv_path}")
    print(f"Using primaries from: {primaries_path}")
    print(f"Saving to: {output_dir}")
    print()

    # Load primaries
    primaries = load_primaries_from_csv(primaries_path)

    # Read RGBO CSV
    rgbo_list = []
    with open(rgbo_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgbo_list.append({
                'observer_index': int(row['observer_index']),
                'pair_index': int(row['pair_index']),
                'metamer_index': int(row['metamer_index']),
                'R': int(row['R']),
                'G': int(row['G']),
                'B': int(row['B']),
                'O': int(row['O'])
            })

    print(f"Found {len(rgbo_list)} RGBO values to simulate")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate simulated spectra
    for rgbo_data in rgbo_list:
        R, G, B, O = rgbo_data['R'], rgbo_data['G'], rgbo_data['B'], rgbo_data['O']

        # Normalize to [0, 1]
        weights = np.array([R, G, B, O]) / 255.0

        # Generate spectrum as weighted sum of primaries
        spectrum_data = sum(w * p.data for w, p in zip(weights, primaries))

        # Add small amount of noise to simulate measurement variation
        noise = np.random.normal(0, 0.001 * np.max(spectrum_data), len(spectrum_data))
        spectrum_data = spectrum_data + noise
        spectrum_data = np.maximum(spectrum_data, 0)  # Ensure non-negative

        # Save as CSV (format compatible with measurement loading functions)
        filename = f"r{R}g{G}b{B}o{O}.csv"
        filepath = output_path / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Wavelength', 'Power'])
            for wl, power in zip(primaries[0].wavelengths, spectrum_data):
                writer.writerow([wl, power])

        print(f"  Generated: {filename}")

    print(f"\nSimulated {len(rgbo_list)} measurements in {output_dir}")
    print()


def run_full_pipeline(
    primaries_path: str,
    metamers_config: str = 'config/display_validation_metamers.json',
    test_output_dir: str = 'test_validation'
):
    """Run the full validation pipeline for testing."""

    test_path = Path(test_output_dir)
    test_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DISPLAY VALIDATION PIPELINE TEST")
    print("="*80)
    print()

    # Step 1: Convert RYGB to RGBO (already done, but show the command)
    print("Step 1: Convert RYGB metamers to RGBO display values")
    print("-" * 80)
    rgbo_targets_path = test_path / "display_targets.csv"

    from convert_rygb_to_rgbo import convert_rygb_to_rgbo
    convert_rygb_to_rgbo(
        metamers_config_path=metamers_config,
        primaries_path=primaries_path,
        output_path=str(rgbo_targets_path)
    )
    print()

    # Step 2: Simulate measurements (in real use, this would be actual PR650 measurements)
    print("Step 2: Simulate measurements (would be actual PR650 in production)")
    print("-" * 80)
    measurements_dir = test_path / "measurements"
    simulate_measurements(
        rgbo_csv_path=str(rgbo_targets_path),
        primaries_path=primaries_path,
        output_dir=str(measurements_dir)
    )

    # Step 3: Validate measurements
    print("Step 3: Validate measurements")
    print("-" * 80)

    from validate_display_measurements import validate_measurements
    report_path = test_path / "validation_report.json"
    plots_dir = test_path / "plots"

    results = validate_measurements(
        metamers_config_path=metamers_config,
        primaries_path=primaries_path,
        measurements_dir=str(measurements_dir),
        output_report_path=str(report_path),
        plots_dir=str(plots_dir)
    )

    print()
    print("="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)
    print(f"Test outputs saved to: {test_path}")
    print(f"  - RGBO targets: {rgbo_targets_path}")
    print(f"  - Simulated measurements: {measurements_dir}")
    print(f"  - Validation report: {report_path}")
    print(f"  - Validation plots: {plots_dir}")
    print()

    # Print key results
    if results['summary']['all_metamers_valid']:
        print("✓ All metamers validated successfully!")
    else:
        print("✗ Some metamers failed validation")

    print(f"  Pairs tested: {results['summary']['total_pairs_tested']}")
    print(f"  Mean LMS RMSE: {results['summary']['mean_lms_rmse']:.4f}")
    print(f"  Mean Q difference: {results['summary']['mean_q_difference']:.4f}")
    print(f"  Mean RYGB RMSE: {results['summary']['mean_rygb_rmse']:.4f}")
    print()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test the full display validation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests the complete workflow:
1. Loads RYGB metamer configuration
2. Converts RYGB to RGBO using measured primaries
3. Simulates measurements (generates synthetic spectra)
4. Validates that measurements produce expected LMSQ responses

Example:
  python test_validation_pipeline.py \\
    --primaries measurements/2025-12-02/primaries_new_method/
        """
    )

    parser.add_argument(
        '--primaries',
        type=str,
        required=True,
        help='Path to directory containing display primary measurements'
    )
    parser.add_argument(
        '--metamers',
        type=str,
        default='config/display_validation_metamers.json',
        help='Path to RYGB metamer configuration (default: config/display_validation_metamers.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_validation',
        help='Output directory for test results (default: test_validation)'
    )

    args = parser.parse_args()

    run_full_pipeline(
        primaries_path=args.primaries,
        metamers_config=args.metamers,
        test_output_dir=args.output
    )


if __name__ == '__main__':
    main()
