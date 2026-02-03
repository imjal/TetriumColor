#!/usr/bin/env python3
"""
Convert RYGB metamer pairs to RGBO display values using today's primaries.

This script takes the fixed RYGB metamer configuration and converts it to RGBO
values that can be displayed, based on the measured display primaries for the day.
"""

from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer.Spectra import Spectra
import argparse
import json
import csv
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# RYGB cutpoints (same as in ColorSpace.py)
RYGB_CUTPOINTS = [493.0, 563.0, 608.0]  # in nanometers


def create_rygb_basis_spectra(wavelengths: np.ndarray) -> list:
    """
    Create the 4 RYGB basis spectra (step functions).

    Args:
        wavelengths: Array of wavelengths

    Returns:
        List of 4 Spectra objects representing R, Y, G, B basis functions
    """
    # Create transitions for RYGB regions
    # Blue: λ < 493nm
    # Green: 493nm <= λ < 563nm
    # Yellow: 563nm <= λ < 608nm
    # Red: λ >= 608nm

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

    return rygb_basis


def compute_rygb_to_rgbo_matrix(primaries: list) -> np.ndarray:
    """
    Compute the transformation matrix from RYGB to RGBO.

    For each RYGB basis function, find the RGBO weights that best reproduce it.

    Args:
        primaries: List of 4 Spectra objects (R, G, B, O primaries)

    Returns:
        4x4 transformation matrix
    """
    # Get wavelengths from primaries
    wavelengths = primaries[0].wavelengths

    # Create RYGB basis spectra
    rygb_basis = create_rygb_basis_spectra(wavelengths)

    # Build primary matrix (each column is a primary spectrum)
    primary_matrix = np.array([p.data for p in primaries]).T  # shape: (n_wavelengths, 4)

    # For each RYGB basis, solve for RGBO weights
    transform_matrix = np.zeros((4, 4))

    for i, rygb_spectrum in enumerate(rygb_basis):
        # Solve: primary_matrix @ weights = rygb_spectrum.data
        # Using least squares since it's overdetermined
        weights, residuals, rank, s = np.linalg.lstsq(
            primary_matrix,
            rygb_spectrum.data,
            rcond=None
        )
        transform_matrix[:, i] = weights

    return transform_matrix


def convert_rygb_to_rgbo(
    metamers_config_path: str,
    primaries_path: str,
    output_path: str = None
):
    """
    Convert RYGB metamers to RGBO display values.

    This performs a direct spectral conversion without needing observer models.
    For each RYGB value, we compute the corresponding spectrum and find the
    RGBO primary weights that best reproduce it.

    Args:
        metamers_config_path: Path to JSON file with RYGB metamers
        primaries_path: Path to directory with display primaries
        output_path: Path to output CSV file (default: stdout)

    Returns:
        List of dictionaries with RGBO values and metadata
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
    primaries_path_obj = Path(primaries_path)

    if primaries_path_obj.is_dir():
        # Directory containing primary measurements
        primaries = load_primaries_from_csv(str(primaries_path))
    elif primaries_path_obj.is_file() and primaries_path_obj.suffix == '.csv':
        raise NotImplementedError("Single CSV file loading not yet implemented. Please provide a directory.")
    else:
        raise ValueError(f"Invalid primaries path: {primaries_path}")

    if len(primaries) < 4:
        raise ValueError(f"Expected 4 primaries (RGBO), but got {len(primaries)}")

    print(f"  Loaded {len(primaries)} primaries")
    print(f"  Wavelength range: {primaries[0].wavelengths[0]}-{primaries[0].wavelengths[-1]} nm")

    # Compute RYGB to RGBO transformation matrix
    print("\nComputing RYGB → RGBO transformation matrix...")
    transform_matrix = compute_rygb_to_rgbo_matrix(primaries)
    print("  Transformation matrix computed")
    print()

    # Convert metamers to RGBO
    rgbo_list = []

    for obs_data in config['observers']:
        genotype = tuple(sorted(tuple(obs_data['genotype'])))  # sort genotype (already includes Q at 547nm)

        # Calculate Q index: Observer will add S cone at 420nm and sort all peaks
        sorted_with_s = tuple(sorted((420,) + genotype))
        q_index = sorted_with_s.index(547)

        print(f"Processing observer {obs_data['observer_index']}: {genotype} (Q at index {q_index})")

        # Convert each metamer pair
        for metamer in obs_data['metamers']:
            # Get RYGB values
            rygb_1 = np.array(metamer['rygb_1'])
            rygb_2 = np.array(metamer['rygb_2'])

            # Convert RYGB to RGBO using the transformation matrix
            # RGBO = transform_matrix @ RYGB
            rgbo_1 = transform_matrix @ rygb_1
            rgbo_2 = transform_matrix @ rygb_2

            # Clip to [0, 1] and convert to 8-bit
            rgbo_1_8bit = np.clip(np.round(rgbo_1 * 255), 0, 255).astype(int)
            rgbo_2_8bit = np.clip(np.round(rgbo_2 * 255), 0, 255).astype(int)

            # Store both metamers
            for metamer_idx, (rgbo_8bit, rygb) in enumerate([(rgbo_1_8bit, rygb_1), (rgbo_2_8bit, rygb_2)]):
                rgbo_list.append({
                    'observer_index': obs_data['observer_index'],
                    'genotype': genotype,
                    'q_cone_index': q_index,
                    'pair_index': metamer['pair_index'],
                    'metamer_index': metamer_idx,
                    'R': int(rgbo_8bit[0]),
                    'G': int(rgbo_8bit[1]),
                    'B': int(rgbo_8bit[2]),
                    'O': int(rgbo_8bit[3]),
                    'rygb_r': float(rygb[0]),
                    'rygb_y': float(rygb[1]),
                    'rygb_g': float(rygb[2]),
                    'rygb_b': float(rygb[3])
                })

            print(f"  Pair {metamer['pair_index']}: "
                  f"RGBO1=({rgbo_1_8bit[0]},{rgbo_1_8bit[1]},{rgbo_1_8bit[2]},{rgbo_1_8bit[3]}), "
                  f"RGBO2=({rgbo_2_8bit[0]},{rgbo_2_8bit[1]},{rgbo_2_8bit[2]},{rgbo_2_8bit[3]})")

        print()

    # Write output
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            fieldnames = ['observer_index', 'genotype', 'q_cone_index', 'pair_index', 'metamer_index',
                          'R', 'G', 'B', 'O', 'rygb_r', 'rygb_y', 'rygb_g', 'rygb_b']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rgbo_list:
                # Convert genotype tuple to string for CSV
                row_copy = row.copy()
                row_copy['genotype'] = str(row['genotype'])
                writer.writerow(row_copy)

        print(f"Saved {len(rgbo_list)} RGBO values to: {output_path}")
    else:
        # Print to stdout
        print("observer_index,genotype,q_cone_index,pair_index,metamer_index,R,G,B,O,rygb_r,rygb_y,rygb_g,rygb_b")
        for row in rgbo_list:
            print(f"{row['observer_index']},{row['genotype']},{row['q_cone_index']},{row['pair_index']},"
                  f"{row['metamer_index']},{row['R']},{row['G']},{row['B']},{row['O']},"
                  f"{row['rygb_r']},{row['rygb_y']},{row['rygb_g']},{row['rygb_b']}")

    return rgbo_list


def main():
    parser = argparse.ArgumentParser(
        description='Convert RYGB metamers to RGBO display values using measured primaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert_rygb_to_rgbo.py \\
    --primaries measurements/2026-02-02/primaries/ \\
    --metamers config/display_validation_metamers.json \\
    --output measurements/2026-02-02/display_targets.csv
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
        help='Path to RYGB metamer configuration JSON (default: config/display_validation_metamers.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: stdout)'
    )

    args = parser.parse_args()

    # Convert metamers
    convert_rygb_to_rgbo(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
