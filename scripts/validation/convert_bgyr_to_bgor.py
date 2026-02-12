#!/usr/bin/env python3
"""
Convert BGYR metamer pairs to BGOR display values using today's primaries.

This script takes the fixed BGYR metamer configuration and converts it to BGOR
values that can be displayed, based on the measured display primaries for the day.
"""

from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.Spectra import Spectra
import argparse
import json
import csv
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def convert_bgyr_to_bgor(
    metamers_config_path: str,
    primaries_path: str,
    output_path: str = None
):
    """
    Convert BGYR metamers to BGOR display values using observer-specific transformations.

    For each observer, this creates an observer-specific ColorSpace and converts
    BGYR → CONE (LMSQ) → DISP (BGOR) using that observer's transformation matrices.
    This ensures each observer uses their own transformation.

    Args:
        metamers_config_path: Path to JSON file with BGYR metamers
        primaries_path: Path to directory with display primaries
        output_path: Path to output CSV file (default: stdout)

    Returns:
        List of dictionaries with BGOR values and metadata
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
        primaries = load_primaries_from_csv(str(primaries_path), extract_zero=False)

    elif primaries_path_obj.is_file() and primaries_path_obj.suffix == '.csv':
        raise NotImplementedError("Single CSV file loading not yet implemented. Please provide a directory.")
    else:
        raise ValueError(f"Invalid primaries path: {primaries_path}")

    if len(primaries) < 4:
        raise ValueError(f"Expected 4 primaries (RGBO), but got {len(primaries)}")

    print(f"  Loaded {len(primaries)} primaries")
    print(f"  Wavelength range: {primaries[0].wavelengths[0]}-{primaries[0].wavelengths[-1]} nm")
    print()

    # Set up observer genotypes for creating observer-specific ColorSpaces
    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata'].get('seed', 42)
    )
    print()

    # Convert metamers to BGOR using observer-specific ColorSpaces
    bgor_list = []

    for obs_data in config['observers']:
        genotype = tuple(sorted(tuple(obs_data['genotype'])))  # sort genotype (already includes Q at 547nm)

        # Calculate Q index: Observer will add S cone at 420nm and sort all peaks
        sorted_with_s = tuple(sorted((420,) + genotype))
        q_index = sorted_with_s.index(547)

        print(f"Processing observer {obs_data['observer_index']}: {genotype} (Q at index {q_index})")

        # Create observer-specific ColorSpace with display primaries
        observer = observer_genotypes.get_observer_for_peaks(genotype)
        metameric_axis = config['metadata'].get('metameric_axis', 2)
        color_space = ColorSpace(observer, display_primaries=primaries, metameric_axis=metameric_axis)
        print(f"  Created observer-specific ColorSpace for conversion")

        # Convert each metamer pair
        for metamer in obs_data['metamers']:
            # Get BGYR values for storage in CSV (always available)
            bgyr_1 = np.array(metamer['bgyr_1'])
            bgyr_2 = np.array(metamer['bgyr_2'])
            
            # Check if RGBO values are stored (from generation in DISP space)
            # If available, use them directly since they're already in the correct normalized [0, 1] range
            # This avoids conversion errors from BGYR → DISP
            if 'rgbo_1' in metamer and 'rgbo_2' in metamer:
                # Use stored RGBO values (already normalized [0, 1] from ColorSampler)
                rgbo_1 = np.array(metamer['rgbo_1'])  # RGBO order
                rgbo_2 = np.array(metamer['rgbo_2'])  # RGBO order
                
                # Convert RGBO to BGOR: RGBO=[R,G,B,O] -> BGOR=[B,G,O,R]
                bgor_1 = np.array([rgbo_1[2], rgbo_1[1], rgbo_1[3], rgbo_1[0]])  # B, G, O, R
                bgor_2 = np.array([rgbo_2[2], rgbo_2[1], rgbo_2[3], rgbo_2[0]])  # B, G, O, R
            else:
                # Fallback: Convert from BGYR (for old configs without RGBO values)
                # Convert BGYR → CONE → DISP (BGOR) using observer-specific ColorSpace
                # ColorSpace.convert() handles the BGYR → CONE → DISP transformation
                bgor_1 = color_space.convert(bgyr_1.reshape(1, -1), ColorSpaceType.BGYR, ColorSpaceType.DISP)[0]
                bgor_2 = color_space.convert(bgyr_2.reshape(1, -1), ColorSpaceType.BGYR, ColorSpaceType.DISP)[0]

            # DISP values should be normalized [0, 1] for display primaries
            # Clip to [0, 1] to ensure valid range, then convert to 8-bit
            bgor_1_clipped = np.clip(bgor_1, 0, 1)
            bgor_2_clipped = np.clip(bgor_2, 0, 1)
            bgor_1_8bit = np.clip(np.round(bgor_1_clipped * 255), 0, 255).astype(int)
            bgor_2_8bit = np.clip(np.round(bgor_2_clipped * 255), 0, 255).astype(int)

            # Store both metamers in BGOR order
            for metamer_idx, (bgor_8bit, bgyr) in enumerate([(bgor_1_8bit, bgyr_1), (bgor_2_8bit, bgyr_2)]):
                bgor_list.append({
                    'observer_index': obs_data['observer_index'],
                    'genotype': genotype,
                    'q_cone_index': q_index,
                    'pair_index': metamer['pair_index'],
                    'metamer_index': metamer_idx,
                    'B': int(bgor_8bit[0]),  # B is at index 0 in BGOR
                    'G': int(bgor_8bit[1]),  # G is at index 1 in BGOR
                    'O': int(bgor_8bit[2]),  # O is at index 2 in BGOR
                    'R': int(bgor_8bit[3]),  # R is at index 3 in BGOR
                    'bgyr_b': float(bgyr[0]),
                    'bgyr_g': float(bgyr[1]),
                    'bgyr_y': float(bgyr[2]),
                    'bgyr_r': float(bgyr[3])
                })

            # Print in BGOR order
            print(f"  Pair {metamer['pair_index']}: "
                  f"BGOR1=({bgor_1_8bit[0]},{bgor_1_8bit[1]},{bgor_1_8bit[2]},{bgor_1_8bit[3]}), "
                  f"BGOR2=({bgor_2_8bit[0]},{bgor_2_8bit[1]},{bgor_2_8bit[2]},{bgor_2_8bit[3]})")

        print()

    # Extract unique RGBO values (convert BGOR to RGBO: RGBO = [R, G, B, O])
    unique_rgbo_set = set()
    for row in bgor_list:
        # CSV has BGOR order: B, G, O, R
        # Convert to RGBO: [R, G, B, O]
        rgbo = (row['R'], row['G'], row['B'], row['O'])
        unique_rgbo_set.add(rgbo)

    # Write output
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            # Write header
            f.write("R,G,B,O\n")
            
            # Write unique RGBO values, sorted for consistency
            for rgbo in sorted(unique_rgbo_set):
                f.write(f"{rgbo[0]},{rgbo[1]},{rgbo[2]},{rgbo[3]}\n")

        print(f"Saved {len(unique_rgbo_set)} unique RGBO values to: {output_path}")
    else:
        # Print to stdout
        print("R,G,B,O")
        for rgbo in sorted(unique_rgbo_set):
            print(f"{rgbo[0]},{rgbo[1]},{rgbo[2]},{rgbo[3]}")

    return bgor_list


def main():
    parser = argparse.ArgumentParser(
        description='Convert BGYR metamers to RGBO display values using measured primaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert_bgyr_to_rgbo.py \\
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
        help='Path to BGYR metamer configuration JSON (default: config/display_validation_metamers.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: stdout)'
    )

    args = parser.parse_args()

    # Convert metamers
    convert_bgyr_to_bgor(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
