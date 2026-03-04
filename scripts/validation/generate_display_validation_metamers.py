#!/usr/bin/env python3
"""
Generate fixed BGYR metamer pairs for display validation using ColorSampler.

This script creates a configuration file containing metamer pairs in BGYR space
for the top 5 tetrachromat observer genotypes. These metamers are fixed and will
be converted to RGBO daily based on measured display primaries.

Uses ColorSampler to efficiently generate a grid of metamer pairs on a cubemap face.
"""

from TetriumColor.ColorSampler import ColorSampler
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer
from typing import Any, Tuple

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_metamers(
    num_observers: int = 8,
    sex: str = 'both',
    grid_size: int = 3,
    luminance: float = 1.0,
    saturation: float = 0.5,
    cube_face: int = 4,
    metameric_axis: int = 2,
    seed: int = 42,
    primaries_path: str = None
):
    """
    Generate fixed metamer pairs for validation using ColorSampler.

    Generates metamers in DISP space (RGBO/BGOR) for each observer using their specific
    ColorSpace, then converts to BGYR for storage. This ensures each observer's metamers
    are generated in their own display space.

    For each observer, this creates a grid×grid array of metamer pairs on a cubemap
    face perpendicular to the Q-metameric direction. The ColorSampler handles the
    grid sampling and metamer finding efficiently.

    Args:
        num_observers: Number of top observers to generate metamers for
        sex: Population to sample from ('male', 'female', 'both')
        grid_size: Size of grid (e.g., 5 for 5×5 = 25 pairs per observer)
        luminance: Luminance level in VSH space for the sampling plane
        saturation: Saturation level in VSH space (controls distance from gray)
        cube_face: Which cubemap face to sample (0-5, default 4 is +Z face)
        metameric_axis: Axis to be metameric over (default: 2 for Q cone)
        seed: Random seed for reproducibility
        primaries_path: Path to directory with display primaries (required)

    Returns:
        Dictionary with structure:
        {
            'metadata': {...},
            'observers': [
                {
                    'genotype': [420, 530, 547, 559],
                    'probability': 0.5,
                    'metamers': [
                        {
                            'pair_index': 0,
                            'grid_position': [2, 2],  # (row, col)
                            'bgyr_1': [...],  # Converted from DISP for storage
                            'bgyr_2': [...],
                            'rgbo_1': [...],  # Original DISP values
                            'rgbo_2': [...],
                            'cone_1': [...],
                            'cone_2': [...]
                        },
                        ...  # 25 total pairs for 5×5 grid
                    ]
                },
                ...
            ]
        }
    """
    # Load display primaries (required)
    if not primaries_path:
        raise ValueError("--primaries-path is required. Metamers must be generated in DISP space.")

    from TetriumColor.Measurement.TetriumMeasurementRoutines import load_primaries_from_csv
    print(f"Loading display primaries from: {primaries_path}")
    display_primaries = load_primaries_from_csv(primaries_path, extract_zero=False, primary_order='BGOR')
    print(f"Loaded {len(display_primaries)} primaries (BGOR order)")
    print(f"Generating metamer grid in DISPLAY PRIMARY space (RGBO)")

    print(f"Top {num_observers} observers using ColorSampler...")
    print(f"Parameters: sex={sex}, grid_size={grid_size}×{grid_size}, seed={seed}")
    print(f"Luminance={luminance}, saturation={saturation}, cube_face={cube_face}")
    print(f"Metameric axis={metameric_axis}")
    print()

    # Use the primaries' wavelength grid so generate and validate use identical observers
    wavelengths = display_primaries[0].wavelengths

    # Initialize ObserverGenotypes for tetrachromats
    observer_genotypes = ObserverGenotypes(wavelengths=wavelengths, dimensions=[3], seed=seed)

    # Get top N observers
    genotypes = list[Any](observer_genotypes.get_pdf(sex).keys())[:num_observers]
    genotypes = [genotype + (547,) for genotype in genotypes]  # add Q cone at 547nm to make tetrachromat
    probabilities = list(observer_genotypes.get_pdf(sex).values())[:num_observers]

    print(f"Selected {len(genotypes)} observers:")
    for i, (genotype, prob) in enumerate(zip(genotypes, probabilities)):
        print(f"  {i+1}. {genotype} (probability: {prob:.4f})")
    print()

    # Total number of metamer pairs per observer
    num_pairs_per_observer = grid_size * grid_size

    print(f"Generating {num_pairs_per_observer} metamer pairs per observer from cube face {cube_face}")
    print()

    observers_data = []

    for observer_idx, (genotype, probability) in enumerate(zip(genotypes, probabilities)):
        print(f"Processing observer {observer_idx+1}/{num_observers}: {genotype}")

        # Create observer (add S cone at 420nm if not present)
        observer = observer_genotypes.get_observer_for_peaks(genotype)

        # Create ColorSpace with display primaries (always use DISP space)
        color_space = ColorSpace(observer, display_primaries=display_primaries, metameric_axis=metameric_axis)

        # Create ColorSampler with the specified grid size
        color_sampler = ColorSampler(color_space, cubemap_size=grid_size, disable=False)

        print(f"  Using ColorSampler with {grid_size}×{grid_size} grid on cube face {cube_face}")

        try:
            # Get metamer pairs for the specified cube face
            # This returns (metamers_in_sampling_space, cones) where:
            # - metamers_in_sampling_space: shape (grid_size^2, 2, 4) - pairs in DISP space (BGOR order)
            # - cones: shape (grid_size^2, 2, 4) - pairs in CONE space
            metamers_in_disp_space, cones = color_sampler.get_metameric_pairs(
                luminance=luminance,
                saturation=saturation,
                cube_idx=cube_face,
                metameric_axis=metameric_axis
            )

            print(f"  Generated {len(metamers_in_disp_space)} metamer pairs in DISP space")

            # Convert DISP (BGOR) to BGYR for storage using observer-specific ColorSpace
            print(f"  Converting DISP → BGYR for storage...")
            # Reshape for conversion: (n_pairs * 2, 4) -> convert -> reshape back
            n_pairs = len(metamers_in_disp_space)
            disp_flat = metamers_in_disp_space.reshape(-1, 4)  # (n_pairs * 2, 4)
            bgyr_flat = color_space.convert(disp_flat, ColorSpaceType.DISP, ColorSpaceType.BGYR)
            bgyr_reshaped = bgyr_flat.reshape(n_pairs, 2, 4)  # (n_pairs, 2, 4)

            n_points = len(metamers_in_disp_space)

            # Package into metamer list
            metamer_pairs = []
            for i in range(n_points):
                # Calculate grid position from flat index
                row = i // grid_size
                col = i % grid_size

                # Get the two metamers in DISP space (BGOR order)
                disp_1 = metamers_in_disp_space[i, 0]  # BGOR order
                disp_2 = metamers_in_disp_space[i, 1]  # BGOR order

                # Get converted BGYR values
                bgyr_1 = bgyr_reshaped[i, 0]
                bgyr_2 = bgyr_reshaped[i, 1]

                cone_1 = cones[i, 0]
                cone_2 = cones[i, 1]

                # Calculate metamer difference (Q channel)
                metamer_diff = abs(cone_1[metameric_axis] - cone_2[metameric_axis])

                # Convert BGOR to RGBO for output
                rgbo_1 = np.array([disp_1[3], disp_1[1], disp_1[0], disp_1[2]])  # R, G, B, O
                rgbo_2 = np.array([disp_2[3], disp_2[1], disp_2[0], disp_2[2]])

                metamer_dict = {
                    'pair_index': i,
                    'grid_position': [int(row), int(col)],
                    'cone_1': cone_1.tolist(),
                    'cone_2': cone_2.tolist(),
                    'metamer_difference': float(metamer_diff),
                    'rgbo_1': rgbo_1.tolist(),  # Original DISP values (RGBO order)
                    'rgbo_2': rgbo_2.tolist(),
                    'bgyr_1': bgyr_1.tolist(),  # Converted to BGYR for storage
                    'bgyr_2': bgyr_2.tolist()
                }

                metamer_pairs.append(metamer_dict)

                if i < 3 or i == n_points // 2:  # Print first few and middle
                    print(f"  Pair {i} at grid ({row}, {col})")
                    print(f"    DISP (BGOR) raw: M1={disp_1}, M2={disp_2}")
                    print(f"    RGBO1: {metamer_dict['rgbo_1']}")
                    print(f"    RGBO2: {metamer_dict['rgbo_2']}")
                    print(f"    BGYR1: {metamer_dict['bgyr_1']}")
                    print(f"    BGYR2: {metamer_dict['bgyr_2']}")
                    print(f"    Metamer diff ({['S', 'M', 'Q', 'L'][metameric_axis]}): {metamer_diff:.4f}")

            observers_data.append({
                'observer_index': observer_idx,
                'genotype': list(genotype),
                'probability': float(probability),
                'metamers': metamer_pairs
            })

            print(f"  Successfully generated {len(metamer_pairs)} metamer pairs")
            print()

        except Exception as e:
            print(f"  Error generating metamers for observer {genotype}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create output structure
    output = {
        'metadata': {
            'num_observers': num_observers,
            'num_pairs_per_observer': num_pairs_per_observer,
            'sex': sex,
            'grid_size': grid_size,
            'luminance': luminance,
            'saturation': saturation,
            'cube_face': cube_face,
            'metameric_axis': metameric_axis,
            'seed': seed,
            'sampling_space': 'RGBO',
            'storage_space': 'BGYR',
            'used_display_primaries': True,
            'wavelength_range': [int(wavelengths[0]), int(wavelengths[-1])],
            'total_metamer_pairs': sum(len(obs['metamers']) for obs in observers_data),
            'description': f'Grid of {grid_size}×{grid_size} metamer pairs per observer. Generated in DISP space (RGBO) using observer-specific ColorSpace, then converted to BGYR for storage. Sampled using ColorSampler on cube face {cube_face} at luminance={luminance}, saturation={saturation}',
            'method': 'ColorSampler.get_metameric_pairs() in DISP space, converted to BGYR via ColorSpace.convert()'
        },
        'observers': observers_data
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed BGYR metamer pairs for display validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_display_validation_metamers.py \\
    --output config/display_validation_metamers.json \\
    --num-observers 5 \\
    --grid-size 5 \\
    --luminance 1.0 \\
    --saturation 0.5 \\
    --cube-face 4
        """
    )

    parser.add_argument(
        '--output',
        type=str,
        default='config/display_validation_metamers.json',
        help='Output JSON file path (default: config/display_validation_metamers.json)'
    )
    parser.add_argument(
        '--num-observers',
        type=int,
        default=5,
        help='Number of top observers to generate metamers for (default: 5)'
    )
    parser.add_argument(
        '--sex',
        type=str,
        default='both',
        choices=['male', 'female', 'both'],
        help='Population to sample observers from (default: both)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=5,
        help='Size of nxn grid (e.g., 5 for 5×5 = 25 pairs per observer) (default: 5)'
    )
    parser.add_argument(
        '--luminance',
        type=float,
        default=1.0,
        help='Luminance level in VSH space (default: 1.0)'
    )
    parser.add_argument(
        '--saturation',
        type=float,
        default=0.5,
        help='Saturation level in VSH space (controls distance from gray) (default: 0.5)'
    )
    parser.add_argument(
        '--cube-face',
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4, 5],
        help='Cubemap face to sample (0-5, default: 4 which is +Z face)'
    )
    parser.add_argument(
        '--metameric-axis',
        type=int,
        default=2,
        help='Metameric axis (0=S, 1=M, 2=Q, 3=L) (default: 2 for Q)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--primaries-path',
        type=str,
        required=True,
        help='Path to directory with display primaries (required). Metamers are generated in DISP space (RGBO) for each observer, then converted to BGYR for storage.'
    )

    args = parser.parse_args()

    # Generate metamers
    output = generate_metamers(
        num_observers=args.num_observers,
        sex=args.sex,
        grid_size=args.grid_size,
        luminance=args.luminance,
        saturation=args.saturation,
        cube_face=args.cube_face,
        metameric_axis=args.metameric_axis,
        seed=args.seed,
        primaries_path=args.primaries_path
    )

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved metamer configuration to: {output_path}")
    print(f"Total observers: {len(output['observers'])}")
    print(f"Total metamer pairs: {output['metadata']['total_metamer_pairs']}")
    print()
    print("Summary:")
    for obs in output['observers']:
        print(f"  Observer {obs['observer_index']}: {obs['genotype']} - {len(obs['metamers'])} pairs")


if __name__ == '__main__':
    main()
