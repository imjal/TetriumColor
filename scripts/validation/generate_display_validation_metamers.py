#!/usr/bin/env python3
"""
Generate fixed RYGB metamer pairs for display validation.

This script creates a configuration file containing metamer pairs in RYGB space
for the top 5 tetrachromat observer genotypes. These metamers are fixed and will
be converted to RGBO daily based on measured display primaries.
"""

from typing import Any


import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TetriumColor.Observer import Observer
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.ColorSampler import ColorSampler


def generate_metamers(
    num_observers: int = 5,
    num_pairs_per_observer: int = 3,
    sex: str = 'both',
    cubemap_size: int = 5,
    luminance: float = 1.0,
    saturation: float = 0.5,
    metameric_axis: int = 2,
    seed: int = 42
):
    """
    Generate fixed RYGB metamer pairs for validation.
    
    Args:
        num_observers: Number of top observers to generate metamers for
        num_pairs_per_observer: Number of metamer pairs per observer (default: 3 for center row)
        sex: Population to sample from ('male', 'female', 'both')
        cubemap_size: Size of cubemap grid (default: 5)
        luminance: Luminance level for metamer generation
        saturation: Saturation level for metamer generation
        metameric_axis: Axis to be metameric over (default: 2 for Q cone)
        seed: Random seed for reproducibility
        
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
                            'rygb_1': [0.5, 0.5, 0.5, 0.5],
                            'rygb_2': [0.5, 0.5, 0.5, 0.5]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    print(f"Generating RYGB metamers for top {num_observers} observers...")
    print(f"Parameters: sex={sex}, cubemap_size={cubemap_size}, seed={seed}")
    print(f"Luminance={luminance}, Saturation={saturation}, Metameric axis={metameric_axis}")
    print()
    
    # Initialize ObserverGenotypes for tetrachromats
    observer_genotypes = ObserverGenotypes(dimensions=[3], seed=seed)
    
    # Get top N observers
    genotypes = list[Any](observer_genotypes.get_pdf(sex).keys())[:num_observers]
    genotypes = [genotype + (547,) for genotype in genotypes] # add Q cone at 547nm to make tetrachromat
    probabilities = list(observer_genotypes.get_pdf(sex).values())[:num_observers]
    
    print(f"Selected {len(genotypes)} observers:")
    for i, (genotype, prob) in enumerate(zip(genotypes, probabilities)):
        print(f"  {i+1}. {genotype} (probability: {prob:.4f})")
    print()
    
    # Generate wavelengths for observers
    wavelengths = np.arange(360, 831, 1)
    
    # Calculate center indices for 5x5 grid
    # Middle row (row 2 of 0-4) has indices: 2*5+0, 2*5+1, 2*5+2, 2*5+3, 2*5+4
    # We want center 3: 2*5+1, 2*5+2, 2*5+3 = indices 11, 12, 13
    center_indices = [11, 12, 13][:num_pairs_per_observer]
    
    print(f"Using center indices: {center_indices} from {cubemap_size}x{cubemap_size} grid")
    print()
    
    observers_data = []
    
    for observer_idx, (genotype, probability) in enumerate(zip(genotypes, probabilities)):
        print(f"Processing observer {observer_idx+1}/{num_observers}: {genotype}")
        
        # Create observer (add S cone at 420nm if not present)
        observer = observer_genotypes.get_observer_for_peaks(genotype)
        
        # Create ColorSpace (no primaries needed for abstract RYGB space)
        color_space = ColorSpace(observer, metameric_axis=metameric_axis)
        
        # Generate metamer pairs directly in RYGB space
        # For a 5x5 grid, center row is at y=2, with x positions 1, 2, 3
        # We'll use these as points in RYGB space
        metamer_pairs = []
        
        for idx, center_idx in enumerate(center_indices):
            # Calculate grid position (row, col) from flat index
            row = center_idx // cubemap_size
            col = center_idx % cubemap_size
            
            # Convert grid position to normalized coordinates [0, 1]
            # Use middle of each cell
            x = (col + 0.5) / cubemap_size
            y = (row + 0.5) / cubemap_size
            
            # Create a point in RYGB space
            # We'll use a simple pattern: vary R and Y based on grid position,
            # keep G and B at middle values
            rygb_pt = np.array([x, y, 0.5, 0.5])
            
            print(f"  Generating pair {idx} at grid position ({row}, {col}), RYGB point: {rygb_pt}")
            
            # Find maximal metamer pair in RYGB space
            try:
                result = color_space.get_maximal_pair_in_disp_from_pt(
                    pt=rygb_pt,
                    metameric_axis=metameric_axis,
                    input_space=ColorSpaceType.RYGB,
                    output_space=ColorSpaceType.RYGB,  # Get results back in RYGB
                    proportion=1.0
                )
                
                if result is None:
                    print(f"    Warning: Could not find metamer pair at {rygb_pt}")
                    continue
                    
                rygb_1, rygb_2, metamer_diff = result
                
                # Also get cone responses for validation
                cone_result = color_space.get_maximal_pair_in_disp_from_pt(
                    pt=rygb_pt,
                    metameric_axis=metameric_axis,
                    input_space=ColorSpaceType.RYGB,
                    output_space=ColorSpaceType.CONE,
                    proportion=1.0
                )
                cone_1, cone_2, _ = cone_result
                
                # Store as lists for JSON serialization
                metamer_pairs.append({
                    'pair_index': idx,
                    'grid_position': [int(row), int(col)],
                    'rygb_center': rygb_pt.tolist(),
                    'rygb_1': rygb_1.tolist(),
                    'rygb_2': rygb_2.tolist(),
                    'cone_1': cone_1.tolist(),
                    'cone_2': cone_2.tolist(),
                    'metamer_difference': float(metamer_diff)
                })
                
                print(f"    RYGB1: {rygb_1}")
                print(f"    RYGB2: {rygb_2}")
                print(f"    Metamer diff (Q): {metamer_diff:.4f}")
                
            except Exception as e:
                print(f"    Error finding metamer pair: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(metamer_pairs) == 0:
            print(f"  Warning: No metamer pairs generated for observer {genotype}")
            continue
        
        observers_data.append({
            'observer_index': observer_idx,
            'genotype': list(genotype),
            'probability': float(probability),
            'metamers': metamer_pairs
        })
        
        print(f"  Generated {len(metamer_pairs)} metamer pairs")
        print()
    
    # Create output structure
    output = {
        'metadata': {
            'num_observers': num_observers,
            'num_pairs_per_observer': num_pairs_per_observer,
            'sex': sex,
            'cubemap_size': cubemap_size,
            'luminance': luminance,
            'saturation': saturation,
            'metameric_axis': metameric_axis,
            'seed': seed,
            'center_indices': center_indices,
            'wavelength_range': [int(wavelengths[0]), int(wavelengths[-1])],
            'total_metamer_pairs': sum(len(obs['metamers']) for obs in observers_data)
        },
        'observers': observers_data
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed RYGB metamer pairs for display validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_display_validation_metamers.py \\
    --output config/display_validation_metamers.json \\
    --num-observers 5 \\
    --num-pairs 3
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
        '--num-pairs',
        type=int,
        default=3,
        help='Number of metamer pairs per observer (default: 3)'
    )
    parser.add_argument(
        '--sex',
        type=str,
        default='both',
        choices=['male', 'female', 'both'],
        help='Population to sample observers from (default: both)'
    )
    parser.add_argument(
        '--cubemap-size',
        type=int,
        default=5,
        help='Size of cubemap grid (default: 5)'
    )
    parser.add_argument(
        '--luminance',
        type=float,
        default=1.0,
        help='Luminance level for metamer generation (default: 1.0)'
    )
    parser.add_argument(
        '--saturation',
        type=float,
        default=0.5,
        help='Saturation level for metamer generation (default: 0.5)'
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
    
    args = parser.parse_args()
    
    # Generate metamers
    output = generate_metamers(
        num_observers=args.num_observers,
        num_pairs_per_observer=args.num_pairs,
        sex=args.sex,
        cubemap_size=args.cubemap_size,
        luminance=args.luminance,
        saturation=args.saturation,
        metameric_axis=args.metameric_axis,
        seed=args.seed
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

