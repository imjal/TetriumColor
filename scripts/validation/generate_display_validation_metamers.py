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
from TetriumColor.Observer import Observer, Spectra, Illuminant
from TetriumColor.Observer.Observer import Cone
from typing import Any, Tuple

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

VALIDATION_OBSERVER_DEGREE = 2.0


def monte_carlo_metamer_robustness(
    peaks, wavelengths, spectrum_1_data, spectrum_2_data,
    n_samples=1000, metameric_axis=2, template='neitz', seed=42,
    mpod_mean=0.908875, mpod_std=0.25, mpod_min=0.0, mpod_max=2.0,
    lens_mean=1.0, lens_half_range_frac=0.25,
    od_lm_mean=0.485, od_lm_half_range=0.1,
    od_s_mean=0.3875, od_s_half_range=0.1,
):
    """
    Monte Carlo simulation of metamer robustness to observer variation.

    For a given metamer pair (two spectra that should look identical to a nominal
    observer), this estimates how much the match breaks when lens density, macular
    pigment optical density, and photopigment optical density vary across the
    population.

    Parameters are sampled as:
        MPOD:  Normal(mpod_mean, mpod_std), clipped to [mpod_min, mpod_max]
        Lens:  Uniform(lens_mean * (1 - frac), lens_mean * (1 + frac))
        OD_LM: Uniform(od_lm_mean - half_range, od_lm_mean + half_range)
        OD_S:  Uniform(od_s_mean - half_range, od_s_mean + half_range)

    Defaults are centered on the nominal observer degree used for validation.

    Returns dict with noise-weighted LMS cone-distance statistics (d_lms)
    across the sampled observers. d_lms > 1 means ~1 JND match breakdown.
    """
    rng = np.random.default_rng(seed)
    sorted_peaks = sorted(peaks)
    n_cones = len(sorted_peaks)
    n_wl = len(wavelengths)

    # Precompute quantal nomogram templates for each cone
    templates_q = []
    for peak in sorted_peaks:
        nom = Cone.templates[template](wavelengths, peak).as_quantal()
        templates_q.append(nom.data.copy())

    # Precompute lens and macular absorption at these wavelengths
    lens_abs = Cone.lens_absorption.interpolate_values(wavelengths).data.copy()
    mac_abs = Cone.macular_absorption.interpolate_values(wavelengths).data.copy()

    # Illuminant for white-point normalization
    illum_data = Illuminant.get('D65').interpolate_values(wavelengths).data.copy()

    # --- Sample physiological parameters (n_samples,) ---
    mpod = rng.normal(mpod_mean, mpod_std, n_samples).clip(mpod_min, mpod_max)
    lens_lo = lens_mean * (1.0 - lens_half_range_frac)
    lens_hi = lens_mean * (1.0 + lens_half_range_frac)
    lens_d = rng.uniform(lens_lo, lens_hi, n_samples)
    od_lm = rng.uniform(od_lm_mean - od_lm_half_range,
                        od_lm_mean + od_lm_half_range, n_samples)
    od_s = rng.uniform(od_s_mean - od_s_half_range,
                       od_s_mean + od_s_half_range, n_samples)

    # --- Build sensor matrices vectorized: (n_samples, n_cones, n_wl) ---
    sensor_matrices = np.zeros((n_samples, n_cones, n_wl))

    # Pre-receptoral filtering exponent (shared across cones within a sample)
    # shape: (n_samples, n_wl)
    pre_rec = np.power(10.0, lens_d[:, None] * lens_abs[None, :]
                       + mpod[:, None] * mac_abs[None, :])

    for j, (peak, tmpl_q) in enumerate(zip(sorted_peaks, templates_q)):
        od = od_s if peak == 420 else od_lm  # (n_samples,)

        # Photopigment self-screening (Beer-Lambert in quantal space)
        # shape: (n_samples, n_wl)
        od_col = od[:, None]
        numerator = 1.0 - np.power(10.0, -od_col * tmpl_q[None, :])
        denominator = 1.0 - np.power(10.0, -od_col)
        od_applied = numerator / denominator

        # Apply pre-receptoral filtering
        filtered = od_applied / pre_rec

        # Normalize (equivalent to Spectra.__invert__)
        fmax = filtered.max(axis=1, keepdims=True)
        fmax = np.maximum(fmax, 1e-30)
        filtered = filtered / fmax

        # Convert quantal -> energy: multiply by wavelength, renormalize
        energy = filtered * wavelengths[None, :]
        emax = energy.max(axis=1, keepdims=True)
        emax = np.maximum(emax, 1e-30)
        energy = energy / emax

        # Normalize by illuminant white-point response
        illum_response = energy @ illum_data  # (n_samples,)
        illum_response = np.maximum(illum_response, 1e-30)

        # Weight by illuminant (matching Observer.get_normalized_sensor_matrix)
        sensor_matrices[:, j, :] = (energy * illum_data[None, :]) / illum_response[:, None]

    # --- Compute cone responses ---
    r1 = np.einsum('scw,w->sc', sensor_matrices, spectrum_1_data)  # (n_samples, n_cones)
    r2 = np.einsum('scw,w->sc', sensor_matrices, spectrum_2_data)

    # Noise-weighted cone contrast distance (same formula as Observer.cone_distance)
    avg = np.maximum((r1 + r2) / 2.0, 1e-10)
    delta = (r1 - r2) / avg

    lms_idx = [j for j in range(n_cones) if j != metameric_axis]
    d_lms = np.sqrt(np.sum(delta[:, lms_idx] ** 2, axis=1))

    if n_cones > 3:
        d_q = np.abs(delta[:, metameric_axis])
    else:
        d_q = np.zeros(n_samples)

    return {
        'mc_mean_d_lms': float(np.mean(d_lms)),
        'mc_std_d_lms': float(np.std(d_lms)),
        'mc_p95_d_lms': float(np.percentile(d_lms, 95)),
        'mc_max_d_lms': float(np.max(d_lms)),
        'mc_mean_d_q': float(np.mean(d_q)),
        'mc_std_d_q': float(np.std(d_q)),
        'mc_n_samples': n_samples,
    }


def _nominal_observer_params(degree=4.0):
    """Compute the degree-adjusted parameters matching Cone.cone() internals."""
    degree = np.clip(degree, 2.0, 10.0)
    macular = 1.0 * (1.0 + (0.271 - 1.0) * (degree**2 - 4) / 96)
    od_lm = 0.50 + (0.38 - 0.50) * (degree**2 - 4) / 96
    od_s = 0.40 + (0.30 - 0.40) * (degree**2 - 4) / 96
    lens = 1.0
    return {'mpod': macular, 'lens': lens, 'od_lm': od_lm, 'od_s': od_s}


def _reconstruct_spectrum(bgor_weights, display_primaries, wavelengths):
    """Reconstruct a spectrum from BGOR display weights and primaries."""
    spectrum_data = np.zeros(len(wavelengths))
    for w, p in zip(bgor_weights, display_primaries):
        spectrum_data += w * p.interpolate_values(wavelengths).data
    return spectrum_data


def _resolve_metameric_axis(genotype, requested_axis: int, peak_to_test: float = 547) -> int:
    """Resolve the validation axis to the sorted cone index of peak_to_test.

    QuestColorGenerator remaps the requested Q-axis to the actual sorted cone
    index for each genotype. Do the same here so validation metamers target the
    same 547 nm cone even when it is not index 2.
    """
    peaks_with_s = sorted(set([420] + list(genotype)))
    if peak_to_test in peaks_with_s:
        return peaks_with_s.index(peak_to_test)
    return requested_axis


def generate_metamers(
    num_observers: int = 8,
    sex: str = 'both',
    grid_size: int = 3,
    luminance: float = 1.0,
    saturation: float = 0.5,
    cube_face: int = 4,
    metameric_axis: int = 2,
    seed: int = 42,
    primaries_path: str = None,
    mc_samples: int = 1000,
    use_display_midpoint: bool = True,
    proportion: float = 0.8,
):
    """
    Generate fixed metamer pairs for validation using ColorSampler.

    Generates metamers in DISP space (RGBO/BGOR) for each observer using their specific
    ColorSpace, then converts to BGYR for storage. This ensures each observer's metamers
    are generated in their own display space.

    When use_display_midpoint=True (default), each observer's single metamer pair is
    generated from the display midpoint [0.5, 0.5, 0.5, 0.5], matching the background
    used by GeneticColorGenerator and QuestColorGenerator in the psychophysics app.

    When use_display_midpoint=False, uses the legacy ColorSampler cubemap path which
    samples a grid×grid array of metamer pairs on a cubemap face. Cubemap points are
    at gamut boundary cusps and are dimmer than the display midpoint.

    Args:
        num_observers: Number of top observers to generate metamers for
        sex: Population to sample from ('male', 'female', 'both')
        grid_size: Size of grid for legacy cubemap mode (ignored when use_display_midpoint=True)
        luminance: Luminance level in VSH space — only used in legacy cubemap mode
        saturation: Saturation level in VSH space — only used in legacy cubemap mode
        cube_face: Which cubemap face to sample — only used in legacy cubemap mode
        metameric_axis: Axis to be metameric over (default: 2 for Q cone)
        seed: Random seed for reproducibility
        primaries_path: Path to directory with display primaries (required)
        mc_samples: Number of Monte Carlo samples for observer variability simulation
        use_display_midpoint: If True (default), generate one pair per observer from
            DISP [0.5,0.5,0.5,0.5]. If False, use legacy cubemap grid sampling.

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

    # Compute degree-adjusted nominal observer parameters (must match Cone.cone internals)
    nom = _nominal_observer_params(degree=VALIDATION_OBSERVER_DEGREE)

    if use_display_midpoint:
        print(f"Top {num_observers} observers using display midpoint [0.5, 0.5, 0.5, 0.5]...")
    else:
        print(f"Top {num_observers} observers using ColorSampler (legacy cubemap mode)...")
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
    num_pairs_per_observer = 1 if use_display_midpoint else grid_size * grid_size

    if use_display_midpoint:
        print(f"Generating 1 metamer pair per observer from display midpoint [0.5, 0.5, 0.5, 0.5]")
    else:
        print(f"Generating {num_pairs_per_observer} metamer pairs per observer from cube face {cube_face}")
    print()

    observers_data = []

    for observer_idx, (genotype, probability) in enumerate(zip(genotypes, probabilities)):
        print(f"Processing observer {observer_idx+1}/{num_observers}: {genotype}")

        observer_metameric_axis = _resolve_metameric_axis(genotype, metameric_axis)
        peaks_with_s = sorted(set([420] + list(genotype)))
        metameric_peak = peaks_with_s[observer_metameric_axis]

        # Create observer (add S cone at 420nm if not present)
        observer = observer_genotypes.get_observer_for_peaks(
            genotype, degree=VALIDATION_OBSERVER_DEGREE)

        # Create ColorSpace with display primaries (always use DISP space)
        color_space = ColorSpace(
            observer,
            display_primaries=display_primaries,
            metameric_axis=observer_metameric_axis)
        print(
            f"  Requested axis={metameric_axis}; resolved validation axis="
            f"{observer_metameric_axis} ({metameric_peak} nm)")

        try:
            if use_display_midpoint:
                # Generate a single metamer pair from the display midpoint DISP [0.5, 0.5, 0.5, 0.5].
                # This matches the background used by GeneticColorGenerator and QuestColorGenerator.
                point = np.ones(color_space.dim) * 0.5
                inside_disp, outside_disp, _ = color_space.get_maximal_pair_in_disp_from_pt(
                    point, metameric_axis=observer_metameric_axis, proportion=proportion,
                    output_space=ColorSpaceType.DISP)
                metamers_in_disp_space = np.array([[inside_disp, outside_disp]])
                disp_pair = np.array([inside_disp, outside_disp])
                cone_pair = color_space.convert(disp_pair, ColorSpaceType.DISP, ColorSpaceType.CONE)
                cones = cone_pair[np.newaxis, :, :]  # shape (1, 2, dim)
                effective_grid_size = 1
                print(f"  Generated midpoint metamer pair in DISP space")
            else:
                # Legacy cubemap path: sample grid_size×grid_size points from the cubemap.
                color_sampler = ColorSampler(color_space, cubemap_size=grid_size, disable=False)
                print(f"  Using ColorSampler with {grid_size}×{grid_size} grid on cube face {cube_face}")
                # Returns (metamers_in_sampling_space, cones):
                # - metamers_in_sampling_space: shape (grid_size^2, 2, 4) in DISP space (BGOR order)
                # - cones: shape (grid_size^2, 2, 4) in CONE space
                metamers_in_disp_space, cones = color_sampler.get_metameric_pairs(
                    luminance=luminance,
                    saturation=saturation,
                    cube_idx=cube_face,
                    metameric_axis=observer_metameric_axis
                )
                effective_grid_size = grid_size
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
                row = i // effective_grid_size
                col = i % effective_grid_size

                # Get the two metamers in DISP space (BGOR order)
                disp_1 = metamers_in_disp_space[i, 0]  # BGOR order
                disp_2 = metamers_in_disp_space[i, 1]  # BGOR order

                # Get converted BGYR values
                bgyr_1 = bgyr_reshaped[i, 0]
                bgyr_2 = bgyr_reshaped[i, 1]

                cone_1 = cones[i, 0]
                cone_2 = cones[i, 1]

                # Calculate metamer difference in the resolved validation channel
                metamer_diff = abs(cone_1[observer_metameric_axis] - cone_2[observer_metameric_axis])

                # Nominal observer d_lms (excluding metameric axis)
                n_cones = len(cone_1)
                avg = np.maximum((cone_1 + cone_2) / 2.0, 1e-10)
                delta = (cone_1 - cone_2) / avg
                lms_idx = [j for j in range(n_cones) if j != observer_metameric_axis]
                nominal_d_lms = float(np.sqrt(np.sum(delta[lms_idx] ** 2)))

                # Convert BGOR to RGBO for output
                rgbo_1 = np.array([disp_1[3], disp_1[1], disp_1[0], disp_1[2]])  # R, G, B, O
                rgbo_2 = np.array([disp_2[3], disp_2[1], disp_2[0], disp_2[2]])

                # Monte Carlo robustness: reconstruct spectra from BGOR weights
                spec_1 = _reconstruct_spectrum(disp_1, display_primaries, wavelengths)
                spec_2 = _reconstruct_spectrum(disp_2, display_primaries, wavelengths)
                mc_result = monte_carlo_metamer_robustness(
                    peaks=genotype, wavelengths=wavelengths,
                    spectrum_1_data=spec_1, spectrum_2_data=spec_2,
                    n_samples=mc_samples, metameric_axis=observer_metameric_axis,
                    template='neitz', seed=seed + i,
                    mpod_mean=nom['mpod'], lens_mean=nom['lens'],
                    od_lm_mean=nom['od_lm'], od_s_mean=nom['od_s'],
                )

                metamer_dict = {
                    'pair_index': i,
                    'grid_position': [int(row), int(col)],
                    'cone_1': cone_1.tolist(),
                    'cone_2': cone_2.tolist(),
                    'metamer_difference': float(metamer_diff),
                    'requested_metameric_axis': int(metameric_axis),
                    'metameric_axis': int(observer_metameric_axis),
                    'metameric_peak_nm': float(metameric_peak),
                    'rgbo_1': rgbo_1.tolist(),  # Original DISP values (RGBO order)
                    'rgbo_2': rgbo_2.tolist(),
                    'bgyr_1': bgyr_1.tolist(),  # Converted to BGYR for storage
                    'bgyr_2': bgyr_2.tolist(),
                    'nominal_d_lms': nominal_d_lms,
                    'observer_variability': mc_result,
                }

                metamer_pairs.append(metamer_dict)

                if i < 3 or i == n_points // 2:  # Print first few and middle
                    print(f"  Pair {i} at grid ({row}, {col})")
                    print(f"    DISP (BGOR) raw: M1={disp_1}, M2={disp_2}")
                    print(f"    RGBO1: {metamer_dict['rgbo_1']}")
                    print(f"    RGBO2: {metamer_dict['rgbo_2']}")
                    print(f"    BGYR1: {metamer_dict['bgyr_1']}")
                    print(f"    BGYR2: {metamer_dict['bgyr_2']}")
                    print(
                        f"    Metamer diff (axis {observer_metameric_axis}, "
                        f"{metameric_peak} nm): {metamer_diff:.4f}")
                    print(f"    Nominal d_lms: {nominal_d_lms:.4f}")
                    print(f"    MC population d_lms: mean={mc_result['mc_mean_d_lms']:.3f}, "
                          f"p95={mc_result['mc_p95_d_lms']:.3f}")

            observers_data.append({
                'observer_index': observer_idx,
                'genotype': list(genotype),
                'probability': float(probability),
                'requested_metameric_axis': int(metameric_axis),
                'metameric_axis': int(observer_metameric_axis),
                'metameric_peak_nm': float(metameric_peak),
                'metamers': metamer_pairs
            })

            print(f"  Successfully generated {len(metamer_pairs)} metamer pairs")
            print()

        except Exception as e:
            print(f"  Error generating metamers for observer {genotype}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Serialize display primaries (BGOR order)
    primaries_serialized = [
        {
            'wavelengths': p.wavelengths.tolist(),
            'data': p.data.tolist(),
        }
        for p in display_primaries
    ]

    # Create output structure
    if use_display_midpoint:
        description = (
            f'1 metamer pair per observer generated from display midpoint DISP [0.5,0.5,0.5,0.5] '
            f'at proportion={proportion} of the maximal metameric extent. '
            f'Matches the background and per-observer 547 nm axis resolution used by QuestColorGenerator. '
            f'Generated in DISP space (RGBO), converted to BGYR for storage.'
        )
        method = f'ColorSpace.get_maximal_pair_in_disp_from_pt(np.ones(dim)*0.5, resolved_547_axis, proportion={proportion}) in DISP space, converted to BGYR via ColorSpace.convert()'
    else:
        description = (
            f'Grid of {grid_size}×{grid_size} metamer pairs per observer. '
            f'Generated in DISP space (RGBO) using observer-specific ColorSpace, then converted to BGYR for storage. '
            f'Sampled using ColorSampler on cube face {cube_face} at luminance={luminance}, saturation={saturation}'
        )
        method = 'ColorSampler.get_metameric_pairs() in DISP space, converted to BGYR via ColorSpace.convert()'

    output = {
        'metadata': {
            'num_observers': num_observers,
            'num_pairs_per_observer': num_pairs_per_observer,
            'sex': sex,
            'grid_size': 1 if use_display_midpoint else grid_size,
            'use_display_midpoint': use_display_midpoint,
            'proportion': proportion if use_display_midpoint else None,
            'luminance': None if use_display_midpoint else luminance,
            'saturation': None if use_display_midpoint else saturation,
            'cube_face': None if use_display_midpoint else cube_face,
            'requested_metameric_axis': metameric_axis,
            'metameric_axis': metameric_axis,
            'metameric_axis_note': (
                'Legacy fallback only. New configs store the resolved per-observer '
                '547 nm validation axis in each observer/metamer as metameric_axis.'
            ),
            'metameric_peak_nm': 547,
            'observer_degree': VALIDATION_OBSERVER_DEGREE,
            'seed': seed,
            'sampling_space': 'RGBO',
            'storage_space': 'BGYR',
            'used_display_primaries': True,
            'primaries_path': str(primaries_path),
            'primaries_order': 'BGOR',
            'wavelength_range': [int(wavelengths[0]), int(wavelengths[-1])],
            'total_metamer_pairs': sum(len(obs['metamers']) for obs in observers_data),
            'description': description,
            'method': method,
            'mc_samples': mc_samples,
            'mc_params': {
                'note': f'means are degree-adjusted to match nominal observer at {VALIDATION_OBSERVER_DEGREE:g} deg',
                'mpod': {'mean': nom['mpod'], 'std': 0.25, 'min': 0.0, 'max': 2.0},
                'lens': {'mean': nom['lens'], 'half_range_frac': 0.25},
                'od_lm': {'mean': nom['od_lm'], 'half_range': 0.1},
                'od_s': {'mean': nom['od_s'], 'half_range': 0.1},
            }
        },
        'display_primaries': primaries_serialized,
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
    parser.add_argument(
        '--mc-samples',
        type=int,
        default=1000,
        help='Number of Monte Carlo samples for observer variability estimation (default: 1000)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating the JND plot'
    )
    parser.add_argument(
        '--use-display-midpoint',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use display midpoint [0.5,0.5,0.5,0.5] as starting point (default: on). '
             'Pass --no-use-display-midpoint for legacy cubemap grid sampling.'
    )
    parser.add_argument(
        '--proportion',
        type=float,
        default=0.8,
        help='Fraction of the maximal metameric extent to use (default: 0.8). '
             'Values < 1.0 keep metamers away from the gamut boundary, making them '
             'robust to day-to-day primaries drift. Only used with --use-display-midpoint.'
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
        primaries_path=args.primaries_path,
        mc_samples=args.mc_samples,
        use_display_midpoint=args.use_display_midpoint,
        proportion=args.proportion,
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

    # Plot observer variability
    if not args.no_plot:
        plot_observer_jnd(output, output_path)


def plot_observer_jnd(output, output_path):
    """Plot nominal and population JND (d_lms) per observer, side by side."""
    observers = output['observers']
    n_obs = len(observers)
    rng = np.random.default_rng(42)

    labels = []
    nominal_means = []
    nominal_all = []
    pop_means = []
    pop_stds = []
    pop_all = []

    for obs in observers:
        labels.append(f"{obs['genotype']}")
        nom = [m['nominal_d_lms'] for m in obs['metamers']]
        pop = [m['observer_variability']['mc_mean_d_lms'] for m in obs['metamers']]
        nominal_all.append(nom)
        nominal_means.append(np.mean(nom))
        pop_all.append(pop)
        pop_means.append(np.mean(pop))
        pop_stds.append(np.std(pop))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_obs * 2), 5), sharey=True)
    x = np.arange(n_obs)

    # Left panel: nominal observer
    ax1.bar(x, nominal_means, color='mediumseagreen', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Mean across pairs')
    for i, noms in enumerate(nominal_all):
        jitter = rng.uniform(-0.2, 0.2, len(noms))
        ax1.scatter(np.full(len(noms), i) + jitter, noms,
                    color='darkgreen', s=15, alpha=0.6, zorder=3)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1 JND')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('d_lms (JND units, excl. 547nm cone)')
    ax1.set_xlabel('Observer genotype (cone peaks)')
    ax1.set_title('Nominal Observer\n(target observer for each metamer)')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: population robustness
    ax2.bar(x, pop_means, yerr=pop_stds, capsize=4, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Mean across pairs')
    for i, pops in enumerate(pop_all):
        jitter = rng.uniform(-0.2, 0.2, len(pops))
        ax2.scatter(np.full(len(pops), i) + jitter, pops,
                    color='coral', s=15, alpha=0.6, zorder=3)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1 JND')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_xlabel('Observer genotype (cone peaks)')
    ax2.set_title('Population Robustness\n(MC over observer variation)')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    plot_path = Path(output_path).with_suffix('.png')
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
