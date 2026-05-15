#!/usr/bin/env python3
"""Verify that saved display-validation metamers are null in raw display cone space."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer import Illuminant
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes

VALIDATION_OBSERVER_DEGREE = 2.0


def _resolve_observer_illuminant(illuminant):
    if isinstance(illuminant, str):
        if illuminant.lower() == 'raw':
            return illuminant, 'raw'
        return illuminant, Illuminant.get(illuminant)
    return None, illuminant


def _rgbo_to_bgor(rgbo):
    rgbo = np.asarray(rgbo, dtype=float)
    return np.array([rgbo[2], rgbo[1], rgbo[3], rgbo[0]], dtype=float)


def _rgbo_to_primary_order(rgbo, primary_order: str):
    primary_order = primary_order.upper()
    if primary_order == 'RGBO':
        return np.asarray(rgbo, dtype=float)
    if primary_order == 'BGOR':
        return _rgbo_to_bgor(rgbo)
    raise ValueError(f"Unsupported primary_order: {primary_order}")


def _non_axis_indices(dim, axis):
    return [i for i in range(dim) if i != axis]


def verify_null_consistency(
    metamers_config_path: str,
    primaries_path: str,
    tolerance: float = 1e-8,
    illuminant: str = 'raw',
    primary_order: str | None = None,
):
    with open(metamers_config_path, 'r') as f:
        config = json.load(f)

    observer_illuminant_label, observer_illuminant = _resolve_observer_illuminant(
        config.get('metadata', {}).get('observer_illuminant', illuminant))
    metadata = config.get('metadata', {})
    primary_order = primary_order or metadata.get('primaries_order', 'BGOR')
    primaries = load_primaries_from_csv(
        primaries_path, extract_zero=False, primary_order=primary_order)
    wavelengths = np.asarray(metadata.get('observer_wavelengths', primaries[0].wavelengths), dtype=float)
    observer_degree = float(metadata.get('observer_degree', VALIDATION_OBSERVER_DEGREE))
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config.get('metadata', {}).get('seed', 42),
        template='baylor',
    )

    rows = []
    failures = []

    for obs_data in config['observers']:
        genotype = tuple(sorted(obs_data['genotype']))
        observer = observer_genotypes.get_observer_for_peaks(
            genotype,
            degree=observer_degree,
            illuminant=observer_illuminant,
            template='baylor',
        )
        metameric_axis = obs_data.get(
            'metameric_axis',
            config.get('metadata', {}).get('metameric_axis', 2),
        )
        color_space = ColorSpace(
            observer,
            display_primaries=primaries,
            metameric_axis=metameric_axis,
        )
        raw_display_to_cone = color_space.get_raw_display_to_cone_matrix()
        background = np.full(raw_display_to_cone.shape[1], 0.5)
        background_cones = np.maximum(raw_display_to_cone @ background, 1e-30)
        non_axis = _non_axis_indices(raw_display_to_cone.shape[0], metameric_axis)

        for metamer in obs_data['metamers']:
            if 'rgbo_1' in metamer and 'rgbo_2' in metamer:
                disp_1 = _rgbo_to_primary_order(metamer['rgbo_1'], primary_order)
                disp_2 = _rgbo_to_primary_order(metamer['rgbo_2'], primary_order)
            elif 'disp_1' in metamer and 'disp_2' in metamer:
                disp_1 = np.asarray(metamer['disp_1'], dtype=float)
                disp_2 = np.asarray(metamer['disp_2'], dtype=float)
            else:
                cone_1 = np.asarray(metamer['cone_1'], dtype=float)
                cone_2 = np.asarray(metamer['cone_2'], dtype=float)
                disp_1 = color_space.convert(
                    cone_1.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0]
                disp_2 = color_space.convert(
                    cone_2.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP)[0]

            raw_contrast_delta = (raw_display_to_cone @ (disp_1 - disp_2)) / background_cones
            converted_cones = color_space.convert(
                np.vstack([disp_1, disp_2]), ColorSpaceType.DISP, ColorSpaceType.CONE)
            converted_delta = converted_cones[0] - converted_cones[1]
            stored_delta = (
                np.asarray(metamer['cone_1'], dtype=float)
                - np.asarray(metamer['cone_2'], dtype=float)
            )

            raw_non_axis = float(np.linalg.norm(raw_contrast_delta[non_axis]))
            converted_non_axis = float(np.linalg.norm(converted_delta[non_axis]))
            stored_non_axis = float(np.linalg.norm(stored_delta[non_axis]))
            row = {
                'observer_index': obs_data['observer_index'],
                'pair_index': metamer['pair_index'],
                'metameric_axis': metameric_axis,
                'raw_non_axis_contrast_norm': raw_non_axis,
                'converted_non_axis_norm': converted_non_axis,
                'stored_non_axis_norm': stored_non_axis,
                'target_axis_contrast_delta': float(raw_contrast_delta[metameric_axis]),
            }
            rows.append(row)

            if raw_non_axis > tolerance or converted_non_axis > tolerance:
                failures.append(row)

    max_raw = max((r['raw_non_axis_contrast_norm'] for r in rows), default=0.0)
    max_converted = max((r['converted_non_axis_norm'] for r in rows), default=0.0)
    max_stored = max((r['stored_non_axis_norm'] for r in rows), default=0.0)

    print(f"Observer illuminant: {observer_illuminant_label or type(observer_illuminant).__name__}")
    print(f"Primary order: {primary_order}")
    print(f"Pairs checked: {len(rows)}")
    print(f"Max raw non-axis cone-contrast norm: {max_raw:.6g}")
    print(f"Max converted non-axis cone norm:    {max_converted:.6g}")
    print(f"Max stored non-axis cone norm:       {max_stored:.6g}")

    if failures:
        print(f"\nFAIL: {len(failures)} pair(s) exceeded tolerance {tolerance:g}")
        for row in failures[:10]:
            print(
                "  obs={observer_index} pair={pair_index} axis={metameric_axis} "
                "raw={raw_non_axis_contrast_norm:.6g} converted={converted_non_axis_norm:.6g}"
                .format(**row)
            )
        return 1

    print(f"\nPASS: all raw/display conversion null checks are <= {tolerance:g}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify display-validation metamer null consistency in raw cone space.')
    parser.add_argument('--metamers', required=True, help='Metamer config JSON to verify')
    parser.add_argument('--primaries', required=True, help='Display primaries directory')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='Failure threshold for non-axis norms (default: 1e-8)')
    parser.add_argument('--illuminant', default='raw', choices=['raw', 'D65'],
                        help='Fallback illuminant for configs without observer_illuminant')
    parser.add_argument('--primary-order', default=None, choices=['RGBO', 'BGOR'],
                        help='Override config metadata primaries_order')
    args = parser.parse_args()
    raise SystemExit(verify_null_consistency(
        metamers_config_path=args.metamers,
        primaries_path=args.primaries,
        tolerance=args.tolerance,
        illuminant=args.illuminant,
        primary_order=args.primary_order,
    ))


if __name__ == '__main__':
    main()
