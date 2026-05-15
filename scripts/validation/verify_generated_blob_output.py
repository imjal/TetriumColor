#!/usr/bin/env python3
"""Generate a real Gaussian blob stimulus and verify PNG/display null consistency."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TetriumColor import ColorSpaceType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.TetraColorPicker import QuestColorGenerator
from TetriumColor.TetraPlate import GaussianBlobGenerator


def _read_disp6_from_pngs(rgb_path: str, ocv_path: str) -> np.ndarray:
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'), dtype=np.float64) / 255.0
    ocv = np.asarray(Image.open(ocv_path).convert('RGB'), dtype=np.float64) / 255.0
    return np.concatenate([rgb, ocv], axis=2)


def _max_departure_pixel(disp6_img: np.ndarray, background: np.ndarray):
    flat = disp6_img.reshape(-1, disp6_img.shape[2])
    distances = np.linalg.norm(flat - background.reshape(1, -1), axis=1)
    idx = int(np.argmax(distances))
    return flat[idx], float(distances[idx]), idx


def _line_fit_alpha(pixel: np.ndarray, background: np.ndarray, foreground: np.ndarray):
    direction = foreground - background
    denom = float(direction @ direction)
    if denom <= 1e-30:
        return 0.0, float(np.linalg.norm(pixel - background))
    alpha = float(((pixel - background) @ direction) / denom)
    fitted = background + alpha * direction
    return alpha, float(np.linalg.norm(pixel - fitted))


def verify_generated_blob_output(
    primaries_path: str,
    output_dir: str,
    observer_index: int = 1,
    metameric_axis: int = 2,
    size: int = 256,
    tolerance: float = 1e-8,
    png_tolerance: float = 1.5 / 255.0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    primaries = load_primaries_from_csv(primaries_path, extract_zero=False)
    color_generator = QuestColorGenerator(
        sex='both',
        percentage_screened=0.995,
        luminance=0.5,
        trials_per_direction=1,
        metameric_axes=[metameric_axis],
        dimensions=[3],
        display_primaries=primaries,
        bipolar=True,
        degree=2.0,
        mcs_k=1,
        observer_indices=[observer_index],
        color_picking_space='raw_cone_excitation',
    )
    blob_generator = GaussianBlobGenerator(
        color_generator,
        seed=42,
        size=size,
        blob_size=1.0,
        constant_disp_background=True,
    )

    trial = blob_generator.NewTest(
        str(output_path / 'blob_verify'),
        hidden_symbol='landolt_right',
        output_space=ColorSpaceType.DISP_6P,
        lum_noise=0.0,
        s_cone_noise=0.0,
    )
    metadata = trial['metadata']

    genotype = color_generator.direction_metadata[color_generator.current_direction_idx]['genotype']
    color_space = color_generator.genotype_mapping[genotype]
    inside_disp = np.asarray(metadata['quest_inside_disp'], dtype=np.float64)
    outside_disp = np.asarray(metadata['quest_outside_disp'], dtype=np.float64)

    contrast_delta = color_space.cone_contrast_delta(
        inside_disp,
        outside_disp,
        background=color_generator.adapting_background,
        sample_space=ColorSpaceType.DISP,
        background_space=color_generator.adapting_background_space,
    )
    axis = int(metadata['quest_metameric_axis'])
    non_axis = [i for i in range(len(contrast_delta)) if i != axis]
    non_axis_norm = float(np.linalg.norm(contrast_delta[non_axis]))

    expected_fg = color_space.convert(
        np.asarray(metadata['inside_cone'], dtype=np.float64).reshape(1, -1),
        ColorSpaceType.CONE,
        ColorSpaceType.DISP_6P,
    )[0]
    expected_bg = np.full(6, 0.5, dtype=np.float64)

    disp6_img = _read_disp6_from_pngs(trial['rgb_path'], trial['ocv_path'])
    delivered_peak, peak_distance, peak_idx = _max_departure_pixel(disp6_img, expected_bg)
    alpha, line_error = _line_fit_alpha(delivered_peak, expected_bg, expected_fg)

    # The blob is Gaussian, so a PNG pixel need not equal foreground exactly; it
    # should lie on the display-space line from background to foreground.
    png_ok = line_error <= png_tolerance and alpha > 0.95
    null_ok = non_axis_norm <= tolerance
    passed = bool(png_ok and null_ok)

    result = {
        'passed': passed,
        'rgb_path': trial['rgb_path'],
        'ocv_path': trial['ocv_path'],
        'genotype': str(genotype),
        'metameric_axis': axis,
        'inside_disp': inside_disp.tolist(),
        'outside_disp': outside_disp.tolist(),
        'cone_contrast_delta': contrast_delta.tolist(),
        'non_axis_contrast_norm': non_axis_norm,
        'expected_foreground_disp6': expected_fg.tolist(),
        'delivered_peak_disp6': delivered_peak.tolist(),
        'delivered_peak_pixel_index': peak_idx,
        'delivered_peak_distance_from_background': peak_distance,
        'foreground_line_alpha': alpha,
        'foreground_line_error': line_error,
        'tolerance': tolerance,
        'png_tolerance': png_tolerance,
    }

    report_path = output_path / 'blob_verify_report.json'
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)

    metamer_config_path = output_path / 'blob_verify_metamer_config.json'
    metamer_config = {
        'metadata': {
            'seed': 42,
            'observer_illuminant': 'raw',
            'observer_degree': 2.0,
            'observer_wavelengths': color_space.observer.wavelengths.tolist(),
            'primaries_order': 'RGBO',
            'metameric_axis': axis,
            'total_metamer_pairs': 1,
            'source': 'verify_generated_blob_output.py',
        },
        'observers': [{
            'observer_index': int(observer_index),
            'genotype': list(genotype),
            'metameric_axis': axis,
            'metamers': [{
                'pair_index': 0,
                'rgbo_1': inside_disp.tolist(),
                'rgbo_2': outside_disp.tolist(),
                'cone_1': metadata['inside_cone'],
                'cone_2': metadata['outside_cone'],
            }],
        }],
    }
    with open(metamer_config_path, 'w') as f:
        json.dump(metamer_config, f, indent=2)

    print(f"Observer/genotype: {result['genotype']}")
    print(f"Metameric axis: {axis}")
    print(f"RGB path: {trial['rgb_path']}")
    print(f"OCV path: {trial['ocv_path']}")
    print(f"Max non-axis cone-contrast norm: {non_axis_norm:.6g}")
    print(f"PNG foreground line alpha: {alpha:.6g}")
    print(f"PNG foreground line error: {line_error:.6g}")
    print(f"Report: {report_path}")
    print(f"Metamer config for verify_display_null_consistency.py: {metamer_config_path}")
    print("\nPASS" if passed else "\nFAIL")

    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(
        description='Generate a Gaussian blob and verify delivered PNG display values.')
    parser.add_argument('--primaries', required=True, help='Display primaries directory')
    parser.add_argument('--output-dir', default='/tmp/tetrium_blob_verify',
                        help='Directory for generated PNGs/report')
    parser.add_argument('--observer-index', type=int, default=1,
                        help='Population-sorted observer index to test (default: 1)')
    parser.add_argument('--metameric-axis', type=int, default=2,
                        help='Requested metameric axis (default: 2)')
    parser.add_argument('--size', type=int, default=256,
                        help='Generated blob image size in pixels (default: 256)')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='Cone-contrast non-axis tolerance (default: 1e-8)')
    parser.add_argument('--png-tolerance', type=float, default=1.5 / 255.0,
                        help='PNG display-line tolerance in normalized units')
    args = parser.parse_args()

    raise SystemExit(verify_generated_blob_output(
        primaries_path=args.primaries,
        output_dir=args.output_dir,
        observer_index=args.observer_index,
        metameric_axis=args.metameric_axis,
        size=args.size,
        tolerance=args.tolerance,
        png_tolerance=args.png_tolerance,
    ))


if __name__ == '__main__':
    main()
