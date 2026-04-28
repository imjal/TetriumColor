#!/usr/bin/env python3
"""End-to-end simulation: AEPsych threshold contour on the null-direction PCA patch.

Usage
-----
    python aepsych_contour_simulation.py --primaries-dir <path> [options]

Runs N_TRIALS adaptive trials using AEPsychThresholdContourGenerator.  The
generator parameterizes directions by the same 2D PCA patch coordinates used in
null_direction_viewer.py: (u, v) in the observer-0 tangent patch, plus amplitude r.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure TetriumColor is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from TetriumColor.TetraColorPicker import (
    AEPsychThresholdContourGenerator,
    GaussianObserverSimulator,
)
from TetriumColor.Utils.CustomTypes import ColorTestResult


def main():
    parser = argparse.ArgumentParser(description='AEPsych threshold contour simulation')
    parser.add_argument('--primaries-dir', type=str,
                        default='measurements/2026-04-16/primaries/',
                        help='Path to display primaries CSV directory')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Total number of adaptive trials')
    parser.add_argument('--n-sobol', type=int, default=20,
                        help='Number of initial Sobol (random) trials')
    parser.add_argument('--n-cmf', type=int, default=100,
                        help='Number of CMF samples for bounds estimation')
    parser.add_argument('--null-u', type=float, default=0.0,
                        help='True null direction u coordinate in PCA patch')
    parser.add_argument('--null-v', type=float, default=0.0,
                        help='True null direction v coordinate in PCA patch')
    parser.add_argument('--null-r', type=float, default=0.08,
                        help='True null marker radius for plot overlay')
    parser.add_argument('--sigma', type=float, default=0.010,
                        help='Observer Gaussian width sigma')
    parser.add_argument('--max-radius', type=float, default=0.65,
                        help='Requested max test radius; clamped to display-safe radius')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='threshold_contour.png',
                        help='Output figure path')
    parser.add_argument('--contour-output', type=str,
                        default='aepsych_threshold_contour.npz',
                        help='Output NPZ path for null_direction_viewer.py')
    parser.add_argument('--model-output', type=str, default=None,
                        help='Optional pickle path for the trained AEPsych generator')
    parser.add_argument('--surface-na', type=int, default=31,
                        help='Number of exported contour samples along model a')
    parser.add_argument('--surface-nb', type=int, default=31,
                        help='Number of exported contour samples along model b')
    args = parser.parse_args()

    from TetriumColor.Measurement import load_primaries_from_csv
    primaries = load_primaries_from_csv(args.primaries_dir, extract_zero=False)
    kwargs = {'display_primaries': primaries}

    print(f"Building AEPsychThresholdContourGenerator "
          f"(n_trials={args.n_trials}, n_sobol={args.n_sobol})")
    generator = AEPsychThresholdContourGenerator(
        center_genotype=None,
        peak_to_test=547,
        n_trials=args.n_trials,
        sex='both',
        luminance=0.5,
        seed=args.seed,
        n_cmf_samples=args.n_cmf,
        threshold_level=0.75,
        n_sobol=args.n_sobol,
        max_radius=args.max_radius,
        **kwargs,
    )

    true_null_direction = generator.direction_from_patch(args.null_u, args.null_v)
    observer = GaussianObserverSimulator(
        null_direction=true_null_direction,
        null_r=args.null_r,
        sigma=args.sigma,
        p_chance=0.25,
        seed=args.seed + 1,
    )

    print(f"True null patch coordinate: u={args.null_u:.4f}, "
          f"v={args.null_v:.4f}, plot_r={args.null_r:.4f}")
    print(f"AEPsych model bounds: a={generator._lb_np[0]:.4f}..{generator._ub_np[0]:.4f}, "
          f"b={generator._lb_np[1]:.4f}..{generator._ub_np[1]:.4f}, "
          f"r={generator._lb_np[2]:.4f}..{generator._ub_np[2]:.4f}")
    print(f"Requested max radius: {args.max_radius:.4f}; "
          f"actual display-safe max radius: {generator.actual_max_radius:.4f}")
    print(f"Patch bounds: u={generator.patch_bounds[0, 0]:.4f}..{generator.patch_bounds[0, 1]:.4f}, "
          f"v={generator.patch_bounds[1, 0]:.4f}..{generator.patch_bounds[1, 1]:.4f}")

    # Run trial loop
    colors = generator.NewColor()
    trial = 0
    while colors is not None:
        _, _, _, r = colors
        resp = observer.simulate_direction_response(generator.last_direction, r)
        result = ColorTestResult.Success if resp == 1 else ColorTestResult.Failure
        colors = generator.GetColor(result)
        trial += 1
        if trial % 25 == 0:
            print(f"  Trial {trial}/{args.n_trials}")

    print(f"Trials complete ({trial} total).")
    if generator.stimulus_disp_log:
        disp_log = np.array(generator.stimulus_disp_log)
        print("Stimulus DISP verification:")
        print(f"  count={len(disp_log)}")
        print(f"  min={np.array2string(disp_log.min(axis=0), precision=5)}")
        print(f"  max={np.array2string(disp_log.max(axis=0), precision=5)}")
        print(f"  all_in_gamut={bool(np.all((disp_log >= 0) & (disp_log <= 1)))}")

    # Fit ellipsoid
    print("Fitting threshold ellipsoid...")
    center, axes, semi_lengths, residuals = generator.fit_threshold_ellipsoid()
    if center is not None:
        print(f"  Ellipsoid center (est. null pt): {center.round(4)}")
        print(f"  Semi-lengths (chromatic): {semi_lengths.round(4)}")
        print(f"  Mean Mahalanobis residual: {np.mean(residuals):.4f}")
    else:
        print("  Insufficient valid surface points for ellipsoid fit.")

    # Export real display-space contour for scripts/simulation/null_direction_viewer.py.
    print(f"Exporting threshold contour to {args.contour_output}")
    contour_data = generator.export_threshold_patch_npz(
        args.contour_output,
        n_a=args.surface_na,
        n_b=args.surface_nb,
    )
    contour_disp = contour_data["disp_points"]
    print("Exported contour DISP verification:")
    print(f"  count={len(contour_disp)}")
    print(f"  grid_shape={tuple(contour_data['grid_shape'].tolist())}")
    print(f"  min={np.array2string(contour_disp.min(axis=0), precision=5)}")
    print(f"  max={np.array2string(contour_disp.max(axis=0), precision=5)}")
    print(f"  all_in_gamut={bool(np.all((contour_disp >= 0) & (contour_disp <= 1)))}")
    print(f"  clipped_count={int(np.sum(contour_data['gamut_clipped']))}")
    print(f"  threshold_found_count={int(np.sum(contour_data['threshold_found']))}")
    print(f"  r_star_range=[{contour_data['r_star'].min():.5f}, "
          f"{contour_data['r_star'].max():.5f}]")
    print(f"  posterior_p_hi_range=[{contour_data['posterior_p_hi'].min():.5f}, "
          f"{contour_data['posterior_p_hi'].max():.5f}]")

    if args.model_output:
        print(f"Saving trained AEPsych model to {args.model_output}")
        generator.save_model_state(args.model_output)

    # Plot
    print(f"Saving plot to {args.output}")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    generator.plot_threshold_contour(ax=ax)

    # Overlay true null point in observer-0 chromatic slice coordinates.
    true_alpha = args.null_r * true_null_direction
    ax.scatter(true_alpha[0], true_alpha[1], true_alpha[2],
               c='lime', s=200, marker='D', zorder=6, label='true null pt')
    ax.legend(fontsize=8)

    plt.savefig(args.output, dpi=150)
    print(f"Done.  Figure saved to {args.output}")


if __name__ == '__main__':
    main()
