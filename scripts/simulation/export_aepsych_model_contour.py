#!/usr/bin/env python3
"""Export a threshold contour from a saved AEPsych model state."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from TetriumColor.TetraColorPicker import AEPsychThresholdContourGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Re-slice a saved AEPsych posterior at a chosen threshold.")
    parser.add_argument("model", help="Pickle saved by aepsych_contour_simulation.py")
    parser.add_argument("--threshold-level", type=float, default=0.75,
                        help="Posterior P(detect) contour to export")
    parser.add_argument("--output", required=True,
                        help="Output NPZ loadable by null_direction_viewer.py")
    parser.add_argument("--surface-na", type=int, default=41,
                        help="Number of exported contour samples along model a")
    parser.add_argument("--surface-nb", type=int, default=41,
                        help="Number of exported contour samples along model b")
    args = parser.parse_args()

    generator = AEPsychThresholdContourGenerator.load_model_state(args.model)
    data = generator.export_threshold_patch_npz(
        args.output,
        n_a=args.surface_na,
        n_b=args.surface_nb,
        threshold_level=args.threshold_level,
    )

    r_star = data["r_star"]
    found = data["threshold_found"]
    clipped = data["gamut_clipped"]
    disp = data["disp_points"]

    print(f"Loaded model {args.model}")
    print(f"Exported {args.output}")
    print(f"  threshold_level={args.threshold_level:.3f}")
    print(f"  grid_shape={tuple(data['grid_shape'].tolist())}")
    print(f"  r_star_range=[{r_star.min():.5f}, {r_star.max():.5f}]")
    print(f"  threshold_found_count={int(np.sum(found))}/{len(found)}")
    print(f"  clipped_count={int(np.sum(clipped))}")
    print(f"  all_disp_in_gamut={bool(np.all((disp >= 0) & (disp <= 1)))}")


if __name__ == "__main__":
    main()
