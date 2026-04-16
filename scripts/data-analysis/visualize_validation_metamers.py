#!/usr/bin/env python3
"""
Visualize validation metamer pairs in polyscope.

Given a metamers.json config (as produced by generate_display_validation_metamers.py),
renders all metamer pairs as point clouds connected by lines in HERING_BGYR space.

Each observer gets a distinct color; M1 and M2 of each pair are shown as solid and
hollow spheres connected by a line.

Usage:
    python visualize_validation_metamers.py \
        --metamers config/display_validation_metamers.json \
        --primaries-dir measurements/2026-03-06/primaries/
"""

import argparse
import json

import numpy as np
import tetrapolyscope as ps
import tetrapolyscope.imgui as psim

from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Visualization.PolyscopeUtils import (
    RenderPointCloud,
    Render3DLine,
    RenderBGYRGamut,
)


DISPLAY_BASIS = PolyscopeDisplayType.HERING_BGYR

OBSERVER_COLORS = [
    np.array([1.0, 0.35, 0.35]),  # red
    np.array([0.35, 0.80, 0.35]),  # green
    np.array([0.35, 0.55, 1.0]),   # blue
    np.array([1.0,  0.80, 0.2]),   # yellow
    np.array([1.0,  0.35, 1.0]),   # magenta
    np.array([0.3,  1.0,  1.0]),   # cyan
    np.array([1.0,  0.55, 0.0]),   # orange
    np.array([0.65, 0.3,  1.0]),   # purple
]


def build_scene(config, primaries, observer_genotypes, metameric_axis):
    """Convert all metamer pairs to display-basis points and render them."""
    # Reference gamut from the first observer
    first_genotype = tuple(sorted(config['observers'][0]['genotype']))
    first_obs = observer_genotypes.get_observer_for_peaks(first_genotype)
    first_cst = ColorSpace(first_obs, display_primaries=primaries,
                           metameric_axis=metameric_axis)
    RenderBGYRGamut('reference_gamut', first_cst, DISPLAY_BASIS, alpha=0.15)

    for enum_idx, obs_data in enumerate(config['observers']):
        genotype = tuple(sorted(obs_data['genotype']))
        obs_index = obs_data['observer_index']
        observer = observer_genotypes.get_observer_for_peaks(genotype)
        cst = ColorSpace(observer, display_primaries=primaries,
                         metameric_axis=metameric_axis)

        color = OBSERVER_COLORS[enum_idx % len(OBSERVER_COLORS)]
        color_m2 = color * 0.55   # dimmer shade for M2 endpoints

        pts_m1, pts_m2 = [], []
        for metamer in obs_data['metamers']:
            c1 = np.array(metamer['cone_1'])
            c2 = np.array(metamer['cone_2'])
            d1 = cst.convert_to_polyscope(
                c1.reshape(1, -1), ColorSpaceType.CONE, DISPLAY_BASIS)[0]
            d2 = cst.convert_to_polyscope(
                c2.reshape(1, -1), ColorSpaceType.CONE, DISPLAY_BASIS)[0]
            pts_m1.append(d1)
            pts_m2.append(d2)
            Render3DLine(
                f"pair_line_obs{obs_index}_pair{metamer['pair_index']}",
                np.array([d1, d2]),
                color,
                radius=0.003,
            )

        pts_m1 = np.array(pts_m1)
        pts_m2 = np.array(pts_m2)
        n = len(pts_m1)

        RenderPointCloud(f"m1_obs{obs_index}", pts_m1,
                         np.tile(color, (n, 1)), radius=0.016)
        RenderPointCloud(f"m2_obs{obs_index}", pts_m2,
                         np.tile(color_m2, (n, 1)), radius=0.016)

        print(f"  Observer {obs_index} {genotype}: rendered {n} pairs "
              f"(color {np.round(color, 2)})")


def make_callback(config):
    """Return a polyscope callback that draws a legend panel."""
    window_open = [True]

    def callback():
        opened, window_open[0] = psim.Begin(
            "Validation Metamers", window_open[0])
        if opened:
            psim.Text(f"Total pairs: {config['metadata']['total_metamer_pairs']}")
            psim.Text(f"Metameric axis: {config['metadata'].get('metameric_axis', 2)}")
            psim.Separator()
            psim.Text("Observer legend  (M1 = bright, M2 = dim)")
            psim.Separator()
            for enum_idx, obs_data in enumerate(config['observers']):
                obs_index = obs_data['observer_index']
                genotype = tuple(sorted(obs_data['genotype']))
                n_pairs = len(obs_data['metamers'])
                color = OBSERVER_COLORS[enum_idx % len(OBSERVER_COLORS)]
                psim.TextColored(
                    tuple(color.tolist() + [1.0]),
                    f"Obs {obs_index}  {genotype}  ({n_pairs} pairs)",
                )
        psim.End()

    return callback


def main():
    parser = argparse.ArgumentParser(
        description='Visualize validation metamer pairs in polyscope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--metamers', required=True,
                        help='Path to metamers JSON config')
    parser.add_argument('--primaries-dir', required=True,
                        help='Directory containing display primary CSVs (BGOR order)')
    args = parser.parse_args()

    # --- Load config ---
    print(f"Loading metamers config: {args.metamers}")
    with open(args.metamers) as f:
        config = json.load(f)
    print(f"  Observers: {len(config['observers'])}")
    print(f"  Total metamer pairs: {config['metadata']['total_metamer_pairs']}")

    # --- Load primaries ---
    print(f"Loading primaries: {args.primaries_dir}")
    primaries = load_primaries_from_csv(
        args.primaries_dir, extract_zero=False, primary_order='BGOR')
    wavelengths = primaries[0].wavelengths
    metameric_axis = config['metadata'].get('metameric_axis', 2)

    # --- Observers ---
    observer_genotypes = ObserverGenotypes(
        wavelengths=wavelengths,
        dimensions=[3],
        seed=config['metadata'].get('seed', 42),
    )

    # --- Polyscope ---
    ps.init()
    ps.set_transparency_render_passes(24)
    ps.set_transparency_peel_epsilon(1e-7)
    ps.set_ground_plane_mode('none')
    ps.set_user_callback(make_callback(config))

    print("\nBuilding scene...")
    build_scene(config, primaries, observer_genotypes, metameric_axis)

    print("Showing polyscope window.")
    ps.show()


if __name__ == '__main__':
    main()
