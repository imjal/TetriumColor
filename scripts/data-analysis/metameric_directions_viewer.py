#!/usr/bin/env python3
"""
Visualization of Q-cone (547nm) metameric directions for trichromatic observers.

Uses tetrapolyscope to plot:
1. Top 10 trichromatic observers (by probability)
2. Add a 547nm cone to each, creating a 3-cone system
3. Find maximal metameric pairs along the Q (547nm) axis
4. Plot these directions in HERING_BGYR opponent color space
5. Show the gamut boundary at the selected luminance
"""

import numpy as np
from typing import List, Tuple
import tetrapolyscope as ps
import tetrapolyscope.imgui as psim

from TetriumColor.Observer import Observer, GetHeringMatrix
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Visualization.PolyscopeUtils import (
    RenderPointCloud,
    Render3DLine,
    RenderGamutSlices,
    RenderBGYRGamut,
)


class MetamericDirectionsViewer:
    def __init__(
        self,
        primaries_dir: str,
        num_observers: int = 10,
        luminance: float = 0.5,
        display_basis: PolyscopeDisplayType = PolyscopeDisplayType.HERING_BGYR,
    ):
        """Initialize the Q-cone metameric directions viewer.

        Takes trichromatic observers, adds a 547nm Q-cone to each, and visualizes
        the maximal metameric pairs along the Q-cone axis.

        Args:
            primaries_dir: Directory containing display primary CSV files
            num_observers: Number of top trichromatic observers to display (will add 547nm Q)
            luminance: Luminance level for visualization
            display_basis: Display basis for visualization
        """
        self.num_observers = num_observers
        self.luminance = luminance
        self.display_basis = display_basis
        self.point_size = 0.02  # Adjustable point radius
        self.window_open = True

        # Store references to point clouds so we can update their properties
        self.point_clouds = {}  # name -> polyscope point cloud object

        # Load display primaries
        print(f"Loading primaries from {primaries_dir}...")
        self.primaries = load_primaries_from_csv(primaries_dir, extract_zero=False)
        print(f"Loaded {len(self.primaries)} primaries")

        # Initialize observer genotypes (trichromats only)
        observer_wavelengths = np.arange(380, 781, 5)
        self.observer_genotypes = ObserverGenotypes(
            wavelengths=observer_wavelengths,
            dimensions=[3],  # Trichromats only (3 cones: S, M, L)
            seed=42
        )

        # Get top N trichromatic genotypes
        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=0.999, sex='both'
        )[:num_observers]

        print(f"Selected {len(self.genotypes)} trichromatic genotypes:")
        for i, g in enumerate(self.genotypes):
            print(f"  {i+1}. {g}")

        # Create observers and color spaces
        self._create_observers()

        print(f"Created {len(self.observers)} observers")

        # Initialize polyscope
        ps.init()
        ps.set_transparency_render_passes(24)
        ps.set_transparency_peel_epsilon(1e-7)
        ps.set_ground_plane_mode("none")
        ps.set_always_redraw(True)

        # Register callback
        ps.set_user_callback(self.callback)

        # Initial render
        self.render_visualization()

    def _create_observers(self):
        """Create observers and color spaces.

        Adds a 547nm cone to each trichromat, creating a 3-cone system where
        the third cone is the Q-like 547nm cone.
        """
        self.observers = []
        self.color_spaces = []
        self.q_axes = []  # Track the axis index for the 547nm Q-cone for each observer

        for genotype in self.genotypes:
            # Add 547nm cone to the trichromatic genotype
            genotype_with_q = tuple(sorted(genotype + (547,)))
            print(f"Creating observer for genotype: {genotype} -> {genotype_with_q} (with 547nm)")

            obs = self.observer_genotypes.get_observer_for_peaks(genotype_with_q, degree=4.0)

            # Find which axis corresponds to the 547nm cone
            q_axis = None
            for axis_idx, cone in enumerate(obs.sensors):
                if abs(cone.peak - 547.0) < 1.0:  # Allow 1nm tolerance
                    q_axis = axis_idx
                    break

            if q_axis is None:
                print(f"  WARNING: Could not find 547nm cone! Sensor peaks: {[s.peak for s in obs.sensors]}")
                # Fall back to axis 2
                q_axis = 2

            print(f"  Q-cone (547nm) is at axis {q_axis}")
            print(f"  All sensor peaks: {[s.peak for s in obs.sensors]}")

            self.observers.append(obs)
            self.q_axes.append(q_axis)
            cst = ColorSpace(obs, self.primaries)
            self.color_spaces.append(cst)

    def render_visualization(self):
        """Render the metameric directions and gamut."""
        print(f"\n=== Rendering visualization (point_size={self.point_size}) ===")

        # Clear previous renders and point cloud references
        self._clear_previous_renders()
        self.point_clouds = {}  # Reset point cloud references

        if len(self.color_spaces) == 0:
            print("No color spaces available")
            return

        cst = self.color_spaces[0]

        # Render reference gamut (full gamut for reference)
        RenderBGYRGamut(
            "reference_gamut",
            cst,
            self.display_basis,
            alpha=0.15,
        )
        print("Rendered reference gamut")

        # Render gamut slice at the selected luminance
        slice_name_base = "gamut_slice"
        RenderGamutSlices(
            slice_name_base,
            cst,
            display_space=ColorSpaceType.DISP,
            display_basis=self.display_basis,
            luminance_values=[self.luminance],
            grid_resolution=25,
            tolerance=0.03,
            alpha=0.35,
        )
        print(f"Rendered gamut slice at L={self.luminance:.2f}")

        # Background point in DISP space
        background = np.ones(4) * 0.5

        # Colors for different observers
        observer_colors = [
            np.array([1.0, 0.3, 0.3]),  # Red
            np.array([0.3, 1.0, 0.3]),  # Green
            np.array([0.3, 0.3, 1.0]),  # Blue
            np.array([1.0, 1.0, 0.3]),  # Yellow
            np.array([1.0, 0.3, 1.0]),  # Magenta
            np.array([0.3, 1.0, 1.0]),  # Cyan
            np.array([1.0, 0.6, 0.0]),  # Orange
            np.array([0.6, 0.3, 1.0]),  # Purple
            np.array([0.3, 0.8, 0.8]),  # Teal
            np.array([0.8, 0.8, 0.3]),  # Olive
        ]

        # Convert background to cone space once for all observers
        background_cone = cst.convert(
            background.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE
        )[0]
        background_disp_viz = cst.convert_to_polyscope(
            background_cone.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
        )[0]

        # Render background point (white)
        RenderPointCloud(
            "background_point",
            background_disp_viz.reshape(1, -1),
            np.array([[1.0, 1.0, 1.0]]),
            radius=self.point_size,
        )
        # Store reference for radius updates
        try:
            self.point_clouds["background_point"] = ps.get_point_cloud("background_point")
        except:
            pass

        # For each observer, find and plot maximal metameric pair in Q (547nm) direction
        for i, (obs, cst_i, q_axis) in enumerate(zip(self.observers, self.color_spaces, self.q_axes)):
            print(f"\nProcessing observer {i+1}/{len(self.observers)}...")
            color = observer_colors[i % len(observer_colors)]

            # Get observer info
            peaks = tuple([s.peak for s in obs.sensors])
            print(f"  Peaks: {peaks}")
            print(f"  Using Q-cone axis: {q_axis}")

            try:
                # Find maximal metamer pair along Q (547nm) axis
                print(f"  Finding maximal metamer in Q (547nm) direction...")

                # Find maximal metamer pair from background in DISP space
                result = cst_i.get_maximal_pair_in_disp_from_pt(
                    pt=background,
                    metameric_axis=q_axis,
                    input_space=ColorSpaceType.DISP,
                    output_space=ColorSpaceType.CONE,
                    proportion=1.0,
                )

                if result is None:
                    print(f"  Could not find metameric pair for Q axis")
                    continue

                cone1, cone2, metamer_diff = result
                print(f"  Found metamer pair, diff: {metamer_diff:.6f}")

                # Convert to display space (3D for HERING_BGYR visualization)
                disp1 = cst_i.convert_to_polyscope(
                    cone1.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]
                disp2 = cst_i.convert_to_polyscope(
                    cone2.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]

                print(f"  Point 1: {disp1}")
                print(f"  Point 2: {disp2}")

                # Render line between metameric pair
                line_name = f"metamer_line_{i:02d}"
                Render3DLine(
                    line_name,
                    np.array([disp1, disp2]),
                    color,
                    radius=0.006,
                )

                # Render metamer endpoints
                RenderPointCloud(
                    f"metamer_{i:02d}_p1",
                    disp1.reshape(1, -1),
                    color.reshape(1, -1),
                    radius=self.point_size,
                )
                try:
                    self.point_clouds[f"metamer_{i:02d}_p1"] = ps.get_point_cloud(f"metamer_{i:02d}_p1")
                except:
                    pass

                RenderPointCloud(
                    f"metamer_{i:02d}_p2",
                    disp2.reshape(1, -1),
                    color.reshape(1, -1),
                    radius=self.point_size,
                )
                try:
                    self.point_clouds[f"metamer_{i:02d}_p2"] = ps.get_point_cloud(f"metamer_{i:02d}_p2")
                except:
                    pass

                # For the first observer, render display point variations (±1 in 8-bit space)
                if i == 0:
                    self._render_display_variations(cone2, cst_i, observer_colors[0])

            except Exception as e:
                print(f"Error processing observer {i+1}: {e}")
                import traceback
                traceback.print_exc()

        print("=== Finished rendering ===\n")

    def _render_display_variations(self, cone_point, cst, base_color):
        """Render ±1 display variations in 8-bit space.

        Takes a point in CONE space, converts to DISP, generates ±1 variations
        in 8-bit [0, 255] space, and renders them as a point cloud in visualization space.

        Args:
            cone_point: Point in CONE space
            cst: ColorSpace for conversions
            base_color: Base color for the point cloud
        """
        print("\nRendering display point variations (±1 in 8-bit)...")

        try:
            # Convert cone point to DISP space (normalized [0, 1])
            disp_point_norm = cst.convert(
                cone_point.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP
            )[0]

            print(f"  Original DISP point (normalized): {disp_point_norm}")

            # Convert to 8-bit [0, 255] space
            disp_point_8bit = np.round(disp_point_norm * 255.0).astype(np.int32)
            print(f"  Original DISP point (8-bit): {disp_point_8bit}")

            # Generate ±1 variations for each channel
            variation_points_8bit = [disp_point_8bit.copy()]  # Include original
            variation_labels = ["original"]

            for channel in range(4):  # RGBO
                # +1 variation
                var_point = disp_point_8bit.copy()
                var_point[channel] = np.clip(var_point[channel] + 1, 0, 255)
                variation_points_8bit.append(var_point)
                channel_names = ["R", "G", "B", "O"]
                variation_labels.append(f"+1 {channel_names[channel]}")

                # -1 variation
                var_point = disp_point_8bit.copy()
                var_point[channel] = np.clip(var_point[channel] - 1, 0, 255)
                variation_points_8bit.append(var_point)
                variation_labels.append(f"-1 {channel_names[channel]}")

            # Convert back to normalized [0, 1]
            variation_points_norm = np.array(variation_points_8bit) / 255.0

            print(f"  Generated {len(variation_points_norm)} variation points:")
            for label, point in zip(variation_labels, variation_points_norm):
                print(f"    {label}: {point}")

            # Convert to cone space
            variation_points_cone = cst.convert(
                variation_points_norm, ColorSpaceType.DISP, ColorSpaceType.CONE
            )

            # Convert to visualization space (HERING_BGYR)
            variation_points_viz = cst.convert_to_polyscope(
                variation_points_cone, ColorSpaceType.CONE, self.display_basis
            )

            # Create colors: original point in white, variations in dimmer version of base color
            colors = np.zeros((len(variation_points_viz), 3))
            colors[0] = np.array([1.0, 1.0, 1.0])  # Original in white
            colors[1:] = base_color * 0.6  # Variations in dimmed base color

            # Render as point cloud
            RenderPointCloud(
                "display_variations",
                variation_points_viz,
                colors,
                radius=self.point_size,
            )
            # Store reference for radius updates
            try:
                self.point_clouds["display_variations"] = ps.get_point_cloud("display_variations")
            except:
                pass

            print(f"  Rendered {len(variation_points_viz)} display variation points")

        except Exception as e:
            print(f"Error rendering display variations: {e}")
            import traceback
            traceback.print_exc()

    def _clear_previous_renders(self):
        """Clear previous visualization elements."""
        print("Clearing previous renders...")
        # Remove metamer lines and points
        max_check = max(self.num_observers, 15)
        for i in range(max_check):
            try:
                ps.remove_curve_network(f"metamer_line_{i:02d}")
                print(f"  Removed metamer_line_{i:02d}")
            except (RuntimeError, KeyError):
                pass

            for suffix in ["p1", "p2"]:
                try:
                    ps.remove_point_cloud(f"metamer_{i:02d}_{suffix}")
                    print(f"  Removed metamer_{i:02d}_{suffix}")
                except (RuntimeError, KeyError):
                    pass

        # Remove background point
        print(f"  Attempting to remove background_point...", flush=True)
        try:
            ps.remove_point_cloud("background_point")
            print(f"  Removed background_point", flush=True)
        except Exception as e:
            print(f"  Could not remove background_point: {e}", flush=True)

        # Remove display variations
        print(f"  Attempting to remove display_variations...", flush=True)
        try:
            ps.remove_point_cloud("display_variations")
            print(f"  Removed display_variations", flush=True)
        except Exception as e:
            print(f"  Could not remove display_variations: {e}", flush=True)

        # Remove gamut slices
        print(f"  Attempting to remove gamut slices...", flush=True)
        for lum_int in range(0, 41):
            lum = lum_int / 20.0
            slice_name = f"gamut_slice_slice_L{lum:.2f}"
            try:
                ps.remove_surface_mesh(slice_name)
            except Exception:
                pass
            try:
                ps.remove_point_cloud(slice_name)
            except Exception:
                pass
        print(f"  Done removing gamut slices", flush=True)

        # Remove reference gamut
        print(f"  Attempting to remove reference_gamut...", flush=True)
        try:
            ps.remove_surface_mesh("reference_gamut")
            print(f"  Removed reference_gamut", flush=True)
        except Exception as e:
            print(f"  Could not remove reference_gamut: {e}", flush=True)

    def callback(self):
        """Polyscope callback for GUI."""

        opened, self.window_open = psim.Begin("Q-Cone Metameric Directions", self.window_open)

        if opened:
            psim.Text(f"Top {len(self.observers)} Trichromats + 547nm Q-cone")
            psim.Text("Maximal metamer directions in HERING_BGYR space")

            changed, self.luminance = psim.SliderFloat(
                "Luminance (L)", self.luminance, 0.0, 2.0
            )
            if changed:
                self.render_visualization()

            changed_size, new_point_size = psim.SliderFloat(
                "Point Size", self.point_size, 0.001, 0.1
            )

            # Check if value actually changed (with larger tolerance)
            if abs(new_point_size - self.point_size) > 1e-5:
                self.point_size = new_point_size
                print(f"Point size changed to {self.point_size:.6f}", flush=True)

                # Update all point cloud radii
                for name, pc in self.point_clouds.items():
                    pc.set_radius(self.point_size)
                print(f"Updated {len(self.point_clouds)} point clouds", flush=True)
            else:
                self.point_size = new_point_size

            psim.Text(f"\nObserver count: {len(self.observers)}")
            psim.Text("Testing Q-cone (547nm) metameric axis")
            psim.Text("White sphere: background point (0.5, 0.5, 0.5, 0.5)")
            psim.Text("Colored lines: maximal metamer pairs for Q direction")

        psim.End()

    def show(self):
        """Show the polyscope window."""
        ps.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Q-cone (547nm) metameric directions for trichromatic observers with added Q-cone"
    )
    parser.add_argument(
        "--primaries_dir",
        type=str,
        default="../../measurements/2026-03-03/primaries/",
        help="Directory containing display primary CSV files",
    )
    parser.add_argument(
        "--num_observers",
        type=int,
        default=10,
        help="Number of trichromatic observers to use (will add 547nm Q-cone to each)",
    )
    parser.add_argument(
        "--initial_luminance",
        type=float,
        default=0.5,
        help="Initial luminance value",
    )

    args = parser.parse_args()

    app = MetamericDirectionsViewer(
        primaries_dir=args.primaries_dir,
        num_observers=args.num_observers,
        luminance=args.initial_luminance,
    )

    app.show()


if __name__ == "__main__":
    main()
