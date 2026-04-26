#!/usr/bin/env python3
"""
Visualization of metameric match point clouds under CMF variation.

Uses Monte Carlo sampling of physiological parameters (od_lm, od_s, macular, lens)
combined with population-weighted genotype sampling to generate a cloud of trichromatic
observers. For each observer, finds the maximal metameric pair along the 547nm Q-axis
and renders the match point cloud in HERING_BGYR space.

Visualizes two fitted ellipsoids corresponding to high and low macular pigment levels,
plus the full point cloud colored by macular value.
"""

import numpy as np
from typing import List, Tuple, Optional
import tetrapolyscope as ps
import tetrapolyscope.imgui as psim

from TetriumColor.Observer import Observer
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.CMFSampler import CMFSampler
from TetriumColor.Observer.Observer import Cone
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.ChromaticityAnalysis import EllipsoidFitter
from TetriumColor.Visualization.PolyscopeUtils import (
    RenderPointCloud,
    RenderNoiseBall,
    RenderBGYRGamut,
    Render3DLine,
)


class CMFMatchVisualizer:
    """
    Visualizer for metameric match point clouds under CMF physiological variation.

    Generates sampled observers via Monte Carlo (genotype + od + macular + lens),
    computes maximal metameric pairs along the 547nm Q-axis, and renders the cloud
    with ellipsoids grouped by macular pigment level.
    """

    def __init__(
        self,
        primaries_dir: str,
        n_samples: int = 500,
        macular_mean: float = 1.0,
        macular_std: float = 0.25,
        od_lm_mean: float = 0.5,
        od_lm_std: float = 0.05,
        od_s_mean: float = 0.4,
        od_s_std: float = 0.04,
        lens_mean: float = 1.0,
        lens_std: float = 0.1,
        display_basis: PolyscopeDisplayType = PolyscopeDisplayType.HERING_BGYR,
        fit_ellipsoids: bool = True,
        point_size: float = 0.02,
        top_n_genotypes: Optional[int] = None,
        num_nominal_observers: int = 10,
    ):
        """
        Initialize the CMF sampler visualizer.

        Args:
            primaries_dir: Directory containing display primary CSV files
            n_samples: Number of Monte Carlo samples
            macular_mean, macular_std: Macular pigment density distribution
            od_lm_mean, od_lm_std: L/M OD distribution
            od_s_mean, od_s_std: S OD distribution
            lens_mean, lens_std: Lens density distribution
            display_basis: Display basis for visualization (default HERING_BGYR)
            fit_ellipsoids: If True, fit and render ellipsoids for high/low macular groups.
                          If False, just render the point cloud (default True)
            point_size: Radius of rendered points (default 0.02)
            top_n_genotypes: If set, only sample from top N most probable genotypes (default None)
            num_nominal_observers: Number of top nominal genotypes to show as reference (default 10)
        """
        self.n_samples = n_samples
        self.macular_mean = macular_mean
        self.macular_std = macular_std
        self.od_lm_mean = od_lm_mean
        self.od_lm_std = od_lm_std
        self.od_s_mean = od_s_mean
        self.od_s_std = od_s_std
        self.lens_mean = lens_mean
        self.lens_std = lens_std
        self.display_basis = display_basis
        self.fit_ellipsoids = fit_ellipsoids
        self.point_size = point_size
        self.top_n_genotypes = top_n_genotypes
        self.num_nominal_observers = num_nominal_observers
        self.window_open = True
        self.point_clouds = {}

        # Load primaries
        print(f"Loading primaries from {primaries_dir}...")
        self.primaries = load_primaries_from_csv(primaries_dir, extract_zero=False)
        print(f"Loaded {len(self.primaries)} primaries")

        # Initialize observer genotypes (trichromats only)
        observer_wavelengths = np.arange(380, 781, 5)
        self.observer_genotypes = ObserverGenotypes(
            wavelengths=observer_wavelengths,
            dimensions=[3],
            seed=42
        )

        # Get reference observer (most common genotype) for visualization basis
        most_common = self.observer_genotypes.get_most_common_genotype('both')
        print(f"Reference genotype (most common): {most_common}")
        self.ref_obs = self.observer_genotypes.get_observer_for_peaks(
            tuple(sorted(most_common + (547,))), degree=4.0
        )
        self.ref_cst = ColorSpace(self.ref_obs, self.primaries)

        # Create sampler and generate samples
        print(f"Generating {n_samples} CMF samples...")
        self.sampler = CMFSampler(
            self.observer_genotypes,
            od_lm_mean=od_lm_mean,
            od_lm_std=od_lm_std,
            od_s_mean=od_s_mean,
            od_s_std=od_s_std,
            macular_mean=macular_mean,
            macular_std=macular_std,
            lens_mean=lens_mean,
            lens_std=lens_std,
            wavelengths=observer_wavelengths,
            seed=42,
            top_n_genotypes=top_n_genotypes,
        )

        self.observers_and_params = self.sampler.sample(n_samples, sex='both')
        print(f"Generated {len(self.observers_and_params)} samples")

        # Compute match points for sampled observers
        self._compute_match_points()

        # Compute match points for top 10 nominal observers
        self._compute_nominal_match_points()

        # Initialize polyscope
        ps.init()
        ps.set_transparency_render_passes(24)
        ps.set_transparency_peel_epsilon(1e-7)
        ps.set_ground_plane_mode("none")
        ps.set_always_redraw(True)
        ps.set_user_callback(self.callback)

        # Initial render
        self.render_visualization()

    def _compute_match_points(self):
        """Compute maximal metameric pairs for all sampled observers."""
        self.match_points = []
        self.macular_values = []
        self.genotypes = []

        background = np.ones(4) * 0.5

        for i, (obs, params) in enumerate(self.observers_and_params):
            if i % 100 == 0:
                print(f"  Computing match point {i+1}/{len(self.observers_and_params)}...")

            # Add 547nm Q-cone to sampled observer
            genotype = params['genotype']
            genotype_with_q = tuple(sorted(genotype + (547,)))

            # Re-create observer with Q-cone at same params
            cst_i = self._create_observer_colorspace(genotype_with_q, params)

            # Find Q-axis index
            q_axis = self._find_q_axis(cst_i.observer)

            try:
                result = cst_i.get_maximal_pair_in_disp_from_pt(
                    pt=background,
                    metameric_axis=q_axis,
                    input_space=ColorSpaceType.DISP,
                    output_space=ColorSpaceType.CONE,
                    proportion=1.0,
                )

                if result is None:
                    continue

                cone1, cone2, metamer_diff = result

                # cone2 is in observer i's cone space; convert through DISP to ref observer's space
                disp_point = cst_i.convert(
                    cone2.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP
                )[0]
                cone_ref = self.ref_cst.convert(
                    disp_point.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE
                )[0]
                pt_viz = self.ref_cst.convert_to_polyscope(
                    cone_ref.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]

                self.match_points.append(pt_viz)
                self.macular_values.append(params['macular'])
                self.genotypes.append(genotype_with_q)

            except Exception as e:
                print(f"    Error on sample {i+1}: {e}")

        self.match_points = np.array(self.match_points)
        self.macular_values = np.array(self.macular_values)
        print(f"Successfully computed {len(self.match_points)} match points")

    def _create_observer_colorspace(
        self, genotype_with_q: Tuple[float, ...], params: dict
    ) -> ColorSpace:
        """Create a ColorSpace for a specific genotype and parameters."""
        # Ensure S-cone is present
        peaks = genotype_with_q if 420 in genotype_with_q else (420,) + genotype_with_q
        peaks = tuple(sorted(peaks))

        cones = []
        for peak in peaks:
            od = params['od_s'] if peak == 420 else params['od_lm']
            cone = Cone.templates['neitz'](
                self.observer_genotypes.wavelengths, peak
            ).with_preceptoral(od=od, macular=params['macular'], lens=params['lens'])
            # Preserve original peak
            cone.peak = int(peak)
            cones.append(cone)

        obs = Observer(cones, illuminant=None)
        return ColorSpace(obs, self.primaries)

    def _compute_nominal_match_points(self):
        """Compute match points for nominal observers with average parameters."""
        self.nominal_match_points = []
        self.nominal_genotypes = []

        # Get top N most probable genotypes
        nominal_genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=0.99, sex='both'
        )[:self.num_nominal_observers]

        background = np.ones(4) * 0.5

        for genotype in nominal_genotypes:
            genotype_with_q = tuple(sorted(genotype + (547,)))

            # Create nominal observer with average parameters
            peaks = genotype_with_q if 420 in genotype_with_q else (420,) + genotype_with_q
            peaks = tuple(sorted(peaks))

            cones = []
            for peak in peaks:
                od = self.od_s_mean if peak == 420 else self.od_lm_mean
                cone = Cone.templates['neitz'](
                    self.observer_genotypes.wavelengths, peak
                ).with_preceptoral(od=od, macular=self.macular_mean, lens=self.lens_mean)
                cone.peak = int(peak)
                cones.append(cone)

            obs = Observer(cones, illuminant=None)
            cst_i = ColorSpace(obs, self.primaries)
            q_axis = self._find_q_axis(obs)

            try:
                result = cst_i.get_maximal_pair_in_disp_from_pt(
                    pt=background,
                    metameric_axis=q_axis,
                    input_space=ColorSpaceType.DISP,
                    output_space=ColorSpaceType.CONE,
                    proportion=1.0,
                )

                if result is None:
                    continue

                cone1, cone2, metamer_diff = result

                # cone2 is in observer i's cone space; convert through DISP to ref observer's space
                disp_point = cst_i.convert(
                    cone2.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP
                )[0]
                cone_ref = self.ref_cst.convert(
                    disp_point.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE
                )[0]
                pt_viz = self.ref_cst.convert_to_polyscope(
                    cone_ref.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]

                self.nominal_match_points.append(pt_viz)
                self.nominal_genotypes.append(genotype_with_q)

            except Exception as e:
                print(f"  Error on nominal genotype {genotype}: {e}")

        self.nominal_match_points = np.array(self.nominal_match_points) if self.nominal_match_points else np.array([])
        print(f"Computed {len(self.nominal_match_points)} nominal match points")

    def _find_q_axis(self, observer: Observer) -> int:
        """Find the axis index of the 547nm cone."""
        for axis_idx, cone in enumerate(observer.sensors):
            if abs(cone.peak - 547.0) < 1.0:
                return axis_idx
        return 2

    def render_visualization(self):
        """Render match point cloud and ellipsoids."""
        print(f"\n=== Rendering visualization ({len(self.match_points)} points) ===")

        self._clear_previous_renders()
        self.point_clouds = {}

        if len(self.match_points) == 0:
            print("No match points available")
            return

        # Render reference gamut
        RenderBGYRGamut(
            "reference_gamut",
            self.ref_cst,
            self.display_basis,
            alpha=0.1,
        )

        # Extract physiological parameters for coloring
        od_lm_values = np.array([p['od_lm'] for _, p in self.observers_and_params])
        lens_values = np.array([p['lens'] for _, p in self.observers_and_params])
        macular_values_all = np.array([p['macular'] for _, p in self.observers_and_params])

        # Normalize each to [0, 1]
        od_lm_norm = (od_lm_values - od_lm_values.min()) / (od_lm_values.max() - od_lm_values.min() + 1e-8)
        lens_norm = (lens_values - lens_values.min()) / (lens_values.max() - lens_values.min() + 1e-8)
        macular_norm = (macular_values_all - macular_values_all.min()) / (macular_values_all.max() - macular_values_all.min() + 1e-8)

        # Color map: R=optical density, G=lens, B=macular
        # White point = high in all three, black = low in all three
        colors = np.zeros((len(self.match_points), 3))
        colors[:, 0] = od_lm_norm  # Red channel = optical density
        colors[:, 1] = lens_norm  # Green channel = lens density
        colors[:, 2] = macular_norm  # Blue channel = macular pigment

        # Split points by 555nm L-cone peak presence
        has_555 = np.array([555 in genotype for genotype in self.genotypes])
        indices_555 = np.where(has_555)[0]
        indices_no_555 = np.where(~has_555)[0]

        # Render points with 555nm at higher opacity
        if len(indices_555) > 0:
            points_555 = self.match_points[indices_555]
            colors_555 = colors[indices_555]
            RenderPointCloud(
                "match_points_555",
                points_555,
                colors_555,
                radius=self.point_size,
            )
            try:
                pc_555 = ps.get_point_cloud("match_points_555")
                pc_555.set_opacity(0.8)
                self.point_clouds["match_points_555"] = pc_555
            except:
                pass

        # Render points without 555nm at lower opacity
        if len(indices_no_555) > 0:
            points_no_555 = self.match_points[indices_no_555]
            colors_no_555 = colors[indices_no_555]
            RenderPointCloud(
                "match_points_no_555",
                points_no_555,
                colors_no_555,
                radius=self.point_size,
            )
            try:
                pc_no_555 = ps.get_point_cloud("match_points_no_555")
                pc_no_555.set_opacity(0.3)
                self.point_clouds["match_points_no_555"] = pc_no_555
            except:
                pass

        # Render nominal observers (top 10) as larger points
        if len(self.nominal_match_points) > 0:
            nominal_colors = np.ones((len(self.nominal_match_points), 3)) * np.array([0.2, 0.2, 0.2])  # Dark gray
            RenderPointCloud(
                "nominal_match_points",
                self.nominal_match_points,
                nominal_colors,
                radius=self.point_size * 3.0,  # 3x larger
            )
            try:
                self.point_clouds["nominal_match_points"] = ps.get_point_cloud("nominal_match_points")
            except:
                pass

            # Render lines from background to each nominal match point
            background_disp = np.ones(4) * 0.5
            background_cone = self.ref_cst.convert(
                background_disp.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE
            )[0]
            background_viz = self.ref_cst.convert_to_polyscope(
                background_cone.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
            )[0]

            for i, nominal_pt in enumerate(self.nominal_match_points):
                line_pts = np.array([background_viz, nominal_pt])
                Render3DLine(
                    f"nominal_line_{i:02d}",
                    line_pts,
                    np.array([0.4, 0.4, 0.4]),  # Dark gray
                    radius=0.002,  # Thin lines
                )

            print(f"Rendered {len(self.nominal_match_points)} nominal points at 3x size with direction lines")

        # Optionally fit and render ellipsoids grouped by macular pigment
        if self.fit_ellipsoids:
            median_macular = np.median(self.macular_values)
            high_mask = self.macular_values >= median_macular
            low_mask = self.macular_values < median_macular

            high_points = self.match_points[high_mask]
            low_points = self.match_points[low_mask]

            print(f"High macular group: {len(high_points)} points")
            print(f"Low macular group: {len(low_points)} points")

            self._fit_and_render_ellipsoid(
                high_points, "ellipsoid_high_macular", np.array([1.0, 0.3, 0.3])
            )
            self._fit_and_render_ellipsoid(
                low_points, "ellipsoid_low_macular", np.array([0.3, 0.3, 1.0])
            )

        print("=== Finished rendering ===\n")

    def _fit_and_render_ellipsoid(
        self, points: np.ndarray, name: str, color: np.ndarray
    ):
        """Fit and render an ellipsoid for a point cloud."""
        if len(points) < 4:
            print(f"  Skipping {name}: insufficient points ({len(points)})")
            return

        try:
            fitter = EllipsoidFitter(dimension=3)
            fitter.fit(points, thresholds=np.ones(len(points)))

            center = fitter.center
            radii = fitter.semi_axes
            rotation = fitter.rotation_matrix

            print(f"  {name}: center={center}, radii={radii}")

            RenderNoiseBall(
                name,
                center=center,
                noise_std=radii,
                rotation=rotation,
                color=color,
                alpha=0.4,
            )

        except Exception as e:
            print(f"  Error fitting {name}: {e}")

    def _clear_previous_renders(self):
        """Clear previous visualization elements."""
        try:
            ps.remove_surface_mesh("reference_gamut")
        except:
            pass

        for pc_name in ["match_points", "match_points_555", "match_points_no_555"]:
            try:
                ps.remove_point_cloud(pc_name)
            except:
                pass

        try:
            ps.remove_point_cloud("nominal_match_points")
        except:
            pass

        # Remove nominal direction lines
        for i in range(self.num_nominal_observers):
            try:
                ps.remove_curve_network(f"nominal_line_{i:02d}")
            except:
                pass

        for name in ["ellipsoid_high_macular", "ellipsoid_low_macular"]:
            try:
                ps.remove_surface_mesh(name)
            except:
                pass

    def callback(self):
        """Polyscope GUI callback."""
        opened, self.window_open = psim.Begin("CMF Match Point Visualizer", self.window_open)

        if opened:
            psim.Text(f"Monte Carlo CMF Samples: {len(self.match_points)} points")
            psim.Text(f"Macular range: {self.macular_values.min():.3f} - {self.macular_values.max():.3f}")

            changed_size, new_point_size = psim.SliderFloat(
                "Point Size", self.point_size, 0.001, 0.1
            )

            if abs(new_point_size - self.point_size) > 1e-5:
                self.point_size = new_point_size
                for name, pc in self.point_clouds.items():
                    pc.set_radius(self.point_size)

            else:
                self.point_size = new_point_size

            psim.Text(f"Sampled cloud: {len(self.match_points)} match points")
            if len(self.nominal_match_points) > 0:
                psim.Text(f"Nominal observers: {len(self.nominal_match_points)} points (3x larger)")
                psim.Text("  with direction lines to background point")
            if self.fit_ellipsoids:
                psim.Text("Red ellipsoid: high macular pigment")
                psim.Text("Blue ellipsoid: low macular pigment")
            psim.Separator()
            psim.Text("Point color (RGB):")
            psim.Text("  R = optical density")
            psim.Text("  G = lens density")
            psim.Text("  B = macular pigment")
            psim.Text("  White = high in all three")
            psim.Separator()
            psim.Text("Point opacity:")
            psim.Text("  Opaque (80%) = 555nm L-cone")
            psim.Text("  Transparent (30%) = other L-cone peaks")

        psim.End()

    def show(self):
        """Show the polyscope window."""
        ps.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize metameric match point clouds under CMF physiological variation"
    )
    parser.add_argument(
        "--primaries_dir",
        type=str,
        default="./measurements/2026-04-16/primaries/",
        help="Directory containing display primary CSV files",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of Monte Carlo samples",
    )
    parser.add_argument(
        "--macular_mean",
        type=float,
        default=1.0,
        help="Mean macular pigment density",
    )
    parser.add_argument(
        "--macular_std",
        type=float,
        default=0.25,
        help="Std dev of macular pigment density",
    )
    parser.add_argument(
        "--od_lm_mean",
        type=float,
        default=0.5,
        help="Mean L/M photopigment OD",
    )
    parser.add_argument(
        "--od_lm_std",
        type=float,
        default=0.05,
        help="Std dev of L/M photopigment OD",
    )
    parser.add_argument(
        "--od_s_mean",
        type=float,
        default=0.4,
        help="Mean S photopigment OD",
    )
    parser.add_argument(
        "--od_s_std",
        type=float,
        default=0.04,
        help="Std dev of S photopigment OD",
    )
    parser.add_argument(
        "--lens_mean",
        type=float,
        default=1.0,
        help="Mean lens density",
    )
    parser.add_argument(
        "--lens_std",
        type=float,
        default=0.1,
        help="Std dev of lens density",
    )
    parser.add_argument(
        "--no_ellipsoids",
        action="store_true",
        help="Skip ellipsoid fitting, just show point cloud",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.02,
        help="Radius of rendered points (default 0.02)",
    )
    parser.add_argument(
        "--top_n_genotypes",
        type=int,
        default=None,
        help="Only sample from top N most probable genotypes (default: all)",
    )
    parser.add_argument(
        "--num_nominal_observers",
        type=int,
        default=10,
        help="Number of nominal observers to show as reference (default: 10)",
    )

    args = parser.parse_args()

    app = CMFMatchVisualizer(
        primaries_dir=args.primaries_dir,
        n_samples=args.n_samples,
        macular_mean=args.macular_mean,
        macular_std=args.macular_std,
        od_lm_mean=args.od_lm_mean,
        od_lm_std=args.od_lm_std,
        od_s_mean=args.od_s_mean,
        od_s_std=args.od_s_std,
        lens_mean=args.lens_mean,
        lens_std=args.lens_std,
        fit_ellipsoids=not args.no_ellipsoids,
        point_size=args.point_size,
        top_n_genotypes=args.top_n_genotypes,
        num_nominal_observers=args.num_nominal_observers,
    )

    app.show()


if __name__ == "__main__":
    main()
