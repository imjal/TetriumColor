#!/usr/bin/env python3
"""
Interactive GUI for visualizing metameric pairs with noise balls for multiple observers.

Features:
- Select luminance plane (L=x)
- Click to select a point on the gamut
- Convert to HERING coordinates and update with selected L value
- Display N observer balls from ObserverGenotypes
- Find opponent metamer point
- Render noise balls with configurable noise_std
- Render metameric line between points
- Label metameric direction with Billboard Text
- Display primaries on the selected luminance plane
"""

import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple
from itertools import product
import tetrapolyscope as ps
import tetrapolyscope.imgui as psim

from TetriumColor.Observer import Observer, GetHeringMatrix
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Visualization.PolyscopeUtils import (
    RenderNoiseBall,
    RenderPointCloud,
    Render3DLine,
    RenderBGYRGamut,
    RenderGamutSlices,
)


class MetamerNoiseGUI:
    def __init__(
        self,
        primaries_dir: str,
        num_observers: int = 4,
        initial_luminance: float = 0.5,
        initial_noise_std: float = 0.01,
        display_basis: PolyscopeDisplayType = PolyscopeDisplayType.HERING_BGYR,
    ):
        """Initialize the GUI application.

        Args:
            primaries_dir: Directory containing display primary CSV files
            num_observers: Number of observer genotypes to display
            initial_luminance: Initial luminance value (L=x)
            initial_noise_std: Initial noise standard deviation
            display_basis: Display basis for visualization
        """
        self.num_observers = num_observers
        self.luminance = initial_luminance
        self.noise_std = initial_noise_std  # Keep for backward compatibility, but use individual values
        # Individual noise parameters for each dimension (L, M, S, Q)
        # Default: S=1.0, others=0.0
        self.L_noise = 0.0001
        self.M_noise = 0.0001
        self.S_noise = 0.1
        self.Q_noise = 0.0001
        # Visual field coverage parameter (affects OD, macular density, etc.)
        self.visual_field_degree = 4  # Visual angle in degrees (2-10 degrees, integer steps)
        self.display_basis = display_basis
        self.doing_interaction = False
        self.window_open = True  # ImGUI window state
        self.show_display_primaries = False  # Toggle for display primaries, disabled by default
        self.show_other_gamut_slice = False  # Toggle for other metamer's gamut slice, disabled by default

        # Load display primaries
        print(f"Loading primaries from {primaries_dir}...")
        self.primaries = load_primaries_from_csv(primaries_dir, extract_zero=False)
        print(f"Loaded {len(self.primaries)} primaries")

        # Initialize observer genotypes
        observer_wavelengths = np.arange(380, 781, 5)
        self.observer_genotypes = ObserverGenotypes(
            wavelengths=observer_wavelengths, dimensions=[3], seed=42
        )

        # Get top N genotypes (create at least 8 so slider can go up to 8)
        max_genotypes = max(num_observers, 8)
        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=0.999, sex="both"
        )[:max_genotypes]
        self.genotypes = [sorted((420, ) + g + (547,)) for g in self.genotypes]

        # Create observers and color spaces with visual field parameters
        self._create_observers()

        print(f"Created {len(self.observers)} observers")

        # Selected point state
        self.selected_point_world: Optional[npt.NDArray] = None
        self.selected_point_bgyr: Optional[npt.NDArray] = None
        self.metamer_pairs: List[Tuple[npt.NDArray, npt.NDArray]] = []
        self.current_gamut_slice_name: Optional[str] = None  # Track current slice name
        self._last_intersection_cone_point: Optional[npt.NDArray] = None  # Store cone point for intersection

        # Initialize polyscope
        ps.init()
        ps.set_transparency_render_passes(24)
        ps.set_transparency_peel_epsilon(1e-7)
        ps.set_ground_plane_mode("none")
        ps.set_always_redraw(True)

        # Register callback
        ps.set_user_callback(self.callback)

        # Render gamut for reference (use first observer's color space)
        if len(self.color_spaces) > 0:
            RenderBGYRGamut(
                "reference_gamut",
                self.color_spaces[0],
                self.display_basis,
                alpha=0.2,
            )
            print("Rendered reference gamut")

            # Render initial gamut slice for the selected luminance
            self.render_gamut_slice()

    def _create_observers(self):
        """Create observers and color spaces with current visual field parameters."""
        self.observers = []
        self.color_spaces = []
        for i, genotype in enumerate(self.genotypes):
            print(f"Creating observer {i+1} for genotype: {genotype}")
            # Use get_observer_for_peaks which accepts od and degree parameters
            # degree affects macular pigment and optical density automatically
            # od=0.5 will be automatically converted to appropriate OD based on degree

            obs = self.observer_genotypes.get_observer_for_peaks(
                genotype,
                degree=float(self.visual_field_degree)
            )
            self.observers.append(obs)
            cst = ColorSpace(obs, self.primaries)
            self.color_spaces.append(cst)

    def screen_coords_to_world_ray(self, screen_coords: Tuple[float, float]) -> Tuple[npt.NDArray, npt.NDArray]:
        """Convert screen coordinates to a world space ray using polyscope's pick_ray.

        Args:
            screen_coords: (x, y) screen coordinates in pixels

        Returns:
            Tuple of (ray_origin, ray_direction) in world space
        """
        # Use polyscope's built-in pick_ray function
        x, y = screen_coords
        try:
            ray_origin, ray_direction = ps.pick_ray(x, y)
            return np.array(ray_origin), np.array(ray_direction)
        except Exception as e:
            print(f"Error in pick_ray: {e}")
            # Fallback: use camera to compute ray
            # Get camera parameters
            camera = ps.get_view_camera_parameters()
            # This is a simplified fallback - in practice you'd compute the ray properly
            # For now, return a default ray
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

    def get_gamut_points_at_luminance(
        self, luminance: float, grid_resolution: int = 20, tolerance: float = 0.03
    ) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
        """Sample gamut points at a specific luminance level.

        Args:
            luminance: Target Hering luminance value
            grid_resolution: Resolution of sampling grid in DISP space
            tolerance: Tolerance for luminance matching

        Returns:
            Tuple of (points in display basis (3D), original cone points), or (None, None) if no points found
        """
        if len(self.color_spaces) == 0:
            return None, None

        cst = self.color_spaces[0]

        # Sample the DISP space [0,1]^4 on a grid
        grid_1d = np.linspace(0, 1, grid_resolution)
        disp_points = np.array(list(product(grid_1d, grid_1d, grid_1d, grid_1d)))

        # Convert to CONE space
        cone_points = cst.convert(disp_points, ColorSpaceType.DISP, ColorSpaceType.CONE)

        # Get Hering transform and compute luminance for each point
        H = GetHeringMatrix(cst.dim)
        hering_points = cone_points @ H.T
        luminances = hering_points[:, 0]  # First coordinate is luminance

        # Find points close to target luminance
        mask = np.abs(luminances - luminance) < tolerance
        slice_points_cone = cone_points[mask]

        if len(slice_points_cone) < 10:
            print(f"Warning: Only {len(slice_points_cone)} points found at luminance {luminance:.2f}")
            return None, None

        # Convert to visualization space (display basis)
        slice_points_viz = cst.convert_to_polyscope(
            slice_points_cone, ColorSpaceType.CONE, self.display_basis
        )

        return slice_points_viz, slice_points_cone

    def intersect_ray_with_gamut(
        self, ray_origin: npt.NDArray, ray_direction: npt.NDArray
    ) -> Optional[npt.NDArray]:
        """Find intersection of ray with gamut surface at the selected luminance.

        Args:
            ray_origin: Ray origin in world space (display basis)
            ray_direction: Ray direction in world space (display basis)

        Returns:
            Intersection point in world space (display basis), or None if no intersection
        """
        # Normalize ray direction
        if np.linalg.norm(ray_direction) < 1e-6:
            print("Warning: Ray direction is too small")
            return None

        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Get gamut points at the selected luminance (both display basis and original cone points)
        gamut_points, gamut_cone_points = self.get_gamut_points_at_luminance(
            self.luminance, grid_resolution=25, tolerance=0.03
        )

        if gamut_points is None or len(gamut_points) < 10:
            print("Warning: Could not sample gamut points, using fallback")
            # Fallback: use a point along the ray at a reasonable distance
            t = 0.3
            point = ray_origin + t * ray_direction
            point = np.clip(point, -0.8, 0.8)
            self._last_intersection_cone_point = None  # Can't convert fallback point
            return point

        # Find the point on the ray that intersects or is closest to the gamut surface
        try:
            from scipy.spatial import Delaunay

            # Create Delaunay triangulation for point-in-hull testing
            delaunay = Delaunay(gamut_points)

            # Sample points along the ray to find intersection
            # Start from a reasonable distance and sample forward
            t_start = 0.0
            t_end = 2.0
            num_samples = 200
            t_values = np.linspace(t_start, t_end, num_samples)
            ray_points = ray_origin + t_values[:, np.newaxis] * ray_direction

            # Find the first point that is inside the gamut (or closest to it)
            best_t = None
            best_point = None
            min_dist_outside = float('inf')

            for i, (t, ray_pt) in enumerate(zip(t_values, ray_points)):
                # Check if point is inside the convex hull
                simplex_idx = delaunay.find_simplex(ray_pt)

                if simplex_idx >= 0:
                    # Point is inside the gamut
                    # Find the point on the gamut surface along this direction
                    # by moving backward until we hit the surface
                    if i > 0:
                        # Binary search between previous point (outside) and current (inside)
                        t_prev = t_values[i-1]
                        t_low, t_high = t_prev, t
                        for _ in range(10):  # Binary search iterations
                            t_mid = (t_low + t_high) / 2
                            pt_mid = ray_origin + t_mid * ray_direction
                            if delaunay.find_simplex(pt_mid) >= 0:
                                t_high = t_mid  # Still inside, move boundary inward
                            else:
                                t_low = t_mid  # Outside, move boundary outward
                        best_t = t_low  # Use the last point that was outside (on surface)
                    else:
                        # Ray starts inside, use origin
                        best_t = t_start
                    break
                else:
                    # Point is outside, track the closest one
                    distances = np.linalg.norm(gamut_points - ray_pt, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < min_dist_outside:
                        min_dist_outside = min_dist
                        best_t = t
                        best_point = gamut_points[np.argmin(distances)]

            # If we found an intersection point
            if best_t is not None:
                intersection_pt = ray_origin + best_t * ray_direction

                # Project onto gamut surface by finding closest gamut point
                distances = np.linalg.norm(gamut_points - intersection_pt, axis=1)
                closest_idx = np.argmin(distances)
                surface_point = gamut_points[closest_idx]

                # Store the corresponding cone point for later conversion
                self._last_intersection_cone_point = gamut_cone_points[closest_idx]

                return surface_point
            elif best_point is not None:
                # Use the closest point we found
                distances = np.linalg.norm(gamut_points - best_point, axis=1)
                closest_idx = np.argmin(distances)
                self._last_intersection_cone_point = gamut_cone_points[closest_idx]
                return best_point
            else:
                # Fallback: use point at fixed distance
                t = 0.3
                point = ray_origin + t * ray_direction
                return point

        except Exception as e:
            print(f"Error computing gamut intersection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: find closest gamut point to ray
            # Sample points along ray and find closest to any gamut point
            t_values = np.linspace(0, 2.0, 100)
            ray_points = ray_origin + t_values[:, np.newaxis] * ray_direction

            min_dist = float('inf')
            best_point = None
            best_idx = None

            for ray_pt in ray_points:
                distances = np.linalg.norm(gamut_points - ray_pt, axis=1)
                min_dist_to_gamut = np.min(distances)
                if min_dist_to_gamut < min_dist:
                    min_dist = min_dist_to_gamut
                    best_idx = np.argmin(distances)
                    best_point = gamut_points[best_idx]

            if best_point is not None:
                self._last_intersection_cone_point = gamut_cone_points[best_idx]
                return best_point

            # Final fallback
            t = 0.3
            point = ray_origin + t * ray_direction
            # For fallback, we'll need to convert differently
            self._last_intersection_cone_point = None
            return point

    def handle_mouse_click(self):
        """Handle mouse click to select a point on the gamut."""
        mouse_pos = psim.GetMousePos()
        screen_coords = (mouse_pos[0], mouse_pos[1])
        print(f"\n=== Mouse click detected at screen coords: {screen_coords} ===")

        # Convert to world position using Polyscope's screen_coords_to_world_position function
        # This uses the depth buffer to get a point on the surface that was clicked
        # Temporarily disable the reference gamut hull so we can read from the gamut slice
        reference_gamut_enabled = True
        reference_gamut_hull_enabled = True
        try:
            # Try to disable reference gamut objects
            try:
                ref_gamut = ps.get_surface_mesh("reference_gamut")
                reference_gamut_enabled = ref_gamut.is_enabled()
                ref_gamut.set_enabled(False)
            except (RuntimeError, KeyError):
                pass
            try:
                ref_gamut_hull = ps.get_surface_mesh("reference_gamut_hull")
                reference_gamut_hull_enabled = ref_gamut_hull.is_enabled()
                ref_gamut_hull.set_enabled(False)
            except (RuntimeError, KeyError):
                pass

            x, y = screen_coords
            # Get the world position from the depth buffer at the clicked location
            world_pos = ps.screen_coords_to_world_position(screen_coords)
            intersection = np.array(world_pos)
            print(f"World position from depth buffer: {intersection}")

            # Re-enable the reference gamut objects
            try:
                ref_gamut = ps.get_surface_mesh("reference_gamut")
                ref_gamut.set_enabled(reference_gamut_enabled)
            except (RuntimeError, KeyError):
                pass
            try:
                ref_gamut_hull = ps.get_surface_mesh("reference_gamut_hull")
                ref_gamut_hull.set_enabled(reference_gamut_hull_enabled)
            except (RuntimeError, KeyError):
                pass

            # Since we got a point directly from the depth buffer, we need to find
            # the corresponding gamut point at the selected luminance
            # The point might not be exactly on the gamut, so we'll find the closest one
            if intersection is not None and len(intersection) > 0:
                # Get gamut points at the selected luminance
                gamut_points, gamut_cone_points = self.get_gamut_points_at_luminance(
                    self.luminance, grid_resolution=25, tolerance=0.03
                )

                if gamut_points is not None and len(gamut_points) > 0:
                    # Find the closest gamut point to the clicked position
                    distances = np.linalg.norm(gamut_points - intersection, axis=1)
                    closest_idx = np.argmin(distances)
                    intersection = gamut_points[closest_idx]
                    self._last_intersection_cone_point = gamut_cone_points[closest_idx]
                    print(f"Closest gamut point: {intersection}")
                else:
                    print("Warning: Could not find gamut points, using depth buffer point directly")
                    self._last_intersection_cone_point = None

            if intersection is not None and len(intersection) > 0:
                self.selected_point_world = intersection
                print(f"Selected point in world space (HERING_BGYR display, 3D): {intersection}")

                # Convert the intersection point to BGYR coordinates
                # The intersection is in HERING_BGYR display space (3D chromatic coordinates)
                # Process: intersection (3D HERING chromatic) -> add L=x -> HERING (4D) -> BGYR -> store
                if len(self.color_spaces) > 0:
                    cst = self.color_spaces[0]
                    try:
                        # The intersection point is 3D HERING chromatic coordinates (luminance dropped)
                        # Add back the luminance to get full 4D HERING coordinates
                        hering_chrom = intersection
                        if len(hering_chrom) == 3:
                            # Add luminance as first component to make full HERING coordinate
                            hering_full = np.array([self.luminance, hering_chrom[0], hering_chrom[1], hering_chrom[2]])
                        elif len(hering_chrom) == 4:
                            # Already 4D, just update the luminance
                            hering_full = hering_chrom.copy()
                            hering_full[0] = self.luminance
                        else:
                            # Pad or truncate as needed
                            if len(hering_chrom) < 4:
                                hering_full = np.concatenate(
                                    [[self.luminance], hering_chrom, np.zeros(4 - len(hering_chrom) - 1)])
                            else:
                                hering_full = np.concatenate([[self.luminance], hering_chrom[:3]])

                        print(f"Reconstructed full HERING coordinate: {hering_full}")

                        # Convert HERING -> BGYR (goes through CONE as intermediate)
                        # Note: hering_full is in HERING space (CONE transformed by Hering matrix)
                        # HERING_BGYR would be BGYR transformed by Hering matrix, which is different
                        bgyr_point = cst.convert(
                            hering_full.reshape(1, -1),
                            ColorSpaceType.HERING_BGYR,
                            ColorSpaceType.BGYR,
                        )[0]

                        self.selected_point_bgyr = bgyr_point
                        print(f"Selected point in BGYR: {bgyr_point}")
                    except Exception as e:
                        print(f"Error converting intersection point to BGYR: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                else:
                    print("ERROR: No color spaces available")
                    return

                # Render the selected point immediately as a visible ball
                # Convert to display space for visualization
                if len(self.color_spaces) > 0:
                    cst = self.color_spaces[0]
                    try:
                        # Convert BGYR -> CONE -> display basis
                        cone_point = cst.convert(
                            self.selected_point_bgyr.reshape(1, -1),
                            ColorSpaceType.BGYR,
                            ColorSpaceType.CONE,
                        )[0]
                        disp_point = cst.convert_to_polyscope(
                            cone_point.reshape(1, -1),
                            ColorSpaceType.CONE,
                            self.display_basis,
                        )[0]
                        # Render selected point as a visible ball
                        # Remove old point first
                        try:
                            ps.remove_point_cloud("selected_point")
                        except:
                            pass

                        RenderPointCloud(
                            "selected_point",
                            disp_point.reshape(1, -1),
                            np.array([[1.0, 1.0, 0.0]]),  # Yellow
                            radius=0.03,
                        )
                        print(f"Rendered selected point at: {disp_point}")

                        # Also render as a larger sphere for visibility
                        try:
                            ps.remove_point_cloud("selected_point_large")
                        except:
                            pass
                        RenderPointCloud(
                            "selected_point_large",
                            disp_point.reshape(1, -1),
                            np.array([[1.0, 1.0, 0.0]]),  # Yellow
                            radius=0.05,
                        )
                    except Exception as e:
                        print(f"Error rendering selected point: {e}")
                        import traceback
                        traceback.print_exc()

                # Update visualization - this will recompute metamer pairs and noise balls
                print("Calling update_visualization() after mouse click...")
                self.update_visualization()
                print("Finished update_visualization() after mouse click")
        except Exception as e:
            print(f"Error handling mouse click: {e}")
            import traceback
            traceback.print_exc()

    def estimate_jacobian(self, func, point, eps=1e-6):
        """Estimate Jacobian via finite differences."""
        n = len(point)
        f0 = func(point)
        m = len(f0)
        J = np.zeros((m, n))
        for i in range(n):
            perturbed = point.copy()
            perturbed[i] += eps
            J[:, i] = (func(perturbed) - f0) / eps
        return J

    def update_visualization(self):
        """Update all visualization elements."""
        print(f"\n=== Updating visualization ===")
        print(f"Selected BGYR point: {self.selected_point_bgyr}")

        # Clear previous metamer pairs and noise balls
        self.clear_previous_renders()

        if self.selected_point_bgyr is None:
            print("No point selected, skipping visualization update")
            return

        # Make sure we only process the number of observers specified
        num_to_process = min(self.num_observers, len(self.observers), len(self.color_spaces))
        print(f"Processing {num_to_process} observers")

        if len(self.color_spaces) == 0:
            print("No color spaces available")
            return

        # Convert the selected BGYR point to display space ONCE using the first observer
        # This gives us a common starting point that all observers will use
        reference_cst = self.color_spaces[0]
        selected_cone_common = reference_cst.convert(
            self.selected_point_bgyr.reshape(1, -1),
            ColorSpaceType.BGYR,
            ColorSpaceType.CONE,
        )[0]
        selected_disp_common = reference_cst.convert_to_polyscope(
            selected_cone_common.reshape(1, -1),
            ColorSpaceType.CONE,
            self.display_basis,
        )[0]
        print(f"Common selected point in display space: {selected_disp_common}")

        # Convert BGYR point to display space for each observer
        for i in range(num_to_process):
            obs = self.observers[i]
            cst = self.color_spaces[i]
            print(f"\nProcessing observer {i+1}...")
            try:
                # Find metameric pair directly from BGYR space (matching notebook approach)
                # This ensures we get the same results as the notebook
                metameric_axis = cst.metameric_axis
                print(f"  Metameric axis: {metameric_axis}")
                result = cst.get_maximal_pair_in_disp_from_pt(
                    pt=self.selected_point_bgyr,
                    metameric_axis=metameric_axis,
                    input_space=ColorSpaceType.BGYR,  # Use BGYR directly, like notebook
                    output_space=ColorSpaceType.CONE,
                    proportion=0.8,
                )

                if result is None:
                    print(f"  Observer {i+1}: Could not find metameric pair")
                    continue

                cone1, cone2, metamer_diff = result
                print(f"  Found metamer pair, diff: {metamer_diff:.4f}")
                print(f"  Cone1: {cone1}, Cone2: {cone2}")

                # Convert both metamer points to display space (matching notebook approach)
                disp1 = cst.convert_to_polyscope(
                    cone1.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]
                disp2 = cst.convert_to_polyscope(
                    cone2.reshape(1, -1), ColorSpaceType.CONE, self.display_basis
                )[0]

                print(f"  Metamer point 1 in display space: {disp1}")
                print(f"  Metamer point 2 in display space: {disp2}")

                # Find which metamer point is furthest from the selected point
                # Convert selected BGYR point to CONE for distance comparison
                selected_cone = cst.convert(
                    self.selected_point_bgyr.reshape(1, -1),
                    ColorSpaceType.BGYR,
                    ColorSpaceType.CONE,
                )[0]
                dist1 = np.linalg.norm(cone1 - selected_cone)
                dist2 = np.linalg.norm(cone2 - selected_cone)

                if dist2 > dist1:
                    # cone2 is further, use it as the "other" metamer point
                    other_cone = cone2
                    other_disp = disp2
                else:
                    # cone1 is further, use it as the "other" metamer point
                    other_cone = cone1
                    other_disp = disp1

                print(f"  Selected point cone: {selected_cone}")
                print(f"  Other metamer cone: {other_cone}")
                print(f"  Distance to cone1: {dist1:.4f}, to cone2: {dist2:.4f}")

                # Use the common selected point for all observers (so they all start from the same place)
                # The line goes from the common selected point to this observer's other metamer point
                print(f"  Selected point in display space (common): {selected_disp_common}")
                print(f"  Other metamer point in display space: {other_disp}")

                # Store the metamer pair (common selected point, other metamer point)
                self.metamer_pairs.append((selected_disp_common, other_disp))

                # Get the luminance of the other metamer point
                H = GetHeringMatrix(cst.dim)
                other_hering = other_cone @ H.T
                other_luminance = other_hering[0]
                print(f"  Other metamer luminance: {other_luminance:.4f}")

                # Use the same color for all elements of this metamer pair
                # Different colors for different observers to distinguish them
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
                metamer_color = observer_colors[i % len(observer_colors)]

                # Render metameric line from common selected point to other metamer point
                # All observers share the same starting point, but have different end points
                Render3DLine(
                    f"metamer_line_{i}",
                    np.array([selected_disp_common, other_disp]),
                    metamer_color,
                    radius=0.003,
                )
                print(f"  Rendered metamer line {i} from common point to observer {i+1}'s metamer")

                # Render noise balls at the metamer pair points (disp1, disp2)
                # Apply noise independently per dimension in cone space (L, M, S, Q)
                # Create noise std array in cone space with individual values
                # Order: S, M, Q, L (matching cone space order)
                noise_std_cone = np.array([self.S_noise, self.M_noise, self.Q_noise, self.L_noise])
                print(f"  Using noise std in cone space: {noise_std_cone}")

                M = self.estimate_jacobian(lambda x: cst.convert_to_polyscope(
                    np.array([x]).reshape(1, -1), ColorSpaceType.CONE, PolyscopeDisplayType.HERING_BGYR)[0], noise_std_cone)  # 3x3 or compute numerically

                # Build covariance in display space
                S = np.diag(noise_std_cone)  # assuming 3D for display
                C = M @ S @ S.T @ M.T

                # Eigendecompose
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                semi_axes = np.sqrt(eigenvalues)

                # Render noise ball only at the other (furthest) metamer point
                print(f"  Rendering noise ball at other metamer point {other_disp} with semi-axes {semi_axes}")
                RenderNoiseBall(
                    f"noise_ball_{i}",
                    other_disp,
                    semi_axes,
                    rotation=eigenvectors,  # columns are principal directions
                    color=metamer_color,
                    num_samples=500,
                    alpha=0.4,
                )
                print(f"  Rendered noise ball {i}")

                # Also render the noisy point as a small marker
                # RenderPointCloud(
                #     f"noisy_point_{i}",
                #     disp_point.reshape(1, -1),
                #     metamer_color.reshape(1, -1),
                #     radius=0.01,
                # )

                # Label the metameric direction
                # Compute direction from the actual metamer line (from selected point to other metamer point)
                # This ensures the direction matches the actual line direction
                metamer_line_vector = other_disp - selected_disp_common
                metamer_dir_display_norm = metamer_line_vector / (np.linalg.norm(metamer_line_vector) + 1e-8)
                print(f"  Metamer line direction (from actual points): {metamer_dir_display_norm}")

                # Position label outside the noise ball
                # Use the maximum semi-axis length plus a margin to ensure it's outside
                max_semi_axis = np.max(semi_axes)
                margin = 0.3  # Additional margin for visibility
                label_offset = max_semi_axis + margin
                label_pos = other_disp + metamer_dir_display_norm * label_offset

                # Render gamut slice at the other metamer's luminance (only if enabled)
                if self.show_other_gamut_slice:
                    # Use unique name per observer
                    other_slice_name_base = f"other_gamut_slice_{i}"
                    other_slice_name = f"{other_slice_name_base}_slice_L{other_luminance:.2f}"

                    # Clear previous "other" gamut slice for this observer if it exists
                    try:
                        ps.remove_surface_mesh(other_slice_name)
                    except (RuntimeError, KeyError):
                        pass
                    try:
                        ps.remove_point_cloud(other_slice_name)
                    except (RuntimeError, KeyError):
                        pass

                    # Render gamut slice at other metamer's luminance
                    RenderGamutSlices(
                        other_slice_name_base,
                        cst,
                        display_space=ColorSpaceType.DISP,
                        display_basis=self.display_basis,
                        luminance_values=[other_luminance],
                        grid_resolution=25,
                        tolerance=0.03,
                        alpha=0.2,  # Slightly more transparent to distinguish from main slice
                    )
                    print(f"  Rendered other gamut slice at L={other_luminance:.2f} (name: {other_slice_name})")

                # Create billboard text label (same color as metamer line)
                label_text = f"L_{int(obs.sensors[metameric_axis].peak)}"
                ps.register_billboard_text(
                    f"label_{i}",
                    label_text,
                    label_pos,
                    enabled=True,
                    font_size=8.0,
                    text_color=metamer_color.tolist(),  # Same color as metamer line
                )

            except Exception as e:
                print(f"Error updating observer {i+1}: {e}")
                import traceback
                traceback.print_exc()

        print(f"=== Finished updating visualization ===\n")

        # Render display primaries on the luminance plane (only if enabled)
        if self.show_display_primaries:
            self.render_display_primaries()

    def render_display_primaries(self):
        """Render display primaries that lie on the selected luminance plane."""
        if self.selected_point_bgyr is None:
            return

        # Sample display primaries
        # We'll sample points in DISP space and filter by luminance
        # Use 8-bit discretization but with a reduced grid for performance
        # Sample every Nth value from 0-255 to keep computation manageable
        grid_step = 16  # Sample every 16th value: 0, 16, 32, ..., 240 (16 values)
        grid_values_8bit = np.arange(0, 256, grid_step)  # 16 values
        grid_1d = grid_values_8bit / 255.0  # Normalize to [0, 1]
        from itertools import product

        # This gives us 16^4 = 65,536 points (manageable)
        disp_points = np.array(list(product(grid_1d, grid_1d, grid_1d, grid_1d)))
        print(
            f"Sampling {len(disp_points)} points from 8-bit quantized grid (step={grid_step}, {len(grid_1d)} values per dimension)")

        # Use the first color space for conversion
        if len(self.color_spaces) == 0:
            return

        cst = self.color_spaces[0]

        # Convert to CONE space
        cone_points = cst.convert(disp_points, ColorSpaceType.DISP, ColorSpaceType.CONE)

        # Get Hering transform and compute luminance
        H = GetHeringMatrix(cst.dim)
        hering_points = cone_points @ H.T
        luminances = hering_points[:, 0]

        # Filter points close to selected luminance
        tolerance = 0.05
        mask = np.abs(luminances - self.luminance) < tolerance
        primary_points_cone = cone_points[mask]

        if len(primary_points_cone) == 0:
            print(f"No display primaries found at L={self.luminance:.2f}")
            return

        # Convert to display space
        primary_points_disp = cst.convert_to_polyscope(
            primary_points_cone, ColorSpaceType.CONE, self.display_basis
        )

        # Further reduce points by subsampling if there are still too many
        max_points = 2000  # Maximum number of points to render
        if len(primary_points_disp) > max_points:
            # Randomly sample points
            indices = np.random.choice(len(primary_points_disp), max_points, replace=False)
            primary_points_disp = primary_points_disp[indices]
            print(f"Subsampled to {max_points} points from {len(cone_points[mask])}")

        # Render as point cloud with smaller radius
        primary_colors = np.ones((len(primary_points_disp), 3)) * 0.5  # Gray
        RenderPointCloud(
            "display_primaries",
            primary_points_disp,
            primary_colors,
            radius=0.0001,  # Much smaller radius (was 0.008)
        )

        print(f"Rendered {len(primary_points_disp)} display primary points at L={self.luminance:.2f}")

    def render_gamut_slice(self):
        """Render the display gamut slice at the current luminance level."""
        if len(self.color_spaces) == 0:
            return

        # Clear previous slice if it exists
        if self.current_gamut_slice_name is not None:
            try:
                ps.remove_surface_mesh(self.current_gamut_slice_name)
            except (RuntimeError, KeyError):
                pass
            try:
                ps.remove_point_cloud(self.current_gamut_slice_name)
            except (RuntimeError, KeyError):
                pass

        # Use first color space for rendering the slice
        cst = self.color_spaces[0]

        # Render the slice at the current luminance
        # RenderGamutSlices creates names like "gamut_slice_slice_L{target_lum:.2f}"
        slice_name_base = "gamut_slice"
        RenderGamutSlices(
            slice_name_base,
            cst,
            display_space=ColorSpaceType.DISP,
            display_basis=self.display_basis,
            luminance_values=[self.luminance],
            grid_resolution=25,
            tolerance=0.03,
            alpha=0.4,
        )

        # Store the actual slice name that was created
        self.current_gamut_slice_name = f"{slice_name_base}_slice_L{self.luminance:.2f}"
        print(f"Rendered gamut slice at L={self.luminance:.2f} (name: {self.current_gamut_slice_name})")

    def clear_previous_renders(self):
        """Clear previously rendered metamer pairs and noise balls."""
        print("Clearing previous renders...")
        # Silently ignore errors when removing structures that don't exist
        # Check up to 10 observers to ensure we remove all
        max_check = max(self.num_observers, 10)
        for i in range(max_check):
            # Remove metamer lines
            try:
                ps.remove_curve_network(f"metamer_line_{i}")
            except (RuntimeError, KeyError):
                pass

            # Remove noise balls (they might be surface meshes or point clouds)
            # Check both old naming (noise_ball_{i}_1, noise_ball_{i}_2) and new naming (noise_ball_{i})
            for suffix in ["_1", "_2", ""]:
                ball_name = f"noise_ball_{i}{suffix}"
                removed = False
                try:
                    ps.remove_surface_mesh(ball_name)
                    removed = True
                except (RuntimeError, KeyError):
                    try:
                        ps.remove_point_cloud(ball_name)
                        removed = True
                    except (RuntimeError, KeyError):
                        pass
                if removed:
                    print(f"  Removed {ball_name}")

            # Remove labels
            try:
                ps.remove_billboard_text(f"label_{i}")
            except (RuntimeError, KeyError):
                pass

        # Remove display primaries
        try:
            ps.remove_point_cloud("display_primaries")
        except (RuntimeError, KeyError):
            pass

        # Remove selected point
        try:
            ps.remove_point_cloud("selected_point")
        except (RuntimeError, KeyError):
            pass
        try:
            ps.remove_point_cloud("selected_point_large")
        except (RuntimeError, KeyError):
            pass

        # Remove other gamut slices (one per observer)
        # Try to remove slices for all observers
        # We'll check a reasonable range of luminance values (0.0 to 2.0 in 0.05 steps)
        max_check = max(self.num_observers, 10)
        for i in range(max_check):
            # Check luminance values from 0.0 to 2.0 in 0.05 steps (41 values)
            for lum_int in range(0, 41):  # 0.00 to 2.00 in 0.05 steps
                lum = lum_int / 20.0
                slice_name = f"other_gamut_slice_{i}_slice_L{lum:.2f}"
                try:
                    ps.remove_surface_mesh(slice_name)
                except (RuntimeError, KeyError):
                    pass
                try:
                    ps.remove_point_cloud(slice_name)
                except (RuntimeError, KeyError):
                    pass

        self.metamer_pairs = []

    def callback(self):
        """Polyscope callback for GUI and interaction."""
        # Handle mouse interaction
        if psim.IsMouseClicked(1):  # Right mouse button
            ps.set_do_default_mouse_interaction(False)
            self.doing_interaction = True
            self.handle_mouse_click()

        if not psim.IsMouseDown(1):
            if self.doing_interaction:
                ps.set_do_default_mouse_interaction(True)
                self.doing_interaction = False

        # GUI
        opened, self.window_open = psim.Begin("Metamer Noise Visualization", self.window_open)

        if opened:
            # Luminance slider
            changed, self.luminance = psim.SliderFloat(
                "Luminance (L)", self.luminance, 0.0, 2.0
            )
            if changed:
                # Update gamut slice
                self.render_gamut_slice()

                # Update selected point if it exists
                # BGYR doesn't have luminance, but we should still update visualization
                if self.selected_point_bgyr is not None:
                    self.update_visualization()

            # Individual noise sliders for each dimension
            psim.Text("Noise Parameters (per dimension):")
            changed_L, self.L_noise = psim.SliderFloat(
                "L Noise", self.L_noise, 0.001, 0.1
            )
            changed_M, self.M_noise = psim.SliderFloat(
                "M Noise", self.M_noise, 0.001, 0.1
            )
            changed_S, self.S_noise = psim.SliderFloat(
                "S Noise", self.S_noise, 0.001, 0.1
            )
            changed_Q, self.Q_noise = psim.SliderFloat(
                "Q Noise", self.Q_noise, 0.001, 0.1
            )
            if changed_L or changed_M or changed_S or changed_Q:
                print(f"\n=== Noise parameters changed ===")
                print(
                    f"L_noise={self.L_noise:.4f}, M_noise={self.M_noise:.4f}, S_noise={self.S_noise:.4f}, Q_noise={self.Q_noise:.4f}")
                self.update_visualization()

            # Display primaries toggle
            changed_primaries, self.show_display_primaries = psim.Checkbox(
                "Show Display Primaries", self.show_display_primaries
            )
            if changed_primaries:
                if self.show_display_primaries:
                    self.render_display_primaries()
                else:
                    # Remove display primaries
                    try:
                        ps.remove_point_cloud("display_primaries")
                    except (RuntimeError, KeyError):
                        pass

            # Other gamut slice toggle
            changed_other_slice, self.show_other_gamut_slice = psim.Checkbox(
                "Show Other Metamer Gamut Slice", self.show_other_gamut_slice
            )
            if changed_other_slice:
                if self.show_other_gamut_slice:
                    # Update visualization to show other gamut slices
                    self.update_visualization()
                else:
                    # Remove other gamut slices when disabled
                    max_check = max(self.num_observers, 10)
                    for i in range(max_check):
                        # Try a range of possible luminance values (0.0 to 2.0 in 0.05 steps)
                        for lum_int in range(0, 41):  # 0.00 to 2.00 in 0.05 steps
                            lum = lum_int / 20.0
                            slice_name = f"other_gamut_slice_{i}_slice_L{lum:.2f}"
                            try:
                                ps.remove_surface_mesh(slice_name)
                            except (RuntimeError, KeyError):
                                pass
                            try:
                                ps.remove_point_cloud(slice_name)
                            except (RuntimeError, KeyError):
                                pass

            # Visual field coverage parameter
            psim.Text("Visual Field Parameters:")
            changed_degree, self.visual_field_degree = psim.SliderInt(
                "Visual Field (degrees)", self.visual_field_degree, 2, 10
            )
            if changed_degree:
                print(f"Visual field degree changed: {self.visual_field_degree} degrees")
                # Recreate observers with new parameters
                self._create_observers()
                # Update visualization if a point is selected
                if self.selected_point_bgyr is not None:
                    self.update_visualization()

            # Number of observers
            max_observers = min(len(self.observers), 8) if len(self.observers) > 0 else 8
            # Ensure num_observers doesn't exceed available observers or limit of 8
            if self.num_observers > max_observers:
                self.num_observers = max_observers
            changed, self.num_observers = psim.SliderInt(
                "Num Observers", self.num_observers, 1, max_observers
            )
            if changed:
                print(f"Number of observers changed to {self.num_observers}")
                # Update visualization to show/hide observers based on new count
                if self.selected_point_bgyr is not None:
                    self.update_visualization()

            psim.Text("Right-click to select a point on the gamut")
            if self.selected_point_bgyr is not None:
                psim.Text(f"Selected point (BGYR): {self.selected_point_bgyr}")

        psim.End()

    def show(self):
        """Show the polyscope window."""
        ps.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive metamer noise visualization GUI"
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
        default=4,
        help="Number of observer genotypes to display",
    )
    parser.add_argument(
        "--initial_luminance",
        type=float,
        default=0.5,
        help="Initial luminance value",
    )
    parser.add_argument(
        "--initial_noise_std",
        type=float,
        default=0.05,
        help="Initial noise standard deviation",
    )

    args = parser.parse_args()

    app = MetamerNoiseGUI(
        primaries_dir=args.primaries_dir,
        num_observers=args.num_observers,
        initial_luminance=args.initial_luminance,
        initial_noise_std=args.initial_noise_std,
    )

    app.show()


if __name__ == "__main__":
    main()
