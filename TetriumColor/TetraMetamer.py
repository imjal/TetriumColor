import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.distance import cdist

from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType


class MetamericOptimizer:
    """
    A class for optimizing metameric color pairs through stochastic sampling and user feedback.

    This optimizer generates candidate metameric pairs by sampling around target colors
    in the display space, avoiding the metameric axis to ensure spectral diversity.
    """

    def __init__(self,
                 color_space: ColorSpace,
                 color_space_output_type: ColorSpaceType = ColorSpaceType.DISP,
                 stochastic_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MetamericOptimizer.

        Args:
            color_space: The ColorSpace object containing observer and display information
            color_space_output_type: The color space type for output (default: DISP)
            stochastic_params: Parameters controlling stochasticity (optional)
        """
        self.color_space = color_space
        self.color_space_output_type = color_space_output_type
        self.observer = color_space.observer
        self.metameric_axis = color_space.metameric_axis

        # Default stochastic parameters
        self.stochastic_params = {
            'direction_noise_std': 0.1,  # Standard deviation for direction sampling
            'distance_noise_std': 0.05,  # Standard deviation for distance sampling
            'num_directions': 20,        # Number of random directions to sample
            'min_distance': 0.01,        # Minimum distance from target
            'max_distance': 0.3,         # Maximum distance from target
            'gamut_tolerance': 0.01,     # Tolerance for gamut boundary checking
            'spectral_diversity_weight': 0.5,  # Weight for spectral diversity in sampling
        }

        # Update with user-provided parameters
        if stochastic_params:
            self.stochastic_params.update(stochastic_params)

    def _sample_random_directions(self, num_directions: int, avoid_axis: int) -> npt.NDArray:
        """
        Sample random directions in the display space, avoiding the metameric axis.

        Args:
            num_directions: Number of directions to sample
            avoid_axis: Axis to avoid (metameric axis)

        Returns:
            Array of normalized direction vectors
        """
        dim = self.observer.dimension

        # Generate random directions
        directions = np.random.randn(num_directions, dim)

        # Zero out the metameric axis to avoid sampling along it
        directions[:, avoid_axis] = 0

        # Normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-8)

        return directions

    def _sample_distances(self, num_samples: int, min_dist: float, max_dist: float) -> npt.NDArray:
        """
        Sample distances from the target color with stochasticity.

        Args:
            num_samples: Number of distance samples
            min_dist: Minimum distance
            max_dist: Maximum distance

        Returns:
            Array of sampled distances
        """
        # Use truncated normal distribution for distance sampling
        mean_dist = (min_dist + max_dist) / 2
        std_dist = self.stochastic_params['distance_noise_std']

        distances = np.random.normal(mean_dist, std_dist, num_samples)
        distances = np.clip(distances, min_dist, max_dist)

        return distances

    def _is_in_gamut(self, point: npt.NDArray, tolerance: float = 0.01) -> bool:
        """
        Check if a point is within the gamut with tolerance.

        Args:
            point: Point to check
            tolerance: Tolerance for boundary checking

        Returns:
            True if point is in gamut
        """
        try:
            # Convert to display basis to check gamut
            display_basis = self.color_space.convert(
                point.reshape(1, -1),
                self.color_space_output_type,
                ColorSpaceType.DISP_6P
            )

            # Check if all components are within [0, 1] with tolerance
            in_gamut = np.all((display_basis >= -tolerance) & (display_basis <= 1 + tolerance))
            return in_gamut
        except Exception:
            return False

    def _project_to_gamut_boundary(self, point: npt.NDArray, direction: npt.NDArray) -> npt.NDArray:
        """
        Project a point to the gamut boundary along a given direction.

        Args:
            point: Starting point
            direction: Direction to project along

        Returns:
            Point projected to gamut boundary
        """
        # Binary search to find gamut boundary
        low, high = 0.0, 1.0
        tolerance = 1e-6

        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            test_point = point + mid * direction

            if self._is_in_gamut(test_point):
                low = mid
            else:
                high = mid

            if high - low < tolerance:
                break

        return point + low * direction

    def _compute_spectral_diversity(self, points: npt.NDArray) -> float:
        """
        Compute spectral diversity score for a set of points.

        Args:
            points: Array of points

        Returns:
            Spectral diversity score
        """
        if len(points) < 2:
            return 0.0

        # Convert to cone space to measure spectral differences
        cone_points = self.color_space.convert(
            points,
            self.color_space_output_type,
            ColorSpaceType.CONE
        )

        # Compute pairwise distances in cone space
        distances = cdist(cone_points, cone_points, metric='euclidean')

        # Return mean pairwise distance as diversity measure
        mask = ~np.eye(len(distances), dtype=bool)
        return np.mean(distances[mask])

    def get_metameric_candidates(self,
                                 inside_color: npt.NDArray,
                                 outside_color: npt.NDArray,
                                 num_candidates_per: int = 10) -> npt.NDArray:
        """
        Generate candidate metameric pairs by sampling around target colors.

        Args:
            inside_color: Target color for inside cone response
            outside_color: Target color for outside cone response
            num_candidates_per: Number of candidates to generate per color

        Returns:
            Array of candidate metameric pairs of size (2, num_candidates_per, dim)
        """
        # Ensure input colors are in the correct format
        if inside_color.ndim == 1:
            inside_color = inside_color.reshape(1, -1)
        if outside_color.ndim == 1:
            outside_color = outside_color.reshape(1, -1)

        # Convert to display space if needed
        if self.color_space_output_type != ColorSpaceType.DISP:
            inside_disp = self.color_space.convert(
                inside_color, self.color_space_output_type, ColorSpaceType.DISP
            )
            outside_disp = self.color_space.convert(
                outside_color, self.color_space_output_type, ColorSpaceType.DISP
            )
        else:
            inside_disp = inside_color
            outside_disp = outside_color

        # Sample random directions avoiding metameric axis
        directions = self._sample_random_directions(
            self.stochastic_params['num_directions'],
            self.metameric_axis
        )

        # Sample distances with stochasticity
        distances = self._sample_distances(
            self.stochastic_params['num_directions'],
            self.stochastic_params['min_distance'],
            self.stochastic_params['max_distance']
        )

        # Generate candidates for inside color
        inside_candidates = []
        for i in range(num_candidates_per):
            # Select random direction and distance
            direction = directions[i % len(directions)]
            distance = distances[i % len(distances)]

            # Add noise to direction
            noise = np.random.normal(0, self.stochastic_params['direction_noise_std'], direction.shape)
            noisy_direction = direction + noise
            noisy_direction = noisy_direction / (np.linalg.norm(noisy_direction) + 1e-8)

            # Generate candidate point
            candidate = inside_disp[0] + distance * noisy_direction

            # Project to gamut if needed
            if not self._is_in_gamut(candidate):
                candidate = self._project_to_gamut_boundary(inside_disp[0], noisy_direction)

            inside_candidates.append(candidate)

        # Generate candidates for outside color
        outside_candidates = []
        for i in range(num_candidates_per):
            # Select random direction and distance
            direction = directions[i % len(directions)]
            distance = distances[i % len(distances)]

            # Add noise to direction
            noise = np.random.normal(0, self.stochastic_params['direction_noise_std'], direction.shape)
            noisy_direction = direction + noise
            noisy_direction = noisy_direction / (np.linalg.norm(noisy_direction) + 1e-8)

            # Generate candidate point
            candidate = outside_disp[0] + distance * noisy_direction

            # Project to gamut if needed
            if not self._is_in_gamut(candidate):
                candidate = self._project_to_gamut_boundary(outside_disp[0], noisy_direction)

            outside_candidates.append(candidate)

        # Convert back to output space if needed
        if self.color_space_output_type != ColorSpaceType.DISP:
            inside_candidates = self.color_space.convert(
                np.array(inside_candidates),
                ColorSpaceType.DISP,
                self.color_space_output_type
            )
            outside_candidates = self.color_space.convert(
                np.array(outside_candidates),
                ColorSpaceType.DISP,
                self.color_space_output_type
            )
        else:
            inside_candidates = np.array(inside_candidates)
            outside_candidates = np.array(outside_candidates)

        # Stack inside_candidates and outside_candidates into a (2, N, 4) array
        return np.stack([inside_candidates, outside_candidates], axis=0)

    def evaluate_metameric_quality(self,
                                   inside_candidates: npt.NDArray,
                                   outside_candidates: npt.NDArray) -> Dict[str, Any]:
        """
        Evaluate the quality of metameric candidates.

        Args:
            inside_candidates: Array of inside color candidates
            outside_candidates: Array of outside color candidates

        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to cone space for evaluation
        inside_cones = self.color_space.convert(
            inside_candidates, self.color_space_output_type, ColorSpaceType.CONE
        )
        outside_cones = self.color_space.convert(
            outside_candidates, self.color_space_output_type, ColorSpaceType.CONE
        )

        # Compute cone response differences
        cone_diffs = np.linalg.norm(inside_cones - outside_cones, axis=1)

        # Compute spectral diversity
        inside_diversity = self._compute_spectral_diversity(inside_candidates)
        outside_diversity = self._compute_spectral_diversity(outside_candidates)

        # Compute gamut coverage
        inside_in_gamut = np.array([self._is_in_gamut(p) for p in inside_candidates])
        outside_in_gamut = np.array([self._is_in_gamut(p) for p in outside_candidates])

        return {
            'mean_cone_difference': np.mean(cone_diffs),
            'std_cone_difference': np.std(cone_diffs),
            'inside_spectral_diversity': inside_diversity,
            'outside_spectral_diversity': outside_diversity,
            'inside_gamut_coverage': np.mean(inside_in_gamut),
            'outside_gamut_coverage': np.mean(outside_in_gamut),
            'total_candidates': len(inside_candidates) + len(outside_candidates)
        }

    def update_stochastic_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update stochastic parameters.

        Args:
            new_params: Dictionary of new parameter values
        """
        self.stochastic_params.update(new_params)

    def get_metameric_axis_direction(self) -> npt.NDArray:
        """
        Get the metameric axis direction in the output color space.

        Returns:
            Normalized metameric axis direction
        """
        return self.color_space.get_metameric_axis_in(
            self.color_space_output_type,
            self.metameric_axis
        )
