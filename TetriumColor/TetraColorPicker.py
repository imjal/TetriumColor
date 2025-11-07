from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict

import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import *
from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes, Observer
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.PsychoPhys.Quest import Quest
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximumWidthAlongDirection


class ColorGenerator(ABC):

    @abstractmethod
    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        pass

    @abstractmethod
    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float] | None:
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        # This is a test generator that doesn't use real color spaces
        # Return dummy values for compatibility
        dummy_cone = np.array([0.5, 0.5, 0.5, 0.5])
        dummy_color_space = None  # This will need to be handled by callers
        dummy_difference = 0.1  # Dummy metamer difference
        return dummy_cone, dummy_cone, dummy_color_space, dummy_difference

    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float] | None:
        # This is a test generator that doesn't use real color spaces
        # Return dummy values for compatibility
        dummy_cone = np.array([0.5, 0.5, 0.5, 0.5])
        dummy_color_space = None  # This will need to be handled by callers
        dummy_difference = 0.1  # Dummy metamer difference
        return dummy_cone, dummy_cone, dummy_color_space, dummy_difference


class QuestColorGenerator(ColorGenerator):
    """Adaptive threshold color generator using Quest in DISP space.

    This generator samples thresholds in different chromatic directions using
    the Quest adaptive algorithm. It can operate in two modes:
    1. Cone-shift-based: Sample metameric axes for top observer genotypes
    2. Full-sphere: Fibonacci sphere in DISP space + cone-shift directions
    """

    def __init__(self, color_space: ColorSpace,
                 background_luminance: float = 0.5,
                 mode: str = 'cone_shift',
                 num_genotypes: int = 8,
                 trials_per_direction: int = 20,
                 sex: str = 'both',
                 quest_params: Optional[Dict] = None,
                 metameric_axes: Optional[List[int]] = [2]):
        """Initialize Quest-based color generator.

        Args:
            color_space: ColorSpace object for the observer being tested
            background_luminance: Background luminance level in VSH space (0-1)
            mode: 'cone_shift' for genotype-based directions or 'full_sphere' for uniform + genotype sampling
            num_genotypes: Number of top genotypes to use (default 8, gives 32 directions for 4D)
            trials_per_direction: Number of trials per direction
            sex: Population to sample genotypes from ('male', 'female', or 'both')
            quest_params: Optional dictionary of Quest parameters (tGuess, tGuessSd, pThreshold, beta, delta, gamma)
            metameric_axes: Optional list of metameric axes to test (e.g., [1] for only testing 547nm cone). If None, tests all axes.
        """
        self.color_space = color_space
        self.background_luminance = background_luminance
        self.mode = mode
        self.num_genotypes = num_genotypes
        self.trials_per_direction = trials_per_direction
        self.sex = sex
        self.metameric_axes = metameric_axes if metameric_axes is not None else list(range(color_space.dim))

        # Default Quest parameters
        default_quest_params = {
            'tGuess': -1.5,  # log10 of initial threshold guess (around 3% saturation)
            'tGuessSd': 0.5,  # standard deviation of initial guess
            'pThreshold': 0.75,  # threshold criterion (75% correct)
            'beta': 3.5,  # steepness of psychometric function
            'delta': 0.01,  # lapse rate
            'gamma': 0.5  # guess rate (2AFC)
        }
        if quest_params:
            default_quest_params.update(quest_params)
        self.quest_params = default_quest_params

        # Initialize ObserverGenotypes for direction generation
        # Note: dimensions is M/L cones only (S cone added automatically)
        self.observer_genotypes = ObserverGenotypes(
            dimensions=[self.color_space.dim - 1],
            seed=42
        )

        self.background = np.ones(self.color_space.dim) * 0.5

        # Generate sampling directions (which include max points)
        self.directions, self.direction_metadata = self._generate_directions()

        # Initialize Quest objects for each direction
        self.quest_objects = [
            Quest(**self.quest_params) for _ in range(len(self.directions))
        ]

        # Track current direction and trial counts
        self.current_direction_idx = 0
        self.trials_completed = [0] * len(self.directions)
        self.total_trials = 0

        # Store threshold estimates
        self.thresholds = {}

    def _generate_directions(self) -> Tuple[List[npt.NDArray], List[Dict]]:
        """Generate chromatic sampling directions based on mode.

        Returns:
            Tuple of (directions, metadata) where directions are in DISP space
            and metadata contains genotype and metameric_axis info
        """
        if self.mode == 'cone_shift':
            return self._generate_cone_shift_directions()
        elif self.mode == 'full_sphere':
            return self._generate_full_sphere_directions()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_cone_shift_directions(self) -> Tuple[List[npt.NDArray], List[Dict]]:
        """Generate directions based on metameric axes of top observer genotypes.

        For each of the top N genotypes, generates a direction for each metameric axis.
        E.g., 8 genotypes × 4 dimensions = 32 directions

        Returns:
            Tuple of (directions, metadata)
        """
        # Get top genotypes by probability
        genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=0.95, sex=self.sex
        )[:self.num_genotypes]

        print(f"Using {len(genotypes)} top genotypes for direction generation")

        directions = []
        metadata = []

        for genotype in genotypes:
            # Create color space for this genotype
            genotype_cs = self.observer_genotypes.get_color_space_for_peaks(
                genotype,
                display_primaries=self.color_space.display_primaries
            )

            # For each metameric axis (each cone dimension)
            for metameric_axis in [2]:
                # Get metameric direction in DISP space
                direction = genotype_cs.get_metameric_axis_in(
                    ColorSpaceType.DISP,
                    metameric_axis_num=metameric_axis
                )

                max_point_in_DISP, _, _ = genotype_cs.get_maximal_pair_in_disp_from_pt(
                    self.background, metameric_axis=metameric_axis, output_space=ColorSpaceType.DISP)

                max_distance = np.linalg.norm(max_point_in_DISP - self.background)

                direction = direction / np.linalg.norm(direction)

                directions.append(direction * max_distance)
                metadata.append({
                    'genotype': genotype,
                    'metameric_axis': metameric_axis,
                    'type': 'cone_shift',
                })

        print(f"Generated {len(directions)} directions from cone shifts")
        return directions, metadata

    def _generate_full_sphere_directions(self) -> Tuple[List[npt.NDArray], List[Dict]]:
        """Generate Fibonacci sphere in DISP space + cone-shift directions.

        Returns:
            Tuple of (directions, metadata)
        """
        # First, get all cone-shift directions
        cone_directions, cone_metadata = self._generate_cone_shift_directions()

        # Then add Fibonacci sphere directions in DISP space
        dim = self.color_space.dim
        num_sphere_points = 50  # Maximum sphere points

        sphere_directions = []
        sphere_metadata = []

        if dim == 3:
            # 2D sphere (circle) for trichromats
            angles = np.linspace(0, 2*np.pi, num_sphere_points, endpoint=False)
            for i, angle in enumerate(angles):
                direction = np.array([np.cos(angle), np.sin(angle), 0])
                direction = direction / np.linalg.norm(direction)
                max_point_in_DISP, _ = np.array(FindMaximumWidthAlongDirection(direction, np.eye(self.dim)))
                max_distance = np.linalg.norm(max_point_in_DISP - self.background)
                sphere_directions.append(direction * max_distance)
                sphere_metadata.append({
                    'genotype': None,
                    'metameric_axis': None,
                    'type': 'sphere',
                    'index': i,
                })

        elif dim == 4:
            # 3D sphere (Fibonacci) for tetrachromats
            phi = np.pi * (3. - np.sqrt(5.))  # golden angle

            for i in range(num_sphere_points):
                y = 1 - (i / float(num_sphere_points - 1)) * 2
                radius = np.sqrt(1 - y * y)
                theta = phi * i

                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                w = y

                direction = np.array([x, z, w, 0])  # Leave one dimension as 0
                direction = direction / np.linalg.norm(direction)
                sphere_directions.append(direction)
                max_point_in_DISP, _ = np.array(FindMaximumWidthAlongDirection(direction, np.eye(self.dim)))
                max_distance = np.linalg.norm(max_point_in_DISP - self.background)
                sphere_metadata.append({
                    'genotype': None,
                    'metameric_axis': None,
                    'type': 'sphere',
                    'index': i,
                    'max_distance': max_distance
                })
        else:
            # Higher dimensions: random sampling
            for i in range(num_sphere_points):
                direction = np.random.randn(dim)
                direction = direction / np.linalg.norm(direction)
                max_point_in_DISP, _ = np.array(FindMaximumWidthAlongDirection(direction, np.eye(self.dim)))
                max_distance = np.linalg.norm(max_point_in_DISP - self.background)
                sphere_directions.append(direction)
                sphere_metadata.append({
                    'genotype': None,
                    'metameric_axis': None,
                    'type': 'sphere',
                    'index': i,
                    'max_distance': max_distance
                })

        # Combine cone-shift and sphere directions
        all_directions = cone_directions + sphere_directions
        all_metadata = cone_metadata + sphere_metadata

        print(
            f"Generated {len(cone_directions)} cone-shift + {len(sphere_directions)} sphere = {len(all_directions)} total directions")

        return all_directions, all_metadata

    def get_num_samples(self) -> int:
        """Get total number of samples (directions × trials per direction)."""
        return len(self.directions) * self.trials_per_direction

    def _select_next_direction(self) -> int:
        """Select next direction to sample (interleaved sampling)."""
        # Find directions that still need trials
        incomplete_directions = [
            i for i, count in enumerate(self.trials_completed)
            if count < self.trials_per_direction
        ]

        if not incomplete_directions:
            return -1  # All directions complete

        # Interleave: cycle through incomplete directions
        return incomplete_directions[self.total_trials % len(incomplete_directions)]

    def _disp_direction_to_point(self, background_disp: npt.NDArray, disp_direction: npt.NDArray, distance: float) -> npt.NDArray:
        """Convert DISP direction + distance to DISP point.

        Args:
            background_disp: Background point in DISP space
            disp_direction: Normalized direction in DISP space
            distance: Distance from background in DISP space

        Returns:
            DISP coordinates
        """
        # Move from background in the direction by the distance
        disp_point = background_disp + disp_direction * distance

        # Clip to valid DISP range [0, 1]
        disp_point = np.clip(disp_point, 0, 1)

        return disp_point

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        """Get first color stimulus."""
        self.current_direction_idx = 0
        return self._get_color_for_direction(self.current_direction_idx)

    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float] | None:
        """Get next color based on previous result.

        Args:
            previous_result: Result from previous trial

        Returns:
            Tuple of (background_cone, test_cone, color_space, saturation) or None if done
        """
        if self.current_direction_idx < 0 or self.current_direction_idx >= len(self.directions):
            return None

        # Update Quest with previous response
        quest = self.quest_objects[self.current_direction_idx]

        # Convert response to Quest format (0=incorrect, 1=correct)
        # ColorTestResult.Success = 1, ColorTestResult.Failure = 0
        response = previous_result.value

        # Get the intensity that was tested (stored in previous trial)
        if hasattr(self, '_last_intensity'):
            quest.update(self._last_intensity, response)

        # Update trial counter
        self.trials_completed[self.current_direction_idx] += 1
        self.total_trials += 1

        # Select next direction
        next_direction_idx = self._select_next_direction()

        if next_direction_idx < 0:
            # All trials complete, compute final thresholds
            self._compute_final_thresholds()
            return None

        self.current_direction_idx = next_direction_idx
        return self._get_color_for_direction(self.current_direction_idx)

    def _get_color_for_direction(self, direction_idx: int) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        """Get color stimulus for a specific direction."""
        direction_vec = self.directions[direction_idx]  # This is already scaled to max_distance
        background_disp = self.background
        quest = self.quest_objects[direction_idx]

        # Get recommended intensity from Quest (in log10 space)
        # Quest returns log10 of proportion of maximum
        log_proportion = quest.quantile()
        proportion = 10 ** log_proportion

        # CRITICAL: Clip proportion to [0, 1] since directions are already scaled to max_distance
        proportion = np.clip(proportion, 0.0, 1.0)

        # Store the CLIPPED log proportion for Quest update (so Quest knows what we actually tested)
        self._last_intensity = np.log10(np.maximum(proportion, 1e-10))  # Avoid log(0)

        # Get test point in DISP space
        # direction_vec is already scaled by max_distance, so proportion directly scales it
        test_disp = self._disp_direction_to_point(background_disp, direction_vec, proportion)

        # Convert both to cone space
        background_cone = self.color_space.convert(
            np.array([background_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
        test_cone = self.color_space.convert(
            np.array([test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

        # Compute DISP distance from background (this is what we're thresholding)
        disp_distance = np.linalg.norm(test_disp - background_disp)

        return background_cone, test_cone, self.color_space, disp_distance

    def _compute_final_thresholds(self):
        """Compute final threshold estimates for all directions."""
        for i, (direction_vec, quest, metadata) in enumerate(
                zip(self.directions, self.quest_objects, self.direction_metadata)):

            # Quest threshold is log10 of proportion
            threshold_log_proportion = quest.mean()
            threshold_proportion = 10 ** threshold_log_proportion

            # Scale to actual distance
            threshold_distance = threshold_proportion * np.linalg.norm(direction_vec)

            sd_log = quest.sd()

            self.thresholds[i] = {
                'direction': direction_vec,  # Normalized direction vector
                'background': self.background,  # Background point (origin)
                'threshold_distance': threshold_distance,  # Actual distance in DISP space
                'threshold_proportion': threshold_proportion,  # Proportion of max distance
                'threshold_log_proportion': threshold_log_proportion,  # Log10 of proportion
                'max_distance': np.linalg.norm(direction_vec),  # Maximum displayable distance
                'sd_log': sd_log,
                'trials': self.trials_completed[i],
                'genotype': metadata.get('genotype'),
                'metameric_axis': metadata.get('metameric_axis'),
                'type': metadata.get('type')
            }

    def get_thresholds(self) -> Dict:
        """Get dictionary of threshold estimates for all directions."""
        if not self.thresholds:
            self._compute_final_thresholds()
        return self.thresholds

    def export_thresholds(self, filename: str):
        """Export thresholds to CSV file."""
        import csv

        thresholds = self.get_thresholds()

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            dim = self.color_space.dim
            max_point_cols = [f'max_pt_{i}' for i in range(dim)]
            direction_cols = [f'dir_{i}' for i in range(dim)]
            background_cols = [f'bg_{i}' for i in range(dim)]
            writer.writerow(['direction_idx'] + max_point_cols + direction_cols + background_cols +
                            ['threshold_distance', 'threshold_proportion', 'threshold_log_proportion',
                             'max_distance', 'sd_log', 'trials',
                             'genotype', 'metameric_axis', 'type'])

            # Data
            for idx in sorted(thresholds.keys()):
                data = thresholds[idx]
                genotype_str = ','.join(map(str, data['genotype'])) if data['genotype'] else 'None'
                row = [idx] + list(data['direction']) + list(data['background']) + [
                    data['threshold_distance'], data['threshold_proportion'],
                    data['threshold_log_proportion'], data['max_distance'],
                    data['sd_log'], data['trials'],
                    genotype_str, data['metameric_axis'], data['type']
                ]
                writer.writerow(row)

        print(f"Thresholds exported to {filename}")


class GeneticCDFTestColorGenerator(ColorGenerator):
    def __init__(self, sex: str, percentage_screened: float, peak_to_test: float = 547, metameric_axis: int = 2, luminance: float = 1.0, saturation: float = 0.5, dimensions: Optional[List[int]] = [3], seed: int = 42, extra_first_genotype: int = 4, **kwargs):
        """Color Generator that samples from the most common trichromatic phenotypes, and tests for the presence of a given peak.

        Args:
            sex (str): 'male' or 'female'
            percentage_screened (float): Percentage of the population to screen
            seed (int): Seed for the random number generator
            dimensions (Optional[List[int]], optional): Dimensions to screen. Defaults to [2], which corresponds to trichromats (S-cone already counted)
            peak_to_test (float, optional): Peak to test for. Defaults to 547, the functional peak.
        """
        self.percentage_screened = percentage_screened
        self.observer_genotypes = ObserverGenotypes(dimensions=dimensions, seed=seed)
        self.metameric_axis = metameric_axis

        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=self.percentage_screened, sex=sex)

        print("Genotypes: ", self.genotypes)

        self.color_spaces = [self.observer_genotypes.get_color_space_for_peaks(
            genotype + (peak_to_test,), **kwargs) for genotype in self.genotypes if peak_to_test not in genotype]

        # quick hack to test the first genotype 4 times
        self.color_spaces = self.color_spaces[:1] * extra_first_genotype + self.color_spaces[1:]

        self.color_samplers = [ColorSampler(color_space, cubemap_size=5).output_cubemap_values(
            luminance, saturation, ColorSpaceType.DISP)[4] for color_space in self.color_spaces]

        self.current_idx = 0

        self.num_samples = len(self.color_spaces)

    def get_num_samples(self) -> int:
        """Get the number of samples in the color generator.

        Returns:
            int: The number of samples in the color generator.
        """
        return self.num_samples

    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace] | None:
        """Get a color from the color generator.

        Args:
            previous_result (ColorTestResult): The previous result of the color test.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, ColorSpace]: return inside/outside cone colors, with the associated color space
        """
        if self.current_idx >= self.num_samples:
            self.current_idx = 0
        return self.NewColor()

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        """Currently, we just return a color in a list down (non-adaptively)
        Raises:
            StopIteration: If no more genotypes to sample

        Returns:
            Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]: return inside/outside cone colors, color space, and metamer difference
        """
        if self.current_idx >= self.num_samples:
            raise StopIteration("No more genotypes to sample")
        color_space = self.color_spaces[self.current_idx]

        # Retry logic for finding valid metamers
        max_retries = 10
        for attempt in range(max_retries):
            random_idx = np.random.randint(0, len(self.color_samplers[self.current_idx]))
            point = self.color_samplers[self.current_idx][random_idx]
            inside_cone, outside_cone, metamer_difference = color_space.get_maximal_pair_in_disp_from_pt(point)
            # inside_cone, outside_cone = color_space.get_maximal_metamer_pair_in_disp(
            #     metameric_axis=color_space.metameric_axis)
            # metamer_difference = abs(inside_cone[color_space.metameric_axis] - outside_cone[color_space.metameric_axis])
            # print("inside cone: ", inside_cone)
            # print("outside cone: ", outside_cone)
            # print("metamer difference: ", metamer_difference)

            if metamer_difference > 0.01:
                print("Metamer difference: ", metamer_difference)
                self.current_idx += 1
                return inside_cone, outside_cone, color_space, metamer_difference

        # If we couldn't find a valid metamer after retries, raise an exception
        # raise RuntimeError(f"Could not find valid metamer after {max_retries} attempts for genotype {self.current_idx}")
        print(f"Could not find valid metamer after {max_retries} attempts for genotype {self.current_idx}")
        return inside_cone, outside_cone, color_space, metamer_difference


class GeneticColorPicker:
    def __init__(self, sex: str, percentage_screened: float, peak_to_test: float = 547,
                 luminance: float = 1.0, saturation: float = 0.5,
                 dimensions: Optional[List[int]] = [3], seed: int = 42, **kwargs):
        """Color picker that samples from the most common trichromatic phenotypes.

        Args:
            sex (str): 'male' or 'female'
            percentage_screened (float): Percentage of the population to screen
            peak_to_test (float, optional): Peak to test for. Defaults to 547, the functional peak.
            metameric_axis (int, optional): Metameric axis to use. Defaults to 2.
            luminance (float, optional): Luminance level. Defaults to 1.0.
            saturation (float, optional): Saturation level. Defaults to 0.5.
            dimensions (Optional[List[int]], optional): Dimensions to screen. Defaults to [1, 2].
            seed (int): Seed for the random number generator
        """
        self.observer_genotypes = ObserverGenotypes(dimensions=dimensions, seed=seed)
        self.luminance = luminance
        self.saturation = saturation

        # Get genotypes covering the target probability
        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=percentage_screened, sex=sex)

        # Create mapping from genotype -> [color_space, color_sampler]
        self.genotype_mapping: Dict[Tuple, Tuple[ColorSpace, List[npt.NDArray]]] = {}

        for genotype in self.genotypes:
            if peak_to_test not in genotype:
                # Create color space with the peak to test added
                color_space = self.observer_genotypes.get_color_space_for_peaks(
                    genotype + (peak_to_test,), **kwargs)

                # Create color sampler and get cubemap values
                color_samplers = []
                for metameric_axis in range(4):
                    color_samplers.append(ColorSampler(color_space, cubemap_size=5)
                                          )
                self.genotype_mapping[genotype] = [color_space, color_samplers]

        self.list_of_genotypes = list(self.genotype_mapping.keys())

    def GetGenotypes(self) -> List[Tuple]:
        """Get the list of genotypes.

        Returns:
            List[Tuple]: The list of genotypes.
        """
        return self.genotypes

    def GetMetamericPair(self, genotype: Tuple, metameric_axis: int = None) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace]:
        """Get a metameric pair for a given genotype.

        Args:
            genotype (Tuple): The genotype to get a metameric pair for
            metameric_axis (int, optional): Metameric axis to use. If None, uses the color space's default.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, ColorSpace]: inside_cone, outside_cone, color_space
        """
        if genotype not in self.genotype_mapping:
            raise ValueError(f"Genotype {genotype} not found in mapping")

        color_space, color_samplers = self.genotype_mapping[genotype]

        grid_points = color_samplers[metameric_axis].output_cubemap_values(
            self.luminance, self.saturation, ColorSpaceType.DISP, metameric_axis=metameric_axis)[4]
        max_diff = 0.0
        max_diff_idx = 0
        for retry in range(10):
            random_idx = np.random.randint(0, len(grid_points))
            point = grid_points[random_idx]
            inside_cone, outside_cone, _ = color_space.get_maximal_pair_in_disp_from_pt(
                point, metameric_axis=metameric_axis)
            if inside_cone[metameric_axis] - outside_cone[metameric_axis] > 0.02:
                return inside_cone, outside_cone, color_space
            else:
                max_diff = max(max_diff, inside_cone[metameric_axis] - outside_cone[metameric_axis])
                max_diff_idx = random_idx
        print(
            f"Could not find valid metamer after 10 retries for genotype {genotype} {metameric_axis}, but we need to return something.")
        point = grid_points[max_diff_idx]
        inside_cone, outside_cone, _ = color_space.get_maximal_pair_in_disp_from_pt(
            point, metameric_axis=metameric_axis)

        return inside_cone, outside_cone, color_space


class CircleGridGenerator:
    def __init__(self, scramble_prob: float, sex: str, percentage_screened: float, peak_to_test: float = 547,
                 luminance: float = 1.0, saturation: float = 0.5,
                 dimensions: Optional[List[int]] = [3], seed: int = 42, **kwargs):
        """Color picker that samples from the most common trichromatic phenotypes.

        Args:
            scramble_prob (float): Probability of scrambling the color.
            sex (str): 'male' or 'female'
            percentage_screened (float): Percentage of the population to screen
            peak_to_test (float, optional): Peak to test for. Defaults to 547, the functional peak.
            metameric_axis (int, optional): Metameric axis to use. Defaults to 2.
            luminance (float, optional): Luminance level. Defaults to 1.0.
            saturation (float, optional): Saturation level. Defaults to 0.5.
            dimensions (Optional[List[int]], optional): Dimensions to screen. Defaults to [1, 2].
            seed (int): Seed for the random number generator
        """
        self.observer_genotypes = ObserverGenotypes(dimensions=dimensions, seed=seed)
        self.luminance = luminance
        self.saturation = saturation
        self.scramble_prob = scramble_prob

        # Get genotypes covering the target probability
        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=percentage_screened, sex=sex)

        # Create mapping from genotype -> [color_space, color_sampler]
        self.genotype_mapping: Dict[Tuple, Tuple[ColorSpace, ColorSampler]] = {}

        for genotype in self.genotypes:
            if peak_to_test not in genotype:
                # Create color space with the peak to test added
                color_space = self.observer_genotypes.get_color_space_for_peaks(
                    genotype + (peak_to_test,), **kwargs)

                # Create color sampler and get cubemap values
                self.genotype_mapping[genotype] = [color_space, ColorSampler(color_space, cubemap_size=5)]

        self.list_of_genotypes = list(self.genotype_mapping.keys())

    def GetGenotypes(self) -> List[Tuple]:
        """Get the list of genotypes.

        Returns:
            List[Tuple]: The list of genotypes.
        """
        return self.genotypes

    def GetImages(self, genotype: Tuple, metameric_axis: int, filenames: List[str], output_space: ColorSpaceType = ColorSpaceType.DISP_6P) -> List[Tuple[int, int]]:
        if genotype not in self.genotype_mapping:
            raise ValueError(f"Genotype {genotype} not found in mapping")

        _, color_sampler = self.genotype_mapping[genotype]

        image_tuples, idxs = color_sampler.get_hue_sphere_scramble(
            self.luminance, self.saturation, 4, metameric_axis=metameric_axis, scramble_prob=self.scramble_prob, output_space=output_space)

        if output_space == ColorSpaceType.DISP_6P:
            for i, (rgb, ocv) in enumerate(image_tuples):
                rgb.save(filenames[i] + "_RGB.png")
                ocv.save(filenames[i] + "_OCV.png")
            return idxs
        else:
            for i, im in enumerate(image_tuples):
                im.save(filenames[i] + "_SRGB.png")
            return idxs


if __name__ == "__main__":
    from TetriumColor.Measurement import load_primaries_from_csv
    import matplotlib.pyplot as plt
    primaries = load_primaries_from_csv("./measurements/2025-10-12/primaries")
    generator = CircleGridGenerator(scramble_prob=0.5,
                                    sex="female", percentage_screened=0.999, peak_to_test=547,
                                    luminance=1.0, saturation=0.5, dimensions=[2], seed=42,
                                    display_primaries=primaries)

    genotypes = generator.GetGenotypes()

    for genotype in genotypes:
        for metameric_axis in range(4):
            print(f"Generating images for genotype {genotype} metameric axis {metameric_axis}")
            filenames = [f"genotype_{genotype}_metameric_axis_{metameric_axis}_unscramble1",
                         f"genotype_{genotype}_metameric_axis_{metameric_axis}_unscramble2", f"genotype_{genotype}_metameric_axis_{metameric_axis}_scramble"]
            generator.GetImages(genotype, metameric_axis, filenames, output_space=ColorSpaceType.SRGB)
