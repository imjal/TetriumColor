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

    def __init__(self, sex: str = 'female',
                 percentage_screened: float = 0.999,
                 peak_to_test: float = 547,
                 luminance: float = 0.5,
                 saturation: float = 0.5,
                 dimensions: Optional[List[int]] = [3],
                 seed: int = 42,
                 mode: str = 'cone_shift',
                 num_genotypes: int = 8,
                 trials_per_direction: int = 20,
                 quest_params: Optional[Dict] = None,
                 metameric_axes: Optional[List[int]] = [2],
                 bipolar: bool = False,
                 degree: float = 4.0,
                 mcs_k: int = 0,
                 **kwargs):
        """Initialize Quest-based color generator.

        Args:
            sex: 'male', 'female', or 'both' for population to sample genotypes from
            percentage_screened: Percentage of population to screen (not used for Quest, kept for API compatibility)
            peak_to_test: Peak wavelength to test (not used for Quest, kept for API compatibility)
            luminance: Background luminance level (0-1)
            saturation: Saturation level (not used for Quest, kept for API compatibility)
            dimensions: Dimensions to use (e.g., [2] for trichromats)
            seed: Random seed
            mode: 'cone_shift' for genotype-based directions or 'full_sphere' for uniform + genotype sampling
            num_genotypes: Number of top genotypes to use (default 8, gives 32 directions for 4D)
            trials_per_direction: Number of trials per direction
            quest_params: Optional dictionary of Quest parameters (tGuess, tGuessSd, pThreshold, beta, delta, gamma)
            metameric_axes: Optional list of metameric axes to test (e.g., [2] for only testing 547nm cone). If None, tests all axes.
            bipolar: If True, sample in both direction and -direction, returning the -direction point instead of the background
            mcs_k: If > 0, use Method of Constant Stimuli with K equally-spaced intensity levels instead of Quest adaptive algorithm.
                   1 = max metamer only, K > 1 = linspace(0, 1, K). Bypasses Quest tracking entirely.
            **kwargs: Additional arguments including display_primaries
        """
        self.background_luminance = luminance
        self.mode = mode
        self.num_genotypes = num_genotypes
        self.trials_per_direction = trials_per_direction
        self.sex = sex
        self.peak_to_test = peak_to_test
        self.metameric_axes = metameric_axes if metameric_axes is not None else list(range(4))
        self.dim = 4
        self.bipolar = bipolar
        self.degree = degree
        self.mcs_k = mcs_k
        if mcs_k > 0:
            self.mcs_proportions = [1.0] if mcs_k <= 1 else list(np.linspace(0.0, 1.0, mcs_k))

        # Add degree to kwargs for observer creation
        kwargs['degree'] = degree

        # Default Quest parameters
        default_quest_params = {
            'tGuess': -0.046,  # log10 of initial threshold guess (90% of max - start at most visible end)
            'tGuessSd': 0.5,  # standard deviation of initial guess
            'pThreshold': 0.5,  # threshold criterion (50% correct), lower bc sensitivity
            'beta': 3.5,  # steepness of psychometric function
            'delta': 0.05,  # lapse rate
            'gamma': 0.25  # guess rate (4AFC)
        }
        if quest_params:
            default_quest_params.update(quest_params)
        self.quest_params = default_quest_params

        # Initialize ObserverGenotypes for direction generation
        # Note: dimensions is M/L cones only (S cone added automatically)
        self.observer_genotypes = ObserverGenotypes(
            dimensions=dimensions,
            seed=seed
        )

        # Get genotypes covering the target probability
        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=percentage_screened, sex=sex)

        # Create mapping from genotype -> [color_space, color_sampler]
        self.genotype_mapping: Dict[Tuple, Tuple[ColorSpace, List[npt.NDArray]]] = {}

        for genotype in self.genotypes:
            if len(genotype) == 1:  # testing for hard dichromats
                if 530 in genotype or 533 in genotype:  # protonope
                    genotype = genotype + (559,)
                else:
                    genotype = (530,) + genotype  # deuteranope
            elif len(genotype) == 2:  # testing for trichromats
                if peak_to_test not in genotype:
                    genotype = genotype + (peak_to_test,)
                else:
                    continue

            # Create color space with the peak to test added
            color_space = self.observer_genotypes.get_color_space_for_peaks(
                genotype, **kwargs)

            # Create color sampler and get cubemap values
            color_sampler = ColorSampler(color_space, cubemap_size=5)
            self.genotype_mapping[genotype] = color_space

        self.background = np.ones(dimensions[0] + 1) * 0.5

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
        directions = []
        metadata = []

        for genotype, genotype_cs in self.genotype_mapping.items():
            # Compute the sorted peak order (mirrors get_observer_for_peaks logic)
            peaks_with_s = sorted(set([420] + list(genotype)))

            # For each metameric axis (each cone dimension)
            for metameric_axis in self.metameric_axes:
                # Remap axis to the actual sorted index of peak_to_test (Q cone).
                # peak_to_test may sort to a different index than the caller assumed
                # (e.g. 547nm sorts to index 1 in [420,547,555,559], not index 2).
                if self.peak_to_test in peaks_with_s:
                    actual_axis = peaks_with_s.index(self.peak_to_test)
                else:
                    actual_axis = metameric_axis

                # Get metameric direction in DISP space
                direction = genotype_cs.get_metameric_axis_in(
                    ColorSpaceType.DISP,
                    metameric_axis_num=actual_axis
                )

                max_point_in_DISP, _, _ = genotype_cs.get_maximal_pair_in_disp_from_pt(
                    self.background, metameric_axis=actual_axis, output_space=ColorSpaceType.DISP)

                max_distance = np.linalg.norm(max_point_in_DISP - self.background)

                direction = direction / np.linalg.norm(direction)

                directions.append(direction * max_distance)
                metadata.append({
                    'genotype': genotype,
                    'metameric_axis': actual_axis,
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
        dim = self.dim
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
        if self.mcs_k > 0:
            return len(self.directions) * len(self.mcs_proportions) * self.trials_per_direction
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

    def _disp_direction_to_point(self, background_disp: npt.NDArray, disp_direction: npt.NDArray, proportion: float) -> npt.NDArray:
        """Convert DISP direction + distance to DISP point.

        Args:
            background_disp: Background point in DISP space
            disp_direction: Normalized direction in DISP space
            distance: Distance from background in DISP space

        Returns:
            DISP coordinates
        """
        # Move from background in the direction by the distance
        disp_point = background_disp + disp_direction * proportion

        # Clip to valid DISP range [0, 1]
        disp_point = np.clip(disp_point, 0, 1)

        return disp_point

    def _generate_mcs_trials(self):
        """Build shuffled MCS trial list: directions × proportions × repetitions."""
        trial_list = []
        for direction_idx in range(len(self.directions)):
            for proportion in self.mcs_proportions:
                for _ in range(self.trials_per_direction):
                    trial_list.append((direction_idx, proportion))
        np.random.shuffle(trial_list)
        self._mcs_trial_list = trial_list
        self._mcs_trial_idx = 0
        print(
            f"MCS trial list: {len(trial_list)} trials "
            f"({len(self.directions)} directions × {len(self.mcs_proportions)} levels "
            f"× {self.trials_per_direction} reps)")

    def _get_color_for_direction_mcs(self, direction_idx: int, proportion: float) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        """Get color stimulus for MCS mode at a fixed proportion of max distance."""
        direction_vec = self.directions[direction_idx]
        background_disp = self.background
        genotype_cs = self.genotype_mapping[self.direction_metadata[direction_idx]['genotype']]

        if self.bipolar:
            test_disp = self._disp_direction_to_point(background_disp, direction_vec, proportion)
            negative_test_disp = self._disp_direction_to_point(background_disp, -direction_vec, proportion)
            background_cone = genotype_cs.convert(
                np.array([negative_test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            test_cone = genotype_cs.convert(
                np.array([test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
        else:
            test_disp = self._disp_direction_to_point(background_disp, direction_vec, proportion)
            background_cone = genotype_cs.convert(
                np.array([background_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            test_cone = genotype_cs.convert(
                np.array([test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

        return background_cone, test_cone, genotype_cs, proportion

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        """Get first color stimulus."""
        if self.mcs_k > 0:
            self._generate_mcs_trials()
            direction_idx, proportion = self._mcs_trial_list[0]
            self.current_direction_idx = direction_idx
            return self._get_color_for_direction_mcs(direction_idx, proportion)
        self.current_direction_idx = 0
        return self._get_color_for_direction(self.current_direction_idx)

    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float] | None:
        """Get next color based on previous result.

        Args:
            previous_result: Result from previous trial

        Returns:
            Tuple of (background_cone, test_cone, color_space, saturation) or None if done
        """
        if self.mcs_k > 0:
            if not hasattr(self, '_mcs_trial_list'):
                self._generate_mcs_trials()
            self._mcs_trial_idx += 1
            if self._mcs_trial_idx >= len(self._mcs_trial_list):
                print("No more MCS trials")
                return None
            direction_idx, proportion = self._mcs_trial_list[self._mcs_trial_idx]
            self.current_direction_idx = direction_idx
            return self._get_color_for_direction_mcs(direction_idx, proportion)

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

    def GetCurrentTestInfo(self) -> Tuple:
        """Get the current info about the test.

        Returns:
            Tuple: The current genotype.
        """
        genotype = self.direction_metadata[self.current_direction_idx]['genotype']
        metameric_axis = self.direction_metadata[self.current_direction_idx]['metameric_axis']
        return genotype, metameric_axis

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

        genotype_cs = self.genotype_mapping[self.direction_metadata[direction_idx]['genotype']]

        if self.bipolar:
            # Sample in both direction and -direction
            # Get test point in positive direction
            test_disp = self._disp_direction_to_point(background_disp, direction_vec, proportion)
            # Get test point in negative direction
            negative_test_disp = self._disp_direction_to_point(background_disp, -direction_vec, proportion)

            # Convert both to cone space
            # Return negative direction point as "background" and positive direction point as "test"
            background_cone = genotype_cs.convert(
                np.array([negative_test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            test_cone = genotype_cs.convert(
                np.array([test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

            # For bipolar, return proportion (0-1) representing proportion of max_distance
            # The distance between the two points is 2 * (proportion * max_distance),
            # but we return the proportion of max_distance for consistency
        else:
            # Original behavior: sample in one direction, return background and test point
            # Get test point in DISP space
            # direction_vec is already scaled by max_distance, so proportion directly scales it
            test_disp = self._disp_direction_to_point(background_disp, direction_vec, proportion)

            # Convert both to cone space
            background_cone = genotype_cs.convert(
                np.array([background_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            test_cone = genotype_cs.convert(
                np.array([test_disp]), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

        # Return proportion (0-1) as intensity, representing proportion of max_distance
        # This is consistent with threshold_proportion and makes intensity comparable across directions
        return background_cone, test_cone, genotype_cs, proportion

    def _compute_final_thresholds(self):
        """Compute final threshold estimates for all directions."""
        for i, (direction_vec, quest, metadata) in enumerate(
                zip(self.directions, self.quest_objects, self.direction_metadata)):

            # Quest threshold is log10 of proportion
            threshold_log_proportion = quest.mean()
            threshold_proportion = 10 ** threshold_log_proportion

            # Scale to actual distance
            threshold_distance = threshold_proportion * np.linalg.norm(direction_vec)
            max_distance = np.linalg.norm(direction_vec)

            # Check if threshold is beyond displayable gamut
            beyond_gamut = threshold_proportion > 1.0

            sd_log = quest.sd()

            self.thresholds[i] = {
                'direction': direction_vec,  # Direction vector (scaled to max_distance)
                'background': self.background,  # Background point (origin)
                # Actual distance in DISP space (may exceed max_distance if beyond_gamut)
                'threshold_distance': threshold_distance,
                # Proportion of max distance (may be > 1.0 if beyond gamut)
                'threshold_proportion': threshold_proportion,
                'threshold_log_proportion': threshold_log_proportion,  # Log10 of proportion
                'max_distance': max_distance,  # Maximum displayable distance
                'beyond_gamut': beyond_gamut,  # True if threshold exceeds displayable gamut
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

            # Header - simplified to only essential columns
            writer.writerow(['direction_idx', 'threshold_distance', 'threshold_proportion',
                            'genotype', 'metameric_axis'])

            # Data
            for idx in sorted(thresholds.keys()):
                data = thresholds[idx]
                genotype_str = ','.join(map(str, data['genotype'])) if data['genotype'] else 'None'
                row = [
                    idx,
                    data['threshold_distance'],
                    data['threshold_proportion'],
                    genotype_str,
                    data['metameric_axis']
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
            luminance (float, optional): Luminance level. Defaults to 1.0.
            saturation (float, optional): Saturation level. Defaults to 0.5.
            dimensions (Optional[List[int]], optional): Dimensions to screen. Defaults to [1, 2].
            trials_per_direction (int, optional): Number of trials per direction. Defaults to 20.
            metameric_axes (List[int], optional): Metameric axes to use. Defaults to [2].
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


class GaussianObserverSimulator:
    """Simulated psychophysical observer for a trichromat tested on a 4D display.

    The observer has a null DIRECTION (null_theta, null_phi): any chromatic stimulus
    along this direction is completely invisible to them, regardless of amplitude.
    Detection is driven by the component of the stimulus PERPENDICULAR to the null
    direction.

    Psychometric function
    --------------------
    signal(θ, φ, r) = r · sin(angle between d(θ,φ) and d_null)

    P(θ, φ, r) = p_chance + (1 - p_chance) · (1 - exp(-signal² / (2·sigma²)))

    Properties
    ----------
    * P → p_chance as r → 0 for every direction ✓
    * P → p_chance along the null direction (sin = 0) for every r ✓
    * Threshold surface r*(θ,φ) = sigma·sqrt(2·ln 3) / sin(angle) — very peaked
      at (null_theta, null_phi) on the sphere.

    Parameters
    ----------
    null_theta, null_phi : float
        Spherical angles of the null direction in the 3D chromatic subspace.
    sigma : float
        Detection sensitivity width in chromatic amplitude units.
        Threshold at 90° from null = sigma·sqrt(2·ln 3) ≈ 1.48·sigma.
    p_chance : float
        Chance-level detection probability (0.25 for 4AFC).
    """

    def __init__(self, null_theta: float = 0.0, null_phi: float = 0.0,
                 null_r: float = None,   # unused, kept for API compatibility
                 sigma: float = 0.05, p_chance: float = 0.25, seed: int = 42,
                 null_direction: Optional[npt.NDArray] = None):
        if null_direction is None:
            self.d_null = np.array([
                np.sin(null_theta) * np.cos(null_phi),
                np.sin(null_theta) * np.sin(null_phi),
                np.cos(null_theta),
            ])
        else:
            self.d_null = np.asarray(null_direction, dtype=float)
            self.d_null = self.d_null / (np.linalg.norm(self.d_null) + 1e-12)
        self.sigma = sigma
        self.p_chance = p_chance
        self.rng = np.random.default_rng(seed)

    def p_detect(self, theta: float, phi: float, r: float) -> float:
        d_test = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        cos_angle = float(np.clip(np.dot(d_test, self.d_null), -1.0, 1.0))
        sin_angle = float(np.sqrt(max(1.0 - cos_angle ** 2, 0.0)))
        signal = r * sin_angle
        return self.p_chance + (1 - self.p_chance) * (1 - np.exp(-signal ** 2 / (2 * self.sigma ** 2)))

    def p_detect_direction(self, direction: npt.NDArray, r: float) -> float:
        d_test = np.asarray(direction, dtype=float)
        d_test = d_test / (np.linalg.norm(d_test) + 1e-12)
        cos_angle = float(np.clip(np.dot(d_test, self.d_null), -1.0, 1.0))
        sin_angle = float(np.sqrt(max(1.0 - cos_angle ** 2, 0.0)))
        signal = r * sin_angle
        return self.p_chance + (1 - self.p_chance) * (1 - np.exp(-signal ** 2 / (2 * self.sigma ** 2)))

    def simulate_response(self, theta: float, phi: float, r: float) -> int:
        p = self.p_detect(theta, phi, r)
        return int(self.rng.binomial(1, p))

    def simulate_direction_response(self, direction: npt.NDArray, r: float) -> int:
        p = self.p_detect_direction(direction, r)
        return int(self.rng.binomial(1, p))


class AEPsychThresholdContourGenerator(ColorGenerator):
    """Adaptive threshold contour estimator using AEPsych's GP classifier.

    Models f(a, b, r) -> P(detect) over the same 2D PCA patch used by
    scripts/simulation/null_direction_viewer.py.  The patch lives in observer-0's
    chromatic slice.  AEPsych samples square coordinates `(a, b) in [-1, 1]^2`;
    those are mapped into the elliptical patch before making stimuli.  Each
    sampled display direction is

        d(u, v) = normalize(pca_mean_null + u * pca_e1 + v * pca_e2)

    and the display point is

        w = w0 + chrom_basis @ (r * d(u, v)).

    `u` and `v` are tangent-plane coordinates inside the PCA ellipse.  `r` is
    the amplitude along that observer-0 slice direction.
    """

    def __init__(
        self,
        center_genotype: Optional[Tuple] = None,
        peak_to_test: float = 547,
        n_trials: int = 300,
        sex: str = 'both',
        luminance: float = 0.5,
        seed: int = 42,
        n_cmf_samples: int = 200,
        threshold_level: float = 0.75,
        n_sobol: int = 20,
        dimensions: Optional[List[int]] = None,
        patch_sigma_scale: float = 5.0,
        min_patch_major: float = 0.15,
        min_patch_minor: float = 0.06,
        max_radius: Optional[float] = 0.65,
        **kwargs,
    ):
        import torch
        from aepsych import GPClassificationModel, Strategy, SequentialStrategy
        from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
        from aepsych.acquisition import MCLevelSetEstimation

        self.n_trials = n_trials
        self.threshold_level = threshold_level
        self.trial_count = 0
        self.luminance = luminance
        self._last_x = None
        self.last_theta = None
        self.last_phi = None
        self.last_u = None
        self.last_v = None
        self.last_a = None
        self.last_b = None
        self.last_direction = None
        self.last_disp = None
        self.stimulus_disp_log = []
        self.last_r = None
        self._peak_to_test = peak_to_test
        self.patch_sigma_scale = patch_sigma_scale
        self.requested_max_radius = max_radius

        dims = dimensions if dimensions is not None else [3]
        self._og_wavelengths = np.arange(380, 781, 5)
        self.observer_genotypes = ObserverGenotypes(
            wavelengths=self._og_wavelengths,
            dimensions=dims,
            seed=seed,
        )
        if center_genotype is None:
            center_genotype = self.observer_genotypes.get_genotypes_covering_probability(
                target_probability=0.999,
                sex=sex,
            )[0]

        display_primaries = kwargs.get('display_primaries', None)
        center_genotype_full = tuple(sorted(tuple(center_genotype) + (peak_to_test,)))
        center_obs = self._create_observer_for_peaks(center_genotype_full)
        self.center_cs = ColorSpace(center_obs, display_primaries=display_primaries)

        q_axis = self._find_axis_for_peak(self.center_cs.observer, peak_to_test)
        self.q_axis = q_axis
        self.chrom_basis = self._build_chromatic_basis(self.center_cs, q_axis)

        # Match scripts/simulation/null_direction_viewer.py: the adapting point
        # is the physical display midgray, not ColorSpace.get_background().
        self.w0 = np.full(self.chrom_basis.shape[0], float(luminance))

        self._compute_patch_from_cmf_samples(
            dims, sex, seed, n_cmf_samples, display_primaries,
            patch_sigma_scale, min_patch_major, min_patch_minor,
        )
        r_min, r_max = self._compute_r_bounds_from_gamut()
        if max_radius is not None:
            r_max = min(float(max_radius), r_max)
        self.actual_max_radius = r_max

        self._lb_np = np.array([-1.0, -1.0, r_min])
        self._ub_np = np.array([1.0, 1.0, r_max])
        self.patch_bounds = np.array([
            [-self.patch_angle_major, self.patch_angle_major],
            [-self.patch_angle_minor, self.patch_angle_minor],
        ])

        lb = torch.tensor(self._lb_np, dtype=torch.float64)
        ub = torch.tensor(self._ub_np, dtype=torch.float64)

        sobol_gen = SobolGenerator(lb=lb, ub=ub, seed=seed)
        model = GPClassificationModel(dim=3)
        opt_gen = OptimizeAcqfGenerator(
            lb=lb, ub=ub,
            acqf=MCLevelSetEstimation,
            acqf_kwargs={'target': threshold_level},
        )
        n_adaptive = max(1, n_trials - n_sobol)
        s1 = Strategy(generator=sobol_gen, lb=lb, ub=ub,
                      outcome_types=['binary'], min_asks=n_sobol)
        s2 = Strategy(generator=opt_gen, lb=lb, ub=ub,
                      outcome_types=['binary'], model=model,
                      min_asks=n_adaptive)
        self.strategy = SequentialStrategy([s1, s2])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_axis_for_peak(observer: Observer, peak: float) -> int:
        for axis_idx, cone in enumerate(observer.sensors):
            if abs(cone.peak - peak) < 1.0:
                return axis_idx
        return 2

    @staticmethod
    def _build_chromatic_basis(cst: ColorSpace, q_axis: int) -> npt.NDArray:
        cone_to_disp = cst._get_cone_to_disp()
        disp_to_cone = np.linalg.inv(cone_to_disp)
        lum_cone = np.ones(cst.dim)
        lum_cone[q_axis] = 0.0
        lum_cone /= np.linalg.norm(lum_cone)
        lum_disp = disp_to_cone.T @ lum_cone
        _, _, Vt = np.linalg.svd(lum_disp.reshape(1, -1), full_matrices=True)
        return Vt[1:, :].T

    def _create_observer_for_peaks(self, peaks_with_q, params=None):
        """Create observers the same way null_direction_viewer.py does."""
        from TetriumColor.Observer.Observer import Cone, Observer as _Observer

        if params is None:
            params = {
                'od_lm': 0.5,
                'od_s': 0.4,
                'macular': 1.0,
                'lens': 1.0,
            }

        peaks = tuple(sorted(peaks_with_q))
        if 420 not in peaks:
            peaks = tuple(sorted((420,) + peaks))

        cones = []
        for peak in peaks:
            od = params['od_s'] if peak == 420 else params['od_lm']
            cone = Cone.templates['neitz'](
                self._og_wavelengths, peak
            ).with_preceptoral(
                od=od,
                macular=params['macular'],
                lens=params['lens'],
            )
            cone.peak = int(peak)
            cones.append(cone)

        return _Observer(cones, illuminant=None)

    def _observer_from_sample_params(self, genotype_3d, params, sampler):
        from TetriumColor.Observer.Observer import Cone, Observer as _Observer

        peaks_4d = tuple(sorted(set(genotype_3d) | {self._peak_to_test}))
        s_peak = 420
        if s_peak not in peaks_4d:
            peaks_4d = (s_peak,) + peaks_4d
        peaks_4d = tuple(sorted(peaks_4d))

        cones = []
        for peak in peaks_4d:
            od = float(params['od_s']) if peak == s_peak else float(params['od_lm'])
            cone = Cone.templates[sampler.template](sampler.wavelengths, peak).with_preceptoral(
                od=od,
                macular=float(params['macular']),
                lens=float(params['lens']),
            )
            cone.peak = int(peak)
            cones.append(cone)
        return _Observer(cones, illuminant=None)

    def _compute_patch_from_cmf_samples(
        self, dims, sex, seed, n_cmf_samples, display_primaries,
        patch_sigma_scale, min_patch_major, min_patch_minor,
    ) -> None:
        """Compute observer-0-slice PCA patch from projected CMF null directions."""
        from TetriumColor.Observer.CMFSampler import CMFSampler
        from TetriumColor import ColorSpace as _ColorSpace

        sampler = CMFSampler(
            self.observer_genotypes,
            wavelengths=self._og_wavelengths,
            seed=seed,
            top_n_genotypes=10,
        )
        try:
            samples_3d = sampler.sample(n_samples=n_cmf_samples, sex=sex)
        except Exception:
            samples_3d = []

        projected_dirs = []
        for _, params in samples_3d:
            try:
                observer_4d = self._observer_from_sample_params(
                    params['genotype'], params, sampler)
                obs_cs = _ColorSpace(
                    observer_4d,
                    display_primaries=display_primaries,
                    metameric_axis=2,
                )
                q_axis = self._find_axis_for_peak(obs_cs.observer, self._peak_to_test)
                meta_dir = obs_cs.get_metameric_axis_in(
                    ColorSpaceType.DISP, metameric_axis_num=q_axis)
                norm = np.linalg.norm(meta_dir)
                if norm < 1e-10:
                    continue
                meta_dir = meta_dir / norm
                alpha = self.chrom_basis.T @ meta_dir  # (3,) chromatic projection
                alpha_norm = float(np.linalg.norm(alpha))
                if alpha_norm < 1e-10:
                    continue
                projected_dirs.append(alpha / alpha_norm)
            except Exception:
                continue

        if len(projected_dirs) < 5:
            center_dir = self._center_null_direction()
            self.pca_mean_null = center_dir
            self.pca_e1, self.pca_e2 = self._orthonormal_tangent_basis(center_dir)
            self.patch_angle_major = min_patch_major
            self.patch_angle_minor = min_patch_minor
            return

        dirs = np.array(projected_dirs)
        mean_dir = dirs.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        tangent = dirs - (dirs @ mean_dir)[:, None] * mean_dir

        _, _, Vt = np.linalg.svd(tangent, full_matrices=False)
        e1 = Vt[0]
        e2 = Vt[1]

        spread_e1 = float(np.std(tangent @ e1) * patch_sigma_scale)
        spread_e2 = float(np.std(tangent @ e2) * patch_sigma_scale)

        self.pca_mean_null = mean_dir
        self.pca_e1 = e1 / np.linalg.norm(e1)
        self.pca_e2 = e2 / np.linalg.norm(e2)
        self.patch_angle_major = max(spread_e1, min_patch_major)
        self.patch_angle_minor = max(spread_e2, min_patch_minor)

    def _center_null_direction(self) -> npt.NDArray:
        meta_dir = self.center_cs.get_metameric_axis_in(
            ColorSpaceType.DISP, metameric_axis_num=self.q_axis)
        meta_dir = meta_dir / (np.linalg.norm(meta_dir) + 1e-12)
        alpha = self.chrom_basis.T @ meta_dir
        return alpha / (np.linalg.norm(alpha) + 1e-12)

    @staticmethod
    def _orthonormal_tangent_basis(mean_dir: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(candidate, mean_dir)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        e1 = candidate - np.dot(candidate, mean_dir) * mean_dir
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(mean_dir, e1)
        e2 /= np.linalg.norm(e2)
        return e1, e2

    def _compute_r_bounds_from_gamut(self) -> Tuple[float, float]:
        """r bounds from gamut extents across the PCA patch."""
        vals = np.linspace(-1.0, 1.0, 21)
        r_max_values = []
        for a in vals:
            for b in vals:
                u, v = self.patch_coords_from_model(a, b)
                d = self.direction_from_patch(u, v)
                r_max_values.append(self._compute_r_max(self.chrom_basis @ d))
        r_max = float(np.min(r_max_values)) if r_max_values else 0.01
        return 0.001, float(min(max(r_max, 0.01), 1.5))

    def _compute_r_max(self, dir_4d: npt.NDArray) -> float:
        """Max r such that w0 + dir_4d * r remains in [0, 1]^4."""
        r_max = np.inf
        for i in range(4):
            if dir_4d[i] > 1e-12:
                r_max = min(r_max, (1.0 - self.w0[i]) / dir_4d[i])
            elif dir_4d[i] < -1e-12:
                r_max = min(r_max, (0.0 - self.w0[i]) / dir_4d[i])
        return float(r_max) if r_max < 1e9 else 1.0

    def _in_patch(self, u: float, v: float) -> bool:
        a = max(self.patch_angle_major, 1e-8)
        b = max(self.patch_angle_minor, 1e-8)
        return (u / a) ** 2 + (v / b) ** 2 <= 1.0

    @staticmethod
    def _square_to_disk(a: float, b: float) -> Tuple[float, float]:
        """Concentric square-to-disk map, avoiding rejected AEPsych asks."""
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            return 0.0, 0.0
        if abs(a) > abs(b):
            r = a
            theta = (np.pi / 4.0) * (b / a)
        else:
            r = b
            theta = (np.pi / 2.0) - (np.pi / 4.0) * (a / b)
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    def patch_coords_from_model(self, a: float, b: float) -> Tuple[float, float]:
        disk_x, disk_y = self._square_to_disk(
            float(np.clip(a, -1.0, 1.0)),
            float(np.clip(b, -1.0, 1.0)),
        )
        return self.patch_angle_major * disk_x, self.patch_angle_minor * disk_y

    def direction_from_patch(self, u: float, v: float) -> npt.NDArray:
        d = self.pca_mean_null + u * self.pca_e1 + v * self.pca_e2
        return d / (np.linalg.norm(d) + 1e-12)

    def _disp_from_patch(self, u: float, v: float, r: float) -> npt.NDArray:
        d3 = self.direction_from_patch(u, v)
        return self.w0 + self.chrom_basis @ (d3 * r)

    def _in_gamut(self, w: npt.NDArray) -> bool:
        return bool(np.all(w >= 0) and np.all(w <= 1))

    def _posterior_detect_prob(self, a: float, b: float, r: float) -> float:
        import torch
        x_q = torch.tensor([[a, b, r]], dtype=torch.float32)
        mean, _ = self.strategy.model.predict(x_q, probability_space=True)
        return float(mean.item())

    def _estimate_threshold_radius(
        self, a: float, b: float, r_hi: float
    ) -> Tuple[float, bool, bool, float, float, float]:
        """Estimate threshold radius and report whether a crossing exists.

        Returns
        -------
        r_star
            Estimated radius, or the nearest sampled bound when no crossing
            exists inside [r_lo, r_hi].
        threshold_found
            True when posterior P(detect) crosses threshold_level.
        lower_saturated
            True when even the lower radius is already above threshold.
        p_lo, p_hi, p_star
            Posterior detection probabilities at the relevant radii.
        """
        r_lo = float(self._lb_np[2])
        r_hi = float(r_hi)
        p_lo = self._posterior_detect_prob(a, b, r_lo)
        p_hi = self._posterior_detect_prob(a, b, r_hi)

        if p_lo >= self.threshold_level:
            return r_lo, False, True, p_lo, p_hi, p_lo
        if p_hi < self.threshold_level:
            return r_hi, False, False, p_lo, p_hi, p_hi

        lo, hi = r_lo, r_hi
        p_star = p_hi
        for _ in range(25):
            mid = (lo + hi) / 2.0
            p_mid = self._posterior_detect_prob(a, b, mid)
            if p_mid < self.threshold_level:
                lo = mid
            else:
                hi = mid
                p_star = p_mid
        return (lo + hi) / 2.0, True, False, p_lo, p_hi, p_star

    # ------------------------------------------------------------------
    # ColorGenerator interface
    # ------------------------------------------------------------------

    def get_num_samples(self) -> int:
        return self.n_trials

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]:
        x = self.strategy.gen()
        a, b, r = float(x[0, 0]), float(x[0, 1]), float(x[0, 2])
        u, v = self.patch_coords_from_model(a, b)
        w = self._disp_from_patch(u, v, r)
        if not np.all(np.isfinite(w)) or not self._in_gamut(w):
            raise RuntimeError(
                f"AEPsych generated non-displayable DISP stimulus: "
                f"model=({a:.4f}, {b:.4f}, {r:.4f}), "
                f"patch=({u:.4f}, {v:.4f}), disp={w}"
            )
        self._last_x = x
        self.last_a = a
        self.last_b = b
        self.last_u = u
        self.last_v = v
        self.last_direction = self.direction_from_patch(u, v)
        self.last_theta = float(np.arccos(np.clip(self.last_direction[2], -1.0, 1.0)))
        self.last_phi = float(np.arctan2(self.last_direction[1], self.last_direction[0]))
        self.last_r = r
        self.last_disp = w.copy()
        self.stimulus_disp_log.append(w.copy())
        bg_cone = self.center_cs.convert(
            self.w0.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
        test_cone = self.center_cs.convert(
            w.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
        return bg_cone, test_cone, self.center_cs, r

    def GetColor(
        self, previous_result: ColorTestResult
    ) -> Optional[Tuple[npt.NDArray, npt.NDArray, ColorSpace, float]]:
        import torch
        if self.trial_count >= self.n_trials or self._last_x is None:
            return None
        response = 1.0 if previous_result == ColorTestResult.Success else 0.0
        y = torch.tensor([response])
        self.strategy.add_data(self._last_x, y)
        self.trial_count += 1
        if self.trial_count >= self.n_trials:
            return None
        return self.NewColor()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_threshold_surface(
        self, n_theta: int = 30, n_phi: int = 60
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Extract threshold surface r*(u, v) from GP posterior.

        Returns
        -------
        xs, ys, zs : (N,) arrays of Cartesian chromatic coordinates
        gamut_clipped : (N,) bool array — True where r* exceeded the gamut extent
        """
        if self.strategy.model is None:
            raise RuntimeError("GP model not fitted yet; run trials first.")

        model_as = np.linspace(self._lb_np[0], self._ub_np[0], n_theta)
        model_bs = np.linspace(self._lb_np[1], self._ub_np[1], n_phi)

        xs, ys, zs, gamut_clipped, r_stars = [], [], [], [], []
        r_lo_bound, r_hi_bound = self._lb_np[2], self._ub_np[2]

        for a in model_as:
            for b in model_bs:
                u, v = self.patch_coords_from_model(a, b)
                d3 = self.direction_from_patch(u, v)
                dir_4d = self.chrom_basis @ d3
                r_max_gamut = self._compute_r_max(dir_4d)

                r_hi = min(r_hi_bound, r_max_gamut)
                r_star, found, lower_sat, _, _, _ = self._estimate_threshold_radius(
                    a, b, r_hi)

                is_clipped = (
                    (not found and not lower_sat) or
                    r_star >= r_max_gamut * 0.97
                )
                gamut_clipped.append(is_clipped)
                r_stars.append(r_star)

                xs.append(d3[0] * r_star)
                ys.append(d3[1] * r_star)
                zs.append(d3[2] * r_star)

        return (np.array(xs), np.array(ys), np.array(zs),
                np.array(gamut_clipped, dtype=bool))

    def get_threshold_patch_data(
        self, n_a: int = 31, n_b: int = 31
    ) -> Dict[str, npt.NDArray]:
        """Exportable threshold contour data in real display coordinates.

        The returned `disp_points` are actual 4-primary display values.  The
        other arrays preserve the parameterization:

            model_ab -> patch_uv -> directions_3d -> r_star -> disp_points
        """
        if self.strategy.model is None:
            raise RuntimeError("GP model not fitted yet; run trials first.")

        model_as = np.linspace(self._lb_np[0], self._ub_np[0], n_a)
        model_bs = np.linspace(self._lb_np[1], self._ub_np[1], n_b)

        model_ab = []
        patch_uv = []
        directions_3d = []
        r_stars = []
        r_max_gamut = []
        gamut_clipped = []
        threshold_found = []
        lower_saturated = []
        posterior_p_lo = []
        posterior_p_hi = []
        posterior_p_star = []
        disp_points = []

        r_lo_bound, r_hi_bound = self._lb_np[2], self._ub_np[2]

        for a in model_as:
            for b in model_bs:
                u, v = self.patch_coords_from_model(a, b)
                d3 = self.direction_from_patch(u, v)
                dir_4d = self.chrom_basis @ d3
                r_max = self._compute_r_max(dir_4d)

                r_hi = min(r_hi_bound, r_max)
                r_star, found, lower_sat, p_lo, p_hi, p_star = (
                    self._estimate_threshold_radius(a, b, r_hi))
                clipped = (not found and not lower_sat) or r_star >= r_max * 0.97
                disp = self.w0 + dir_4d * r_star

                model_ab.append([a, b])
                patch_uv.append([u, v])
                directions_3d.append(d3)
                r_stars.append(r_star)
                r_max_gamut.append(r_max)
                gamut_clipped.append(clipped)
                threshold_found.append(found)
                lower_saturated.append(lower_sat)
                posterior_p_lo.append(p_lo)
                posterior_p_hi.append(p_hi)
                posterior_p_star.append(p_star)
                disp_points.append(disp)

        return {
            "format_version": np.array([1], dtype=np.int32),
            "model_ab": np.array(model_ab, dtype=float),
            "patch_uv": np.array(patch_uv, dtype=float),
            "directions_3d": np.array(directions_3d, dtype=float),
            "r_star": np.array(r_stars, dtype=float),
            "r_max_gamut": np.array(r_max_gamut, dtype=float),
            "gamut_clipped": np.array(gamut_clipped, dtype=bool),
            "threshold_found": np.array(threshold_found, dtype=bool),
            "lower_saturated": np.array(lower_saturated, dtype=bool),
            "posterior_p_lo": np.array(posterior_p_lo, dtype=float),
            "posterior_p_hi": np.array(posterior_p_hi, dtype=float),
            "posterior_p_star": np.array(posterior_p_star, dtype=float),
            "disp_points": np.array(disp_points, dtype=float),
            "w0": np.array(self.w0, dtype=float),
            "chrom_basis": np.array(self.chrom_basis, dtype=float),
            "pca_mean_null": np.array(self.pca_mean_null, dtype=float),
            "pca_e1": np.array(self.pca_e1, dtype=float),
            "pca_e2": np.array(self.pca_e2, dtype=float),
            "patch_bounds": np.array(self.patch_bounds, dtype=float),
            "model_bounds": np.stack([self._lb_np, self._ub_np], axis=0),
            "requested_max_radius": np.array(
                [-1.0 if self.requested_max_radius is None else self.requested_max_radius],
                dtype=float,
            ),
            "actual_max_radius": np.array([self.actual_max_radius], dtype=float),
            "threshold_level": np.array([self.threshold_level], dtype=float),
            "grid_shape": np.array([n_a, n_b], dtype=np.int32),
        }

    def export_threshold_patch_npz(
        self, filename: str, n_a: int = 31, n_b: int = 31
    ) -> Dict[str, npt.NDArray]:
        """Write threshold contour data loadable by null_direction_viewer.py."""
        data = self.get_threshold_patch_data(n_a=n_a, n_b=n_b)
        np.savez_compressed(filename, **data)
        return data

    def fit_threshold_ellipsoid(
        self, n_theta: int = 30, n_phi: int = 60
    ) -> Tuple:
        """Fit a PCA ellipsoid to the valid (non-gamut-clipped) threshold surface points.

        Returns
        -------
        center : (3,) Cartesian chromatic coordinates of ellipsoid center
        axes   : (3, 3) eigenvectors as columns (principal axes)
        semi_lengths : (3,) semi-axis lengths (sqrt of eigenvalues)
        residuals    : (N,) Mahalanobis residuals from ellipsoid surface
        """
        xs, ys, zs, clipped = self.get_threshold_surface(n_theta, n_phi)
        valid = ~clipped
        if valid.sum() < 4:
            return None, None, None, None

        pts = np.stack([xs[valid], ys[valid], zs[valid]], axis=1)
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        semi_lengths = np.sqrt(np.maximum(eigenvalues, 0.0))

        pts_c = pts - center
        if np.all(semi_lengths > 1e-12):
            pts_scaled = (pts_c @ eigenvectors) / semi_lengths
            residuals = np.abs(np.linalg.norm(pts_scaled, axis=1) - 1.0)
        else:
            residuals = np.zeros(len(pts))

        return center, eigenvectors, semi_lengths, residuals

    def plot_threshold_contour(self, ax=None):
        """3-D scatter of threshold surface coloured by r*, with fitted ellipsoid center."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        xs, ys, zs, clipped = self.get_threshold_surface()
        mag = np.sqrt(xs**2 + ys**2 + zs**2)
        valid = ~clipped

        sc = ax.scatter(xs[valid], ys[valid], zs[valid],
                        c=mag[valid], cmap='viridis', s=20, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='r* (threshold amplitude)')

        if clipped.any():
            ax.scatter(xs[clipped], ys[clipped], zs[clipped],
                       c='red', s=25, marker='x', alpha=0.5, label='gamut-clipped')

        scale = float(np.max(mag[valid])) if valid.any() else 0.2
        ax.quiver(0, 0, 0,
                  self.pca_mean_null[0], self.pca_mean_null[1], self.pca_mean_null[2],
                  length=scale, color='orange',
                  linewidth=2, label='patch center null dir')

        center, _, _, _ = self.fit_threshold_ellipsoid()
        if center is not None:
            ax.scatter(*center, c='yellow', s=150, marker='*', zorder=5,
                       label='ellipsoid center (null pt est.)')

        ax.set_xlabel('Chromatic e1')
        ax.set_ylabel('Chromatic e2')
        ax.set_zlabel('Chromatic e3')
        ax.set_title('Threshold contour in chromatic subspace')
        ax.legend(fontsize=8)
        plt.tight_layout()
        return ax


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
