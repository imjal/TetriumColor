from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict

import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import *
from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes, Observer
from TetriumColor.Measurement import load_primaries_from_csv


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


class GeneticCDFTestColorGenerator(ColorGenerator):
    def __init__(self, sex: str, percentage_screened: float,  peak_to_test: float = 547, metameric_axis: int = 2, luminance: float = 1.0, saturation: float = 0.5, dimensions: Optional[List[int]] = [2], seed: int = 42, extra_first_genotype: int = 4, **kwargs):
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
                 dimensions: Optional[List[int]] = [2], seed: int = 42, **kwargs):
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
                 dimensions: Optional[List[int]] = [2], seed: int = 42, **kwargs):
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
                                    cst_display_type='led', display_primaries=primaries)

    genotypes = generator.GetGenotypes()

    for genotype in genotypes:
        for metameric_axis in range(4):
            print(f"Generating images for genotype {genotype} metameric axis {metameric_axis}")
            filenames = [f"genotype_{genotype}_metameric_axis_{metameric_axis}_unscramble1",
                         f"genotype_{genotype}_metameric_axis_{metameric_axis}_unscramble2", f"genotype_{genotype}_metameric_axis_{metameric_axis}_scramble"]
            generator.GetImages(genotype, metameric_axis, filenames, output_space=ColorSpaceType.SRGB)
