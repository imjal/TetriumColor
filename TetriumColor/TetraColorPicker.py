from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

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


class CircleGridGenerator:
    def __init__(self, primary_path: str, num_samples: int, scramble_prob: float = 0.5):
        self.scramble_prob = scramble_prob
        self.num_samples = num_samples

        primaries = load_primaries_from_csv(primary_path)
        self.color_space = ColorSpace(Observer.tetrachromat(), cst_display_type='led', display_primaries=primaries)
        self.color_sampler = ColorSampler(self.color_space, cubemap_size=5)

    def GetImages(self, luminance: float, saturation: float, filenames: List[str], output_space: ColorSpaceType = ColorSpaceType.DISP_6P) -> List[Tuple[int, int]]:
        image_tuples, idxs = self.color_sampler.get_hue_sphere_scramble(
            luminance, saturation, 4, scramble_prob=self.scramble_prob, output_space=output_space)

        if output_space == ColorSpaceType.DISP_6P:
            for i, (rgb, ocv) in enumerate(image_tuples):
                rgb.save(filenames[i] + "_RGB.png")
                ocv.save(filenames[i] + "_OCV.png")
            return idxs
        else:
            for i, im in enumerate(image_tuples):
                im.save(filenames[i] + ".png")
            return idxs


if __name__ == "__main__":
    generator = CircleGridGenerator(
        primary_path="../measurements/2025-10-12/primaries", num_samples=10, scramble_prob=0.5)
    generator.GetImages(1.0, 0.4, ["unscramble1", "unscramble2", "scramble"], output_space=ColorSpaceType.SRGB)
