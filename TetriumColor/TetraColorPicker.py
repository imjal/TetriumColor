from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import *
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType


class NoiseGenerator(ABC):
    @abstractmethod
    def GenerateNoiseFunction(self, plate_color: PlateColor | npt.NDArray) -> Callable[[], npt.NDArray]:
        pass


class ColorGenerator(ABC):

    @abstractmethod
    def NewColor(self) -> PlateColor:
        pass

    @abstractmethod
    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class ScreeningTestColorGenerator(ColorGenerator):
    """
    Screening Test Color Generator

    Attributes:
        num_tests (int): The number of tests to generate per transform directory
        transform_dirs (List[str]): The list of directories to load the ColorSpaceTransform from
        pre_generated_filenames (List[str]): The list of pre-generated metamers already serialized to pickle files
    """

    def __init__(self, num_tests: int):
        if num_tests > 10:
            raise ValueError(
                "Number of tests must be less than or equal to 10 per transform directory")
        self.num_tests = num_tests
        self.current_idx = 0

        self.metamer_list: List[PlateColor] = []

    def NewColor(self) -> PlateColor:
        plate_color = self.metamer_list[self.current_idx]
        self.current_idx += 1
        return plate_color

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        # TODO: incorporate feedback but for now not necessary
        if self.current_idx >= self.num_tests:
            self.current_idx = 0  # loop back to the first for now
        return self.NewColor()


class TargetedTestColorGenerator(ColorGenerator):
    def __init__(self, color_space: ColorSpace, num_trials: int, luminance: float, num_saturation_levels: int):
        self.current_idx: int = 0
        self.num_trials: int = num_trials

        self.color_space = color_space

        # Get metameric direction, and find the maximal saturation, and then sample along that direction
        metameric_vsh = self.color_space.get_metameric_axis_in(ColorSpaceType.VSH)
        self.angles = tuple(metameric_vsh[2:])  # Keep the hue angle components

        self.luminance = luminance
        self.max_sat_at_luminance = self.color_space.max_sat_at_luminance(luminance, self.angles)

        # Generate saturations list (high to low)
        self.saturations: npt.NDArray = np.linspace(0, self.max_sat_at_luminance, num_saturation_levels + 1)
        self.saturations = self.saturations[::-1]

    def NewColor(self) -> PlateColor:
        # Create VSH point with current saturation level
        vsh = np.array([self.luminance, self.saturations[self.current_idx], *self.angles])

        # Convert to plate color
        color = self.color_space.to_plate_color(vsh, self.luminance)
        print(color)
        self.current_idx += 1
        return color

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        if self.current_idx >= len(self.saturations):
            return None
        return self.NewColor()


class InDepthTestColorGenerator(ColorGenerator):
    def __init__(self, color_space: ColorSpace, luminance: float, num_trials: int, num_hue_samples: int, num_saturation_levels: int):
        self.current_idx: int = 0
        self.num_trials: int = num_trials

        self.color_space = color_space

        # Equally sample directions in hue space based on num_hue_samples
        self.hue_angles = [tuple(x) for x in color_space.sample_hue_manifold(luminance, 1, num_hue_samples)]
        self.luminance = luminance
        self.max_sats_at_luminance = self.color_space.max_sat_at_luminance(luminance, self.hue_angles)

        # Generate saturations list (high to low)
        self.saturations: npt.NDArray = np.linspace(
            0, self.max_sats_at_luminance, num_saturation_levels + 1)
        self.saturations = self.saturations[::-1]

        # Generate all combinations of hue angles and saturation levels
        self.vsh_combinations = np.array([
            [luminance, saturation, hue_angle[0], hue_angle[1]]
            for hue_angle in self.hue_angles
            for saturation in self.saturations
        ])
        # Randomize the list
        np.random.shuffle(self.vsh_combinations)

    def NewColor(self) -> PlateColor:
        # Create VSH point with current saturation level
        vsh = self.vsh_combinations[self.current_idx]
        # Convert to plate color
        color = self.color_space.to_plate_color(vsh, self.luminance)
        print(color)
        self.current_idx += 1
        return color

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        if self.current_idx >= len(self.saturations):
            return None
        return self.NewColor()
