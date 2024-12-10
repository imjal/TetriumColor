from abc import ABC, abstractmethod
from typing import List

from matplotlib import colors
from TetriumColor.Utils.CustomTypes import *
import numpy as np
import pickle
import os

from TetriumColor.PsychoPhys.Quest import Quest
from TetriumColor.Utils.IO import LoadColorSpaceTransform
import TetriumColor.ColorMath.GamutMath as GamutMath


# TODO: Implement the following classes
class BackgroundNoiseGenerator:
    def __init__(self):
        pass

    def NoiseGenerator(self) -> PlateColor:
        pass


class ColorGenerator(ABC):

    @staticmethod
    def __generateQuestObject(t_guess, t_guess_sd) -> Quest:
        # TODO: For now don't mess with quest, just sample at an interval to get idea of parameters
        return Quest(t_guess, t_guess_sd, 0.8, beta=3.5, delta=0.01, gamma=0.05, grain=0.01, range=None)

    @abstractmethod
    def NewColor(self) -> PlateColor:
        pass

    @abstractmethod
    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class ScreeningTestColorGenerator(ColorGenerator):
    """
    Screening Test Color Generator

    Attributes:
        num_tests (int): The number of tests to generate per transform directory
        transform_dirs (List[str]): The list of directories to load the ColorSpaceTransform from
        pre_generated_filenames (List[str]): The list of pre-generated metamers already serialized to pickle files
    """

    def __init__(self, num_tests: int, transform_dirs: List[str], pre_generated_filenames: List[str] = None):
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

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        # TODO: incorporate feedback but for now not necessary
        if self.current_idx >= self.num_tests:
            self.current_idx = 0  # loop back to the first for now
        return self.NewColor()


class TargetedTestColorGenerator(ColorGenerator):

    def __init__(self, num_trials: int, transform_dir: str, luminance: float, num_saturation_levels: int):
        self.current_idx: int = 0
        self.num_trials: int = num_trials
        self.color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(transform_dir)

        # get parameters for a single direction (kind of terrible, might need to refactor)
        max_L: float = (np.linalg.inv(self.color_space_transform.hering_to_disp) @
                        np.ones(self.color_space_transform.cone_to_disp.shape[0]))[0]
        metameric_vshh: npt.NDArray = GamutMath.GetMetamericAxisInVSH(self.color_space_transform)
        self.angles: npt.NDArray = metameric_vshh[0, 2:]
        lum_cusp, sat_cusp = GamutMath.FindMaxSaturationForVSH(
            metameric_vshh, self.color_space_transform)
        self.luminance: float = luminance
        self.max_sat_at_luminance: float = GamutMath.SolveForBoundary(luminance, max_L, lum_cusp, sat_cusp)

        self.saturations: npt.NDArray = np.linspace(0, self.max_sat_at_luminance, num_saturation_levels + 1)
        self.saturations = self.saturations[::-1]

    def NewColor(self) -> PlateColor:
        vsh = np.concatenate([[self.luminance, self.saturations[self.current_idx]], self.angles])
        color: PlateColor = GamutMath.ConvertVSHToPlateColor(vsh, self.luminance, self.color_space_transform)
        print(color)
        self.current_idx += 1
        return color

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        if self.current_idx >= len(self.saturations):
            return None  # how should I communicate with you @Tian?
        return self.NewColor()


class InDepthTestColorGenerator(ColorGenerator):

    def __init__(self, transform_dir: str, luminance: float, saturation: float, num_directions: int = 25, num_saturation_levels: int = 5):
        self.color_space_transform: ColorSpaceTransform = LoadColorSpaceTransform(transform_dir)
        self.num_directions: int = num_directions
        self.luminance: float = luminance
        self.saturation: float = saturation
        self.hue_space: npt.NDArray = GamutMath.SampleHueManifold(
            luminance, saturation, self.color_space_transform.dim, num_directions)
        map_angle_to_cusp = GamutMath.GenerateGamutLUT(self.hue_space, self.color_space_transform)
        self.map_angle_to_lum_plane = GamutMath.GetEquiluminantPlane(
            luminance, self.color_space_transform, map_angle_to_cusp)

        self.current_direction = 0
        self.current_saturation_per_direction = [num_saturation_levels] * num_directions
        self.num_saturation_levels = num_saturation_levels

    def NewColor(self) -> PlateColor:
        hue = tuple(self.hue_space[self.current_direction][2:])
        lum, sat = self.map_angle_to_lum_plane[hue]
        current_sat_level = self.current_saturation_per_direction[self.current_direction]/self.num_saturation_levels
        color = GamutMath.ConvertVSHToPlateColor(
            np.array([lum, sat*current_sat_level, *hue]), self.luminance, self.color_space_transform)
        self.current_saturation_per_direction[self.current_direction] -= 1
        self.current_direction = (self.current_direction + 1) % self.num_directions
        return color

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor:
        if np.all(np.array(self.current_saturation_per_direction) == 0):
            return None
        return self.NewColor()
