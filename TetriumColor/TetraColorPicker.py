from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import *
from TetriumColor.ColorSpace import ColorSpace
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes


class ColorGenerator(ABC):

    @abstractmethod
    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace]:
        pass

    @abstractmethod
    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace] | None:
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        pass


class TestColorGenerator(ColorGenerator):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def NewColor(self) -> PlateColor:
        return PlateColor(shape=TetraColor(np.array([0, 255, 0], dtype=int), np.array([255, 0, 0], dtype=int)), background=TetraColor(np.array([255, 255, 0], dtype=int), np.array([0, 255, 255], dtype=int)))

    def GetColor(self, previous_result: ColorTestResult) -> PlateColor | None:
        return PlateColor(shape=TetraColor(np.array([255, 0, 0]), np.array([0, 255, 0])), background=TetraColor(np.array([255, 255, 0]), np.array([0, 255, 255])))


class GeneticCDFTestColorGenerator(ColorGenerator):
    def __init__(self, sex: str, percentage_screened: float, **kwargs):
        self.percentage_screened = percentage_screened
        self.observer_genotypes = ObserverGenotypes()
        self.kwargs = kwargs  # Store kwargs for passing to color space creation

        self.genotypes = self.observer_genotypes.get_genotypes_covering_probability(
            target_probability=self.percentage_screened, sex=sex)

        fig = self.observer_genotypes.plot_cdf(sex=sex)
        fig.savefig(f"cdf_{sex}.png")

        self.color_spaces = self.observer_genotypes.get_color_spaces_covering_probability(
            target_probability=self.percentage_screened, sex=sex, **kwargs)
        self.current_idx = 0
        self.meta_idx = 0

        self.num_samples = len(self.color_spaces)

    def get_num_samples(self) -> int:
        return self.num_samples

    def GetColor(self, previous_result: ColorTestResult) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace] | None:
        if self.current_idx >= self.num_samples:
            self.current_idx = 0
        return self.NewColor()

    def NewColor(self) -> Tuple[npt.NDArray, npt.NDArray, ColorSpace]:
        if self.current_idx >= self.num_samples:
            raise StopIteration("No more genotypes to sample")
        color_space = self.color_spaces[self.current_idx]

        inside_cone, outside_cone = color_space.get_maximal_metamer_pair_in_disp(metameric_axis=self.meta_idx)

        self.meta_idx = (self.meta_idx + 1) % color_space.dim
        if self.meta_idx == 0:
            print(f"Meta axis {self.meta_idx} reached for genotype {self.current_idx}")
            print("Moving onto Next Genotype")
            self.current_idx += 1
        return inside_cone, outside_cone, color_space
