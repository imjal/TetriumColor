import numpy.typing as npt
from typing import List

from dataclasses import dataclass
from enum import Enum


class ColorTestResult(Enum):
    Success = 1
    Failure = 0
    NoAnswer = -1


class TestType(Enum):
    """
    Three Different Types of Screening Tests Exist.
        Screening: Maximal Metamers
        Targeted: Targeted Quest Along N=1 Axis, the Q-axis
        InDepth: In Depth Quest Along N Hue Directions
    """
    Screening = 0
    Targeted = 1
    InDepth = 2


@dataclass
class TetraColor:
    """
    A class to hold RGB/OCV color data for even-odd rendering.
        RGB (npt.NDArray): The RGB color data.
        OCV (npt.NDArray): The OCV color data.
    """
    RGB: npt.NDArray
    OCV: npt.NDArray


@dataclass
class PlateColor:
    """
    A class to hold the plate color data.

        shape (TetraColor): The color of the shape inside being "hidden".
        background (TetraColor): The color of the background on the plate.
    """
    shape: TetraColor
    background: TetraColor
