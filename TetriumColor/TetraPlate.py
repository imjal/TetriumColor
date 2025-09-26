import numpy.typing as npt
from typing import List

from TetriumColor.Utils.CustomTypes import ColorTestResult
from TetriumColor.Observer import *
from TetriumColor import ColorSpace, ColorSpaceType, ColorSampler, TetraColor, PlateColor
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.TetraColorPicker import ColorGenerator

# Control Test


class PseudoIsochromaticPlateGenerator:

    def __init__(self, color_generator: ColorGenerator, color_space: ColorSpace, seed: int = 42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given color generator, color space and seed

        Args:
            color_generator (ColorGenerator): The color generator to use for plate colors
            color_space (ColorSpace): The color space for color conversions
            seed (int): The seed for the plate pattern generation.
        """
        self.seed: int = seed
        self.color_generator: ColorGenerator = color_generator
        self.color_space: ColorSpace = color_space
        self.current_plate: IshiharaPlateGenerator = IshiharaPlateGenerator(color_space, seed=self.seed)

    # must be called before GetPlate
    def NewPlate(self, inside_cone: npt.NDArray, outside_cone: npt.NDArray, 
                filenames: List[str], hidden_number: int, 
                output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0):
        """
        Generates a new plate with the given hidden number and cone space colors

        Args:
            inside_cone (npt.NDArray): Inside color in cone space
            outside_cone (npt.NDArray): Outside color in cone space
            filenames (List[str]): List of filenames to save the output images
            hidden_number (int): The hidden number to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
        """
        images = self.current_plate.GeneratePlate(
            inside_cone, outside_cone, hidden_number, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise
        )
        self.current_plate.ExportPlate(images, filenames)

    def GetPlate(self, previous_result: ColorTestResult, inside_cone: npt.NDArray, outside_cone: npt.NDArray,
                filenames: List[str], hidden_number: int, 
                output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0):
        """
        Generates a new plate and saves it to files with the given hidden number and cone space colors

        Args:
            previous_result (ColorTestResult): The result of the previous test (did they get it right or not)
            inside_cone (npt.NDArray): Inside color in cone space
            outside_cone (npt.NDArray): Outside color in cone space
            filenames (List[str]): List of filenames to save the output images
            hidden_number (int): The hidden number to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
        """
        # Note: We're not using previous_result yet, but it could be used for adaptive algorithms
        images = self.current_plate.GeneratePlate(
            inside_cone, outside_cone, hidden_number, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise
        )
        self.current_plate.ExportPlate(images, filenames)
