import numpy.typing as npt
from typing import List

from TetriumColor.Utils.CustomTypes import ColorTestResult
from TetriumColor.Observer import *
from TetriumColor import ColorSpaceType
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.TetraColorPicker import ColorGenerator


class PseudoIsochromaticPlateGenerator:

    def __init__(self, color_generator: ColorGenerator, seed: int = 42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given color generator, color space and seed

        Args:
            color_generator (ColorGenerator): The color generator to use for plate colors
            seed (int): The seed for the plate pattern generation.
        """
        self.seed: int = seed
        self.color_generator: ColorGenerator = color_generator
        self.plate_generator: IshiharaPlateGenerator = IshiharaPlateGenerator(seed=self.seed)

    # must be called before GetPlate
    def NewPlate(self, filename: str, hidden_number: int,
                 output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0):
        """
        Generates a new plate with the given hidden number and cone space colors

        Args:
            hidden_number (int): The hidden number to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
        """
        inside_cone, outside_cone, color_space = self.color_generator.NewColor()
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_number, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise
        )
        self.plate_generator.ExportPlateTo6P(image, filename)

    def GetPlate(self, previous_result: ColorTestResult,
                 filename: str, hidden_number: int,
                 output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0):
        """
        Generates a new plate and saves it to files with the given hidden number and cone space colors

        Args:
            previous_result (ColorTestResult): The result of the previous test (did they get it right or not)
            hidden_number (int): The hidden number to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
        """
        inside_cone, outside_cone, color_space = self.color_generator.GetColor(previous_result)
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_number, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise
        )
        self.plate_generator.ExportPlateTo6P(image, filename)

        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_number, ColorSpaceType.SRGB,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise
        )
        image[0].save(f"{filename}_srgb.png")

    if __name__ == "__main__":

        from TetriumColor.TetraColorPicker import GeneticCDFTestColorGenerator
        from TetriumColor import PseudoIsochromaticPlateGenerator
        from TetriumColor.Measurement import load_primaries_from_csv

        primaries = load_primaries_from_csv("./measurements/2025-10-01/primaries/")

        # for p in primaries:
        #     p.plot()
        # plt.show()

        color_generator = GeneticCDFTestColorGenerator(
            sex='female', percentage_screened=0.999, cst_display_type='led', display_primaries=primaries, dimensions=[2])

        print("Number of Genotypes: ", color_generator.get_num_samples())
        number_of_tests = color_generator.get_num_samples()
        plate_generator = PseudoIsochromaticPlateGenerator(color_generator)

        lum_noise = 0.001
        s_cone_noise = 0.000

        dirname = f"./measurements/2025-10-01/tests_noise_{lum_noise}_scone_noise_{s_cone_noise}"
        os.makedirs(dirname, exist_ok=True)
        for i in range(number_of_tests):
            print(f"Generating plate {i}")
            plate_generator.GetPlate(
                None, os.path.join(dirname, f"test_{i}"), 10, ColorSpaceType.DISP_6P, lum_noise=lum_noise, s_cone_noise=s_cone_noise)
