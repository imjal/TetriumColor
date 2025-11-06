import numpy.typing as npt
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from PIL import Image

from TetriumColor.Utils.CustomTypes import ColorTestResult
from TetriumColor.Observer import *
from TetriumColor import ColorSpaceType, ColorSpace
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.TetraColorPicker import ColorGenerator, GeneticColorPicker
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid


class PlateGenerator(ABC):
    def __init__(self):
        self.plate_generator = IshiharaPlateGenerator()

    @abstractmethod
    def GetPlate(self, inside_cone: npt.NDArray, outside_cone: npt.NDArray, color_space: ColorSpace, filename: str, hidden_symbol: Union[int, str],
                 output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0, corner_label: str = None):
        pass

    def GetControlPlate(self, filename: str, color_space: ColorSpace, lum_noise: float = 0, s_cone_noise: float = 0, output_space: ColorSpaceType = ColorSpaceType.SRGB, corner_label: str = None):

        inside_cone, _, _ = color_space.get_maximal_pair_in_disp_from_pt(np.array([0.5, 0.5, 0.5, 0.5]))

        image = self.plate_generator.GeneratePlate(
            inside_cone, inside_cone, color_space,
            10, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=corner_label
        )
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
        else:
            image[0].save(f"{filename}_SRGB.png")
        return image

    def GetLuminancePlate(self, filename: str, hidden_symbol: Union[int, str], color_space: ColorSpace, lum_noise: float = 0, s_cone_noise: float = 0, output_space: ColorSpaceType = ColorSpaceType.SRGB, corner_label: str = None):
        vshh_points = np.array([[1.5, 0, 0.0, 0.0], [0.5, 0, 0.0, 0.0]])
        cones = color_space.convert(vshh_points, ColorSpaceType.VSH, ColorSpaceType.CONE)

        image = self.plate_generator.GeneratePlate(
            cones[0], cones[1], color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=corner_label
        )
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
        else:
            image[0].save(f"{filename}_SRGB.png")
        return image


class PseudoIsochromaticPlateGenerator(PlateGenerator):

    def __init__(self, color_generator: ColorGenerator, seed: int = 42):
        """
        Initializes the PseudoIsochromaticPlateGenerator with the given color generator, color space and seed

        Args:
            color_generator (ColorGenerator): The color generator to use for plate colors
            seed (int): The seed for the plate pattern generation.
        """
        np.random.seed(seed)
        self.color_generator: ColorGenerator = color_generator
        self.plate_generator: IshiharaPlateGenerator = IshiharaPlateGenerator()

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
        inside_cone, outside_cone, color_space, _ = self.color_generator.NewColor()
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_number, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise,
            seed=np.random.randint(0, 1000000)
        )
        self.plate_generator.ExportPlateTo6P(image, filename)

    def GetPlate(self, previous_result: ColorTestResult,
                 filename: str, hidden_symbol: Union[int, str],
                 output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0, corner_label: str = None):
        """
        Generates a new plate and saves it to files with the given hidden number and cone space colors

        Args:
            previous_result (ColorTestResult): The result of the previous test (did they get it right or not)
            hidden_number (int): The hidden number to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
        """
        inside_cone, outside_cone, color_space, _ = self.color_generator.GetColor(previous_result)
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=corner_label,
            seed=np.random.randint(0, 1000000)
        )
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
        else:
            image[0].save(f"{filename}_SRGB.png")

        return image


class GeneticColorPickerPlateGenerator(PlateGenerator):
    def __init__(self, color_generator: GeneticColorPicker):
        super().__init__()
        self.color_generator = color_generator

    def GetPlate(self, genotype: Tuple, metameric_axis: int, filename: str, hidden_symbol: Union[int, str],
                 output_space: ColorSpaceType, lum_noise: float = 0, s_cone_noise: float = 0, corner_label: str = None):
        inside_cone, outside_cone, color_space = self.color_generator.GetMetamericPair(genotype, metameric_axis)
        image = self.plate_generator.GeneratePlate(inside_cone, outside_cone, color_space,
                                                   hidden_symbol, output_space, lum_noise, s_cone_noise, corner_label)
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
        else:
            image[0].save(f"{filename}_SRGB.png")
        return image


if __name__ == "__main__":

    def genetic_cdf_test():

        from TetriumColor.TetraColorPicker import GeneticCDFTestColorGenerator
        from TetriumColor import PseudoIsochromaticPlateGenerator
        from TetriumColor.Measurement import load_primaries_from_csv

        primaries = load_primaries_from_csv("./measurements/2025-10-10/primaries/")

        color_generator = GeneticCDFTestColorGenerator(
            sex='female', percentage_screened=0.999, dimensions=[2], display_primaries=primaries)

        print("Number of Genotypes: ", color_generator.get_num_samples())
        number_of_tests = color_generator.get_num_samples()
        plate_generator = PseudoIsochromaticPlateGenerator(color_generator)

        lum_noise = 0.0
        s_cone_noise = 0.1
        output_space = ColorSpaceType.DISP_6P
        output_filename = "metamer_difference_noise_all"

        dirname = f"./measurements/2025-11-4/tests_noise_{lum_noise}_scone_noise_{s_cone_noise}"
        os.makedirs(dirname, exist_ok=True)

        # Get the list of uppercase alphabet letters (A-Z)
        import string
        alphabet = list(string.ascii_uppercase)

        control_plate = plate_generator.GetControlPlate(os.path.join(
            dirname, "control"), color_generator.color_spaces[0], lum_noise=lum_noise, s_cone_noise=s_cone_noise, output_space=output_space, corner_label=alphabet[0])
        images = [control_plate]

        landolt_symbols = ['landolt_up', 'landolt_down', 'landolt_left', 'landolt_right']

        for i in range(1, number_of_tests + 1):
            random_landolt_symbol = np.random.choice(landolt_symbols)
            print(f"Generating plate {i}")
            images.append(plate_generator.GetPlate(
                None, os.path.join(dirname, f"test_{i}"), random_landolt_symbol, output_space=output_space, lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=alphabet[i]))
        if output_space == ColorSpaceType.DISP_6P:
            rgb_images = [image[0] for image in images]
            ocv_images = [image[1] for image in images]
            rgb_grid = CreatePaddedGrid(rgb_images, padding=0, channels=3, square_grid=False)
            ocv_grid = CreatePaddedGrid(ocv_images, padding=0, channels=3, square_grid=False)
            rgb_grid.save(os.path.join(dirname, f"{output_filename}_RGB.png"))
            ocv_grid.save(os.path.join(dirname, f"{output_filename}_OCV.png"))
        else:
            images = [image[0] for image in images]
            grid = CreatePaddedGrid(images, padding=0, channels=3, square_grid=False)
            grid.save(os.path.join(dirname, f"{output_filename}_sRGB.png"))

    def genetic_color_picker_test():

        from TetriumColor import PseudoIsochromaticPlateGenerator
        from TetriumColor.Measurement import load_primaries_from_csv

        primaries = load_primaries_from_csv("./measurements/2025-10-10/primaries/")

        color_generator = GeneticColorPicker(
            sex='female', percentage_screened=0.999, display_primaries=primaries, dimensions=[2])

        genotypes = color_generator.GetGenotypes()
        plate_generator = GeneticColorPickerPlateGenerator(color_generator)

        lum_noise = 0.0
        s_cone_noise = 0.1
        output_space = ColorSpaceType.DISP_6P
        output_filename = "genetic_color_picker_scone_noise"

        dirname = f"./measurements/2025-11-4/tests_noise_{lum_noise}_scone_noise_{s_cone_noise}"
        os.makedirs(dirname, exist_ok=True)

        # Get the list of uppercase alphabet letters (A-Z)
        import string
        alphabet = list(string.ascii_uppercase) * 2

        control_plate = plate_generator.GetControlPlate(os.path.join(
            dirname, "control"), color_generator.genotype_mapping[genotypes[0]][0], lum_noise=lum_noise, s_cone_noise=s_cone_noise, output_space=output_space, corner_label=alphabet[0])
        images = [control_plate]

        landolt_symbols = ['landolt_up', 'landolt_down', 'landolt_left', 'landolt_right']

        for i, genotype in enumerate(genotypes):
            for j, metameric_axis in enumerate(range(4)):
                idx = i * 4 + j
                random_landolt_symbol = np.random.choice(landolt_symbols)
                print(f"Generating plate {genotype} {metameric_axis}")
                images.append(plate_generator.GetPlate(
                    genotype, metameric_axis, os.path.join(dirname, f"test_{idx}"), random_landolt_symbol, output_space=output_space, lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=alphabet[idx]))
        if output_space == ColorSpaceType.DISP_6P:
            rgb_images = [image[0] for image in images]
            ocv_images = [image[1] for image in images]
            rgb_grid = CreatePaddedGrid(rgb_images, padding=0, channels=3, square_grid=False)
            ocv_grid = CreatePaddedGrid(ocv_images, padding=0, channels=3, square_grid=False)
            rgb_grid.save(os.path.join(dirname, f"{output_filename}_RGB.png"))
            ocv_grid.save(os.path.join(dirname, f"{output_filename}_OCV.png"))
        else:
            images = [image[0] for image in images]
            grid = CreatePaddedGrid(images, padding=0, channels=3, square_grid=False)
            grid.save(os.path.join(dirname, f"{output_filename}_sRGB.png"))

    genetic_color_picker_test()
