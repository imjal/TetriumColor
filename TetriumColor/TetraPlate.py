import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from PIL import Image

from TetriumColor.Utils.CustomTypes import ColorTestResult
from TetriumColor.Observer import *
from TetriumColor import ColorSpaceType, ColorSpace
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator
from TetriumColor.TetraColorPicker import ColorGenerator, GeneticColorGenerator
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid


class TestGenerator(ABC):
    def __init__(self, color_generator: ColorGenerator):
        self.color_generator = color_generator

    @abstractmethod
    def NewTest(self, filename: str, hidden_symbol: Union[int, str], output_space: ColorSpaceType, **kwargs):
        """Generate first test and return trial data as dict"""
        pass

    @abstractmethod
    def GetTest(self, previous_result: ColorTestResult, filename: str, hidden_symbol: Union[int, str], output_space: ColorSpaceType, **kwargs):
        """Generate next test based on previous result, return dict or None if complete"""
        pass


class PlateGenerator(TestGenerator):
    def __init__(self, color_generator: ColorGenerator):
        super().__init__(color_generator)
        self.plate_generator = IshiharaPlateGenerator()

    @abstractmethod
    def GetTest(self, previous_result: ColorTestResult, filename: str, hidden_symbol: Union[int, str], output_space: ColorSpaceType = ColorSpaceType.DISP_6P) -> List[int]:
        pass

    @abstractmethod
    def NewTest(self, filename: str, hidden_symbol: Union[int, str], output_space: ColorSpaceType = ColorSpaceType.DISP_6P):
        pass

    def GetControlPlate(self, filename: str, color_space: ColorSpace, lum_noise: float = 0, s_cone_noise: float = 0, output_space: ColorSpaceType = ColorSpaceType.DISP_6P, **kwargs):

        inside_cone, _, _ = color_space.get_maximal_pair_in_disp_from_pt(np.array([0.5, 0.5, 0.5, 0.5]))

        image = self.plate_generator.GeneratePlate(
            inside_cone, inside_cone, color_space,
            10, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, **kwargs
        )
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
        else:
            image[0].save(f"{filename}_SRGB.png")
        return image

    def GetLuminancePlate(self, filename: str, hidden_symbol: Union[int, str], color_space: ColorSpace,
                          lum_noise: float = 0, s_cone_noise: float = 0, output_space: ColorSpaceType = ColorSpaceType.DISP_6P, **kwargs):
        vshh_points = np.array([[1.5, 0, 0.0, 0.0], [0.5, 0, 0.0, 0.0]])
        cones = color_space.convert(vshh_points, ColorSpaceType.VSH, ColorSpaceType.CONE)

        image = self.plate_generator.GeneratePlate(
            cones[0], cones[1], color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, **kwargs
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
        super().__init__(color_generator)
        self.plate_generator: IshiharaPlateGenerator = IshiharaPlateGenerator()

    def NewTest(self, filename: str, hidden_symbol: Union[int, str],
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P, lum_noise: float = 0, s_cone_noise: float = 0):
        """
        Generates a new plate with the given hidden symbol and returns trial data as dict

        Args:
            filename (str): Base filename to save the plate images
            hidden_symbol (Union[int, str]): The hidden symbol to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount

        Returns:
            dict: Trial data with paths, metadata, and trial information
        """
        inside_cone, outside_cone, color_space, intensity = self.color_generator.NewColor()
        # Generate the plate image
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise,
            seed=np.random.randint(0, 1000000)
        )

        # Save images to disk
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
            rgb_path = f"{filename}_RGB.png"
            ocv_path = f"{filename}_OCV.png"
        else:
            image[0].save(f"{filename}_SRGB.png")
            rgb_path = f"{filename}_SRGB.png"
            ocv_path = rgb_path  # For SRGB, both paths are the same

        # Extract genotype if available
        genotype = getattr(color_space, 'genotype', None)
        if genotype:
            genotype_str = str(genotype)
        else:
            genotype_str = "unknown"

        # Extract metameric axis if available
        metameric_axis = getattr(color_space, 'metameric_axis', -1)

        # Return trial data as dictionary
        return {
            'trial_type': 'pseudo_isochromatic',
            'genotype': genotype_str,
            'metameric_axis': metameric_axis,
            'rgb_path': rgb_path,
            'ocv_path': ocv_path,
            'hidden_symbol': str(hidden_symbol),
            'intensity': intensity,
            'metadata': {
                'inside_cone': inside_cone.tolist(),
                'outside_cone': outside_cone.tolist(),
                'lum_noise': lum_noise,
                's_cone_noise': s_cone_noise
            }
        }

    def GetTest(self, previous_result: ColorTestResult,
                filename: str, hidden_symbol: Union[int, str],
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P, lum_noise: float = 0, s_cone_noise: float = 0.1, **kwargs):
        """
        Generates a new plate based on previous result and returns trial data as dict or None if complete

        Args:
            previous_result (ColorTestResult): The result of the previous test (did they get it right or not)
            filename (str): Base filename to save the plate images
            hidden_symbol (Union[int, str]): The hidden symbol to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount

        Returns:
            dict or None: Trial data dict if test continues, None if test is complete
        """
        # Get next color from color generator
        result = self.color_generator.GetColor(previous_result)

        # If None returned, test is complete
        if result is None:
            return None

        inside_cone, outside_cone, color_space, intensity = result

        # Generate the plate image
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, **kwargs,
            seed=np.random.randint(0, 1000000)
        )

        # Save images to disk
        if output_space == ColorSpaceType.DISP_6P:
            self.plate_generator.ExportPlateTo6P(image, filename)
            rgb_path = f"{filename}_RGB.png"
            ocv_path = f"{filename}_OCV.png"
        else:
            image[0].save(f"{filename}_SRGB.png")
            rgb_path = f"{filename}_SRGB.png"
            ocv_path = rgb_path  # For SRGB, both paths are the same

        # Extract genotype if available
        genotype = getattr(color_space, 'genotype', None)
        if genotype:
            genotype_str = str(genotype)
        else:
            genotype_str = "unknown"

        # Extract metameric axis if available
        metameric_axis = getattr(color_space, 'metameric_axis', -1)

        # Return trial data as dictionary
        return {
            'trial_type': 'pseudo_isochromatic',
            'genotype': genotype_str,
            'metameric_axis': metameric_axis,
            'rgb_path': rgb_path,
            'ocv_path': ocv_path,
            'hidden_symbol': str(hidden_symbol),
            'intensity': intensity,
            'metadata': {
                'inside_cone': inside_cone.tolist(),
                'outside_cone': outside_cone.tolist(),
                'lum_noise': lum_noise,
                's_cone_noise': s_cone_noise
            }
        }


class CircleGridGenerator(TestGenerator):
    def __init__(self, color_generator: ColorGenerator, scramble_prob: float = 0.5, luminance: float = 1.0, saturation: float = 0.5):
        super().__init__(color_generator)
        self.scramble_prob = scramble_prob
        self.luminance = luminance
        self.saturation = saturation

    def GetGenotypes(self) -> List[Tuple]:
        """Get the list of genotypes from the color generator.

        Returns:
            List[Tuple]: The list of genotypes.
        """
        if hasattr(self.color_generator, 'GetGenotypes'):
            return self.color_generator.GetGenotypes()
        elif hasattr(self.color_generator, 'genotypes'):
            return self.color_generator.genotypes
        else:
            raise AttributeError("ColorGenerator does not have GetGenotypes() method or genotypes attribute")

    def GetImages(self, genotype: Tuple, metameric_axis: int, filename: Union[str, List[str]], output_space: ColorSpaceType = ColorSpaceType.DISP_6P) -> List[Tuple[int, int]]:
        """Get images for a given genotype and metameric axis.

        Args:
            genotype: Genotype tuple
            metameric_axis: Metameric axis index
            filename: Either a single base filename (str) or list of 3 filenames (List[str])
            output_space: Output color space

        Returns:
            List of (int, int) tuples representing scramble indices
        """
        if genotype not in self.color_generator.genotype_mapping:
            raise ValueError(f"Genotype {genotype} not found in mapping")

        _, color_sampler = self.color_generator.genotype_mapping[genotype]

        image_tuples, idxs = color_sampler.get_hue_sphere_scramble(
            self.luminance, self.saturation, 4, metameric_axis=metameric_axis, scramble_prob=self.scramble_prob, output_space=output_space)

        # Handle both single filename (base) and list of filenames
        if isinstance(filename, str):
            # Single base filename - save as {base}_0, {base}_1, {base}_2
            base_filename = filename
            filenames = [f"{base_filename}_{i}" for i in range(3)]
        else:
            # List of filenames provided
            filenames = filename
            if len(filenames) != 3:
                raise ValueError(f"Expected 3 filenames, got {len(filenames)}")

        if output_space == ColorSpaceType.DISP_6P:
            for i, (rgb, ocv) in enumerate(image_tuples):
                rgb.save(f"{filenames[i]}_RGB.png")
                ocv.save(f"{filenames[i]}_OCV.png")
            return idxs
        else:
            for i, im in enumerate(image_tuples):
                im.save(f"{filenames[i]}_SRGB.png")
            return idxs

    def NewTest(self, filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                genotype: Tuple = None, metameric_axis: int = None, **kwargs) -> dict:
        """
        Generate a new test and return trial data as dict.

        Args:
            filename (str): Base filename to save the plate images
            hidden_symbol (Union[int, str], optional): Not used for circle grid, kept for API compatibility
            output_space (ColorSpaceType): Target color space for output
            genotype (Tuple, optional): Genotype to use. If None, gets from color_generator.GetDirection()
            metameric_axis (int, optional): Metameric axis to use. If None, gets from color_generator.GetDirection()

        Returns:
            dict: Trial data dict with paths, genotype, metameric_axis, and scramble_indices
        """
        # If genotype/metameric_axis not provided, get from color generator
        if genotype is None or metameric_axis is None:
            if hasattr(self.color_generator, 'GetDirection'):
                gen, axis = self.color_generator.GetDirection()
                if genotype is None:
                    genotype = gen
                if metameric_axis is None:
                    metameric_axis = axis
            else:
                raise ValueError(
                    "genotype and metameric_axis must be provided if color_generator does not have GetDirection()")

        # Generate images
        idxs = self.GetImages(genotype, metameric_axis, filename, output_space)

        # Build image paths (base filenames - app will add _RGB/_OCV suffixes via GetTexturePaths)
        # GetImages() saves as {filename}_0_RGB.png, {filename}_1_RGB.png, etc.
        # So we return base filenames {filename}_0, {filename}_1, {filename}_2
        # App will use GetTexturePaths() to add _RGB.png or _SRGB.png suffix
        image_paths = []
        for i in range(3):
            image_paths.append(f"{filename}_{i}")

        # Convert scramble indices to flat list
        # idxs is actually a numpy array of ints (from np.random.choice), not List[Tuple[int, int]]
        # Convert numpy array to Python list of ints
        if isinstance(idxs, np.ndarray):
            # It's a numpy array - convert to list of Python ints
            scramble_indices = [int(idx) for idx in idxs.tolist()]
        elif hasattr(idxs, '__iter__') and not isinstance(idxs, str):
            # It's an iterable (list, tuple, etc.)
            scramble_indices = [int(idx) for idx in idxs]
        else:
            # Single value (shouldn't happen)
            scramble_indices = [int(idxs)]

        # Convert genotype tuple to string
        genotype_str = str(genotype)

        return {
            'trial_type': 'circle_grid',
            'genotype': genotype_str,
            'metameric_axis': metameric_axis,
            'image_paths': image_paths,
            'scramble_indices': scramble_indices,
            'metadata': {
                'luminance': self.luminance,
                'saturation': self.saturation,
                'scramble_prob': self.scramble_prob
            }
        }

    def GetTest(self, previous_result: ColorTestResult, filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                genotype: Tuple = None, metameric_axis: int = None, **kwargs) -> dict:
        """
        Get the test images for the given filename and output space.

        Args:
            previous_result (ColorTestResult): The result of the previous test (not used for circle grid, kept for API compatibility)
            filename (str): Base filename to save the images to
            hidden_symbol (Union[int, str], optional): Not used for circle grid, kept for API compatibility
            output_space (ColorSpaceType): The output space to save the images to
            genotype (Tuple, optional): Genotype to use. If None, gets from color_generator.GetDirection()
            metameric_axis (int, optional): Metameric axis to use. If None, gets from color_generator.GetDirection()

        Returns:
            dict: Trial data dict with paths, genotype, metameric_axis, and scramble_indices
        """
        return self.NewTest(filename, hidden_symbol, output_space, genotype, metameric_axis, **kwargs)


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

        color_generator = GeneticColorGenerator(
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
