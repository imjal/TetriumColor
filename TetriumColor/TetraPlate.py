import numpy as np
import numpy.typing as npt
from typing import Any, List, Tuple, Union
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
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P, lum_noise: float = 0, s_cone_noise: float = 0,
                background_luminance: float = 0.5, dot_size: float = 1.0, degree: float = 4.0):
        """
        Generates a new plate with the given hidden symbol and returns trial data as dict

        Args:
            filename (str): Base filename to save the plate images
            hidden_symbol (Union[int, str]): The hidden symbol to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
            background_luminance (float): Background luminance level (0.0 to 1.0)

        Returns:
            dict: Trial data with paths, metadata, and trial information
        """
        inside_cone, outside_cone, color_space, intensity = self.color_generator.NewColor()
        # Generate the plate image
        image = self.plate_generator.GeneratePlate(
            inside_cone, outside_cone, color_space,
            hidden_symbol, output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise,
            background_luminance=background_luminance,
            dot_size=dot_size,
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
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P, lum_noise: float = 0, s_cone_noise: float = 0.1,
                background_luminance: float = 0.5, dot_size: float = 1.0, degree: float = 4.0, **kwargs):
        """
        Generates a new plate based on previous result and returns trial data as dict or None if complete

        Args:
            previous_result (ColorTestResult): The result of the previous test (did they get it right or not)
            filename (str): Base filename to save the plate images
            hidden_symbol (Union[int, str]): The hidden symbol to save to the plate
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Luminance noise amount
            s_cone_noise (float): S-cone noise amount
            background_luminance (float): Background luminance level (0.0 to 1.0)

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
            lum_noise=lum_noise, s_cone_noise=s_cone_noise,
            background_luminance=background_luminance,
            dot_size=dot_size, **kwargs,
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

        genotype, metameric_axis = self.color_generator.GetCurrentTestInfo()
        genotype_str = str(genotype)

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


class BipartiteFieldGenerator(TestGenerator):
    """
    Generator for bipartite field tests - simple circles split vertically in half
    with two different colors (metamers).
    """

    def __init__(self, color_generator: ColorGenerator, seed: int = 42, size: int = 512):
        """
        Initializes the BipartiteFieldGenerator with the given color generator.

        Args:
            color_generator (ColorGenerator): The color generator to use for colors
            seed (int): The seed for random generation
            size (int): Size of the output image (width and height)
        """
        np.random.seed(seed)
        super().__init__(color_generator)
        self.size = size

    def _create_bipartite_circle(self, left_color, right_color, size: int):
        """
        Create a bipartite circle image split vertically.

        Args:
            left_color: RGB color array for left half [R, G, B] in range [0, 1]
            right_color: RGB color array for right half [R, G, B] in range [0, 1]
            size: Size of the output image

        Returns:
            PIL Image with bipartite circle
        """
        # Create image array
        img_array = np.zeros((size, size, 3), dtype=np.float32)

        # Create circular mask
        center = size / 2
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        circle_mask = dist_from_center <= (size / 2)

        # Create left/right split mask
        left_mask = x < center
        right_mask = x >= center

        # Apply colors
        # Left half
        left_region = circle_mask & left_mask
        img_array[left_region] = left_color[:3]

        # Right half
        right_region = circle_mask & right_mask
        img_array[right_region] = right_color[:3]

        # Convert to 8-bit image
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')

    def NewTest(self, filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                lum_noise: float = 0, s_cone_noise: float = 0, **kwargs):
        """
        Generates a new bipartite field test and returns trial data as dict.

        Args:
            filename (str): Base filename to save the images
            hidden_symbol (Union[int, str], optional): Not used, kept for API compatibility
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Not used for bipartite field
            s_cone_noise (float): Not used for bipartite field

        Returns:
            dict: Trial data with paths, metadata, and trial information
        """
        inside_cone, outside_cone, color_space, intensity = self.color_generator.NewColor()

        # Convert colors to display space
        if output_space == ColorSpaceType.DISP_6P:
            # Convert cone values to display primaries
            inside_disp = color_space.convert(inside_cone.reshape(
                1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP_6P)[0]
            outside_disp = color_space.convert(outside_cone.reshape(
                1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP_6P)[0]

            # Create RGB image (first 3 channels)
            left_rgb = inside_disp[:3]
            right_rgb = outside_disp[:3]
            img_rgb = self._create_bipartite_circle(left_rgb, right_rgb, self.size)

            # Create OCV image (last 3 channels)
            left_ocv = inside_disp[3:]
            right_ocv = outside_disp[3:]
            img_ocv = self._create_bipartite_circle(left_ocv, right_ocv, self.size)

            # Save images
            rgb_path = f"{filename}_RGB.png"
            ocv_path = f"{filename}_OCV.png"
            img_rgb.save(rgb_path)
            img_ocv.save(ocv_path)
        else:
            # SRGB output
            inside_srgb = color_space.convert(inside_cone.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.SRGB)[0]
            outside_srgb = color_space.convert(outside_cone.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.SRGB)[0]

            img = self._create_bipartite_circle(inside_srgb, outside_srgb, self.size)
            rgb_path = f"{filename}_SRGB.png"
            ocv_path = rgb_path
            img.save(rgb_path)

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
            'trial_type': 'bipartite_field',
            'genotype': genotype_str,
            'metameric_axis': metameric_axis,
            'rgb_path': rgb_path,
            'ocv_path': ocv_path,
            'intensity': intensity,
            'metadata': {
                'inside_cone': inside_cone.tolist(),
                'outside_cone': outside_cone.tolist(),
                'size': self.size
            }
        }

    def GetTest(self, previous_result: ColorTestResult,
                filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                lum_noise: float = 0, s_cone_noise: float = 0, **kwargs):
        """
        Generates a bipartite field test based on previous result.

        Args:
            previous_result (ColorTestResult): The result of the previous test
            filename (str): Base filename to save the images
            hidden_symbol (Union[int, str], optional): Not used, kept for API compatibility
            output_space (ColorSpaceType): Target color space for output
            lum_noise (float): Not used for bipartite field
            s_cone_noise (float): Not used for bipartite field

        Returns:
            dict or None: Trial data dict if test continues, None if test is complete
        """
        # Get next color from color generator
        result = self.color_generator.GetColor(previous_result)

        # If None returned, test is complete
        if result is None:
            return None

        inside_cone, outside_cone, color_space, intensity = result

        # Convert colors to display space
        if output_space == ColorSpaceType.DISP_6P:
            # Convert cone values to display primaries
            inside_disp = color_space.convert(inside_cone.reshape(
                1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP_6P)
            outside_disp = color_space.convert(outside_cone.reshape(
                1, -1), ColorSpaceType.CONE, ColorSpaceType.DISP_6P)[0]

            # Create RGB image (first 3 channels)
            left_rgb = inside_disp[:3]
            right_rgb = outside_disp[:3]
            img_rgb = self._create_bipartite_circle(left_rgb, right_rgb, self.size)

            # Create OCV image (last 3 channels)
            left_ocv = inside_disp[3:]
            right_ocv = outside_disp[3:]
            img_ocv = self._create_bipartite_circle(left_ocv, right_ocv, self.size)

            # Save images
            rgb_path = f"{filename}_RGB.png"
            ocv_path = f"{filename}_OCV.png"
            img_rgb.save(rgb_path)
            img_ocv.save(ocv_path)
        else:
            # SRGB output
            inside_srgb = color_space.convert(inside_cone.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.SRGB)[0]
            outside_srgb = color_space.convert(outside_cone.reshape(1, -1), ColorSpaceType.CONE, ColorSpaceType.SRGB)[0]

            img = self._create_bipartite_circle(inside_srgb, outside_srgb, self.size)
            rgb_path = f"{filename}_SRGB.png"
            ocv_path = rgb_path
            img.save(rgb_path)

        genotype, metameric_axis = self.color_generator.GetCurrentTestInfo()
        genotype_str = str(genotype)

        # Return trial data as dictionary
        return {
            'trial_type': 'bipartite_field',
            'genotype': genotype_str,
            'metameric_axis': metameric_axis,
            'rgb_path': rgb_path,
            'ocv_path': ocv_path,
            'intensity': intensity,
            'metadata': {
                'inside_cone': inside_cone.tolist(),
                'outside_cone': outside_cone.tolist(),
                'size': self.size
            }
        }


class GaussianBlobGenerator(TestGenerator):
    """
    4AFC gaussian blob detection stimulus, matching the anomaloscope blob mode.

    A uniform circle of the background (outside/metamer) color fills the stimulus area.
    A small Gaussian blob of the foreground (inside) color is placed at one of 4 cardinal
    positions (up/down/left/right) within the circle at half-radius offset.
    The blob diameter matches the Landolt-C gap (1/5 of circle diameter).

    The hidden_symbol (e.g. "landolt_up") determines which position the blob appears at.

    All compositing and noise are applied in cone space before conversion to display
    space, so RGB and OCV images share a single consistent noise realisation.
    """

    _DIRECTION_MAP = {
        'landolt_up': 'up', 'landolt_down': 'down',
        'landolt_left': 'left', 'landolt_right': 'right',
        'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
    }

    BASE_DEGREE = 4.0

    def __init__(self, color_generator: ColorGenerator, seed: int = 42, size: int = 1024):
        np.random.seed(seed)
        super().__init__(color_generator)
        self.size = size

    def _build_cone_image(self, fg_cone, bg_cone, direction: str, degree: float,
                          lum_noise: float, s_cone_noise: float):
        """Build the stimulus circle in cone space (H, W, n_cones).

        Pixels outside the circle are left as NaN so the caller can fill them
        with the correct display-space background colour.

        Returns:
            (cone_image, circle_mask)
        """
        size = self.size
        n_cones = len(bg_cone)
        center = size / 2.0
        radius = size * 0.475

        Y, X = np.ogrid[:size, :size]
        dist_sq = (X - center) ** 2 + (Y - center) ** 2
        circle_mask = dist_sq <= radius ** 2

        cone_img = np.full((size, size, n_cones), np.nan, dtype=np.float64)

        for c in range(n_cones):
            cone_img[:, :, c] = np.where(circle_mask, bg_cone[c], np.nan)

        # Per-pixel luminance noise (applied equally to all cone channels)
        if lum_noise > 0:
            lum_noise_map = np.random.normal(0.0, lum_noise, (size, size))
            for c in range(n_cones):
                cone_img[:, :, c] += np.where(circle_mask, lum_noise_map, 0.0)

        # Per-pixel S-cone noise (channel 0 only, matching IshiharaPlateGenerator)
        if s_cone_noise > 0:
            s_noise_map = np.random.normal(0.0, s_cone_noise, (size, size))
            cone_img[:, :, 0] += np.where(circle_mask, s_noise_map, 0.0)

        # Gaussian blob at one of 4 cardinal positions
        gap_px = radius * 2.0 * 0.2
        blob_sigma = max(gap_px / 4.0, 1.0)

        offset = radius * 0.5
        positions = {
            'up':    (center, center - offset),
            'down':  (center, center + offset),
            'left':  (center - offset, center),
            'right': (center + offset, center),
        }
        bx, by = positions[direction]

        hw = int(np.ceil(4 * blob_sigma))
        x0, x1 = max(int(bx) - hw, 0), min(int(bx) + hw + 1, size)
        y0, y1 = max(int(by) - hw, 0), min(int(by) + hw + 1, size)

        xs = np.arange(x0, x1, dtype=np.float64) - bx
        ys = np.arange(y0, y1, dtype=np.float64) - by
        alpha = np.exp(-ys[:, None] ** 2 / (2 * blob_sigma ** 2)) \
              * np.exp(-xs[None, :] ** 2 / (2 * blob_sigma ** 2))
        alpha *= circle_mask[y0:y1, x0:x1]

        for c in range(n_cones):
            patch = cone_img[y0:y1, x0:x1, c]
            cone_img[y0:y1, x0:x1, c] = patch * (1.0 - alpha) + fg_cone[c] * alpha

        np.clip(cone_img, 0, None, out=cone_img)
        return cone_img, circle_mask

    @staticmethod
    def _cone_to_images(cone_img, circle_mask, color_space, output_space,
                        background_luminance: float):
        """Convert a (H, W, n_cones) cone image to PIL images.

        Pixels outside *circle_mask* are filled with a neutral gray matching the
        app background (background_luminance / max_L), identical to how the
        IshiharaPlateGenerator computes its background colour.

        Returns (img_a, img_b): two PIL images (RGB+OCV for DISP_6P, or same for SRGB).
        """
        h, w, _ = cone_img.shape

        # Only convert circle pixels through the colour-space pipeline
        flat = cone_img.reshape(-1, cone_img.shape[2])
        mask_flat = circle_mask.ravel()
        circle_flat = flat[mask_flat]
        disp_circle = color_space.convert(circle_flat, ColorSpaceType.CONE, output_space)

        n_disp = disp_circle.shape[1]
        disp_flat = np.zeros((h * w, n_disp), dtype=np.float64)
        disp_flat[mask_flat] = disp_circle
        disp_img = disp_flat.reshape(h, w, n_disp)

        if output_space == ColorSpaceType.DISP_6P:
            rgb = np.clip(disp_img[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            ocv = np.clip(disp_img[:, :, 3:] * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb, 'RGB'), Image.fromarray(ocv, 'RGB')
        else:
            srgb = np.clip(disp_img[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(srgb, 'RGB')
            return img, img

    def _parse_direction(self, hidden_symbol) -> str:
        if hidden_symbol is None:
            return np.random.choice(['up', 'down', 'left', 'right'])
        s = str(hidden_symbol)
        return self._DIRECTION_MAP.get(s, 'right')

    def NewTest(self, filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                lum_noise: float = 0, s_cone_noise: float = 0,
                background_luminance: float = 0.5, degree: float = 4.0, **kwargs):
        inside_cone, outside_cone, color_space, intensity = self.color_generator.NewColor()
        return self._generate(inside_cone, outside_cone, color_space, intensity,
                              filename, hidden_symbol, output_space,
                              lum_noise, s_cone_noise, background_luminance, degree)

    def GetTest(self, previous_result, filename: str, hidden_symbol: Union[int, str] = None,
                output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                lum_noise: float = 0, s_cone_noise: float = 0,
                background_luminance: float = 0.5, degree: float = 4.0, **kwargs):
        result = self.color_generator.GetColor(previous_result)
        if result is None:
            return None
        inside_cone, outside_cone, color_space, intensity = result
        return self._generate(inside_cone, outside_cone, color_space, intensity,
                              filename, hidden_symbol, output_space,
                              lum_noise, s_cone_noise, background_luminance, degree)

    def _generate(self, inside_cone, outside_cone, color_space, intensity,
                  filename, hidden_symbol, output_space,
                  lum_noise, s_cone_noise, background_luminance, degree):
        direction = self._parse_direction(hidden_symbol)

        cone_img, circle_mask = self._build_cone_image(
            inside_cone, outside_cone, direction, degree, lum_noise, s_cone_noise)

        img_a, img_b = self._cone_to_images(
            cone_img, circle_mask, color_space, output_space, background_luminance)

        if output_space == ColorSpaceType.DISP_6P:
            rgb_path = f"{filename}_RGB.png"
            ocv_path = f"{filename}_OCV.png"
            img_a.save(rgb_path)
            img_b.save(ocv_path)
        else:
            rgb_path = f"{filename}_SRGB.png"
            ocv_path = rgb_path
            img_a.save(rgb_path)

        genotype, metameric_axis = self.color_generator.GetCurrentTestInfo()

        return {
            'trial_type': 'gaussian_blob',
            'genotype': str(genotype),
            'metameric_axis': metameric_axis,
            'rgb_path': rgb_path,
            'ocv_path': ocv_path,
            'hidden_symbol': str(hidden_symbol) if hidden_symbol else f'landolt_{direction}',
            'intensity': intensity,
            'metadata': {
                'inside_cone': inside_cone.tolist(),
                'outside_cone': outside_cone.tolist(),
                'size': self.size,
                'direction': direction,
                'degree': degree,
                'lum_noise': lum_noise,
                's_cone_noise': s_cone_noise,
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
            sex='female', percentage_screened=0.999, dimensions=[2], display_primaries=primaries, trials_per_direction=1)

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
        testing_dim = 3
        color_generator = GeneticColorGenerator(
            sex='both', percentage_screened=0.99, display_primaries=primaries, dimensions=[testing_dim],
            metameric_axes=[2],
            # list(range(1, testing_dim + 1)),
            trials_per_direction=1, randomize_genotypes=False)

        print("Number of Genotypes: ", len(color_generator.genotypes))

        genotypes = color_generator.GetGenotypes()
        print(f"Genotypes: {genotypes}")
        plate_generator = PseudoIsochromaticPlateGenerator(color_generator)

        lum_noise = 0.0
        s_cone_noise = 0.1
        output_space = ColorSpaceType.SRGB
        output_filename = "genetic_color_picker_scone_noise"

        dirname = f"./measurements/2025-11-10/noise_{lum_noise}_scone_{s_cone_noise}_dimension_{testing_dim}"
        os.makedirs(dirname, exist_ok=True)

        # Get the list of uppercase alphabet letters (A-Z)
        import string
        alphabet = list(string.ascii_uppercase) * 10
        images = []

        # control_plate = plate_generator.GetControlPlate(os.path.join(
        #     dirname, "control"), color_generator.genotype_mapping[genotypes[0]][0], lum_noise=lum_noise, s_cone_noise=s_cone_noise, output_space=output_space, corner_label=alphabet[0])
        # images.append(control_plate)

        landolt_symbols = ['landolt_up', 'landolt_down', 'landolt_left', 'landolt_right']

        idx = 0
        while True:
            random_landolt_symbol = np.random.choice(landolt_symbols)
            image = plate_generator.GetTest(None, os.path.join(
                dirname, f"test_{idx}"), random_landolt_symbol, output_space=output_space,
                lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=alphabet[idx])
            print(f"Generated plate {idx}")
            if image is None:
                break
            images.append(image)
            idx += 1

        if output_space == ColorSpaceType.DISP_6P:
            from PIL import Image
            rgb_images = [Image.open(image['rgb_path']) for image in images]
            ocv_images = [Image.open(image['ocv_path']) for image in images]
            rgb_grid = CreatePaddedGrid(rgb_images, padding=0, channels=3, square_grid=False)
            ocv_grid = CreatePaddedGrid(ocv_images, padding=0, channels=3, square_grid=False)
            rgb_grid.save(os.path.join(dirname, f"{output_filename}_RGB.png"))
            ocv_grid.save(os.path.join(dirname, f"{output_filename}_OCV.png"))

            # For each generated plate (indexed by idx/alphabet), save genotype and metameric axis to a .txt file
            genotype_axis_txt_path = os.path.join(dirname, f"{output_filename}_genotype_axis.txt")
            with open(genotype_axis_txt_path, "w") as f:
                f.write("Letter\tGenotype\tMetamericAxis\n")
                for i, image_info in enumerate(images):
                    # Each 'image_info' should have metameric_axis and genotype (from GetTest)
                    letter = alphabet[i]
                    genotype = image_info.get('genotype', None)
                    axis = image_info.get('metameric_axis', None)
                    # Format genotype nicely as tuple or string
                    if genotype is not None:
                        genotype_str = str(tuple[Any, ...](genotype)) if not isinstance(genotype, str) else genotype
                    else:
                        genotype_str = "N/A"
                    axis_str = str(axis) if axis is not None else "N/A"
                    f.write(f"{letter}\t{genotype_str}\t{axis_str}\n")
            print(f"Saved genotype/axis mapping to {genotype_axis_txt_path}")
        else:
            from PIL import Image
            images = [Image.open(image['rgb_path']) for image in images]
            grid = CreatePaddedGrid(images, padding=0, channels=3, square_grid=False)
            grid.save(os.path.join(dirname, f"{output_filename}_sRGB.png"))
            print(f"Saved grid to {os.path.join(dirname, output_filename + '_sRGB.png')}")

    genetic_color_picker_test()
