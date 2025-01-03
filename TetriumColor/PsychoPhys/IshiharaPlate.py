import numpy as np
import packcircles
from importlib import resources

from typing import Callable, List
import numpy.typing as npt

from PIL import Image, ImageDraw
from TetriumColor.Utils.CustomTypes import PlateColor, TetraColor
from PIL import ImageFont


class IshiharaPlate:
    _secrets = [27, 35, 39, 64, 67, 68, 72, 73, 85, 87, 89, 96]

    def __init__(self, plate_color: PlateColor = None, secret: int = _secrets[0],
                 num_samples: int = 100, dot_sizes: List[int] = [16, 22, 28],
                 image_size: int = 1024, directory: str = '.', seed: int = 0,
                 lum_noise: float = 0, noise: float = 0, gradient: bool = False):
        """
        :param plate_color: A PlateColor object with shape and background colors (RGB/OCV tuples).

        :param secret:  May be either a string or integer, specifies which
                        secret file to use from the secrets directory.
        """
        if not gradient:
            # only need to sample center of circles for once
            num_samples = 1
            if noise != 0:
                raise ValueError("None-zero noise is not supported for non-gradient plates -- it doesn't make sense!")
        self.num_samples: int = num_samples
        self.dot_sizes: List[int] = dot_sizes
        self.image_size: int = image_size
        self.directory: str = directory
        self.seed: int = seed
        self.noise: float = noise
        self.gradient: bool = gradient
        self.lum_noise: float = lum_noise
        self.circles: List = None

        self.__generateGeometry()
        self.__setSecretImage(secret)

    def __setSecretImage(self, secret: int):
        if secret in IshiharaPlate._secrets:
            with resources.path("TetriumColor.Assets.HiddenImages", f"{str(secret)}.png") as data_path:
                self.secret = Image.open(data_path)
            self.secret = self.secret.resize(
                [self.image_size, self.image_size])
            self.secret = np.asarray(self.secret)
        else:
            raise ValueError(f"Invalid Hidden Number {secret}")

    def GeneratePlate(self, seed: int | None = None, hidden_number: int | None = None, plate_color: PlateColor | None = None, noise_generator: Callable[[], npt.NDArray] | None = None):
        """
        Generate the Ishihara Plate with specified inside/outside colors and secret.
        A new seed can be specified to generate a different plate pattern.
        New inside or outside colors may be specified to recolor the plate
        without modifying the geometry.

        :param seed: A seed for RNG when creating the plate pattern.
        :param hidden_number: The hidden number to embed in the plate.
        :param inside_color: A 6-tuple RGBOCV color.
        :param outside_color: A 6-tuple RGBOCV color.
        """
        def helper_generate():
            self.__generateGeometry()
            self.__computeInsideOutside()
            self.__drawPlate(noise_generator)

        if plate_color:
            self.inside_color = self.__standardizeColor(plate_color.shape)
            self.outside_color = self.__standardizeColor(
                plate_color.background)

        if hidden_number:
            if hidden_number < 0:  # pick random if negative.
                hidden_number = np.random.choice(IshiharaPlate._secrets)
            self.__setSecretImage(hidden_number)

        # Plate doesn't exist; set seed and colors and generate whole plate.
        if self.circles is None:
            self.seed = seed or self.seed
            helper_generate()
            return

        # Need to generate new geometry and re-color.
        if seed and seed != self.seed:
            self.seed = seed
            self.__resetPlate()
            helper_generate()
            return

        # Don't regenerate geometry but recolor w/new hidden number
        if hidden_number:
            self.__computeInsideOutside()
            self.__resetImages()
            self.__drawPlate(noise_generator)
            return

        # Need to re-color, but don't need to re-generate geometry or hidden number.
        if plate_color:
            self.__resetImages()
            self.__drawPlate(noise_generator)
            return

    def ExportPlate(self, filename_RGB: str, filename_OCV: str):
        """
        This method saves two images - RGB and OCV encoded image.

        :param save_name: Name of directory to save plate to.
        :param ext: File extension to use, such as 'png' or 'tif'.
        """
        self.channels[0].save(filename_RGB)
        self.channels[1].save(filename_OCV)

    def DrawCorner(self, label: str, color: npt.ArrayLike = np.array([255/2, 255/2, 255/2, 255/2, 0, 0]).astype(int)):
        """
        Draw a colored corner on the plate.

        :param label: what you want displayed
        :param color: RGBOCV tuple represented as float [0, 1].
        """
        font = ImageFont.load_default(size=150)

        self.channel_draws[0].text((10, 10), label, fill=tuple(color[:3]), font=font)
        self.channel_draws[1].text((10, 10), label, fill=tuple(color[3:]), font=font)

    def __standardizeColor(self, color: TetraColor):
        """
        :param color: Ensure a TetraColor is a float in [0, 1].
        """
        if np.issubdtype(color.RGB.dtype, np.integer):
            color.RGB = color.RGB.astype(float) / 255.0

        if np.issubdtype(color.OCV.dtype, np.integer):
            color.OCV = color.OCV.astype(float) / 255.0

        return np.concatenate([color.RGB, color.OCV])

    def __generateGeometry(self):
        """
        :return output_circles: List of [x, y, r] sequences, where (x, y)
                                are the center coordinates of a circle and r
                                is the radius.
        """
        np.random.seed(self.seed)

        # Create packed_circles, a list of (x, y, r) tuples.
        radii = self.dot_sizes * 2000
        np.random.shuffle(radii)
        packed_circles = packcircles.pack(radii)

        # Generate output_circles.
        center = self.image_size // 2
        output_circles = []

        for (x, y, radius) in packed_circles:
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
                r = radius - np.random.randint(2, 5)
                output_circles.append([x, y, r])

        self.circles = output_circles

    def __computeInsideOutside(self):
        """
        For each circle, estimate the proportion of its area that is inside or outside.
        Take num_sample point samples within each circle, generated by rejection sampling.
        """
        # Inside corresponds to numbers; outside corresponds to background
        outside = np.int32(np.sum(self.secret == 255, -1) == 4)
        inside = None

        if self.gradient:
            # TODO: is this necessary for gradient? it's cursed to be both inside and outside
            inside = np.int32((self.secret[:, :, 3] == 255)) - outside

        inside_props = []
        outside_props = []
        n = np.random.rand(len(self.circles))

        for i, [x, y, r] in enumerate(self.circles):
            x, y = int(round(x)), int(round(y))

            if self.gradient:
                assert inside is not None
                inside_count, outside_count = 0, 0

                for _ in range(self.num_samples):
                    while True:
                        dx = np.random.uniform(-r, r)
                        dy = np.random.uniform(-r, r)
                        if (dx**2 + dy**2) <= r**2:
                            break

                    x_grid = int(np.clip(np.round(x + dx), 0, self.image_size - 1))
                    y_grid = int(np.clip(np.round(y + dy), 0, self.image_size - 1))
                    if inside[y_grid, x_grid]:
                        inside_count += 1
                    elif outside[y_grid, x_grid]:
                        outside_count += 1

                in_p = np.clip(inside_count / self.num_samples *
                               (1 - (n[i] * self.noise / 100)), 0, 1)
                out_p = np.clip(outside_count / self.num_samples *
                                (1 - (n[i] * self.noise / 100)), 0, 1)

                inside_props.append(in_p)
                outside_props.append(out_p)
            else:  # non-gradient sampling -- only sample center of circles
                x = np.clip(x, 0, self.image_size - 1)
                y = np.clip(y, 0, self.image_size - 1)
                is_outside: int = 1 if outside[y, x] else 0
                is_inside: int = 1 - is_outside

                inside_props.append(is_inside)
                outside_props.append(is_outside)

        self.inside_props = inside_props
        self.outside_props = outside_props

    def __drawPlate(self, noise_generator: Callable[[], npt.NDArray] | None = None):
        """
        Using generated geometry data and computed inside/outside proportions,
        draw the plate.
        """
        assert None not in [self.circles,
                            self.inside_props, self.outside_props]

        for i, [x, y, r] in enumerate(self.circles):
            in_p, out_p = self.inside_props[i], self.outside_props[i]
            # only apply to vector that are on
            if noise_generator:
                new_color = np.clip(noise_generator(), 0, 1)
                if in_p:
                    new_color = new_color[0]
                else:
                    new_color = new_color[1]
            else:
                circle_color = in_p * self.inside_color + out_p * self.outside_color
                # noise apply to the six channel, scale the entire vector
                lum_noise = np.random.normal(0, self.lum_noise)
                # only apply to vector that are on
                new_color = np.clip(
                    circle_color + (lum_noise * (circle_color > 0)), 0, 1)
            self.__drawEllipse([x-r, y-r, x+r, y+r], new_color)

    def __drawEllipse(self, bounding_box: List, fill: npt.ArrayLike):
        """
        Wrapper function for PIL ImageDraw. Draws to each of the
        R, G1, G2, and B channels; each channel is represented as
        a grayscale image.

        :param bounding_box: Four points to define the bounding box.
            Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1].
        :param fill: RGBOCV tuple represented as float [0, 1].
        """
        ellipse_color = (fill * 255).astype(int)
        self.channel_draws[0].ellipse(
            bounding_box, fill=tuple(ellipse_color[:3]), width=0)
        self.channel_draws[1].ellipse(
            bounding_box, fill=tuple(ellipse_color[3:]), width=0)

    def __resetGeometry(self):
        """
        Reset plate geometry. Useful if we want to regenerate the plate pattern
        with a different seed.
        """
        self.circles = None
        self.inside_props = None
        self.outside_props = None

    def __resetImages(self):
        """
        Reset plate images. Useful if we want to regenerate the plate with
        different inside/outside colors.
        """
        self.channels = [Image.new(mode='RGB', size=(
            self.image_size, self.image_size)) for _ in range(4)]
        self.channel_draws = [ImageDraw.Draw(ch) for ch in self.channels]

    def __resetPlate(self):
        """
        Reset geometry and images.
        """
        self.__resetGeometry()
        self.__resetImages()
