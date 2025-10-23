import numpy as np
import packcircles
import importlib.resources as resources
from importlib.resources import as_file

from typing import Callable, List, Tuple, Optional, Union
import numpy.typing as npt

from PIL import Image, ImageDraw
from TetriumColor.Utils.CustomTypes import PlateColor, TetraColor
from TetriumColor import ColorSpace, ColorSpaceType
from PIL import ImageFont
from pathlib import Path


# Available hidden numbers and Landolt C directions
_SECRETS = list(range(10, 100)) + [
    'landolt_up', 'landolt_down', 'landolt_left', 'landolt_right',
    'landolt_up-right', 'landolt_down-right', 'landolt_up-left', 'landolt_down-left'
]


class IshiharaPlateGenerator:
    """
    Ishihara plate generator with geometry caching for improved performance.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the plate generator.

        Args:
            color_space: ColorSpace object for color conversions
            seed: Random seed for geometry generation
        """
        self.seed = seed

    def _get_geometry_cache_path(self, seed: int, dot_sizes: List[int], image_size: int) -> Path:
        """Get the cache file path for geometry."""
        cache_key = f"geometry_{seed}_{hash(tuple(dot_sizes))}_{image_size}"
        with resources.path("TetriumColor.Assets.Cache.geometry-ishihara", f"{cache_key}.npy") as path:
            return path

    def _get_geometry(self, seed: int, dot_sizes: List[int], image_size: int) -> List[List[float]]:
        """
        Get cached geometry or generate and cache new geometry.

        Args:
            seed: Random seed for geometry generation
            dot_sizes: List of dot sizes to use
            image_size: Size of the output image

        Returns:
            List of circle definitions [x, y, r]
        """
        cache_path = self._get_geometry_cache_path(seed, dot_sizes, image_size)

        if cache_path.exists():
            return np.load(cache_path).tolist()
        else:
            geometry = _generate_geometry(dot_sizes, image_size, seed)
            np.save(cache_path, np.array(geometry))
            return geometry

    def GeneratePlate(self, inside_cone: npt.NDArray, outside_cone: npt.NDArray,
                      color_space: ColorSpace,
                      hidden_symbol: Union[int, str], output_space: ColorSpaceType,
                      lum_noise: float = 0, s_cone_noise: float = 0,
                      corner_label: Optional[str] = None,
                      metamer_difference: Optional[float] = None,
                      **kwargs) -> List[Image.Image]:
        """
        Generate plate with specified output color space.

        Args:
            inside_cone: Inside color in cone space
            outside_cone: Outside color in cone space  
            hidden_number: Number to embed in plate
            output_space: Target color space (DISP_6P, PRINT_4D, SRGB, etc.)
            lum_noise: Luminance noise amount
            s_cone_noise: S-cone noise amount
            metamer_difference: Metamer difference for adaptive noise calculation
            **kwargs: Additional arguments passed to generate_ishihara_plate
        """
        # Get default values or use provided kwargs
        dot_sizes = kwargs.get("dot_sizes", [16, 22, 28])
        image_size = kwargs.get("image_size", 1024)

        # Get cached geometry
        circles = self._get_geometry(self.seed, dot_sizes, image_size)

        # Generate plate using cached geometry
        return generate_ishihara_plate(
            inside_cone, outside_cone, color_space,
            secret=hidden_symbol, circles=circles,  # Use cached geometry
            output_space=output_space,
            lum_noise=lum_noise, s_cone_noise=s_cone_noise, corner_label=corner_label,
            metamer_difference=metamer_difference,
            seed=self.seed,  # Pass seed for consistency
            **kwargs
        )

    def ExportPlateTo6P(self, plate_imgs: List[Image.Image], filename: str):
        """Export plate images to files."""
        exts = ["RGB", "OCV"]
        for i, img in enumerate(plate_imgs):
            img.save(f"{filename}_{exts[i]}.png")


def _generate_geometry(dot_sizes: List[int], image_size: int, seed: int) -> List[List[float]]:
    """
    Generate the geometry for the Ishihara plate.

    :param dot_sizes: List of dot sizes to use
    :param image_size: Size of the output image
    :param seed: Random seed for reproducibility
    :return: List of circle definitions [x, y, r]
    """
    np.random.seed(seed)

    # Create packed_circles, a list of (x, y, r) tuples
    radii = dot_sizes * 2000
    if image_size > 1024:
        radii = radii * 8
    np.random.shuffle(radii)
    packed_circles = packcircles.pack(radii)

    # Generate output_circles
    center = image_size // 2
    output_circles = []

    for (x, y, radius) in packed_circles:
        if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
            r = radius - np.random.randint(2, 5)
            output_circles.append([x, y, r])

    return output_circles


def _compute_inside_outside(
    circles: List[List[float]],
    secret_img: np.ndarray,
    image_size: int,
    num_samples: int,
    noise: float,
    gradient: bool
) -> Tuple[List[float], List[float]]:
    """
    Compute which circles are inside vs outside the secret shape.

    :param circles: List of circle definitions [x, y, r]
    :param secret_img: Secret image as numpy array
    :param image_size: Size of the image
    :param num_samples: Number of samples to take for gradient plates
    :param noise: Amount of noise to add
    :param gradient: Whether to use gradient sampling
    :return: Tuple of (inside_props, outside_props)
    """
    # Inside corresponds to numbers; outside corresponds to background
    outside = np.int32(np.sum(secret_img == 255, -1) == 4)
    inside = None

    if gradient:
        inside = np.int32((secret_img[:, :, 3] == 255)) - outside

    inside_props = []
    outside_props = []
    n = np.random.rand(len(circles))

    for i, [x, y, r] in enumerate(circles):
        x, y = int(round(x)), int(round(y))

        if gradient:
            assert inside is not None
            inside_count, outside_count = 0, 0

            for _ in range(num_samples):
                while True:
                    dx = np.random.uniform(-r, r)
                    dy = np.random.uniform(-r, r)
                    if (dx**2 + dy**2) <= r**2:
                        break

                x_grid = int(np.clip(np.round(x + dx), 0, image_size - 1))
                y_grid = int(np.clip(np.round(y + dy), 0, image_size - 1))
                if inside[y_grid, x_grid]:
                    inside_count += 1
                elif outside[y_grid, x_grid]:
                    outside_count += 1

            in_p = np.clip(inside_count / num_samples * (1 - (n[i] * noise / 100)), 0, 1)
            out_p = np.clip(outside_count / num_samples * (1 - (n[i] * noise / 100)), 0, 1)
        else:
            # Non-gradient sampling -- only sample center of circles
            x = int(np.clip(x, 0, image_size - 1))
            y = int(np.clip(y, 0, image_size - 1))
            is_outside = 1 if outside[y, x] else 0
            is_inside = 1 - is_outside
            in_p, out_p = is_inside, is_outside

        inside_props.append(in_p)
        outside_props.append(out_p)

    return inside_props, outside_props


def _draw_plate(
    circles: List[List[float]],
    inside_props: List[float],
    outside_props: List[float],
    inside_color: np.ndarray,  # cone stimulation
    outside_color: np.ndarray,  # cone stimulation
    color_space: ColorSpace,
    channel_draws: List[ImageDraw.ImageDraw],
    lum_noise: float,
    s_cone_noise: float = 0.0,
    input_space: ColorSpaceType = ColorSpaceType.CONE,
    output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
    metamer_difference: Optional[float] = None,
) -> None:
    """
    Draw the plate with the computed circle positions and colors.

    :param circles: List of circle definitions [x, y, r]
    :param inside_props: List of inside proportions for each circle
    :param outside_props: List of outside proportions for each circle
    :param inside_color: Color for shape elements
    :param outside_color: Color for background elements
    :param channel_draws: ImageDraw objects for each channel
    :param lum_noise: Luminance noise amount (ignored if metamer_difference provided)
    :param s_cone_noise: S-cone noise amount (ignored if metamer_difference provided)
    :param metamer_difference: Metamer difference for adaptive noise (applied to all channels except metameric axis)
    """
    for i, [x, y, r] in enumerate(circles):
        in_p, out_p = inside_props[i], outside_props[i]

        # chooses inside or outside color
        circle_color = in_p * inside_color + out_p * outside_color

        noise_vector = np.zeros(color_space.dim)

        # Determine which noise to apply
        if metamer_difference is not None:
            # Use adaptive noise based on metamer difference
            adaptive_noise = metamer_difference / 2
            # Apply noise to all channels except the metameric axis
            for channel in range(color_space.dim):
                if channel != color_space.metameric_axis:
                    noise_vector[channel] += np.random.normal(0, adaptive_noise)
        else:
            # Use traditional noise parameters
            if s_cone_noise > 0:
                noise_vector[0] += np.random.normal(0, s_cone_noise)

            if lum_noise > 0:
                # Add luminance noise to all channels except metameric axis
                for channel in range(color_space.dim):
                    if channel != color_space.metameric_axis:
                        noise_vector[channel] += np.random.normal(0, lum_noise)

        circle_color = np.clip(circle_color + noise_vector, 0, None)
        circle_color = color_space.convert(np.array([circle_color]), input_space, output_space)[0]

        # Draw the ellipse
        bounding_box = [x-r, y-r, x+r, y+r]
        if input_space != output_space:
            # Use rounding to nearest integer when discretizing to int
            circle_color = np.round(circle_color * 255).astype(int)
        else:
            noise_vector = np.full((4), np.random.normal(0, lum_noise))
            noise_vector = color_space.convert(noise_vector, ColorSpaceType.CONE, ColorSpaceType.PRINT)
            # lum_dir = lum_dir / np.linalg.norm(lum_dir) * noise_vector
            circle_color = circle_color + noise_vector
            circle_color = np.round((circle_color)).astype(int)

        if len(circle_color) > 4:
            for i in range(len(channel_draws)):
                channel_draws[i].ellipse(bounding_box, fill=tuple(circle_color[3*i:3*i + 3]), width=0)
        else:  # should match the rgb/cmyk length
            channel_draws[0].ellipse(bounding_box, fill=tuple(circle_color), width=0)


def generate_ishihara_plate(
    inside_cone: npt.NDArray,
    outside_cone: npt.NDArray,
    color_space: ColorSpace,
    secret: Union[int, str] = _SECRETS[0],
    circles: Optional[List[List[float]]] = None,
    num_samples: int = 100,
    dot_sizes: List[int] = [16, 22, 28],
    image_size: int = 1024,
    seed: int = 0,
    lum_noise: float = 0,
    s_cone_noise: float = 0,
    noise: float = 0,
    input_space: ColorSpaceType = ColorSpaceType.CONE,
    output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
    gradient: bool = False,
    corner_label: Optional[str] = None,
    corner_color: npt.ArrayLike = np.array([255/2, 255/2, 255/2, 255/2, 0, 0]).astype(int),
    background_color: npt.NDArray = np.array([0, 0, 0, 0, 0, 0]),
    blur_radius: float = 1.0,
    metamer_difference: Optional[float] = None
) -> List[Image.Image]:
    """
    Generate an Ishihara Plate with specified properties.

    Parameters:
    -----------
    inside_cone: npt.NDArray
        Color for the inside of the plate (shape elements).
    outside_cone: npt.NDArray
        Color for the outside of the plate (background elements).
    secret : Union[int, str]
        Specifies which secret file to use from the secrets directory.
    circles : Optional[List[List[float]]]
        Pre-computed circle geometry. If None, geometry will be generated.
    num_samples : int
        Number of samples to take for gradient plates.
    dot_sizes : List[int]
        List of dot sizes to use in the plate.
    image_size : int
        Size of the output image.
    seed : int
        RNG seed for plate generation.
    lum_noise : float
        Amount of luminance noise to add to all channels except metameric axis.
        Ignored if metamer_difference is provided.
    s_cone_noise : float
        Amount of S-cone specific noise to add.
        Ignored if metamer_difference is provided.
    noise : float
        Legacy noise parameter (deprecated).
    metamer_difference : Optional[float]
        Metamer difference for adaptive noise calculation. When provided,
        adaptive noise (metamer_difference/2) is applied to all channels except
        the metameric axis, and lum_noise/s_cone_noise are ignored.
    output_space : ColorSpaceType
        Target color space for output.
    gradient : bool
        Whether to generate a gradient plate.
    corner_label : str
        Optional label text to draw in the corner of the plate.
    corner_color : npt.ArrayLike
        Color for the corner label.

    Returns:
    --------
    List[Image.Image]
        A list of images, one for each channel.
    """
    # Validate inputs
    if isinstance(secret, int) and secret not in _SECRETS:
        raise ValueError(f"Invalid Hidden Symbol {secret}")
    elif isinstance(secret, str) and secret not in _SECRETS:
        raise ValueError(f"Invalid Hidden Symbol {secret}")

    if not gradient:
        num_samples = 1
        if noise != 0:
            raise ValueError("None-zero noise is not supported for non-gradient plates -- it doesn`t make sense!")
    dim = inside_cone.shape[0]

    # Load secret image
    with resources.path("TetriumColor.Assets.HiddenImages", f"{str(secret)}.png") as data_path:
        secret_img = Image.open(data_path)

    secret_img = secret_img.resize([image_size, image_size])
    secret_img = np.asarray(secret_img)

    # Generate or use provided geometry
    if circles is None:
        circles = _generate_geometry(dot_sizes, image_size, seed)

    # Calculate inside/outside proportions
    inside_props, outside_props = _compute_inside_outside(
        circles, secret_img, image_size, num_samples, noise, gradient
    )

    # Become fancier eventually - determine the # of channels / type of image based on the output space -- we need some function that maps each of the color spaces to a specific number
    background_color = tuple(background_color.tolist())
    if output_space.num_channels() > 4:  # 6P disp
        # INSERT_YOUR_CODE
        channels: List[Image.Image] = [Image.new(mode="RGB", size=(
            image_size, image_size), color=background_color[i * 3: i * 3 + 3]) for i in range(2)]
    elif output_space.num_channels() == 4:  # any 4P space like CMYK
        channels: List[Image.Image] = [Image.new(mode="RGBA", size=(
            image_size, image_size), color=(0, 0, 0, 0)) for i in range(1)]
    else:  # any 3P space like sRGB, XYZ, OKLAB, OKLABM1, CIELAB
        channels: List[Image.Image] = [Image.new(mode="RGB", size=(
            image_size, image_size), color=background_color[0:3]) for i in range(1)]

    channel_draws = [ImageDraw.Draw(ch) for ch in channels]

    # Draw plate
    _draw_plate(
        circles, inside_props, outside_props, inside_cone, outside_cone, color_space,
        channel_draws, lum_noise, s_cone_noise, input_space, output_space, metamer_difference
    )

    # Draw corner label if provided
    if corner_label:
        font = ImageFont.load_default(size=150)
        corner_color_array = np.array(corner_color)
        if np.issubdtype(corner_color_array.dtype, np.floating):
            corner_color_array = (corner_color_array * 255).astype(int)
        for i in range(len(channels)):
            channel_draws[i].text((10, 10), corner_label, fill=tuple(corner_color_array[:3]), font=font)

    # INSERT_YOUR_CODE
    # Blur the image(s) in channels using PIL's GaussianBlur filter
    from PIL import ImageFilter
    if blur_radius > 0:
        for i in range(len(channels)):
            channels[i] = channels[i].filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return channels


def export_plate(rgb_img: Image.Image, ocv_img: Image.Image, filename_rgb: str, filename_ocv: str):
    """
    Export the generated plate images to files.

    :param rgb_img: RGB channel image
    :param ocv_img: OCV channel image
    :param filename_rgb: Filename for the RGB image
    :param filename_ocv: Filename for the OCV image
    """
    rgb_img.save(filename_rgb)
    ocv_img.save(filename_ocv)


def GenerateLandoltC(output_dir: str, gap_directions: List[str] = None):
    """
    Generate Landolt C optotypes with different gap orientations.

    The Landolt C is a standardized optotype that looks like a circle with a gap.
    Gap directions: 'up', 'down', 'left', 'right', 'up-right', 'down-right', 'up-left', 'down-left'

    :param output_dir: Directory to save the generated images.
    :param gap_directions: List of gap directions to generate. If None, generates all 8 directions.
    """
    if gap_directions is None:
        gap_directions = ['up', 'down', 'left', 'right', 'up-right', 'down-right', 'up-left', 'down-left']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Landolt C specifications
    image_size = 1024
    outer_radius = image_size // 2 - 100  # Leave more margin to make C smaller
    # inner_radius = outer_radius * 0.6     # Standard Landolt C proportions
    gap_width = outer_radius * 2 * 0.2        # Gap width is 1/5 of outer diameter
    stroke_width = gap_width

    for i, direction in enumerate(gap_directions):
        # Create a transparent image
        img = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        center_x, center_y = image_size // 2, image_size // 2

        # Calculate gap position based on direction
        gap_angle = _get_gap_angle(direction)
        gap_start_angle = gap_angle - np.degrees(np.arcsin(gap_width / (2 * outer_radius)))
        gap_end_angle = gap_angle + np.degrees(np.arcsin(gap_width / (2 * outer_radius)))

        # Draw the outer ring with gap
        _draw_landolt_c_ring(draw, center_x, center_y, outer_radius, stroke_width,
                             gap_start_angle, gap_end_angle, (255, 255, 255, 255))

        # Save the image with direction identifier
        img.save(output_path / f"landolt_{direction}.png")
        print(f"Generated Landolt C with gap direction: {direction}")


def _get_gap_angle(direction: str) -> float:
    """Get the angle in degrees for the gap direction."""
    # Note: In image coordinates, Y increases downward, so we need to flip the angles
    angle_map = {
        'up': 270,      # Gap pointing up (toward top of image)
        'up-right': 315,
        'right': 0,     # Gap pointing right (toward right of image)
        'down-right': 45,
        'down': 90,     # Gap pointing down (toward bottom of image)
        'down-left': 135,
        'left': 180,    # Gap pointing left (toward left of image)
        'up-left': 225
    }
    return angle_map.get(direction, 0)


def _draw_landolt_c_ring(draw: ImageDraw.ImageDraw, center_x: int, center_y: int,
                         outer_radius: float, stroke_width: float,
                         gap_start_angle: float, gap_end_angle: float,
                         color: Tuple[int, int, int, int]):
    """Draw a Landolt C ring with a gap."""
    # Convert angles to radians
    gap_start_rad = np.radians(gap_start_angle)
    gap_end_rad = np.radians(gap_end_angle)

    # Draw the ring in segments, skipping the gap
    num_segments = 64
    segment_angle = 2 * np.pi / num_segments

    for i in range(num_segments):
        start_angle = i * segment_angle
        end_angle = (i + 1) * segment_angle

        # Skip the gap segment
        if _angle_in_gap(start_angle, end_angle, gap_start_rad, gap_end_rad):
            continue

        # Calculate arc endpoints
        x1_start = center_x + outer_radius * np.cos(start_angle)
        y1_start = center_y + outer_radius * np.sin(start_angle)
        x1_end = center_x + outer_radius * np.cos(end_angle)
        y1_end = center_y + outer_radius * np.sin(end_angle)

        x2_start = center_x + (outer_radius - stroke_width) * np.cos(start_angle)
        y2_start = center_y + (outer_radius - stroke_width) * np.sin(start_angle)
        x2_end = center_x + (outer_radius - stroke_width) * np.cos(end_angle)
        y2_end = center_y + (outer_radius - stroke_width) * np.sin(end_angle)

        # Draw the arc segment as a polygon
        points = [
            (x1_start, y1_start),
            (x1_end, y1_end),
            (x2_end, y2_end),
            (x2_start, y2_start)
        ]
        draw.polygon(points, fill=color)


def _angle_in_gap(start_angle: float, end_angle: float, gap_start: float, gap_end: float) -> bool:
    """Check if a segment overlaps with the gap."""
    # Normalize angles to [0, 2π]
    start_angle = start_angle % (2 * np.pi)
    end_angle = end_angle % (2 * np.pi)
    gap_start = gap_start % (2 * np.pi)
    gap_end = gap_end % (2 * np.pi)

    # Check if the segment overlaps with the gap
    # A segment overlaps if either its start or end is within the gap range
    if gap_start <= gap_end:
        # Normal case: gap doesn't wrap around
        return (start_angle >= gap_start and start_angle <= gap_end) or \
               (end_angle >= gap_start and end_angle <= gap_end) or \
               (start_angle <= gap_start and end_angle >= gap_end)
    else:
        # Wrap-around case: gap crosses 0 degrees
        return (start_angle >= gap_start or start_angle <= gap_end) or \
               (end_angle >= gap_start or end_angle <= gap_end)


def GenerateHiddenImages(output_dir: str):
    """
    Generate a series of images from 1-99 that resemble the style of Assets/HiddenImages/27.png.
    Each image will have a transparent background, a white circle, and a black number centered.

    :param output_dir: Directory to save the generated images.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for number in range(10, 100):
        # Create a transparent image
        image_size = 1024
        img = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw a white circle
        circle_radius = image_size // 2
        circle_bbox = [
            (image_size // 2 - circle_radius, image_size // 2 - circle_radius),
            (image_size // 2 + circle_radius, image_size // 2 + circle_radius),
        ]
        draw.ellipse(circle_bbox, fill=(255, 255, 255, 255))

        # Draw the black number centered
        font_size = 700
        resource = resources.files("TetriumColor.Assets.Fonts") / "Rubik-Medium.ttf"
        with as_file(resource) as font_path:
            font = ImageFont.truetype(str(font_path), size=font_size)
        text = str(number)
        draw.text((image_size/2, image_size/2), text, font=font, anchor="mm", fill=(0, 0, 0, 255))

        # Save the image
        img.save(output_path / f"{number}.png")


if __name__ == "__main__":
    # Generate both numeric hidden images and Landolt C optotypes
    # GenerateHiddenImages("TetriumColor/Assets/HiddenImages")
    GenerateLandoltC("TetriumColor/Assets/HiddenImages")
