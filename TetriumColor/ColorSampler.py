import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple, Union, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import pickle
import hashlib

from importlib import resources
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer import Spectra
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlateGenerator, generate_ishihara_plate
from TetriumColor.Utils.CustomTypes import TetraColor, PlateColor
import TetriumColor.ColorMath.Geometry as Geometry
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation, FindMaximumIn1DimDirection, FindMaximumWidthAlongDirection, excitations_to_contrast, receptor_isolate_spectral

from TetriumColor.Measurement import load_primaries_from_csv


class ColorSampler:
    """
    A class for sampling colors in various ways from a color space.

    This class centralizes color sampling functionality and handles efficient
    gamut mapping by computing and caching the gamut boundary information.
    """

    def __init__(self, color_space: ColorSpace, cubemap_size: int = 64):
        """
        Initialize the ColorSampler with a ColorSpace.

        Parameters:
            color_space (ColorSpace): The color space to sample from
        """
        self.color_space = color_space
        self._gamut_cubemap = None
        self._lum_range = None
        self._sat_range = None
        self._cubemap_size = cubemap_size  # Default size for the cubemap
        self._max_L = color_space.max_L

        # Try to load cubemap from cache during initialization
        if not self._load_from_cache():
            print("Failed to load cubemap from cache, generating new cubemap")
            self.get_gamut_lut()

    def _get_cache_filename(self) -> str:
        """
        Generate a unique filename for caching the cubemap based on the color space.

        Returns:
            str: Cache filename
        """
        # Add cubemap size to the hash input
        hash_input = f"{str(self.color_space)}|cubemap_size:{self._cubemap_size}"

        # Create MD5 hash
        hash_obj = hashlib.md5(hash_input.encode())
        hash_str = hash_obj.hexdigest()

        return f"cubemap_{hash_str}.pkl"

    def _save_to_cache(self) -> None:
        """Save cubemap and range data to cache."""
        if self._gamut_cubemap is None:
            return

        # Save data
        cache_data = {
            'cubemap': self._gamut_cubemap,
            'lum_range': self._lum_range,
            'sat_range': self._sat_range
        }

        _cache_file = self._get_cache_filename()
        try:
            with resources.path("TetriumColor.Assets.Cache", _cache_file) as path:
                with open(path, "wb") as f:
                    pickle.dump(cache_data, f)
            print(f"Saved cubemap cache to {_cache_file}")
        except Exception as e:
            print(f"Failed to save cubemap cache: {e}")

    def _load_from_cache(self) -> bool:
        """
        Load cubemap and range data from cache.

        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        # Get cache filename
        _cache_file = self._get_cache_filename()
        with resources.path("TetriumColor.Assets.Cache", _cache_file) as path:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        cache_data = pickle.load(f)
                    self._gamut_cubemap = cache_data['cubemap']
                    self._lum_range = cache_data['lum_range']
                    self._sat_range = cache_data['sat_range']
                    return True
                except Exception as e:
                    print(f"Failed to load cubemap from cache: {e}")
                    return False
            else:
                return False

    def _generate_gamut_cubemap(self, cubemap_size: Optional[int] = None) -> Tuple[Dict, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Generate a cubemap representation of the gamut boundaries.

        Parameters:
            cubemap_size (int, optional): Size of the cubemap images

        Returns:
            Tuple containing:
                - Dict of cubemap images
                - Tuple of luminance and saturation ranges
        """
        if cubemap_size:
            self._cubemap_size = cubemap_size

        # Generate grid of UV coordinates
        all_us = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        all_vs = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        cube_u, cube_v = np.meshgrid(all_us, all_vs)
        flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

        # Get metameric direction matrix
        metamericDirMat = self._get_transform_chrom_to_metameric_dir()
        invMetamericDirMat = np.linalg.inv(metamericDirMat)

        # Process each face of the cube
        lut_dicts = []
        for i in tqdm(range(6), desc="Generating cubemap"):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size)
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self.color_space._hering_to_vsh(vxyz)

            # Generate gamut LUT for this face
            face_dict = {}
            for j in range(len(flattened_u)):
                u, v = flattened_v[j], flattened_u[j]
                angle = tuple(vshh[j, 2:])
                # Find cusp point for this hue angle
                hue_cartesian = self.color_space._vsh_to_hering(np.array([[0, 1, *angle]]))
                max_sat_point = self._find_maximal_saturation(
                    (self.color_space.transform.hering_to_disp @ hue_cartesian.T).T[0])
                max_sat_hering = np.linalg.inv(self.color_space.transform.hering_to_disp) @ max_sat_point
                max_sat_vsh = self.color_space._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
                lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]
                face_dict[(u, v)] = (lum_cusp, sat_cusp)

            lut_dicts.append(face_dict)

        # Compute overall ranges for normalization
        all_values = np.array([list(lut_dicts[i].values()) for i in range(6)]).reshape(-1, 2)
        lum_min, lum_max = np.min(all_values[:, 0]), np.max(all_values[:, 0])
        sat_min, sat_max = np.min(all_values[:, 1]), np.max(all_values[:, 1])

        # Generate cubemap images
        cubemap_images = {}
        for i in range(6):
            img = Image.new('RGB', (self._cubemap_size, self._cubemap_size))
            draw = ImageDraw.Draw(img)

            for j in range(len(flattened_u)):
                u, v = flattened_v[j], flattened_u[j]
                lum_cusp, sat_cusp = lut_dicts[i][(u, v)]
                normalized_lum = (lum_cusp - lum_min) / (lum_max - lum_min)
                normalized_sat = (sat_cusp - sat_min) / (sat_max - sat_min)
                rgb_color = (int(normalized_lum * 255), int(normalized_sat * 255), 0)
                draw.point((int(u * self._cubemap_size), int(v * self._cubemap_size)), fill=rgb_color)

            cubemap_images[i] = img

        return cubemap_images, ((lum_min, lum_max), (sat_min, sat_max))

    def get_gamut_lut(self, force_recompute: bool = False) -> None:
        """
        Get the gamut lookup table, represented as a cubemap.

        Tries to load from cache first, or computes and caches if not available.

        Parameters:
            force_recompute (bool): Whether to force recomputation
        """
        if force_recompute:
            # Skip cache if forced to recompute
            self._gamut_cubemap, ranges = self._generate_gamut_cubemap()
            self._lum_range, self._sat_range = ranges
            # Save to cache for future use
            self._save_to_cache()
        elif self._gamut_cubemap is None:
            # Try loading from cache first (already tried during init)
            # If not loaded, generate and save
            self._gamut_cubemap, ranges = self._generate_gamut_cubemap()
            self._lum_range, self._sat_range = ranges
            self._save_to_cache()

    def _get_transform_chrom_to_metameric_dir(self) -> npt.NDArray:
        """
        Get the transformation matrix from chromatic coordinates to metameric direction.

        Returns:
            npt.NDArray: Transformation matrix
        """
        normalized_direction = self.color_space.get_metameric_axis_in(ColorSpaceType.HERING)
        return Geometry.RotateToZAxis(normalized_direction[1:])

    def _angles_to_cube_uv(self, angles: tuple[float, float]):
        """
        Convert angles to cube face index and UV coordinates.

        Parameters:
            angles (tuple): Angles to convert

        Returns:
            Tuple[int, float, float]: Cube face index and UV coordinates
        """
        angles_with_ones = np.array([[0, 1, *angles]])
        x, y, z = self.color_space.convert(angles_with_ones, ColorSpaceType.VSH, ColorSpaceType.HERING)[0, 1:]
        face_id, u, v = Geometry.ConvertXYZToCubeUV(x, y, z)
        return int(face_id), float(u), float(v)

    def _interpolate_from_cubemap(self, angles: tuple) -> Tuple[float, float]:
        """
        Interpolate luminance and saturation values from the cubemap for given angles.

        Parameters:
            angles (tuple): Angles for which to interpolate values

        Returns:
            Tuple[float, float]: Interpolated luminance and saturation values
        """
        if self._gamut_cubemap is None:
            self.get_gamut_lut()  # Initialize the cubemap if not already done

        face_idx, u, v = self._angles_to_cube_uv(angles)

        lum_min, lum_max = self._lum_range
        sat_min, sat_max = self._sat_range

        img = self._gamut_cubemap[face_idx]
        # Convert UV to pixel coordinates
        x, y = u * img.width, v * img.height

        # Get integer pixel coordinates and fractional parts
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, img.width - 1), min(y0 + 1, img.height - 1)
        dx, dy = x - x0, y - y0

        # Get pixel values for the four surrounding pixels
        r00, g00, _ = img.getpixel((x0, y0))
        r10, g10, _ = img.getpixel((x1, y0))
        r01, g01, _ = img.getpixel((x0, y1))
        r11, g11, _ = img.getpixel((x1, y1))

        # Perform bilinear interpolation
        r = (1 - dx) * (1 - dy) * r00 + dx * (1 - dy) * r10 + (1 - dx) * dy * r01 + dx * dy * r11
        g = (1 - dx) * (1 - dy) * g00 + dx * (1 - dy) * g10 + (1 - dx) * dy * g01 + dx * dy * g11

        # Convert normalized values back to actual luminance and saturation
        lum = (r / 255.0) * (lum_max - lum_min) + lum_min
        sat = (g / 255.0) * (sat_max - sat_min) + sat_min

        return lum, sat

    def _find_maximal_saturation(self, hue_direction: npt.NDArray) -> npt.NDArray:
        """
        Find the point with maximal saturation in the given hue direction.

        Parameters:
            hue_direction (npt.NDArray): Hue direction vector

        Returns:
            npt.NDArray: Point with maximal saturation
        """
        from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation
        result = FindMaximalSaturation(hue_direction, np.eye(self.color_space.dim))
        if result is None:
            raise ValueError("Failed to find maximal saturation point")
        return result

    def compute_max_sat_at_luminance(self, luminance: float, angle: tuple) -> float:
        """
        Compute the maximum saturation at a given luminance for a specific angle directly.

        This method avoids using the cubemap and calculates the value precisely.

        Parameters:
            luminance (float): Luminance value
            angle (tuple): Hue angle

        Returns:
            float: Maximum saturation at the given luminance and angle
        """
        # Find the cusp point for this hue angle
        hue_cartesian = self.color_space._vsh_to_hering(np.array([[0, 1, *angle]]))
        max_sat_point = self._find_maximal_saturation(
            (self.color_space.transform.hering_to_disp @ hue_cartesian.T).T[0]
        )
        max_sat_hering = np.linalg.inv(self.color_space.transform.hering_to_disp) @ max_sat_point
        max_sat_vsh = self.color_space._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
        lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]

        # Calculate the maximum saturation at the given luminance
        return self._solve_for_boundary(luminance, self._max_L, lum_cusp, sat_cusp)

    def _solve_for_boundary(self, L: float, max_L: float, lum_cusp: float, sat_cusp: float) -> float:
        """
        Solve for the boundary of the gamut at a given luminance.

        Parameters:
            L (float): Luminance value to solve for
            max_L (float): Maximum luminance value
            lum_cusp (float): Luminance value at the cusp
            sat_cusp (float): Saturation value at the cusp

        Returns:
            float: Saturation value at the boundary
        """
        if L >= lum_cusp:
            slope = -(max_L - lum_cusp) / sat_cusp
            return (L - max_L) / slope
        else:
            slope = lum_cusp / sat_cusp
            return L / slope

    def max_sat_at_luminance(self, luminance: float,
                             angles: Union[List[tuple], tuple]) -> Union[float, List[float]]:
        """
        Get the maximum saturation at a given luminance for specific angle(s).

        Uses the cubemap for interpolation if available, otherwise computes directly.

        Parameters:
            luminance (float): Luminance value
            angles (tuple or List[tuple]): Hue angle(s)

        Returns:
            float or List[float]: Maximum saturation at the given luminance for each angle
        """
        # Handle single angle vs list of angles
        is_single = isinstance(angles, tuple)
        angle_list = [angles] if is_single else angles

        # If cubemap is available, use interpolation (faster)
        if self._gamut_cubemap is not None:
            sat_maxes = []
            for angle in angle_list:
                lum_cusp, sat_cusp = self._interpolate_from_cubemap(angle)
                sat_max = self._solve_for_boundary(luminance, self._max_L, lum_cusp, sat_cusp)
                sat_maxes.append(sat_max)
        # Otherwise compute directly (more accurate)
        else:
            sat_maxes = [self.compute_max_sat_at_luminance(luminance, angle) for angle in angle_list]

        return sat_maxes[0] if is_single else sat_maxes

    def remap_to_gamut(self, vshh: npt.NDArray) -> npt.NDArray:
        """
        Remap points to be within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Remapped points that are in gamut
        """
        # Copy the input to avoid modifying it
        remapped_vshh = vshh.copy()

        # Remap each point
        for i in range(len(remapped_vshh)):
            angle = tuple(remapped_vshh[i, 2:])

            # Calculate the maximum saturation at the given luminance
            sat_max = self.max_sat_at_luminance(remapped_vshh[i, 0], angle)

            # Clamp the saturation to the maximum
            remapped_vshh[i, 1] = min(sat_max, remapped_vshh[i, 1])

        return remapped_vshh

    def is_in_gamut(self, vshh: npt.NDArray) -> Union[bool, npt.NDArray]:
        """
        Check if points are within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            bool or npt.NDArray: Boolean indicating if point(s) are in gamut
        """
        remapped = self.remap_to_gamut(vshh)

        # If single point
        if vshh.ndim == 1:
            return np.allclose(vshh, remapped, rtol=1e-05, atol=1e-08)

        # Check if any coordinates changed for multiple points
        return np.all(np.isclose(vshh, remapped, rtol=1e-05, atol=1e-08), axis=1)

    def sample_hue_manifold(self, luminance: float, saturation: float, num_points: int) -> npt.NDArray:
        """
        Sample hue directions at a given luminance and saturation.

        Parameters:
            luminance (float): Luminance value
            saturation (float): Saturation value
            num_points (int): Number of points to sample

        Returns:
            npt.NDArray: Array of sampled points in VSH space
        """
        all_angles = Geometry.SampleAnglesEqually(num_points, self.color_space.dim-1)
        all_vshh = np.zeros((len(all_angles), self.color_space.dim))
        all_vshh[:, 0] = luminance
        all_vshh[:, 1] = saturation
        all_vshh[:, 2:] = all_angles
        return all_vshh

    def sample_equiluminant_plane(self, luminance: float, num_points: int = 100,
                                  remap_to_gamut: bool = True) -> npt.NDArray:
        """
        Sample points on an equiluminant plane.

        Parameters:
            luminance (float): Luminance value for the plane
            num_points (int): Number of points to sample
            remap_to_gamut (bool): Whether to remap points to be within the gamut

        Returns:
            npt.NDArray: Array of sampled points in VSH space
        """
        # Sample hue directions
        vshh = self.sample_hue_manifold(luminance, 1.0, num_points)

        # Remap to gamut if requested
        if remap_to_gamut:
            vshh = self.remap_to_gamut(vshh)

        return vshh

    # need to write an equivalent function for sampling the boundary of the OBS solid
    def sample_full_colors(self, num_points=10000) -> npt.NDArray:
        """Generate the Full Colors Boundary of the Object Color Solid

        Args:
            num_points (int, optional): number of points to generate the boundary of. Defaults to 10000.

        Returns:
            npt.NDArray: Array of full colors in Hering Space
        """
        # Create a hash based on observer and num_points
        observer_hash = hashlib.md5(str(self.color_space.observer).encode()).hexdigest()
        cache_filename = f"full_colors_{observer_hash}_{num_points}.pkl"

        # Try to load from cache first
        try:
            with resources.path("TetriumColor.Assets.Cache", cache_filename) as path:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        cached_data = pickle.load(f)
                    print(f"Loaded full colors from cache: {cache_filename}")
                    return cached_data
        except Exception as e:
            print(f"Failed to load full colors from cache: {e}")

        # If not in cache, compute the full colors
        # for every hue
        vshh: npt.NDArray = self.sample_hue_manifold(1, 0.5, num_points)
        all_disp_points = self.color_space.convert(vshh, ColorSpaceType.VSH, ColorSpaceType.DISP)

        # For every point, find the reflectance of maximum saturation
        generating_vecs = self.color_space.observer.get_normalized_sensor_matrix(wavelengths=np.arange(360, 831, 1)).T
        pts = []
        for pt in tqdm(all_disp_points):
            res = FindMaximalSaturation(pt, generating_vecs=generating_vecs)
            if res is not None:
                pts += [res]
        max_sat_cartesian_per_angle = np.array(pts)

        # return point of max saturation for every hue
        hering_points = self.color_space.convert(
            max_sat_cartesian_per_angle, ColorSpaceType.DISP, ColorSpaceType.HERING)

        # Save to cache
        try:
            with resources.path("TetriumColor.Assets.Cache", cache_filename) as path:
                with open(path, "wb") as f:
                    pickle.dump(hering_points, f)
            print(f"Saved full colors to cache: {cache_filename}")
        except Exception as e:
            print(f"Failed to save full colors to cache: {e}")

        return hering_points

    @staticmethod
    def _concatenate_cubemap(faces):
        """
        Concatenate cubemap textures into a single cross-layout image with correct orientation.

        Parameters:
            basename (str): The base name of the input files, e.g., "texture". Files are assumed to follow the format "<basename>_i.png".
                        `i` corresponds to the index: 0 (+X), 1 (-X), 2 (+Y), 3 (-Y), 4 (+Z), 5 (-Z).
        """
        # Assume all faces are the same size
        face_width, face_height = faces[0].size

        # Create a blank image for the cross layout
        width = 4 * face_width
        height = 3 * face_height
        cubemap_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # +X (0)
        cubemap_image.paste(faces[0], (2 * face_width, face_height))
        # -X (1) flipped horizontally
        cubemap_image.paste(faces[1], (0, face_height))
        # +Y (2) flipped vertically
        cubemap_image.paste(faces[3], (face_width, 0))  # swap 2 and 3 because of the flipped orientation i think
        # -Y (3) flipped vertically
        cubemap_image.paste(faces[2], (face_width, 2 * face_height))
        # +Z (4)
        cubemap_image.paste(faces[4], (face_width, face_height))
        # -Z (5) flipped horizontally
        cubemap_image.paste(faces[5], (3 * face_width, face_height))

        # Save the concatenated image
        return cubemap_image

    def generate_cubemap(self, luminance: float, saturation: float,
                         display_color_space: ColorSpaceType = ColorSpaceType.SRGB) -> List[Image.Image]:
        """Generate a cubemap within the gamut boundaries

        Args:
            luminance (float): luminance
            saturation (float): saturation
            display_color_space (ColorSpaceType, optional): color space that you want to transform to. Defaults to ColorSpaceType.SRGB.

        Returns:
            PIL.Image.Image: Image of the cubemap
        """

        # Generate grid of UV coordinates
        all_us = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        all_vs = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        cube_u, cube_v = np.meshgrid(all_us, all_vs)

        # Get metameric direction matrix
        metamericDirMat = self._get_transform_chrom_to_metameric_dir()
        invMetamericDirMat = np.linalg.inv(metamericDirMat)

        # Process each face of the cube
        cubemap_images = []
        for i in tqdm(range(6), desc="Generating cubemap"):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            max_saturations = np.array(self._gamut_cubemap[i]).reshape(-1, 3)[:, 1]/255
            normalized_saturations = (
                max_saturations * (self._sat_range[1] - self._sat_range[0])) + self._sat_range[0]

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size) * luminance
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self.color_space.convert(vxyz, ColorSpaceType.HERING, ColorSpaceType.VSH)
            vshh[:, 1] = np.min(
                np.vstack((np.full(normalized_saturations.shape, saturation), normalized_saturations)), axis=0)
            remapped_points = self.remap_to_gamut(vshh)
            corresponding_colors = self.color_space.convert(remapped_points, ColorSpaceType.VSH, display_color_space)
            # Convert colors to 8-bit format and reshape for image saving
            corresponding_colors = np.clip(corresponding_colors, 0, 1) * 255
            corresponding_colors = corresponding_colors.astype(np.uint8)
            corresponding_colors = corresponding_colors.reshape(
                self._cubemap_size, self._cubemap_size, 3).transpose(1, 0, 2)

            # Create an image from the array
            cubemap_images += [Image.fromarray(corresponding_colors, 'RGB')]

        return cubemap_images

    def generate_concatenated_cubemap(self, luminance: float, saturation: float,
                                      display_color_space: ColorSpaceType = ColorSpaceType.SRGB) -> Image.Image:
        """Generate a cubemap and return it concatenated"""

        cubemap_images = self.generate_cubemap(luminance, saturation, display_color_space)
        return self._concatenate_cubemap(cubemap_images)

    def output_cubemap_values(self, luminance: float, saturation: float,
                              display_color_space: ColorSpaceType = ColorSpaceType.SRGB) -> List[npt.NDArray]:
        """Generate a cubemap within the gamut boundaries

        Args:
            luminance (float): luminance
            saturation (float): saturation
            display_color_space (ColorSpaceType, optional): color space that you want to transform to. Defaults to ColorSpaceType.SRGB.

        Returns:
            PIL.Image.Image: Image of the cubemap
        """

        # Generate grid of UV coordinates
        all_us = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        all_vs = (np.arange(self._cubemap_size) + 0.5) / self._cubemap_size
        cube_u, cube_v = np.meshgrid(all_us, all_vs)

        # Get metameric direction matrix
        metamericDirMat = self._get_transform_chrom_to_metameric_dir()
        invMetamericDirMat = np.linalg.inv(metamericDirMat)

        # Process each face of the cube
        colors = []
        for i in tqdm(range(6), desc="Generating cubemap"):
            # Convert UV to XYZ coordinates for this cube face
            xyz = Geometry.ConvertCubeUVToXYZ(i, cube_u, cube_v, 1).reshape(-1, 3)
            xyz = np.dot(invMetamericDirMat, xyz.T).T

            max_saturations = np.array(self._gamut_cubemap[i]).reshape(-1, 3)[:, 1]/255
            normalized_saturations = (
                max_saturations * (self._sat_range[1] - self._sat_range[0])) + self._sat_range[0]

            # Create hering coordinates with unit luminance
            lum_vector = np.ones(self._cubemap_size * self._cubemap_size) * luminance
            vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))

            # Convert to VSH space
            vshh = self.color_space.convert(vxyz, ColorSpaceType.HERING, ColorSpaceType.VSH)
            vshh[:, 1] = np.min(
                np.vstack((np.full(normalized_saturations.shape, saturation), normalized_saturations)), axis=0)
            remapped_points = self.remap_to_gamut(vshh)
            colors += [self.color_space.convert(remapped_points, ColorSpaceType.VSH, display_color_space)]

        return colors

    def get_metameric_pairs(self, luminance: float, saturation: float, cube_idx: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """ Get the metamer points for a given luminance and cube index
        Args:
            luminance (float): luminance value
            saturation (float): saturation value
            cube_idx (int): cube index
            grid_size (int): grid size
            color_space_transform (ColorSpaceTransform): color space transform object

        Returns:
            npt.NDArray: The metamer points
        """
        disp_points = self.output_cubemap_values(luminance, saturation, ColorSpaceType.DISP)[cube_idx]
        metamer_dir_in_disp = self.color_space.get_metameric_axis_in(ColorSpaceType.DISP)

        vec = np.zeros(self.color_space.dim)
        vec[0] = luminance

        metamers_in_disp = np.zeros((disp_points.shape[0], 2, self.color_space.dim))
        for i in tqdm(range(metamers_in_disp.shape[0]), desc="Generating plates"):
            # points in contention in disp space, bounded by unit cube scaled by vectors, direction is the metameric axis
            metamers_in_disp[i] = np.array(FindMaximumIn1DimDirection(
                disp_points[i], metamer_dir_in_disp, np.eye(self.color_space.dim)))

        cones = self.color_space.convert(metamers_in_disp.reshape(-1, self.color_space.dim),
                                         ColorSpaceType.DISP, ColorSpaceType.CONE)
        return metamers_in_disp, cones.reshape(-1, 2, self.color_space.dim)

    def get_maximal_metameric_pairs(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get Maximal Metameric Color Pairs

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: metamers_in_disp, cones
        """
        metamer_dir_in_disp = self.color_space.get_metameric_axis_in(ColorSpaceType.DISP)
        metamers_in_disp = np.array(FindMaximumWidthAlongDirection(metamer_dir_in_disp, np.eye(self.color_space.dim)))
        cones = self.color_space.convert(metamers_in_disp.reshape(-1, self.color_space.dim),
                                         ColorSpaceType.DISP, ColorSpaceType.CONE)
        return metamers_in_disp, cones.reshape(-1, 2, self.color_space.dim)

    def get_metameric_grid_plates(self, luminance: float, saturation: float,
                                  cube_idx: int, secrets: Optional[List[int]] = None,
                                  lum_noise=0.0,
                                  s_cone_noise=0.0,
                                  output_space: ColorSpaceType = ColorSpaceType.DISP_6P,
                                  background: None | npt.NDArray = None,
                                  isSRGB: bool = False) -> List[Tuple[Image.Image, Image.Image]]:
        """ Get the metamer points for a given luminance and cube index
        Args:
            luminance (float): luminance value
            saturation (float): saturation value
            cube_idx (int): cube index
            grid_size (int): grid size
            color_space_transform (ColorSpaceTransform): color space transform object

        Returns:
            npt.NDArray: The metamer points
        """
        disp_points = self.output_cubemap_values(luminance, saturation, ColorSpaceType.DISP)[cube_idx]
        metamer_dir_in_disp = self.color_space.get_metameric_axis_in(ColorSpaceType.DISP)
        if secrets is None:
            secrets = np.random.randint(10, 100, size=len(disp_points)).tolist()

        metamers_in_disp = np.zeros((disp_points.shape[0], 2, self.color_space.dim))
        plates = []
        plate_generator = IshiharaPlateGenerator(seed=0)
        for i in tqdm(range(metamers_in_disp.shape[0]), desc="Generating plates"):
            # points in contention in disp space, bounded by unit cube scaled by vectors, direction is the metameric axis
            # Find metamers in unit cube space first, then scale to display space
            # metamers_unit_cube = np.array(FindMaximumIn1DimDirection(
            #     disp_points[i] / self.color_space.transform.white_weights,
            #     metamer_dir_in_disp / self.color_space.transform.white_weights,
            #     np.eye(self.color_space.dim)))
            metamers_in_disp[i] = np.clip(FindMaximumIn1DimDirection(
                disp_points[i],
                metamer_dir_in_disp,
                np.eye(self.color_space.dim)), 0, 1)

            # Scale back to display space
            # metamers_in_disp[i] = metamers_unit_cube * self.color_space.transform.white_weights
            colors = self.color_space.convert(metamers_in_disp[i], ColorSpaceType.DISP, ColorSpaceType.CONE)
            # Output as a copyable numpy array

            if False:
                primaries = load_primaries_from_csv("../../measurements/2025-10-11/primaries")

                bgor = self.color_space.convert(colors, ColorSpaceType.CONE, ColorSpaceType.DISP)
                metamers = [Spectra.add(Spectra.multiply(primaries, bgor[0])),
                            Spectra.add(Spectra.multiply(primaries, bgor[1]))]
                cones = self.color_space.observer.observe_spectras(metamers)
                print(cones)
                print("Q diff: ", np.abs(cones[0][self.color_space.metameric_axis] -
                      cones[1][self.color_space.metameric_axis]))
                assert np.all((metamers_in_disp[i] >= 0) & (metamers_in_disp[i] <= 1)
                              ), f"metamer_vals values not in [0, 1]: {metamers_in_disp[i]}"

            plates += [plate_generator.GeneratePlate(colors[0], colors[1], self.color_space, secrets[i], output_space,
                                                     lum_noise=lum_noise, s_cone_noise=s_cone_noise)]

        return plates

    def get_metameric_grid_plate(self, luminance: float, saturation: float,
                                 cube_idx: int, grid_idx: tuple[int, int],
                                 secret: Optional[int] = None, lum_noise: float = 0.0,
                                 s_cone_noise: float = 0.0) -> Tuple[Image.Image, Image.Image]:
        """ Get the metamer points for a given luminance and cube index
        Args:
            luminance (float): luminance value
            saturation (float): saturation value
            cube_idx (int): cube index
            grid_idx (int): grid index into the cubemap (out of six)
            secret (Optional[int]): secret number for the plate
            lum_noise (float): noise to add to the luminance channel
            s_cone_noise (float): noise to add to the S-cone channel

        Returns:
            Tuple[Image.Image, Image.Image]: The metamer plates in RGB/OCV
        """
        disp_points = self.output_cubemap_values(luminance, saturation, ColorSpaceType.DISP)[cube_idx]
        metamer_dir_in_disp = self.color_space.get_metameric_axis_in(ColorSpaceType.DISP)
        if secret is None:
            secret = np.random.randint(10, 100)

        vec = np.zeros(self.color_space.dim)
        vec[0] = luminance
        background = self.color_space.convert(vec, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

        metamers_in_disp = np.zeros((disp_points.shape[0], 2, self.color_space.dim))
        plates = []
        i = grid_idx[0] * self._cubemap_size + grid_idx[1]

        metamers_in_disp[i] = np.array(FindMaximumIn1DimDirection(
            disp_points[i], metamer_dir_in_disp, np.eye(self.color_space.dim)))

        cones = self.color_space.convert(metamers_in_disp[i], ColorSpaceType.DISP, ColorSpaceType.CONE)

        return generate_ishihara_plate(cones[0], cones[1], self.color_space, secret, lum_noise=lum_noise, s_cone_noise=s_cone_noise)

    def get_cone_contrast_metamers_brainard(self, target_contrasts: npt.NDArray,  # shape: [n_primaries]
                                            background_primary: npt.NDArray,  # shape: [n_primaries]
                                            isPlotResults=False) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compute the cone contrast metamers using Brainard's modulation method

        Args:
            target_contrasts (npt.NDArray): the target cone contrasts to achieve
            background_primary (npt.NDArray): the background primary choice

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: two primary modulations
        """
        if self.color_space.display_primaries is None:
            raise ValueError("Display primaries are not defined in the color space.")

        n_wavelengths = len(self.color_space.observer.wavelengths)
        n_primaries = len(self.color_space.display_primaries)
        n_receptors = len(self.color_space.observer.sensor_matrix)
        n_basis = n_primaries

        observer = self.color_space.observer
        primaries = self.color_space.display_primaries

        # Synthetic example values
        wls = np.linspace(380, 780, n_wavelengths)
        T_receptors = observer.sensor_matrix
        B_primary = np.array([p.data for p in primaries]).T
        ambientSpd = np.zeros(wls.shape)  # np.sum(B_primary / 2.0, axis=1)
        targetBasis = np.random.rand(n_wavelengths, n_basis)  # not using this
        projectIndices = np.arange(n_wavelengths)  # not using this either
        # targetContrasts = np.array([0, 0, -0.03, 0])
        # backgroundPrimary = 0.5 * np.ones(n_primaries)
        initialPrimary = np.zeros(n_primaries)
        primaryHeadroom = 0.0
        targetLambda = 0.0

        # --- Run Optimization ---
        modulatingPrimary, upperPrimary, lowerPrimary = receptor_isolate_spectral(
            T_receptors, target_contrasts,
            B_primary, background_primary, initialPrimary,
            primaryHeadroom, targetBasis, projectIndices,
            targetLambda, ambientSpd,
            POSITIVE_ONLY=False,
            EXCITATIONS=False
        )

        if isPlotResults:
            backgroundSpd = B_primary @ background_primary + ambientSpd
            upperIsolatingSpd = B_primary @ upperPrimary + ambientSpd
            lowerIsolatingSpd = B_primary @ lowerPrimary + ambientSpd

            backgroundResp = T_receptors @ backgroundSpd
            upperIsolatingResp = T_receptors @ upperIsolatingSpd
            lowerIsolatingResp = T_receptors @ lowerIsolatingSpd
            obtainedUpperContrasts = excitations_to_contrast(upperIsolatingResp, backgroundResp)
            obtainedLowerContrasts = excitations_to_contrast(lowerIsolatingResp, backgroundResp)

            print("Target contrasts: ", np.round(target_contrasts, 2))
            print("Obtained Upper contrasts: ", np.round(obtainedUpperContrasts, 2))
            print("Obtained Lower Contrasts", np.round(obtainedLowerContrasts, 2))

            import matplotlib.pyplot as plt
            # --- Plot Spectra ---
            plt.figure()
            plt.plot(wls, backgroundSpd, 'k-', label='Background', linewidth=2)
            plt.plot(wls, upperIsolatingSpd, 'r-', label='Upper Isolating', linewidth=2)
            plt.plot(wls, lowerIsolatingSpd, 'b--', label='Lower Isolating', linewidth=2)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Power')
            plt.title('Upper and Lower Isolating Modulations vs Background')
            plt.legend()
            plt.show()

        return upperPrimary, lowerPrimary

    def get_cone_contrast_plate(self, target_contrasts: npt.NDArray,
                                background_primary: npt.NDArray,
                                secret: int | None = None
                                ) -> Tuple[Image.Image, Image.Image]:

        upper, lower = self.get_cone_contrast_metamers_brainard(target_contrasts, background_primary)
        upper_sixd_color = upper[[0, 1, 3, 2, 1, 3]]  # need something better to reuse colorspace code
        lower_sixd_color = lower[[0, 1, 3, 2, 1, 3]]
        plate_color = PlateColor(TetraColor(upper_sixd_color[:3], upper_sixd_color[3:]), TetraColor(
            lower_sixd_color[:3], lower_sixd_color[3:]))

        # need something better to reuse colorspace code
        sixd_background = background_primary[[0, 1, 3, 2, 1, 3]]
        background_tetracolor: TetraColor = TetraColor(sixd_background[:3], sixd_background[3:])

        if secret is None:
            secret = np.random.randint(10, 100)
        return generate_ishihara_plate(plate_color, secret, background_color=background_tetracolor)

    def to_tetra_color(self, vsh_points: npt.NDArray) -> List[TetraColor]:
        """
        Convert VSH points to TetraColor objects.

        Parameters:
            vsh_points (npt.NDArray): Points in VSH space

        Returns:
            List[TetraColor]: List of TetraColor objects
        """
        # Convert to RGB_OCV space
        six_d_color = self.color_space.convert(vsh_points, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

        # Create TetraColor objects
        return [TetraColor(six_d_color[i, :3], six_d_color[i, 3:])
                for i in range(six_d_color.shape[0])]

    def to_plate_color(self, vsh_point: npt.NDArray, background_luminance: float = 0.5) -> PlateColor:
        """
        Create a PlateColor object from a VSH point and a background luminance.

        Parameters:
            vsh_point (npt.NDArray): Point in VSH space for the foreground
            background_luminance (float): Luminance value for the background

        Returns:
            PlateColor: PlateColor object with foreground and background
        """
        # Create a background point with the same hue but different luminance
        background_vsh = np.array([background_luminance, 0.0, *vsh_point[2:]])

        # Convert both points to RGB_OCV
        points = np.vstack([vsh_point, background_vsh])
        six_d_colors = self.color_space.convert(points, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

        # Create TetraColor objects for foreground and background
        foreground = TetraColor(six_d_colors[0, :3], six_d_colors[0, 3:])
        background = TetraColor(six_d_colors[1, :3], six_d_colors[1, 3:])

        # Return the PlateColor
        return PlateColor(foreground, background)
