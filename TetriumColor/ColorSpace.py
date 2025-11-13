import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple
from enum import Enum
from scipy.linalg import orth

from TetriumColor.Observer import Observer, MaxBasisFactory, GetHeringMatrix
from TetriumColor.Observer.Spectra import Illuminant, Spectra
from TetriumColor.Observer.Inks import InkGamut
from TetriumColor.Observer.ColorSpaceTransform import GetConeToXYZPrimaries
from TetriumColor.ColorMath import ConeToDisplay

from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_Lab
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximumWidthAlongDirection, FindMaximumIn1DimDirection
import TetriumColor.ColorMath.Geometry as Geometry


OKLAB_M1 = np.array([
    [0.8189330101, 0.0329845436, 0.0482003018],
    [0.3618667424, 0.9293118715, 0.2643662691],
    [-0.1288597137, 0.0361456387, 0.6338517070]
]).T

OKLAB_M2 = np.array([
    [0.210454, 0.793617, -0.004072],
    [1.977998, -2.428592, 0.450593],
    [0.025904, 0.782771, -0.808675]
])

IPT_M1 = np.array([
    [0.4002, 0.7075, -0.0807],
    [-0.2280, 1.1500, 0.0612],
    [0.0000, 0.0000, 0.9184]
])

M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB


class ColorSpaceType(Enum):
    """ColorSpaceType is the core of the color space system. It defines the different types of color spaces
    that can be used in the system. Each color space type is represented by a string value.
    """
    VSH = "vsh"  # Value-Saturation-Hue
    HERING = "hering"  # Hering opponent color space
    MAXBASIS = "maxbasis"  # Display space (RYGB)
    CONE = "cone"  # Cone responses (SMQL)

    DISP_6P = "disp_6p"  # RGO/BGO 6D representation
    DISP = "disp"  # Display space (RGBO)

    SRGB = "srgb"  # sRGB display
    XYZ = "xyz"  # CIE XYZ color space
    OKLAB = 'oklab'
    OKLABM1 = 'oklabm1'  # OKLAB color space with M1 matrix
    CIELAB = 'cielab'  # CIE L*a*b* color space

    CHROM = "chrom"  # Chromaticity space
    HERING_CHROM = "hering_chrom"  # Hering chromaticity space
    MACLEOD_CHROM = "macleod_chrom"  # MacLeod-Boynton chromaticity space

    # Printer gamut primaries (percentages/area coverages)
    PRINT = "print"

    def __str__(self):
        return self.value

    def num_channels(self) -> int:
        if self == ColorSpaceType.DISP_6P:
            return 6
        elif self == ColorSpaceType.PRINT:
            return 4
        elif self == ColorSpaceType.SRGB or self == ColorSpaceType.XYZ or self == ColorSpaceType.OKLAB or self == ColorSpaceType.OKLABM1 or self == ColorSpaceType.CIELAB:
            return 3
        else:
            return 1


class PolyscopeDisplayType(Enum):
    """To display in polyscope, we only allow a subset of the ColorSpaceType Enum to be displayed.
    """
    # this is because it makes no sense to display DISP_6P
    MAXBASIS = ColorSpaceType.MAXBASIS
    CONE = ColorSpaceType.CONE
    DISP = ColorSpaceType.DISP
    OKLAB = ColorSpaceType.OKLAB
    OKLABM1 = ColorSpaceType.OKLABM1
    CIELAB = ColorSpaceType.CIELAB

    HERING_MAXBASIS = ColorSpaceType.HERING  # this is HERING_MAXBASIS
    HERING_DISP = "hering_disp"
    HERING_CONE = "hering_cone"


class ColorSpace:
    """
    A class that represents a color space, combining an observer model with a display.

    This class encapsulates the functionality of ColorSpaceTransform and provides methods
    for sampling colors and transforming between different color spaces.
    """

    def __init__(self, observer: Observer,
                 display_primaries: List[Spectra] | None = None,
                 print_gamut: InkGamut | None = None,
                 metameric_axis: int = 2,
                 led_mapping: List[int] | None = [0, 1, 3, 2, 1, 3],
                 disp_method: str = 'direct'):
        """
        Initialize a ColorSpace with an observer and optional display.

        Parameters:
            observer (Observer): The observer model
            display_primaries (List[Spectra], optional): Display primary spectra
            print_gamut (InkGamut, optional): Printer gamut for PRINT color space
            metameric_axis (int, optional): Axis to be metameric over (default: 2)
            led_mapping (List[int], optional): LED mapping for 6P display (default: [0,1,3,2,1,3])
            disp_method (str, optional): Method for CONE->DISP ('direct', 'lsq', 'optimized', 'subset')
        """
        self.observer = observer
        self.metameric_axis = metameric_axis
        if display_primaries is not None and observer.dimension == 3:
            display_primaries = [display_primaries[i] for i in [0, 1, 2]]  # RGB , presumably RGBO
            led_mapping = [0, 1, 2, 0, 1, 2]  # RGB / RGB
        self.led_mapping = led_mapping
        self.display_primaries = display_primaries
        self.print_gamut = print_gamut
        self.disp_method = disp_method

        # Store the dimensionality of the color space
        self.dim = self.observer.dimension

        # Lazy-computed transforms (cached after first computation)
        self._cone_to_maxbasis = None
        self._cone_to_hering = None
        self._cone_to_disp = None
        self._cone_to_xyz = None
        self._disp_metadata = None

        # Lazy-computed gamut properties
        self._max_L = None

    def _get_cone_to_maxbasis(self) -> npt.NDArray:
        """Lazy compute CONE->MAXBASIS transformation matrix."""
        if self._cone_to_maxbasis is None:
            max_basis = MaxBasisFactory.get_object(self.observer, denom=1, verbose=False)
            self._cone_to_maxbasis = max_basis.cone_to_maxbasis
        return self._cone_to_maxbasis

    def _get_cone_to_hering(self) -> npt.NDArray:
        """Lazy compute CONE->HERING transformation matrix."""
        if self._cone_to_hering is None:
            # Reuse the same max_basis object to ensure consistency
            max_basis = MaxBasisFactory.get_object(self.observer, denom=1, verbose=False)
            cone_to_maxbasis = max_basis.cone_to_maxbasis
            hering_matrix = max_basis.HMatrix
            self._cone_to_hering = hering_matrix @ cone_to_maxbasis
        return self._cone_to_hering

    def _get_cone_to_disp(self) -> npt.NDArray:
        """Lazy compute CONE->DISP transformation matrix."""
        if self._cone_to_disp is None:
            if self.display_primaries is None:
                raise ValueError(
                    "Cannot compute CONE->DISP transform without display_primaries. "
                    "Provide display_primaries when creating ColorSpace."
                )

            # Use appropriate method based on disp_method
            if self.disp_method == 'direct':
                matrix, metadata = ConeToDisplay.compute_cone_to_display_direct(
                    self.observer, self.display_primaries
                )
            elif self.disp_method == 'lsq':
                matrix, metadata = ConeToDisplay.compute_cone_to_display_lsq(
                    self.observer, self.display_primaries
                )
            elif self.disp_method == 'optimized':
                matrix, metadata = ConeToDisplay.compute_cone_to_display_optimized(
                    self.observer, self.display_primaries
                )
            elif self.disp_method == 'subset':
                # For subset method, we would need indices passed separately
                # Default to direct if no specific indices provided
                matrix, metadata = ConeToDisplay.compute_cone_to_display_direct(
                    self.observer, self.display_primaries
                )
            else:
                raise ValueError(f"Unknown disp_method: {self.disp_method}")

            self._cone_to_disp = matrix
            self._disp_metadata = metadata

        return self._cone_to_disp

    @classmethod
    def init_RGB(self, observer: Observer, primaries: List[Spectra]) -> 'ColorSpace':
        """Initialize a ColorSpace with an observer and RGB display primaries."""
        return ColorSpace(observer, primaries, led_mapping=[0, 1, 2, 0, 1, 2])

    def _get_cone_to_xyz(self) -> npt.NDArray:
        """Lazy compute CONE->XYZ transformation matrix."""
        if self._cone_to_xyz is None:
            self._cone_to_xyz = GetConeToXYZPrimaries(self.observer, self.metameric_axis)
        return self._cone_to_xyz

    def _get_hering_to_disp(self) -> npt.NDArray:
        """Compute HERING->DISP transformation matrix."""
        cone_to_hering = self._get_cone_to_hering()
        cone_to_disp = self._get_cone_to_disp()
        # hering_to_disp = cone_to_disp @ inv(cone_to_hering)
        return cone_to_disp @ np.linalg.inv(cone_to_hering)

    def _get_maxbasis_to_disp(self) -> npt.NDArray:
        """Compute MAXBASIS->DISP transformation matrix."""
        cone_to_maxbasis = self._get_cone_to_maxbasis()
        cone_to_disp = self._get_cone_to_disp()
        # maxbasis_to_disp = cone_to_disp @ inv(cone_to_maxbasis)
        return cone_to_disp @ np.linalg.inv(cone_to_maxbasis)

    @property
    def max_L(self) -> float:
        """
        Get the maximum luminance value in HERING space.

        This is computed as the luminance (first component) when all display primaries
        are at maximum (white point in DISP space).

        Returns:
            float: Maximum luminance value
        """
        if self._max_L is None:
            if self.display_primaries is None:
                # Without display primaries, use unit cone response
                white_cone = np.ones(self.dim)
            else:
                # Convert maximum display point to cone space
                max_disp = np.ones(self.dim)
                white_cone = self.convert(max_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)

            # Convert to HERING and take luminance (first component)
            white_hering = self.convert(white_cone, ColorSpaceType.CONE, ColorSpaceType.HERING)
            self._max_L = white_hering[0]

        return self._max_L

    def get_metameric_axis_in(self, color_space_type: ColorSpaceType, metameric_axis_num: Optional[int] = None) -> npt.NDArray:
        """
        Get the metameric axis in the specified color space.

        Args:
            color_space_type: Target color space type
            metameric_axis_num: Index of metameric axis (defaults to self.metameric_axis)

        Returns:
            npt.NDArray: Normalized direction of the metameric axis
        """
        if metameric_axis_num is None:
            metameric_axis_num = self.metameric_axis
        metameric_axis = np.zeros(self.dim)
        metameric_axis[metameric_axis_num] = 1

        direction = self.convert(metameric_axis, ColorSpaceType.CONE, color_space_type)
        if color_space_type == ColorSpaceType.VSH:
            normalized_direction = direction
            normalized_direction[1] = 1.0  # make saturation 1
        else:
            normalized_direction = direction / np.linalg.norm(direction)
        return normalized_direction

    def get_maximal_metamer_pair_in_disp(self, metameric_axis: int, output_space: ColorSpaceType = ColorSpaceType.CONE) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get Maximal Metameric Color Pairs

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: cone1, cone2
        """
        metamer_dir_in_disp = self.get_metameric_axis_in(ColorSpaceType.DISP, metameric_axis_num=metameric_axis)

        metamers_in_disp = np.array(FindMaximumWidthAlongDirection(metamer_dir_in_disp, np.eye(self.dim)))

        points = self.convert(metamers_in_disp.reshape(-1, self.dim),
                              ColorSpaceType.DISP, output_space).reshape(-1, 2, self.dim)
        return points[0][0], points[0][1]

    def get_maximal_pair_in_disp_from_pt(self, pt: npt.NDArray, metameric_axis: int = 2, output_space: ColorSpaceType = ColorSpaceType.CONE, proportion: float = 1.0) -> Optional[Tuple[npt.NDArray, npt.NDArray, float]]:
        """Get Maximal Metameric Color Pairs from a Point

        Args:
            pt: Point in display space
            metameric_axis: Metameric axis to use
            proportion: Proportion of maximum distance to use (0.0 to 1.0). Default is 1.0 (maximum).

        Returns:
            Optional[Tuple[npt.NDArray, npt.NDArray, float]]: (cone1, cone2, metamer_difference) or None if rejected
        """
        metamer_dir_in_disp = self.get_metameric_axis_in(ColorSpaceType.DISP, metameric_axis_num=metameric_axis)
        disp_pts = np.clip(FindMaximumIn1DimDirection(
            pt,
            metamer_dir_in_disp,
            np.eye(self.dim)), 0, 1)

        # Scale by proportion if not using full distance
        if proportion < 1.0:
            # Interpolate between center point and maximal points
            disp_pts = pt + (disp_pts - pt) * proportion

        cones = self.convert(disp_pts, ColorSpaceType.DISP, ColorSpaceType.CONE)

        # Calculate metamer difference in the metameric channel
        metamer_difference = abs(cones[0][self.metameric_axis] - cones[1][self.metameric_axis])

        output = self.convert(cones, ColorSpaceType.CONE, output_space)

        return output[0], output[1], metamer_difference

    def convert(self, points: npt.NDArray,
                from_space: str | ColorSpaceType,
                to_space: str | ColorSpaceType,
                ink_gamut: InkGamut | None = None) -> npt.NDArray:
        """
        Transform points from one color space to another.

        All transformations route through CONE as the central space.

        Parameters:
            points (npt.NDArray): Points to transform
            from_space (str or ColorSpaceType): Source color space
            to_space (str or ColorSpaceType): Target color space
            ink_gamut (InkGamut, optional): Ink gamut for PRINT conversions

        Returns:
            npt.NDArray: Transformed points
        """
        # Convert string to enum if necessary
        if isinstance(from_space, str):
            from_space = ColorSpaceType(from_space.lower())
        if isinstance(to_space, str):
            to_space = ColorSpaceType(to_space.lower())

        # If source and target are the same, return the input
        if from_space == to_space:
            return points

        # CONE-CENTRIC ROUTING: All conversions go through CONE

        # Step 1: Convert FROM source space TO CONE
        if from_space == ColorSpaceType.CONE:
            cone_points = points
        elif from_space == ColorSpaceType.PRINT:
            # PRINT -> CONE via InkGamut
            if ink_gamut is None:
                ink_gamut = self.print_gamut
            if ink_gamut is None:
                raise ValueError("InkGamut required for PRINT conversions")
            if ink_gamut.neugebauer.num_inks != self.observer.dimension:
                raise ValueError("InkGamut inks must match observer dimension")
            cone_points = ink_gamut.primaries_to_cone(points, self.observer)
        elif from_space == ColorSpaceType.MAXBASIS:
            # MAXBASIS -> CONE
            cone_to_maxbasis = self._get_cone_to_maxbasis()
            cone_points = (np.linalg.inv(cone_to_maxbasis) @ points.T).T
        elif from_space == ColorSpaceType.HERING:
            # HERING -> CONE
            cone_to_hering = self._get_cone_to_hering()
            cone_points = (np.linalg.inv(cone_to_hering) @ points.T).T
        elif from_space == ColorSpaceType.DISP:
            # DISP -> CONE
            cone_to_disp = self._get_cone_to_disp()
            cone_points = (np.linalg.inv(cone_to_disp) @ points.T).T
        elif from_space == ColorSpaceType.VSH:
            # VSH -> HERING -> CONE
            hering_points = self._vsh_to_hering(points)
            cone_to_hering = self._get_cone_to_hering()
            cone_points = (np.linalg.inv(cone_to_hering) @ hering_points.T).T
        elif from_space == ColorSpaceType.XYZ:
            # XYZ -> CONE
            if self.dim != 3:
                raise ValueError("XYZ conversions only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            cone_points = (np.linalg.inv(cone_to_xyz) @ points.T).T
        elif from_space == ColorSpaceType.SRGB:
            # sRGB -> XYZ -> CONE (decode gamma first)
            # Decode sRGB gamma
            linear_rgb = np.where(points <= 0.04045,
                                  points / 12.92,
                                  np.power((points + 0.055) / 1.055, 2.4))
            xyz_points = (np.linalg.inv(M_XYZ_to_RGB) @ linear_rgb.T).T
            cone_to_xyz = self._get_cone_to_xyz()
            cone_points = (np.linalg.inv(cone_to_xyz) @ xyz_points.T).T
        elif from_space == ColorSpaceType.OKLAB:
            # OKLAB -> XYZ -> CONE
            if self.dim != 3:
                raise ValueError("OKLAB only defined for 3D color spaces")
            m2_points = np.linalg.inv(OKLAB_M2) @ points.T
            m2_cubed = np.power(m2_points, 3)
            xyz_points = (np.linalg.inv(OKLAB_M1) @ m2_cubed).T
            cone_to_xyz = self._get_cone_to_xyz()
            cone_points = (np.linalg.inv(cone_to_xyz) @ xyz_points.T).T
        elif from_space == ColorSpaceType.OKLABM1:
            # OKLABM1 -> XYZ -> CONE
            if self.dim != 3:
                raise ValueError("OKLABM1 only defined for 3D color spaces")
            m1_cubed = points
            xyz_points = (np.linalg.inv(OKLAB_M1) @ (m1_cubed**3).T).T
            cone_to_xyz = self._get_cone_to_xyz()
            cone_points = (np.linalg.inv(cone_to_xyz) @ xyz_points.T).T
        elif from_space == ColorSpaceType.CIELAB:
            # CIELAB -> XYZ -> CONE
            if self.dim != 3:
                raise ValueError("CIELAB only defined for 3D color spaces")
            from colour import Lab_to_XYZ
            xyz_points = Lab_to_XYZ(points * np.array([100, 400, 400]))
            cone_to_xyz = self._get_cone_to_xyz()
            cone_points = (np.linalg.inv(cone_to_xyz) @ xyz_points.T).T
        elif from_space == ColorSpaceType.DISP_6P:
            # DISP_6P -> DISP -> CONE
            disp_points = self._map_6d_to_4d(points)
            cone_to_disp = self._get_cone_to_disp()
            cone_points = (np.linalg.inv(cone_to_disp) @ disp_points.T).T
        elif from_space == ColorSpaceType.CHROM or from_space == ColorSpaceType.HERING_CHROM or from_space == ColorSpaceType.MACLEOD_CHROM:
            raise ValueError(f"Cannot convert from chromaticity space {from_space}")
        else:
            raise ValueError(f"Unknown source color space: {from_space}")

        # Step 2: Convert FROM CONE TO target space
        if to_space == ColorSpaceType.CONE:
            return cone_points
        # Special case: if converting from HERING to VSH, use direct path (no roundtrip through CONE)
        elif from_space == ColorSpaceType.HERING and to_space == ColorSpaceType.VSH:
            return self._hering_to_vsh(points)
        elif to_space == ColorSpaceType.PRINT:
            # CONE -> PRINT via InkGamut
            if ink_gamut is None:
                ink_gamut = self.print_gamut
            if ink_gamut is None:
                raise ValueError("InkGamut required for PRINT conversions")
            if ink_gamut.neugebauer.num_inks != self.observer.dimension:
                raise ValueError("InkGamut inks must match observer dimension")
            return ink_gamut.cone_to_primaries(cone_points, self.observer, method='optimization')
        elif to_space == ColorSpaceType.MAXBASIS:
            # CONE -> MAXBASIS
            cone_to_maxbasis = self._get_cone_to_maxbasis()
            return (cone_to_maxbasis @ cone_points.T).T
        elif to_space == ColorSpaceType.HERING:
            # CONE -> HERING
            cone_to_hering = self._get_cone_to_hering()
            return (cone_to_hering @ cone_points.T).T
        elif to_space == ColorSpaceType.DISP:
            # CONE -> DISP
            cone_to_disp = self._get_cone_to_disp()
            return (cone_to_disp @ cone_points.T).T
        elif to_space == ColorSpaceType.VSH:
            # CONE -> HERING -> VSH
            cone_to_hering = self._get_cone_to_hering()
            hering_points = (cone_to_hering @ cone_points.T).T
            return self._hering_to_vsh(hering_points)
        elif to_space == ColorSpaceType.XYZ:
            # CONE -> XYZ
            if self.dim != 3:
                raise ValueError("XYZ conversions only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            return (cone_to_xyz @ cone_points.T).T
        elif to_space == ColorSpaceType.SRGB:
            # CONE -> XYZ -> sRGB (encode gamma)
            # if self.dim != 3:
            #     raise ValueError("sRGB conversions only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            if self.dim > 3:
                cone_points = cone_points[:, [i for i in range(self.dim) if i != self.metameric_axis]]
            xyz_points = (cone_to_xyz @ cone_points.T).T
            linear_rgb = M_XYZ_to_RGB @ xyz_points.T
            # sRGB gamma encoding
            encoded_rgb = np.where(linear_rgb <= 0.0031308,
                                   12.92 * linear_rgb,
                                   1.055 * np.power(linear_rgb, 1/2.2) - 0.055)
            return encoded_rgb.T
        elif to_space == ColorSpaceType.OKLAB:
            # CONE -> XYZ -> OKLAB
            if self.dim != 3:
                raise ValueError("OKLAB only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            xyz_points = (cone_to_xyz @ cone_points.T).T
            m1_points = OKLAB_M1 @ xyz_points.T
            m1_cubed = np.cbrt(m1_points)
            m2_points = OKLAB_M2 @ m1_cubed
            return m2_points.T
        elif to_space == ColorSpaceType.OKLABM1:
            # CONE -> XYZ -> OKLABM1
            if self.dim != 3:
                raise ValueError("OKLABM1 only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            xyz_points = (cone_to_xyz @ cone_points.T).T
            m1_points = OKLAB_M1 @ xyz_points.T
            m1_cubed = np.cbrt(m1_points)
            return m1_cubed.T
        elif to_space == ColorSpaceType.CIELAB:
            # CONE -> XYZ -> CIELAB
            if self.dim != 3:
                raise ValueError("CIELAB only defined for 3D color spaces")
            cone_to_xyz = self._get_cone_to_xyz()
            xyz_points = (cone_to_xyz @ cone_points.T).T
            return XYZ_to_Lab(xyz_points) / np.array([100, 400, 400])
        elif to_space == ColorSpaceType.DISP_6P:
            # CONE -> DISP -> DISP_6P
            cone_to_disp = self._get_cone_to_disp()
            disp_points = (cone_to_disp @ cone_points.T).T
            return self._map_4d_to_6d(disp_points)
        elif to_space == ColorSpaceType.CHROM:
            # CONE -> chromaticity (drop first coordinate)
            return (cone_points.T / (np.sum(cone_points.T, axis=0) + 1e-9))[1:].T
        elif to_space == ColorSpaceType.MACLEOD_CHROM:
            # CONE -> MacLeod-Boynton chromaticity (drop M cone)
            return (cone_points.T / (np.sum(cone_points[:, 1:].T, axis=0) + 1e-9))[[i for i in range(self.dim) if i != 1]].T
        elif to_space == ColorSpaceType.HERING_CHROM:
            # CONE -> MAXBASIS -> Hering chromaticity
            cone_to_maxbasis = self._get_cone_to_maxbasis()
            maxbasis_points = (cone_to_maxbasis @ cone_points.T).T
            hering_matrix = GetHeringMatrix(self.dim)
            return (hering_matrix @ (maxbasis_points.T / (np.sum(maxbasis_points.T, axis=0) + 1e-9)))[1:].T
        else:
            raise ValueError(f"Unknown target color space: {to_space}")

    def _vsh_to_hering(self, vsh: npt.NDArray) -> npt.NDArray:
        """Convert from Value-Saturation-Hue to Hering opponent space."""
        if vsh.shape[1] == 4:
            return np.hstack([vsh[:, [0]], Geometry.ConvertSphericalToCartesian(vsh[:, 1:])])
        elif vsh.shape[1] == 3:
            return np.hstack([vsh[:, [0]], Geometry.ConvertPolarToCartesian(vsh[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

    def _hering_to_vsh(self, hering: npt.NDArray) -> npt.NDArray:
        """Convert from Hering opponent space to Value-Saturation-Hue."""
        if hering.shape[1] == 4:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToSpherical(hering[:, 1:])])
        elif hering.shape[1] == 3:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToPolar(hering[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

    def _map_4d_to_6d(self, points: npt.NDArray) -> npt.NDArray:
        """Map 4D display points to 6D even-odd representation."""
        # Get white weights from metadata if available
        if self._disp_metadata is not None:
            white_weights = self._disp_metadata.get('white_weights', np.ones(self.dim))
        else:
            white_weights = np.ones(self.dim)

        # Apply led_mapping to create 6D representation
        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        six_d = np.zeros((points.shape[0], 6))
        for i in range(6):
            six_d[:, i] = points[:, self.led_mapping[i]] * white_weights[self.led_mapping[i]]

        return six_d if points.shape[0] > 1 else six_d[0]

    def _map_6d_to_4d(self, points: npt.NDArray) -> npt.NDArray:
        """Map 6D even-odd representation to 4D display points."""
        # Get white weights from metadata if available
        if self._disp_metadata is not None:
            white_weights = self._disp_metadata.get('white_weights', np.ones(self.dim))
        else:
            white_weights = np.ones(self.dim)

        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        four_d = np.zeros((points.shape[0], self.dim))
        # Average the even and odd frames for each LED
        for led_idx in range(self.dim):
            # Find which 6D indices correspond to this LED
            indices = [i for i, x in enumerate(self.led_mapping) if x == led_idx]
            if len(indices) > 0:
                four_d[:, led_idx] = np.mean(points[:, indices], axis=1) / white_weights[led_idx]

        return four_d if points.shape[0] > 1 else four_d[0]

    def convert_to_polyscope(self, points: npt.NDArray,
                             from_space: str | ColorSpaceType,
                             to_space: PolyscopeDisplayType | str) -> npt.NDArray:
        """Convert from ColorSpaceType to a PolyscopeDisplayType for display in polyscope.

        Args:
            points (npt.NDArray): points in from_space
            from_space (str | ColorSpaceType): basis defined in ColorSpaceType
            to_space (PolyscopeDisplayType): basis defined in PolyscopeDisplayType

        Returns:
            npt.NDArray: array of points in the given PolyscopeDisplayType
        """
        if isinstance(to_space, str):
            to_space = PolyscopeDisplayType[to_space]

        # split the name into two - if it's hering, add a flag, and remove it
        name_split = to_space.name.split("_")
        isHering = True if name_split[0] == 'HERING' else False
        name_split = name_split[1:] if isHering else name_split

        # Convert to the target space
        points = self.convert(points, from_space, "_".join(name_split))

        if isHering:
            # Transform to Hering opponent space
            hering_matrix = GetHeringMatrix(self.dim)
            points = points @ hering_matrix.T

            if self.dim > 3:
                # For 4D, drop luminance and return chromatic coordinates
                return points[:, 1:]
            else:
                # For 3D, swap first two coordinates
                tmp = points[:, 1].copy()
                points[:, 1] = points[:, 0]
                points[:, 0] = tmp
                return points

        return points

    def get_maxbasis_parallelepiped(self, display_basis: PolyscopeDisplayType) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the maxbasis parallelepiped for the given display basis.

        Args:
            display_basis (PolyscopeDisplayType): The display basis type

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: (points, rgbs, lines)
        """
        np.set_printoptions(precision=3, suppress=True)
        display_name = display_basis.name.split("_")
        denom = 1
        maxbasis = MaxBasisFactory.get_object(self.observer, denom=denom)
        refs, _, rgbs, lines = maxbasis.GetDiscreteRepresentation()

        if "OKLAB" in display_name or "CIELAB" in display_name:
            display_name = display_name[:-1] if len(display_name) > 1 else display_name

        cones = self.observer.observe_spectras(refs)
        points = self.convert_to_polyscope(cones, ColorSpaceType.CONE, "_".join(display_name))

        # Compute the perpendicular distance from the lines (1, 0, 0)
        reference_point = np.array([1, 0, 0])
        projections = np.dot(points, reference_point) / np.linalg.norm(reference_point)
        perpendicular_distances = np.linalg.norm(points - np.outer(projections, reference_point), axis=1)

        print(perpendicular_distances[1:4], points[1:4, 0])

        # Measure the angles on the yz-plane of points
        yz_points = points[1:4, [0, 2]]  # Extract y and z coordinates
        angles = np.degrees(np.arctan2(yz_points[:, 1], yz_points[:, 0])) % 360
        angle_diffs = [(angles[i] - angles[(i+1) % len(angles)]) % 360 for i in range(len(angles))]
        print("Angles on the yz-plane (degrees):", angles)
        print("Angles Diffs: ", angle_diffs)

        return points, rgbs, lines

    def get_background(self, luminance, output_space: ColorSpaceType):
        vec = np.zeros(self.dim)
        vec[0] = luminance
        return self.convert(np.array([vec]), from_space=ColorSpaceType.VSH, to_space=output_space)[0]

    def __str__(self) -> str:
        """
        Generate a unique string representation of this color space for hashing.

        Returns:
            str: Hash string that uniquely identifies this color space's configuration
        """
        # Collect relevant properties that affect color gamut
        components = [
            # Observer properties
            f"observer_dim:{self.observer.dimension}",
            f"{str(self.observer)}",

            # Display Properties
            f"metameric_axis:{self.metameric_axis}",
            f"subset_leds:{self.led_mapping}",
        ]
        # Join all components with a separator
        return "|".join(components)


if __name__ == "__main__":
    # Test if Oklab implementation is correct
    observer = Observer.trichromat()

    import matplotlib.pyplot as plt
    from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
    from colour import SpectralShape

    shape = SpectralShape(min(observer.wavelengths), max(observer.wavelengths),
                          int(observer.wavelengths[1] - observer.wavelengths[0]))
    xyz_cmfs = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].copy().align(shape).values
    xyz_d65 = xyz_cmfs.T @ Illuminant.get("D65").to_colour().align(shape).values
    xyz_ones = xyz_cmfs.T @ np.ones(len(observer.wavelengths))
    # xyz_d65 = xyz_d65/xyz_d65[1]
    xyz = xyz_cmfs / (xyz_ones/(xyz_d65/xyz_d65[1]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    cst = ColorSpace(Observer.trichromat(wavelengths=np.arange(360, 830, 10)))
    cone_to_xyz = cst._get_cone_to_xyz()
    print(repr(OKLAB_M1@cone_to_xyz))

    print(repr(OKLAB_M2))
    # cones = cst.convert(xyz, ColorSpaceType.XYZ, ColorSpaceType.CONE)
    # oklab = cst.convert(cones, ColorSpaceType.CONE, ColorSpaceType.OKLAB)

    # Plot the original XYZ color matching functions
    axes[0].plot(observer.wavelengths, xyz)
    axes[0].set_title("CIE 1931 XYZ Color Matching Functions")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Value")

    # Plot the transformed OKLAB M1 values
    axes[1].plot(observer.wavelengths, (observer.normalized_sensor_matrix.T @ cone_to_xyz.T) @ OKLAB_M1.T)
    axes[1].set_title("Transformed OKLAB M1 Values")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Value")

    # Plot the transformed OKLAB M1 values
    axes[2].plot(observer.wavelengths, np.cbrt(xyz @ OKLAB_M1.T))
    axes[2].set_title("Transformed OKLAB M1 Values (Cube Root)")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Value")

    axes[3].plot(observer.wavelengths, np.cbrt(xyz @ OKLAB_M1.T)@OKLAB_M2.T)
    axes[3].set_title("Transformed OKLAB M2 Values")
    axes[3].set_xlabel("Wavelength (nm)")
    axes[3].set_ylabel("Value")

    plt.tight_layout()
    plt.show()

    cs = ColorSpace(observer)

    white_pt = np.array([1, 1, 1])  # observer.get_whitepoint()
    # Convert white point to XYZ
    print("White point in Cone", white_pt)
    print("White point in XYZ", cs.convert(white_pt, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.XYZ))
    print("White point in OKlab", cs.convert(white_pt, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.OKLAB))

    print(np.round(cs.convert(np.array([0.950, 1.0, 1.089]),
          from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([1, 0, 0]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([0, 1, 0]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))
    print(np.round(cs.convert(np.array([0, 0, 1]), from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3))

    # Store the results of the conversions
    oklab_results = [
        np.round(cs.convert(np.array([0.950, 1.0, 1.089]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([1, 0, 0]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([0, 1, 0]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3),
        np.round(cs.convert(np.array([0, 0, 1]),
                            from_space=ColorSpaceType.XYZ, to_space=ColorSpaceType.OKLAB), 3)
    ]

    # Convert back to XYZ and check equivalence
    for original, oklab in zip(
        [np.array([0.950, 1.0, 1.089]),
         np.array([1, 0, 0]),
         np.array([0, 1, 0]),
         np.array([0, 0, 1])],
        oklab_results
    ):
        converted_back = np.round(cs.convert(oklab, from_space=ColorSpaceType.OKLAB, to_space=ColorSpaceType.XYZ), 3)
        print(f"Original: {original}, Converted Back: {converted_back}, Equivalent: {np.allclose(original, converted_back)}")
