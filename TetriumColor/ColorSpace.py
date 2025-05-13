from re import I
import numpy as np
import numpy.typing as npt
from typing import List
from enum import Enum

from TetriumColor.Observer import Observer, MaxBasisFactory, GetHeringMatrix, GetPerceptualHering
from TetriumColor.Observer.Spectra import Illuminant, Spectra
from TetriumColor.Utils.CustomTypes import TetraColor, PlateColor
from TetriumColor.Observer.ColorSpaceTransform import (
    GetColorSpaceTransform, CSTDisplayType, GetMaxBasisToDisplayTransform
)

from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_Lab
import TetriumColor.ColorMath.Geometry as Geometry
import TetriumColor.ColorMath.Conversion as Conversion
import TetriumColor.Utils.BasisMath as BasisMath


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

M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB


class ColorSpaceType(Enum):
    """ColorSpaceType is the core of the color space system. It defines the different types of color spaces
    that can be used in the system. Each color space type is represented by a string value.
    """
    VSH = "vsh"  # Value-Saturation-Hue
    HERING = "hering"  # Hering opponent color space
    MAXBASIS = "maxbasis"  # Display space (RYGB)

    MAXBASIS243 = "maxbasis243"  # Perceptual space w denom 3
    MAXBASIS300 = "maxbasis300"  # Perceptual space w denom 2.43
    MAXBASIS_PERCEPTUAL_243 = "maxbasis_perceptual_243"  # Perceptual space w denom 3
    MAXBASIS_PERCEPTUAL_300 = "maxbasis_perceptual_300"  # Perceptual space w denom 2.43
    MAXBASIS243_PERCEPTUAL_243 = "maxbasis243_perceptual_243"  # Perceptual space w denom 3
    MAXBASIS300_PERCEPTUAL_300 = "maxbasis300_perceptual_300"  # Perceptual space w denom 2.43

    CONE = "cone"  # Cone responses (SMQL)
    CONE_PERCEPTUAL_243 = "cone_perceptual_243"  # Perceptual space w denom 3
    CONE_PERCEPTUAL_300 = "cone_perceptual_300"  # Perceptual space w denom 2.43

    DISP_6P = "disp_6p"  # RGO/BGO 6D representation
    DISP = "disp"  # Display space (RGBO)

    SRGB = "srgb"  # sRGB display
    XYZ = "xyz"  # CIE XYZ color space
    OKLAB = 'oklab'
    OKLABM1 = 'oklabm1'  # OKLAB color space with M1 matrix
    CIELAB = 'cielab'  # CIE L*a*b* color space

    CHROM = "chrom"  # Chromaticity space
    HERING_CHROM = "hering_chrom"  # Hering chromaticity space

    def __str__(self):
        return self.value


class PolyscopeDisplayType(Enum):
    """To display in polyscope, we only allow a subset of the ColorSpaceType Enum to be displayed.
    """
    # this is because it makes no sense to display DISP_6P
    MAXBASIS = ColorSpaceType.MAXBASIS
    MAXBASIS_PERCEPTUAL_243 = ColorSpaceType.MAXBASIS_PERCEPTUAL_243
    MAXBASIS_PERCEPTUAL_300 = ColorSpaceType.MAXBASIS_PERCEPTUAL_300
    MAXBASIS243_PERCEPTUAL_243 = ColorSpaceType.MAXBASIS243_PERCEPTUAL_243
    MAXBASIS300_PERCEPTUAL_300 = ColorSpaceType.MAXBASIS300_PERCEPTUAL_300

    CONE = ColorSpaceType.CONE
    CONE_PERCEPTUAL_243 = ColorSpaceType.CONE_PERCEPTUAL_243
    CONE_PERCEPTUAL_300 = ColorSpaceType.CONE_PERCEPTUAL_300
    DISP = ColorSpaceType.DISP
    OKLAB = ColorSpaceType.OKLAB
    OKLABM1 = ColorSpaceType.OKLABM1
    OKLAB_PERCEPTUAL_243 = "oklab_perceptual_243"
    CIELAB = ColorSpaceType.CIELAB
    CIELAB_PERCEPTUAL_243 = "cielab_perceptual_243"

    HERING_MAXBASIS = ColorSpaceType.HERING  # this is HERING_MAXBASIS
    HERING_MAXBASIS_PERCEPTUAL_243 = "hering_maxbasis_perceptual_243"
    HERING_MAXBASIS_PERCEPTUAL_300 = "hering_maxbasis_perceptual_300"
    HERING_MAXBASIS243_PERCEPTUAL_243 = "hering_maxbasis243_perceptual_243"
    HERING_MAXBASIS300_PERCEPTUAL_3 = "hering_maxbasis300_perceptual_300"

    HERING_DISP = "hering_disp"
    HERING_CONE = "hering_cone"
    HERING_CONE_PERCEPTUAL_243 = "hering_cone_perceptual_243"
    HERING_CONE_PERCEPTUAL_300 = "hering_cone_perceptual_300"


class ColorSpace:
    """
    A class that represents a color space, combining an observer model with a display.

    This class encapsulates the functionality of ColorSpaceTransform and provides methods
    for sampling colors and transforming between different color spaces.
    """

    def __init__(self, observer: Observer,
                 cst_display_type: CSTDisplayType | str = CSTDisplayType.NONE,
                 display_primaries: List[Spectra] | None = None,
                 metameric_axis: int = 2,
                 luminance_per_channel: List[float] = [1/np.sqrt(3)] * 3,
                 chromas_per_channel: List[float] = [np.sqrt(2/3)] * 3,
                 led_mapping: List[int] | None = [0, 1, 3, 2, 1, 3]):
        """
        Initialize a ColorSpace with an observer and optional display.

        Parameters:
            observer (Observer): The observer model
            display (List[Spectra] or str, optional): Either a list of display primary spectra or
                                                    a string identifying a predefined display
            scaling_factor (float, optional): Scaling factor for the display primaries
            metameric_axis (int, optional): Axis to be metameric over
            subset_leds (List[int], optional):
        """
        self.observer = observer
        self.metameric_axis = metameric_axis
        self.led_mapping = led_mapping
        self.lums_per_channel = luminance_per_channel
        self.chromas_per_channel = chromas_per_channel

        if isinstance(cst_display_type, str):
            cst_display_type = CSTDisplayType[cst_display_type.upper()]

        self.transform = GetColorSpaceTransform(observer, cst_display_type, display_primaries,
                                                metameric_axis, led_mapping)
        # Store the dimensionality of the color space
        self.dim = self.observer.dimension

        self.max_L = (np.linalg.inv(self.transform.hering_to_disp) @
                      np.ones(self.transform.cone_to_disp.shape[0]))[0]

    def get_metameric_axis_in(self, color_space_type: ColorSpaceType) -> npt.NDArray:
        """
        Get the metameric axis in display space.

        Returns:
            npt.NDArray: Normalized direction of the metameric axis
        """
        metameric_axis = np.zeros(self.dim)
        metameric_axis[self.transform.metameric_axis] = 1

        direction = self.convert(metameric_axis, ColorSpaceType.CONE, color_space_type)
        if color_space_type == ColorSpaceType.VSH:
            normalized_direction = direction
            normalized_direction[1] = 1.0  # make saturation 1
        else:
            normalized_direction = direction / np.linalg.norm(direction)
        return normalized_direction

    def _find_maximal_saturation(self, hue_direction: npt.NDArray) -> npt.NDArray:
        """
        Find the point with maximal saturation in the given hue direction.

        Parameters:
            hue_direction (npt.NDArray): Hue direction vector

        Returns:
            npt.NDArray: Point with maximal saturation
        """
        from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation
        result = FindMaximalSaturation(hue_direction, np.eye(self.dim))
        if result is None:
            raise ValueError("Failed to find maximal saturation point")
        else:

            return result

    def _vsh_to_hering(self, vsh: npt.NDArray) -> npt.NDArray:
        """
        Convert from Value-Saturation-Hue to Hering opponent space.

        Parameters:
            vsh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Points in Hering space
        """
        if vsh.shape[1] == 4:
            return np.hstack([vsh[:, [0]], Geometry.ConvertSphericalToCartesian(vsh[:, 1:])])
        elif vsh.shape[1] == 3:
            return np.hstack([vsh[:, [0]], Geometry.ConvertPolarToCartesian(vsh[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

    def _hering_to_vsh(self, hering: npt.NDArray) -> npt.NDArray:
        """
        Convert from Hering opponent space to Value-Saturation-Hue.

        Parameters:
            hering (npt.NDArray): Points in Hering space

        Returns:
            npt.NDArray: Points in VSH space
        """
        if hering.shape[1] == 4:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToSpherical(hering[:, 1:])])
        elif hering.shape[1] == 3:
            return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToPolar(hering[:, 1:])])
        else:
            raise NotImplementedError("Not implemented for dimensions other than 3 or 4")

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
            return (L - max_L) / (slope)
        else:
            slope = lum_cusp / sat_cusp
            return L / slope

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
        all_angles = Geometry.SampleAnglesEqually(num_points, self.dim-1)
        all_vshh = np.zeros((len(all_angles), self.dim))
        all_vshh[:, 0] = luminance
        all_vshh[:, 1] = saturation
        all_vshh[:, 2:] = all_angles
        return all_vshh

    def sample_equiluminant_plane(self,
                                  luminance: float,
                                  num_points: int = 100,
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

    def _solve_for_cusp(self, angle):
        # Compute the cusp point dynamically if not in the LUT
        hue_cartesian = self._vsh_to_hering(np.array([[0, 1, *angle]]))
        max_sat_point = self._find_maximal_saturation(
            (self.transform.hering_to_disp @ hue_cartesian.T).T[0]
        )
        max_sat_hering = np.linalg.inv(self.transform.hering_to_disp) @ max_sat_point
        max_sat_vsh = self._hering_to_vsh(max_sat_hering[np.newaxis, :])[0]
        lum_cusp, sat_cusp = max_sat_vsh[0], max_sat_vsh[1]
        return lum_cusp, sat_cusp

    def remap_to_gamut(self, vshh: npt.NDArray) -> npt.NDArray:
        """
        Remap points to be within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Remapped points that are in gamut
        """
        # Ensure the cubemap is generated
        remapped_vshh = vshh.copy()

        # Remap each point
        for i in range(len(remapped_vshh)):
            angle = tuple(remapped_vshh[i, 2:])

            # Get cusp values by interpolating from the cubemap
            lum_cusp, sat_cusp = self._solve_for_cusp(angle)

            # Calculate the maximum saturation at the given luminance
            sat_max = self._solve_for_boundary(remapped_vshh[i, 0], self.max_L, lum_cusp, sat_cusp)

            # Clamp the saturation to the maximum
            remapped_vshh[i, 1] = min(sat_max, remapped_vshh[i, 1])

        return remapped_vshh

    def max_sat_at_luminance(self, luminance: float, angles: List[tuple[float, float]] | tuple[float, float]) -> float | List[float]:
        """
        Get the maximum saturation at a given luminance.

        Parameters:
            luminance (float): Luminance value

        Returns:
            float: Maximum saturation at the given luminance
        """
        # Ensure the cubemap is generated
        isOneD = False
        if isinstance(angles, tuple):
            isOneD = True
            angles = [angles]
        sat_maxes = []
        for angle in angles:
            # Get cusp values by interpolating from the cubemap
            lum_cusp, sat_cusp = self._solve_for_cusp(angle)
            # Calculate the maximum saturation at the given luminance
            sat_maxes += [self._solve_for_boundary(luminance, self.max_L, lum_cusp, sat_cusp)]

        if isOneD:
            return sat_maxes[0]
        else:
            return sat_maxes

    def is_in_gamut(self, points: npt.NDArray, color_space_type: ColorSpaceType) -> bool:
        """
        Check if points are within the gamut.

        Parameters:
            vshh (npt.NDArray): Points in VSH space

        Returns:
            npt.NDArray: Boolean array indicating if each point is in gamut
        """
        # Get the remapped points
        display_basis = self.convert(points, color_space_type, ColorSpaceType.DISP_6P)
        in_gamut = np.all((display_basis >= 0) & (display_basis <= 1), axis=1)

        return in_gamut

    @staticmethod
    def _remove_perceptual_name(name: str):
        name_split = name.split("_")
        if "PERCEPTUAL" in name_split:
            index = name_split.index("PERCEPTUAL")
            new_name = "".join(name_split[:index]).lower()
            return ColorSpaceType(new_name)
        else:
            return ColorSpaceType(name.lower())

    def convert(self, points: npt.NDArray,
                from_space: str | ColorSpaceType,
                to_space: str | ColorSpaceType) -> npt.NDArray:
        """
        Transform points from one color space to another. --> only handles linear conversions. For perceptual, use convert_to_perceptual

        Parameters:
            points (npt.NDArray): Points to transform
            from_space (str or ColorSpaceType): Source color space
            to_space (str or ColorSpaceType): Target color space

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

        # handle only the linear portions of perceptual spaces --> basically if it's cone_perceptual, just convert it to cone.
        from_space = self._remove_perceptual_name(from_space.name)
        to_space = self._remove_perceptual_name(to_space.name)

        # Handle 3D only points separately -- only one directional

        if to_space == ColorSpaceType.SRGB:
            # Convert to cone space first, then to sRGB
            cone_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            return (M_XYZ_to_RGB @ cone_points.T).T
        elif to_space == ColorSpaceType.OKLAB:
            if self.transform.dim != 3:
                raise ValueError("OKLAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to OKLAB
            xyz_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            m1_points = OKLAB_M1 @ xyz_points.T
            m1_cubed = np.cbrt(m1_points)
            m2_points = OKLAB_M2 @ m1_cubed
            return m2_points.T
        elif to_space == ColorSpaceType.OKLABM1:
            if self.transform.dim != 3:
                raise ValueError("OKLAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to OKLAB
            xyz_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            m1_points = OKLAB_M1 @ xyz_points.T
            m1_cubed = np.cbrt(m1_points)
            return m1_cubed.T
        elif to_space == ColorSpaceType.CIELAB:
            if self.transform.dim != 3:
                raise ValueError("CIELAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to CIELAB
            xyz_points = self.convert(points, from_space, ColorSpaceType.XYZ)
            return XYZ_to_Lab(xyz_points) / np.array([100, 400, 400])

         # chromaticity based color transforms
        if from_space == ColorSpaceType.CHROM or from_space == ColorSpaceType.HERING_CHROM:
            raise ValueError("Cannot transform from chromaticity back to another color space")
        elif to_space == ColorSpaceType.CHROM:
            cone_pts = self.convert(points, from_space, ColorSpaceType.CONE)
            return (cone_pts.T / (np.sum(cone_pts.T, axis=0) + 1e-9))[1:].T  # auto drop first coordinate
        elif to_space == ColorSpaceType.HERING_CHROM:
            maxbasis_pts = self.convert(points, from_space, ColorSpaceType.MAXBASIS)
            return (GetHeringMatrix(self.transform.dim) @
                    (maxbasis_pts.T / (np.sum(maxbasis_pts.T, axis=0) + 1e-9)))[1:].T

        # Handle the basic linear transforms
        if from_space == ColorSpaceType.VSH:
            return self.convert(self._vsh_to_hering(points), ColorSpaceType.HERING, to_space)
        elif from_space == ColorSpaceType.SRGB:
            # Convert from sRGB to cone space, then proceed with normal conversions
            cone_points = (np.linalg.inv(M_XYZ_to_RGB) @ points.T).T
            return self.convert(cone_points, ColorSpaceType.XYZ, to_space)
        elif from_space == ColorSpaceType.OKLAB:
            if self.transform.dim != 3:
                raise ValueError("OKLAB color space is only defined for 3D color spaces")
            # Convert to cone space first, then to OKLAB
            m2_points = np.linalg.inv(OKLAB_M2) @ points.T
            m2_cubed = np.power(m2_points, 3)
            xyz_points = np.linalg.inv(OKLAB_M1) @ m2_cubed
            return self.convert(xyz_points.T, ColorSpaceType.XYZ, to_space)
        elif from_space == ColorSpaceType.XYZ:
            if self.transform.dim != 3:
                raise ValueError("transforming from XYZ to another color space is only defined for 3D color spaces")
            cone_pts = np.linalg.inv(self.transform.cone_to_XYZ) @ points.T  # can't do this inverse if it's not 3D
            return self.convert(cone_pts.T, ColorSpaceType.CONE, to_space)
        elif from_space == ColorSpaceType.HERING:
            disp_points = self.transform.hering_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.MAXBASIS:
            disp_points = self.transform.maxbasis_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.MAXBASIS243:
            disp_points = self.transform.maxbasis_243_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.MAXBASIS300:
            disp_points = self.transform.maxbasis_3_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.CONE:
            disp_points = self.transform.cone_to_disp @ points.T
            return self.convert(disp_points.T, ColorSpaceType.DISP, to_space)
        elif from_space == ColorSpaceType.DISP:
            if to_space == ColorSpaceType.VSH:
                hering = (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
                return self._hering_to_vsh(hering)
            elif to_space == ColorSpaceType.HERING:
                return (np.linalg.inv(self.transform.hering_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.MAXBASIS:
                return (np.linalg.inv(self.transform.maxbasis_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.MAXBASIS243:
                return (np.linalg.inv(self.transform.maxbasis_243_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.MAXBASIS300:
                return (np.linalg.inv(self.transform.maxbasis_3_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.CONE:
                return (np.linalg.inv(self.transform.cone_to_disp) @ points.T).T
            elif to_space == ColorSpaceType.DISP_6P:
                return Conversion.Map4DTo6D(points, self.transform)
            elif to_space == ColorSpaceType.XYZ:
                M_DISP_TO_CONE = np.linalg.inv(self.transform.cone_to_disp)
                M_CONE_TO_XYZ = self.transform.cone_to_XYZ
                if self.dim == 3:
                    M_DISP_TO_XYZ = M_CONE_TO_XYZ @ M_DISP_TO_CONE
                else:
                    M_DISP_TO_XYZ = M_CONE_TO_XYZ @ M_DISP_TO_CONE[[
                        i for i in range(self.dim) if i != self.metameric_axis]]

                return points @ M_DISP_TO_XYZ.T

        elif from_space == ColorSpaceType.DISP_6P:
            display = Conversion.Map6DTo4D(points, self.transform)
            return self.convert(display, ColorSpaceType.DISP, to_space)

        # If we reach here, the transformation is not defined
        raise ValueError(f"Transformation from {from_space} to {to_space} not implemented")

    def convert_to_perceptual_new(self, points: npt.NDArray, from_space: str | ColorSpaceType,
                                  denom_of_nonlin: float = 3) -> npt.NDArray:
        white_point = np.array([1, 1, 1])  # in cone space

        # 0.5 Get all of the basis vectors, and transform to this space
        max_basis = MaxBasisFactory.get_object(self.observer, denom=denom_of_nonlin)
        refs, _, _, _ = max_basis.GetDiscreteRepresentation()
        maxbasis_points = self.observer.observe_spectras(refs[1:4])

        # 1. first convert to cone
        points = self.convert(points, from_space, ColorSpaceType.CONE)

        # 2. apply non-linearity in cone space
        points = np.power(points, 1/denom_of_nonlin)
        maxbasis_points = np.power(maxbasis_points, 1/denom_of_nonlin)

        # 3. Transform to luminance space --> using cone long-wavelength sum as luminance
        mat = GetPerceptualHering(self.transform.dim)
        points = points@mat.T
        maxbasis_points = maxbasis_points@mat.T
        white_point = white_point@mat.T

        # get the transformation points -- use some heuristics
        lums = maxbasis_points[:, 0]  # luminance value
        lums = [lums[0], lums[1], lums[2] * 1.3]
        chromas = np.ones(3) * np.sqrt(2/3) * np.array([1.0, 0.5, 0.5])  # chromas as fractions of a basis
        # chromas = [np.sqrt(2/3)] * 3
        vshh = self.convert(maxbasis_points, ColorSpaceType.HERING, ColorSpaceType.VSH)
        vshh[:, 0] = lums
        vshh[:, 1] = chromas
        target_points = BasisMath.construct_angle_basis(maxbasis_points.shape[1], white_point, lums, chromas)
        hering_target_points = self.convert(vshh, ColorSpaceType.VSH, ColorSpaceType.HERING)

        transform_mat, _ = BasisMath.solve_transformation_matrix(maxbasis_points, target_points[:, [0, 2, 1]])
        points = points @ transform_mat.T
        # points = points * [1, 1.5, 1.5]

        transformed_max_basis = maxbasis_points @ transform_mat.T

        print("Max basis points:", maxbasis_points)
        print("Target points:", target_points[:, [0, 2, 1]])
        print("Transform matrix:", transform_mat)
        print("Transformed points:", maxbasis_points@transform_mat.T)

        BasisMath.visualize_transformation(
            maxbasis_points, target_points[:, [0, 2, 1]], maxbasis_points@transform_mat.T)

        # no_lum_blue = transformed_max_basis[0]
        # no_lum_blue[0] = 0
        # stretch_mat = BasisMath.stretch_matrix_along_direction(no_lum_blue, 1.5)

        # points = points @ stretch_mat.T

        # Create a rotation matrix for 60 degrees around the (1, 0, 0) axis
        angle = np.radians(50)
        axis = np.array([1, 0, 0])
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        # Compute the rotation matrix using the Rodrigues' rotation formula
        rotation_matrix = np.array([
            [
                cos_angle + axis[0] * axis[0] * one_minus_cos,
                axis[0] * axis[1] * one_minus_cos - axis[2] * sin_angle,
                axis[0] * axis[2] * one_minus_cos + axis[1] * sin_angle
            ],
            [
                axis[1] * axis[0] * one_minus_cos + axis[2] * sin_angle,
                cos_angle + axis[1] * axis[1] * one_minus_cos,
                axis[1] * axis[2] * one_minus_cos - axis[0] * sin_angle
            ],
            [
                axis[2] * axis[0] * one_minus_cos - axis[1] * sin_angle,
                axis[2] * axis[1] * one_minus_cos + axis[0] * sin_angle,
                cos_angle + axis[2] * axis[2] * one_minus_cos
            ]
        ])

        # Apply the rotation matrix to the points
        points = points @ rotation_matrix.T

        if self.dim == 3:
            points = points[:, [1, 0, 2]]
        return points / np.sqrt(2)

    def convert_to_polyscope(self, points: npt.NDArray,
                             from_space: str | ColorSpaceType,
                             to_space: PolyscopeDisplayType | str) -> npt.NDArray:
        """Convert from ColorSpaceType to a PolyscopeDisplayType to vastly simplify display basis type

        Args:
            points (npt.NDArray): points in from_space
            from_space (str | ColorSpaceType): basis defined in ColorSpaceType
            to_space (PolyscopeDisplayType): basis defined in PolyscopeDisplayType

        Returns:
            npt.NDArray: array of points in the given PolyscopeDisplayType
        """
        if isinstance(to_space, str):
            to_space = PolyscopeDisplayType[to_space]
        # split the name into two - if it's hering, add a flag, and remove it. Otherwise, take the rest of the name
        name_split = to_space.name.split("_")
        isHering = True if name_split[0] == 'HERING' else False
        name_split = name_split[1:] if isHering else name_split

        # just apply the non-linearity first, then do all the basis transforms manually here
        if len(name_split) > 1 and name_split[1] == 'PERCEPTUAL':
            return self.convert_to_perceptual(points, from_space, ColorSpaceType.CONE, denom_of_nonlin=float(name_split[2])/100)
            # return self.convert_to_perceptual_new(points, from_space, denom_of_nonlin=float(name_split[2])/100)

        if isHering:
            mat = GetPerceptualHering(self.transform.dim, isLumY=True)
            points = points @ mat.T
            return points

        points = self.convert(points, from_space, "_".join(name_split))
        if to_space == PolyscopeDisplayType.OKLAB or to_space == PolyscopeDisplayType.CIELAB:
            return points[:, [1, 0, 2]]

        if to_space == PolyscopeDisplayType.CONE:
            return points[:, [2, 1, 0]]

        return points

    def get_maxbasis_parallelepiped(self, display_basis: PolyscopeDisplayType) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the maxbasis parallelepiped for the given display basis

        Args:
            display_basis (PolyscopeDisplayType):

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: _description_
        """
        np.set_printoptions(precision=3, suppress=True)
        display_name = display_basis.name.split("_")
        denom = 1 if len(display_name) < 2 or display_name[-2] != "PERCEPTUAL" else float(display_name[-1])/100
        maxbasis = MaxBasisFactory.get_object(self.observer, denom=1)  # , denom=denom)
        refs, _, rgbs, lines = maxbasis.GetDiscreteRepresentation()

        if "OKLAB" in display_name or "CIELAB" in display_name:
            display_name = display_name[:-2]
        cones = self.observer.observe_spectras(refs)
        points = self.convert_to_polyscope(cones, ColorSpaceType.CONE, "_".join(display_name))

        # Compute the perpendicular distance from the lines (1, 0, 0)
        reference_point = np.array([1, 0, 0])
        projections = np.dot(points, reference_point) / np.linalg.norm(reference_point)
        perpendicular_distances = np.linalg.norm(points - np.outer(projections, reference_point), axis=1)

        print(perpendicular_distances[1:4], points[1:4, 0])

        # Measure the angles on the yz-plane of points
        yz_points = points[1:4, [0, 2]]  # Extract y and z coordinates
        angles = np.degrees(np.arctan2(yz_points[:, 1], yz_points[:, 0])) % 360  # Compute angles in radians
        angle_diffs = [(angles[i] - angles[(i+1) % len(angles)]) % 360 for i in range(len(angles))]
        print("Angles on the yz-plane (degrees):", angles)
        print("Angles Diffs: ", angle_diffs)

        return points, rgbs, lines

    def convert_to_perceptual(self, points: npt.NDArray,
                              from_space: str | ColorSpaceType,
                              M_basis: str | ColorSpaceType,
                              denom_of_nonlin: float) -> npt.NDArray:
        """Convert points from some linear space to a perceptual space

        Args:
            points (npt.NDArray): input points in from_space, that will be converted to to_space
            from_space (str | ColorSpaceType): The space to convert from
            M_basis (str | ColorSpaceType): M_basis to use as the basis of the transform
            denom_of_nonlin (float): denominator for the non-linearity

        Returns:
            npt.NDArray: Converted points in "perceptual space"
        """
        # convert points
        max_basis = MaxBasisFactory.get_object(self.observer, denom=1)
        refs, _, _, _ = max_basis.GetDiscreteRepresentation()
        maxbasis_points = self.observer.observe_spectras([refs[x] for x in [1, 6]])
        dists = [0.7, 0.7]
        # Adjust maxbasis_points such that their perpendicular distance away from np.ones(3) is dists
        reference_point = np.ones(3)
        target_points = np.zeros((len(maxbasis_points), 3))
        for i in range(len(maxbasis_points)):
            projection = np.dot(maxbasis_points[i], reference_point) / np.linalg.norm(reference_point)
            direction = maxbasis_points[i] - projection * reference_point
            direction /= np.linalg.norm(direction)  # Normalize the direction
            target_points[i] = projection * reference_point + direction * dists[i]

        # target_points = np.append(target_points, , axis=0, )
        # source_points = np.append(maxbasis_points, np.ones((1, 3)), axis=0)

        M = np.array(
            [[0, 0,  1],
             [0,  1, 0],
             [0.8,  0,  0.2]]
        )
        # spectral_locus = (M@self.observer.normalized_sensor_matrix).T
        new_white_pt = M@np.ones(3)
        new_transform_mat = (np.diag(1/new_white_pt))@M
        # new_spectral_locus = (new_transform_mat@self.observer.normalized_sensor_matrix).T  # @ new_transform_mat.T
        points = (new_transform_mat@points.T).T
        # # A, _, _, _ = np.linalg.lstsq(maxbasis_points, target_points, rcond=None)

        # # A = target_points@np.linalg.inv(maxbasis_points)

        # # transform_mat = target_points@np.linalg.inv(maxbasis_points)

        # print(maxbasis_points)
        # print(target_points)
        # print(transform_mat)

        # import pdb
        # pdb.set_trace()

        # assert np.allclose(target_points, transform_mat@maxbasis_points.T,
        #                    atol=1e-3), f"Transformation matrix is incorrect {target_points} != {transform_mat@maxbasis_points}"

        # new_white_pt = transform_mat@np.ones(3)
        # new_transform_mat = (np.diag(1/new_white_pt))@transform_mat
        # points = (transform_mat@points.T).T  # @ new_transform_mat.T

        points = np.power(points, 1/denom_of_nonlin)
        # return self.convert(points, from_space, M_basis)
        return points

    def convert_to_hering(self, points: npt.NDArray):
        """Convert points to Hering space -- make it easier to display

        Args:
            points (npt.NDArray): input points in from_space, that will be converted to to_space

        Returns:
            npt.NDArray: Converted points in Hering space
        """
        return points@GetHeringMatrix(self.transform.dim).T

    def convert_to_linear(self, points: npt.NDArray,
                          to_space: str | ColorSpaceType,
                          M_basis: str | ColorSpaceType,
                          denom_of_nonlin: float) -> npt.NDArray:
        # revert the power
        points = np.power(points, denom_of_nonlin)
        # Convert to the basis space
        return self.convert(points, M_basis, to_space)

    def to_tetra_color(self, points: npt.NDArray, from_space: ColorSpaceType) -> List[TetraColor]:
        """
        Convert VSH points to TetraColor objects.

        Parameters:
            vsh_points (npt.NDArray): Points in VSH space

        Returns:
            List[TetraColor]: List of TetraColor objects
        """
        # Convert to RGB_OCV space
        six_d_color = self.convert(points, ColorSpaceType.VSH, ColorSpaceType.DISP_6P)

        # Create TetraColor objects
        return [TetraColor(six_d_color[i, :3], six_d_color[i, 3:])
                for i in range(six_d_color.shape[0])]

    def to_plate_color(self, foreground: npt.NDArray, background: npt.NDArray, from_space: ColorSpaceType) -> PlateColor:
        """
        Create a PlateColor object from a VSH point and a background luminance.

        Parameters:
            vsh_point (npt.NDArray): Point in VSH space for the foreground
            background_luminance (float): Luminance value for the background

        Returns:
            PlateColor: PlateColor object with foreground and background
        """

        # Convert both points to RGB_OCV
        points = np.vstack([foreground, background])
        six_d_colors = self.convert(points, from_space, ColorSpaceType.DISP_6P)

        # Create TetraColor objects for foreground and background
        front = TetraColor(six_d_colors[0, :3], six_d_colors[0, 3:])
        back = TetraColor(six_d_colors[1, :3], six_d_colors[1, 3:])

        # Return the PlateColor
        return PlateColor(front, back)

    def get_RYGB_to_DISP_6P(self):
        """
        Get the transformation matrix from RYGB to RGB/OCV

        Returns:
            npt.NDArray: Transformation matrix
        """
        return GetMaxBasisToDisplayTransform(self.transform)

    def get_RYGB_to_sRGB(self):
        """
        Get the transformation matrix from RYGB to sRGB

        Returns:
            npt.NDArray: Transformation matrix
        """
        max_to_cone = np.linalg.inv(self.transform.cone_to_disp) @ self.transform.maxbasis_to_disp
        max_to_sRGB = M_XYZ_to_RGB @ self.transform.cone_to_XYZ @ max_to_cone
        print(max_to_sRGB)
        return max_to_sRGB

    def get_background(self, luminance):

        vec = np.zeros(self.dim)
        vec[0] = luminance
        return self.to_tetra_color(np.array([vec]), from_space=ColorSpaceType.VSH)[0]

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
    print(repr(OKLAB_M1@cst.transform.cone_to_XYZ))
    # cones = cst.convert(xyz, ColorSpaceType.XYZ, ColorSpaceType.CONE)
    # oklab = cst.convert(cones, ColorSpaceType.CONE, ColorSpaceType.OKLAB)

    # Plot the original XYZ color matching functions
    axes[0].plot(observer.wavelengths, xyz)
    axes[0].set_title("CIE 1931 XYZ Color Matching Functions")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Value")

    # Plot the transformed OKLAB M1 values
    axes[1].plot(observer.wavelengths, (observer.normalized_sensor_matrix.T @ cst.transform.cone_to_XYZ.T) @ OKLAB_M1.T)
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
