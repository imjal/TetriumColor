import numpy as np

import numpy.typing as npt
from typing import List
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_RGB, wavelength_to_XYZ, SpectralShape

from . import Observer, Spectra, MaxBasis, MaxBasisFactory, Illuminant
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, CSTDisplayType


def GetsRGBfromWavelength(wavelength):
    try:
        return XYZ_to_RGB(wavelength_to_XYZ(wavelength), "sRGB")
    except Exception as e:
        return np.array([0, 0, 0])


def GetConeTosRGBPrimaries(observer: Observer, metameric_axis: int = 2):
    M_XYZ_to_RGB = RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB
    return M_XYZ_to_RGB@GetConeToXYZPrimaries(observer, metameric_axis)


def GetConeToXYZPrimaries(observer: Observer, metameric_axis: int = 2) -> npt.NDArray:
    """Get the 3xobserver.dim matrix that transforms from cone space to XYZ space

    Args:
        observer (Observer): Observer to transform from
        metameric_axis (int, optional): the dimension to drop to transform to XYZ. Use the idxs of the non LMS cones. Defaults to 2.

    Returns:
        npt.NDArray: the transformation matrix from cone space to XYZ space
    """
    subset = list(range(observer.dimension))
    if observer.dimension > 3:
        subset = [i for i in range(4) if i != metameric_axis]

    shape = SpectralShape(min(observer.wavelengths), max(observer.wavelengths),
                          int(observer.wavelengths[1] - observer.wavelengths[0]))
    xyz_cmfs = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].copy().align(shape).values
    xyz_d65 = xyz_cmfs.T @ Illuminant.get("D65").to_colour().align(shape).values
    xyz_d65 = xyz_d65/xyz_d65[1]

    # 1. Calculate initial transformation matrix
    # (using pseudoinverse with your sample colors)
    M_initial = xyz_cmfs.T @ np.linalg.pinv(observer.normalized_sensor_matrix[subset])
    # SML -> XYZ

    # 2. Apply to D65 white point
    xyz_d65_transformed = M_initial @ np.ones(len(subset))

    # 3. Calculate scaling factors
    scaling_factors = xyz_d65 / xyz_d65_transformed

    # 4. Apply scaling to transformation matrix
    # (scaling each row of the matrix)
    M_scaled = np.diag(scaling_factors) @ M_initial
    return M_scaled


def GetColorSpaceTransform(observer: Observer, display_basis: CSTDisplayType,
                           display_primaries: List[Spectra] | None = None,
                           metameric_axis: int = 2,
                           led_mapping: List[int] | None = [0, 1, 3, 2, 1, 3],
                           scaling_factor: float = 100, generate_max_basis=False) -> ColorSpaceTransform:
    """Get ColorSpaceTransform for the observer

    Args:
        observer (Observer): Observer  
        display_basis (CSTDisplayType): display_basis type for CST
        display_primaries (List[Spectra] | None, optional): display primaries. Defaults to None.
        metameric_axis (int, optional): metameric axis for which cone to "drop". Defaults to 2.
        led_mapping (List[int] | None, optional): based on the idxs of dispaly_primaries,
            how to fill in the 6p even-odd display. Defaults to [0, 1, 3, 2, 1, 3], which is the RGO/BGO display on an input of RGBO.
        scaling_factor (float, optional): scale the measurements of the display for better linalg. Defaults to 1000.

    Raises:
        ValueError: Display primaries and led_mapping must be provided for LED display basis together.
        ValueError: Observer dimension must be 3 for sRGB display basis.

    Returns:
        ColorSpaceTransform: the ColorSpaceTransform object that represents all transform matrices of the observer to display basis
    """
    # Set M_Cone_To_Primaries and white_weights according to the display type wanted
    if display_basis == CSTDisplayType.LED:
        if display_primaries is None or led_mapping is None:
            raise ValueError("Display primaries and led_mapping must be provided for LED display basis together.")
        disp = observer.observe_spectras(display_primaries)
        intensities = disp.T * scaling_factor
        white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
        white_weights = np.linalg.inv(intensities)@white_pt
        # rescaled_white_weights = white_weights / np.max(white_weights)

        new_intensities = intensities * np.max(white_weights)
        rescaled_white_weights = np.ones(observer.dimension)
        M_Cone_To_Primaries = np.linalg.inv(new_intensities)  # something is fucked
    else:
        rescaled_white_weights = np.ones(observer.dimension)
        led_mapping = [i for i in range(observer.dimension)]
        if display_basis == CSTDisplayType.NONE:
            M_Cone_To_Primaries = np.eye(observer.dimension)
        elif display_basis == CSTDisplayType.SRGB:
            if observer.dimension != 3:
                raise ValueError("Observer dimension must be 3 for sRGB display basis.")
            M_Cone_To_Primaries = GetConeTosRGBPrimaries(observer, metameric_axis)

    # Get all of the max_basis that we will use -> at most these many
    # print("Calculating MaxBasis")
    max_basis = MaxBasisFactory.get_object(observer, denom=1, verbose=True)
    # max_basis = MaxBasis(observer, denom=1, verbose=True)
    # print("MaxBasis calculated w/denom=1")
    # print(max_basis.cutpoints)
    # max_basis_243 = MaxBasisFactory.get_object(observer, denom=2.43 if generate_max_basis else 1, verbose=False)
    # print("MaxBasis calculated w/denom=2.43")
    # max_basis_3 = MaxBasisFactory.get_object(observer, denom=3 if generate_max_basis else 1, verbose=False)
    # print("MaxBasis calculated w/denom=3")

    # print("MaxBasis calculated")

    # Get all transforms from cone to maxbasis to hering
    M_PrimariesToCone = np.linalg.inv(M_Cone_To_Primaries)
    M_ConeToMaxBasis = max_basis.cone_to_maxbasis
    # M_ConeToMaxBasis243 = max_basis_243.cone_to_maxbasis
    # M_ConeToMaxBasis3 = max_basis_3.cone_to_maxbasis
    M_MaxBasisToHering = max_basis.HMatrix

    # transform the cone to the disp basis
    M_ConeToHering = M_MaxBasisToHering@M_ConeToMaxBasis
    M_PrimariesToMaxBasis = M_ConeToMaxBasis@M_PrimariesToCone
    # M_PrimariesToMaxBasis243 = M_ConeToMaxBasis243@M_PrimariesToCone
    # M_PrimariesToMaxBasis3 = M_ConeToMaxBasis3@M_PrimariesToCone
    M_PrimariesToHering = M_MaxBasisToHering@M_PrimariesToMaxBasis

    # Get the cone to XYZ matrix
    M_Cone_to_XYZ = GetConeToXYZPrimaries(observer, metameric_axis)

    return ColorSpaceTransform(
        observer.dimension,
        M_Cone_To_Primaries,
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToMaxBasis),
        np.linalg.inv(M_PrimariesToHering),
        np.linalg.inv(M_ConeToHering),
        metameric_axis,
        led_mapping,
        rescaled_white_weights,
        M_Cone_to_XYZ
    )


def GetMaxBasisToDisplayTransform(color_space_transform: ColorSpaceTransform) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate from a 4x4 matrix that converts from RYGB to display primaries, a two 4x3 matrices
    that represent the conversion from max basis to display primaries in order to interface with Tetrium Paint Program

    Args:
        color_space_transform (ColorSpaceTransform): the colorspace transform to use.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: a tuple of 4x3 matrices that converts from max basis to display primaries
    """
    # We want to now go from RYGB to RGO / BGO
    disp_weights = np.identity(4) * color_space_transform.white_weights
    # go from bgyr to rygb
    rygb_to_disp = disp_weights @ np.flip(color_space_transform.maxbasis_to_disp, axis=1)

    rygb_to_disp1 = np.zeros((3, 4))
    rygb_to_disp1 = rygb_to_disp[color_space_transform.led_mapping[:3]]

    rygb_to_disp2 = np.zeros((3, 4))
    rygb_to_disp2 = rygb_to_disp[color_space_transform.led_mapping[3:]]

    return rygb_to_disp1.T, rygb_to_disp2.T
