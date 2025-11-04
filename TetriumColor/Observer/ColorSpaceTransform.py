import numpy as np

import numpy.typing as npt
from typing import List
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.models import RGB_COLOURSPACE_BT709
from colour import XYZ_to_RGB, wavelength_to_XYZ, SpectralShape

from . import Observer, Spectra, Illuminant


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
