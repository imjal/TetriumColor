from typing import List
import numpy as np
import numpy.typing as npt

from TetriumColor.Utils.CustomTypes import PlateColor, TetraColor


def Map6DTo4D(colors: npt.NDArray, transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Nx6 Array in Plate Color Coordinates, transform into 4D Array

    :param colors: Nx6 Array in Plate Color Coordinates
    :param transform: ColorSpaceTransform to use for the conversion to a Plate Color
    """
    mat = np.zeros((colors.shape[0], 4))
    for i, mapped_idx in enumerate(transform.led_mapping):
        mat[:, i] = colors[:, mapped_idx] / transform.white_weights[i]
    return mat


def Map4DTo6D(colors: npt.NDArray, transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Nx4 Array in Display Space Coordinates, transform into 6D Array

    :param colors: Nx4 Array in Display Space Coordinates
    :param transform: ColorSpaceTransform to use for the conversion to a Plate Color
    """
    mat = np.zeros((colors.shape[0], 6))
    for i, mapped_idx in enumerate(transform.led_mapping):
        # Multiply the color by the white point weight
        # All colors are [0, 1] inside of the color space to make things "nice"
        # But when we need to transform to a display weight, we need to rescale them back
        # in their dynamic range -- need to double check that this is right theoretically!
        mat[:, i] = colors[:, mapped_idx] * transform.white_weights[mapped_idx]
    return mat
