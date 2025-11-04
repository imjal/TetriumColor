
import pdb
from typing import List
import numpy as np
from TetriumColor import ColorSpace
from TetriumColor.Observer import Spectra, Observer
from TetriumColor.Measurement import load_primaries_from_csv


def print_glm_format(matrix: np.ndarray):
    print("glm::mat4x4{")
    for row in matrix:
        print(f"{{{row[0]}, {row[1]}, {row[2]}, {0.0}}},")
    print("},")


def generate_tetrium_matrices(observer, primaries):

    cs = ColorSpace(observer, display_primaries=primaries, metameric_axis=2)

    rygb_to_rgb, rygb_to_ocv = cs.get_RYGB_to_DISP_6P()

    print(rygb_to_rgb.T)
    print(rygb_to_ocv.T)

    print_glm_format(rygb_to_rgb)

    print_glm_format(rygb_to_ocv)

    # rygb_to_sRGB = np.flip(cs.get_RYGB_to_sRGB().T, axis=0)

    # print_glm_format(rygb_to_sRGB)


if __name__ == "__main__":
    np.set_printoptions(precision=8)
    wavelengths = np.arange(360, 831, 1)
    observer = Observer.custom_observer(wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559, template="neitz")

    primaries: List[Spectra] = load_primaries_from_csv("../../measurements/2025-05-06/")
    generate_tetrium_matrices(observer, primaries)
