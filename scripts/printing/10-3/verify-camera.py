from TetriumColor.Measurement.Nix import read_nix_csv
from TetriumColor.Observer import Spectra, Illuminant
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import TetriumColor.Measurement.SpecimIQ as specim


def inspect_camera(ref_num: int, type_image: str = "reflectance"):

    if type_image == "reflectance":
        selected_spectra = specim.inspect_hyperspectral_image(
            f"{ref_num}/results/REFLECTANCE_{ref_num}.hdr", output_wavelengths=np.arange(400, 701, 5))
    elif type_image == "raw":
        selected_spectra = specim.inspect_hyperspectral_image(
            f"{ref_num}/capture/{ref_num}.hdr", plot_spectra_color=np.zeros(3))

    # plt.suptitle("Nix Scan w/Specim IQ Picture")
    for name, spectra in selected_spectra.items():
        spectra.plot(name=name, color=np.zeros(3))

    plt.legend()
    plt.savefig("reflectance_paper_w_d50_light.png")


if __name__ == "__main__":

    import os
    os.environ["SPECTRAL_DATA"] = "/Users/jessicalee/Projects/generalized-colorimetry/code/TetriumColor/measurements/2025-10-03/hypespectral/"
    specim.inspect_raw_data("1340/capture/1340.hdr")
    # inspect_camera(1340, type_image="raw")
