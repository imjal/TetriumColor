from TetriumColor.Measurement.Nix import read_nix_csv
from TetriumColor.Observer import Spectra, Illuminant
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import TetriumColor.Measurement.SpecimIQ as specim


def compare_nix_and_camera():
    nix_csv_path = "../../../measurements/2025-09-22/nix-scan-2.csv"
    scans, _ = read_nix_csv(nix_csv_path)
    nix_wavelengths = list(scans.values())[0].wavelengths
    # plt.savefig("scan.png")

    selected_spectra = specim.inspect_hyperspectral_image(
        "1338/results/REFLECTANCE_1338.hdr", output_wavelengths=np.arange(400, 701, 5))

    # plt.suptitle("Nix Scan w/Specim IQ Picture")
    for name, spectra in scans.items():
        spectra.plot(name=name, color=spectra.to_rgb())

    for i, spectra in enumerate(selected_spectra):
        spectra.plot(name=f"camera_measured_spectra_{i}", color=spectra.to_rgb())

    plt.legend()
    plt.savefig("nix_and_camera_together.png")


def extract_grid_from_camera_image(nix_wavelengths: List[float]):

    # Generate all combinations of 3 values in 0.25 increments from 0 to 1 (inclusive)
    increments = np.arange(0, 1.01, 0.25)
    combinations = [(a, b, c) for a, b, c in product(increments, repeat=3)]
    measured_grid = specim.extract_grid_from_image("1338/results/REFLECTANCE_1338.hdr",
                                                   "../../../measurements/2025-09-22/hyperspectral/1338/results/RGBSCENE_1338.png", halftones=combinations)

    for name, spectra in measured_grid.items():
        interpolated_spectra = spectra.interpolate(nix_wavelengths)
        interpolated_spectra.plot(name=name, color=interpolated_spectra.to_rgb())

    plt.legend()
    plt.savefig("measured_grid.png")


if __name__ == "__main__":

    import os
    os.environ["SPECTRAL_DATA"] = "/Users/jessicalee/Projects/generalized-colorimetry/code/TetriumColor/measurements/2025-09-22/hyperspectral/"

    illum = Illuminant.get("d50")
    illum.plot(name="d50")
    plt.title("d50")
    plt.show(block=True)

    specim.inspect_raw_data("1338/capture/1338.hdr")
    compare_nix_and_camera()
    # extract_grid_from_camera_image(nix_wavelengths)
