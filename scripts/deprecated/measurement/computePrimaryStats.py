import pandas as pd
import numpy as np
import os
# Add this import at the top of the file if not already present
from colour import SpectralDistribution, SpectralShape, XYZ_to_xy, XYZ_to_Luv, xy_to_Luv_uv, sd_to_XYZ, dominant_wavelength
from colour.plotting import plot_chromaticity_diagram_CIE1931
import dataframe_image as dfi

from TetriumColor.Observer import Illuminant
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def computeStatsRoutine(led_spectrums_path):
    # Get the directory path of led_spectrums_path
    save_path = os.path.join(os.path.dirname(led_spectrums_path), "results")
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(led_spectrums_path)

    # Extract wavelength and spectra
    wavelengths = data.iloc[:, 0].values
    # spectra = data.iloc[:, 1:]
    excluded = [6, 7, 8]
    spectra = data.iloc[:, [x for x in range(1, data.shape[1]) if x not in excluded]]
    # spectra = our_primaries[:, [x for x in range(1, our_primaries.shape[1]) if x not in excluded]]

    # Function to calculate FWHM

    def calculate_fwhm(wavelengths, spectrum):
        half_max = np.max(spectrum) / 2
        indices = np.where(spectrum >= half_max)[0]
        if len(indices) < 2:
            return None
        fwhm = wavelengths[indices[-1]] - wavelengths[indices[0]]
        return fwhm

    # Initialize results
    results = []

    # Compute the xy chromaticity coordinates of D65
    Illuminant_D65 = Illuminant.get("D65")
    d65_sd = Illuminant_D65.to_colour().align(SpectralShape(380, 780, 1))
    d65_xyz = sd_to_XYZ(d65_sd)
    d65_xy = XYZ_to_xy(d65_xyz)

    # Process each spectrum
    for column in spectra.columns:
        spectrum = spectra[column].values

        # Find peak wavelength
        peak_idx = np.argmax(spectrum)
        peak_wavelength = wavelengths[peak_idx]

        # Calculate FWHM
        fwhm = calculate_fwhm(wavelengths, spectrum)
        # Fit a Gaussian to the spectrum

        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        try:
            popt, _ = curve_fit(gaussian, wavelengths, spectrum, p0=[np.max(spectrum), peak_wavelength, fwhm / 2.355])
            fitted_spectrum = gaussian(wavelengths, *popt)

            # Compare the Gaussian fit to the actual spectrum
            plt.figure()
            plt.plot(wavelengths, spectrum, label="Actual Spectrum")
            plt.plot(wavelengths, fitted_spectrum, label="Gaussian Fit", linestyle="--")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.title(f"Spectrum: {column}")
            plt.legend()
            plt.grid(True)
            # plt.savefig(os.path.join(save_path, f"{column}_gaussian_fit.png"), dpi=300)
            # plt.close()
            plt.show()
            # conclusion is that they are fine
        except RuntimeError:
            print(f"Gaussian fit failed for spectrum: {column}")

        # Create a spectral distribution for colorimetry
        sd = SpectralDistribution(dict(zip(wavelengths, spectrum))).interpolate(
            SpectralShape(wavelengths.min(), wavelengths.max() + 1, 1))

        # Calculate dominant wavelength and chromaticity coordinates

        xyz = sd_to_XYZ(sd)
        xy = XYZ_to_xy(xyz)
        luv = XYZ_to_Luv(xyz)
        uv = xy_to_Luv_uv(xy)
        dm_wv = float(dominant_wavelength(xy, d65_xy)[0])

        # Append results
        results.append({
            "Spectrum": column,
            "Peak Wavelength (nm)": peak_wavelength,
            "Dominant Wavelength (nm)": dm_wv,
            "FWHM (nm)": fwhm,
            "x": xy[0],
            "y": xy[1],
            "L": luv[0],
            "u": uv[0],
            "v": uv[1]
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results as a table
    print(results_df)

    # Optionally save the results to a CSV file
    results_df.to_csv(os.path.join(save_path, "spectrum_analysis_results.csv"), index=False)

    # Pretty print the results as an image

    # Save the DataFrame as an image
    dfi.export(results_df, os.path.join(save_path, "spectrum_analysis_results.png"))

    # Calculate the average FWHM of all LEDs under 56 nm bc those are unusable to some extent
    average_fwhm = results_df[results_df["FWHM (nm)"] < 56]["FWHM (nm)"].mean()
    print(f"Average FWHM (nm) of LEDs under 56 nm: {average_fwhm}")  # 22.4 for current set

    # Plot the chromaticity coordinates on the CIE xy color diagram

    # Create the plot
    plot_chromaticity_diagram_CIE1931(standalone=False)

    # Add the chromaticity points
    for _, row in results_df.iterrows():
        # plt.plot(row["x"], row["y"], 'o', label=row["Spectrum"])
        # Plot the color as the x, y color
        plt.scatter(row["x"], row["y"], color=(row["x"], row["y"], 1 - row["x"] - row["y"]),
                    s=100, edgecolor='black', label=row["Spectrum"])

    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CIE 1931 Chromaticity Diagram")
    plt.legend(loc="upper right", fontsize="small", title="Spectra")
    plt.grid(True)

    # Show the plot
    plt.savefig(os.path.join(save_path, "chromaticity_diagram.png"), dpi=300)


if __name__ == "__main__":
    # Example usage
    led_spectrums_path = "../../measurements/2025-04-04/led-spectrums.csv"
    computeStatsRoutine(led_spectrums_path)
