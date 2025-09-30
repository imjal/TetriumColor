from typing import List, Tuple
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

from TetriumColor.Observer import Spectra
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType


def save_primaries_into_csv(primaries_dir: str, primaries_filename: str):
    spectras = load_primaries_from_csv(primaries_dir)

    wavelengths = np.arange(380, 781, 4)  # Assuming a fixed wavelength range
    with open(primaries_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Wavelength'] + [f'Primary {i+1}' for i in range(len(spectras))])
        for i, wavelength in enumerate(wavelengths):
            row = [wavelength] + [spectras[j].data[i] if spectras[j]
                                  is not None else None for j in range(len(spectras))]
            writer.writerow(row)


def load_primaries_from_csv(primaries_dir: str) -> List[Spectra]:
    """Load primaries from a csv file

    Args:
        primary_csv (str): path to the csv file

    Returns:
        List[Spectra]: list of Spectra objects representing the Primaries measured
    """

    try:
        return get_spectras_from_rgbo_list(primaries_dir, [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)])
    except Exception as e:
        raise Exception(f"Error loading primaries from {primaries_dir}: {e}")


def get_spectras_from_rgbo_list(
    directory: str, rgbo_list: List[Tuple[int, int, int, int]]
) -> List[Spectra]:
    """Given a list of (r, g, b, o) tuples, read the corresponding power data if available.

    Returns a list of power value lists in the same order as the input RGBO list.
    If a file is missing, the corresponding entry is None and a warning is printed.
    """
    wavelengths = np.arange(380, 781, 4)  # Assuming a fixed wavelength range
    results = []

    for rgbo in rgbo_list:
        r, g, b, o = rgbo
        filename = f"r{r}g{g}b{b}o{o}.csv"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            print(f"Warning: CSV file for {rgbo} not found at {filepath}")
            results.append(None)
            continue

        power_values = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header if present
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    power = float(row[1])
                    power_values.append(power)
                except ValueError:
                    continue  # Skip malformed rows

        if len(power_values) > len(wavelengths):
            power_values = power_values[-len(wavelengths):]  # take the soonest measurements
        results.append(Spectra(wavelengths=wavelengths, data=np.array(power_values)))
    return results


def get_rgbo_from_filename(filename: str) -> Tuple[int, int, int, int]:
    """Extract RGBO tuple from filename like r12g34b56o78.csv"""
    name = os.path.splitext(filename)[0]
    parts = name.strip("rgbop")
    r = int(name[name.index("r") + 1: name.index("g")])
    g = int(name[name.index("g") + 1: name.index("b")])
    b = int(name[name.index("b") + 1: name.index("o")])
    o = int(name[name.index("o") + 1:])
    return r, g, b, o


def renormalize_spectra(observer, primaries: List[Spectra], scaling_factor: float = 10000):

    disp = observer.observe_spectras(primaries)  # each row is a cone_vec
    intensities = disp.T * scaling_factor  # each column is a cone_vec
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    return white_weights


def compare_dataset_to_primaries(
    measurements_dir: str,
    rgbo_list: List[Tuple[int, int, int, int]],
    primary_spectra: List[Spectra],
    exclude_primaries: bool = True
) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    """
    Compares all spectra in a directory to their predicted linear combinations from primaries.

    Args:
        measurements_dir (str): Path to the directory containing measured CSVs
        primary_spectra (List[Spectra]): List of 4 Spectra objects for (R, G, B, O) at 255
        exclude_primaries (bool): Whether to exclude pure primaries from evaluation

    Returns:
        List[Tuple[RGBO, RMSE, diff_spectrum]]
    """
    spectrums = get_spectras_from_rgbo_list(measurements_dir, rgbo_list)
    results = []
    for idx, (rgbo, spectrum) in enumerate(zip(rgbo_list, spectrums)):
        if spectrum is None:
            print(f"Warning: Spectrum for {rgbo} not found.")
            continue

        if exclude_primaries and rgbo in [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)]:
            continue

        scaling_factors = np.array(rgbo) / 255.0

        predicted = Spectra(wavelengths=primary_spectra[0].wavelengths, data=np.array(sum(
            scale * primary.data for scale, primary in zip(scaling_factors, primary_spectra)
        )))

        diff = spectrum.data - predicted.data

        rmse = np.sqrt(np.mean(diff ** 2))
        plot_measured_vs_predicted(idx, rgbo, spectrum, predicted, rmse)
        results.append((rgbo, rmse, diff))

    return results


def export_predicted_vs_measured_with_square_coords(
    measurements_dir: str,
    rgbo_list: List[Tuple[int, int, int, int]],
    primary_spectra: List["Spectra"],
    output_dir: str,
    exclude_primaries: bool = True
):
    """
    Exports predicted vs measured spectra to CSV using filenames:
    square_<top_or_bottom>_<i>_<j>.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    spectrums = get_spectras_from_rgbo_list(measurements_dir, rgbo_list)

    assert len(rgbo_list) == 50, "Expected exactly 50 RGBOs."

    for idx, (rgbo, spectrum) in enumerate(zip(rgbo_list, spectrums)):
        if spectrum is None:
            print(f"Skipping {rgbo}: Measured spectrum not found.")
            continue

        if exclude_primaries and rgbo in [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)]:
            continue

        top_or_bottom = 0 if idx < 25 else 1
        relative_idx = idx if top_or_bottom == 0 else idx - 25
        i, j = divmod(relative_idx, 5)  # 5x5 grid
        if top_or_bottom:
            i = 4 - i

        scaling_factors = np.array(rgbo) / 255.0
        predicted_data = sum(scale * primary.data for scale, primary in zip(scaling_factors, primary_spectra))
        predicted = Spectra(wavelengths=primary_spectra[0].wavelengths, data=predicted_data)

        if not np.array_equal(spectrum.wavelengths, predicted.wavelengths):
            print(f"Skipping {rgbo}: Wavelength mismatch.")
            continue

        # top or bottom, then transposed j and i, where i is reversed on the bottom half of the cube
        filename = f"square_{top_or_bottom}_{j}_{i}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Wavelength", "Measured", "Predicted"])
            for wl, p, m in zip(predicted.wavelengths, spectrum.data, predicted.data):
                if wl < 400 or wl > 700:
                    continue
                writer.writerow([wl, p, m])


def export_metamer_difference(
    observer,
    cs,
    measurements_dir: str,
    rgbo_list: List[Tuple[int, int, int, int]],
    primary_spectra: List["Spectra"],
    output_dir: str,
):
    """
    Exports predicted vs measured spectra to CSV using filenames:
    square_<top_or_bottom>_<i>_<j>.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    spectrums = get_spectras_from_rgbo_list(measurements_dir, rgbo_list)

    assert len(rgbo_list) == 50, "Expected exactly 50 RGBOs."

    for idx in range(0, 50, 2):  # only top half
        cone_response = np.zeros((2, observer.dimension))
        measured_spectras = []
        for j in range(2):
            spectrum = spectrums[idx + j]
            measured_spectras.append(spectrum)
            rgbo = rgbo_list[idx + j]

            scaling_factors = np.array(rgbo) / 255.0
            predicted_data = sum(scale * primary.data for scale, primary in zip(scaling_factors, primary_spectra))
            predicted = Spectra(wavelengths=primary_spectra[0].wavelengths, data=predicted_data)

            if not np.array_equal(spectrum.wavelengths, predicted.wavelengths):
                print(f"Skipping {rgbo}: Wavelength mismatch.")
                continue

            # top or bottom, then transposed j and i, where i is reversed on the bottom half of the cube
            filename = f"metamer_{idx//2}_{j}.csv"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Wavelength", "Measured", "Predicted"])
                for wl, p, m in zip(predicted.wavelengths, spectrum.data, predicted.data):
                    if wl < 400 or wl > 700:
                        continue
                    writer.writerow([wl, p, m])

            cone_response[j] = observer.observe_spectras([spectrum])[0]

        lmsq_filename = f"LMSQ_{idx//2}.csv"
        lmsq_filepath = os.path.join(output_dir, lmsq_filename)

        measured = observer.observe_spectras(measured_spectras) * 10000
        white_weights = renormalize_spectra(observer, primary_spectra)
        disp_vals = cs.convert(measured, from_space=ColorSpaceType.CONE,
                               to_space=ColorSpaceType.DISP) * white_weights
        hering_vals_new = cs.convert(disp_vals, from_space=ColorSpaceType.DISP,
                                     to_space=ColorSpaceType.HERING)[:, 1:]
        sRGBvals_new = cs.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.SRGB)
        cone_vals = cs.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.CONE)

        with open(lmsq_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["L", "M", "S", "Q"])
            s, m, q, l = np.abs(cone_vals[0] - cone_vals[1])
            writer.writerow([l, m, s, q])


def plot_measured_vs_predicted(
    idx: str,
    rgbo: Tuple[int, int, int, int],
    measured: Spectra,
    predicted: Spectra,
    rmse: float,
    save_path: str | None = None
):
    """
    Plot measured vs predicted spectra and their difference.

    Args:
        rgbo (Tuple): RGBO tuple
        measured (Spectra): Measured spectrum
        predicted (np.ndarray): Predicted spectrum from primaries
        rmse (float): Root mean square error
        save_path (str, optional): If given, save the plot to this path instead of showing
    """
    wavelengths = measured.wavelengths
    diff = measured.data - predicted.data

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, measured.data, label="Measured", color='blue')
    plt.plot(wavelengths, predicted.data, label="Predicted", color='orange')
    plt.plot(wavelengths, diff, label="Difference", color='red', linestyle='--')
    plt.title(f"Number- {idx} - RGBO {rgbo} - RMSE: {rmse:.4f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
