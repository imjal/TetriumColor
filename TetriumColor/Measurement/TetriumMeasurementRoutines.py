from typing import List, Tuple, Optional
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


def load_primaries_from_csv(primaries_dir: str,
                            extract_zero: bool = False,
                            smooth_method: Optional[str] = 'gaussian',
                            primary_order: str = 'RGBO') -> List[Spectra]:
    """Load primaries from a csv file with optional zero extraction and smoothing.

    Args:
        primaries_dir (str): path to the directory containing primary measurements
        extract_zero (bool): If True, uses consecutive measurements to extract and
            subtract the zero/offset that PR650 can't measure directly.
        smooth_method (str, optional): Interpolation method to smooth/upsample to 1nm.
            Options: 'asymmetric_gaussian', 'gaussian', 'cubic', 'linear', etc.
        primary_order (str): Ordering of returned primaries list. Options:
            'RGBO' (default) - [Red, Green, Blue, Orange], matches led_mapping=[0,1,3,2,1,3]
                for correct 6P frame output (RGB frame=RGO, OCV frame=BGO).
            'BGOR' - [Blue, Green, Orange, Red], used by validation scripts that zip
                BGOR-ordered weights directly against the primaries list.

    Returns:
        List[Spectra]: list of Spectra objects in the requested primary_order
    """
    try:
        if extract_zero:
            # Use consecutive measurements to extract zero
            primaries, _ = load_primaries_with_zero_extraction(
                primaries_dir,
                smooth_method=smooth_method
            )
        else:
            # Standard loading: average last 4 measurements in RGBO order
            primaries = get_spectras_from_rgbo_list(
                primaries_dir,
                [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)]
            )

            # Apply smoothing/interpolation to 1nm resolution
            if smooth_method is not None:
                output_wavelengths = np.arange(380, 781, 1)
                primaries = [
                    p.interpolate(output_wavelengths, method=smooth_method) if p is not None else None
                    for p in primaries
                ]

        # primaries is now in RGBO order: [R=0, G=1, B=2, O=3]
        if primary_order == 'BGOR':
            # Reorder RGBO=[R,G,B,O] → BGOR=[B,G,O,R]
            primaries = [primaries[2], primaries[1], primaries[3], primaries[0]]
        elif primary_order != 'RGBO':
            raise ValueError(f"Unknown primary_order '{primary_order}'. Use 'RGBO' or 'BGOR'.")

        return primaries
    except Exception as e:
        raise Exception(f"Error loading primaries from {primaries_dir}: {e}")


def load_single_measurement(filepath: str) -> Spectra:
    """Load a single measurement CSV file into a Spectra object.

    Args:
        filepath: Path to the CSV file

    Returns:
        Spectra object with the measurement data
    """
    wavelengths = np.arange(380, 781, 4)
    power_values = []

    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                power_values.append(float(row[1]))
            except ValueError:
                continue

    if len(power_values) > len(wavelengths):
        power_values = power_values[-len(wavelengths):]

    return Spectra(wavelengths=wavelengths, data=np.array(power_values))


def get_measurement_files_for_rgbo(directory: str, rgbo: Tuple[int, int, int, int]) -> List[str]:
    """Get all measurement files for a given RGBO, sorted by timestamp.

    Args:
        directory: Directory containing measurement files
        rgbo: (R, G, B, O) tuple

    Returns:
        List of filenames sorted by timestamp (oldest first)
    """
    r, g, b, o = rgbo
    pattern = f"r{r}g{g}b{b}o{o}"

    matching_files = []
    for filename in os.listdir(directory):
        if filename.startswith(pattern) and filename.endswith('.csv'):
            matching_files.append(filename)

    matching_files.sort()  # Sorts by timestamp embedded in filename
    return matching_files


def extract_zero_from_consecutive_measurements(
    primaries_dir: str,
    smooth_method: Optional[str] = 'asymmetric_gaussian'
) -> Tuple[List[Spectra], List[Spectra]]:
    """Extract zero/offset by subtracting consecutive measurements of the same LED.

    Since PR650 can't measure true zero (too dim), we estimate it by taking
    two back-to-back measurements of the same LED setting. If the LED is stable,
    their difference reveals the measurement noise floor and systematic offset.

    Method:
    - For each primary (R, G, B, O), find all timestamped measurements
    - Take two consecutive measurements: M1 and M2
    - The average (M1 + M2) / 2 is our best estimate of the signal
    - The difference |M1 - M2| / 2 estimates the noise/zero level
    - Subtract the estimated zero from the signal

    Args:
        primaries_dir: Directory with measurement CSVs
        smooth_method: Interpolation method ('asymmetric_gaussian', 'gaussian', 'cubic', etc.)

    Returns:
        Tuple of (corrected_primaries, estimated_zeros)
    """
    primary_patterns = [
        (255, 0, 0, 0),  # R
        (0, 255, 0, 0),  # G
        (0, 0, 255, 0),  # B
        (0, 0, 0, 255),  # O
    ]

    corrected_primaries = []
    estimated_zeros = []

    for rgbo in primary_patterns:
        files = get_measurement_files_for_rgbo(primaries_dir, rgbo)

        if len(files) < 2:
            print(f"Warning: Need at least 2 measurements for {rgbo}, found {len(files)}")
            if len(files) == 1:
                filepath = os.path.join(primaries_dir, files[0])
                corrected_primaries.append(load_single_measurement(filepath))
                estimated_zeros.append(None)
            else:
                corrected_primaries.append(None)
                estimated_zeros.append(None)
            continue

        # Use the last two consecutive measurements
        m1_path = os.path.join(primaries_dir, files[-2])
        m2_path = os.path.join(primaries_dir, files[-1])

        m1 = load_single_measurement(m1_path)
        m2 = load_single_measurement(m2_path)

        # Average is our best signal estimate
        signal_estimate = (m1.data + m2.data) / 2

        # Half the absolute difference estimates the zero/noise floor
        zero_estimate = np.abs(m1.data - m2.data) / 2

        # Subtract zero estimate from signal
        corrected_data = signal_estimate - zero_estimate
        corrected_data = np.clip(corrected_data, 0, None)

        corrected_primaries.append(Spectra(wavelengths=m1.wavelengths, data=corrected_data))
        estimated_zeros.append(Spectra(wavelengths=m1.wavelengths, data=zero_estimate))

        print(f"{rgbo}: Used {files[-2]} and {files[-1]}, zero estimate max={np.max(zero_estimate):.4f}")

    # Apply smoothing/interpolation to 1nm resolution
    if smooth_method is not None:
        output_wavelengths = np.arange(380, 781, 1)
        corrected_primaries = [
            p.interpolate(output_wavelengths, method=smooth_method) if p is not None else None
            for p in corrected_primaries
        ]

    return corrected_primaries, estimated_zeros


def load_primaries_with_zero_extraction(
    primaries_dir: str,
    measurement_indices: Tuple[int, int] = (-2, -1),
    smooth_method: Optional[str] = 'asymmetric_gaussian'
) -> Tuple[List[Spectra], List[Spectra]]:
    """Load primaries by extracting zero from consecutive measurements.

    Takes two measurements of each LED setting and subtracts them to remove
    the PR650's systematic offset that can't be directly measured.

    Args:
        primaries_dir: Directory containing measurement CSVs
        measurement_indices: Which measurements to use (default: last two)
        smooth_method: Interpolation method to smooth/upsample to 1nm.
            Options: 'asymmetric_gaussian', 'gaussian', 'cubic', 'linear', etc.

    Returns:
        Tuple of (primaries, zero_estimates)
    """
    primary_patterns = [
        (255, 0, 0, 0),  # R
        (0, 255, 0, 0),  # G
        (0, 0, 255, 0),  # B
        (0, 0, 0, 255),  # O
    ]

    primaries = []
    zero_estimates = []
    idx1, idx2 = measurement_indices

    for rgbo in primary_patterns:
        files = get_measurement_files_for_rgbo(primaries_dir, rgbo)

        if len(files) < 2:
            print(f"Warning: Need 2+ measurements for {rgbo}, found {len(files)}")
            if files:
                primaries.append(load_single_measurement(os.path.join(primaries_dir, files[-1])))
                zero_estimates.append(None)
            else:
                primaries.append(None)
                zero_estimates.append(None)
            continue

        # Load the two specified measurements
        m1 = load_single_measurement(os.path.join(primaries_dir, files[idx1]))
        m2 = load_single_measurement(os.path.join(primaries_dir, files[idx2]))

        # Subtract to remove common offset: (Signal + Zero) - (Signal + Zero')
        # If Zero ≈ Zero', the signals should be similar, difference shows drift
        # Average gives best signal estimate, difference shows zero level
        avg_signal = (m1.data + m2.data) / 2
        zero_level = np.abs(m1.data - m2.data) / 2

        corrected = np.clip(avg_signal - zero_level, 0, None)
        primaries.append(Spectra(wavelengths=m1.wavelengths, data=corrected))
        zero_estimates.append(Spectra(wavelengths=m1.wavelengths, data=zero_level))

        print(f"{rgbo}: Used {files[idx1]} and {files[idx2]}, zero estimate max={np.max(zero_level):.4f}")

    # Apply smoothing/interpolation using Spectra's interpolate method
    if smooth_method is not None:
        output_wavelengths = np.arange(380, 781, 1)  # 1nm resolution
        primaries = [
            p.interpolate(output_wavelengths, method=smooth_method) if p is not None else None
            for p in primaries
        ]

    return primaries, zero_estimates


def get_spectras_from_rgbo_list(
    directory: str,
    rgbo_list: List[Tuple[int, int, int, int]],
    smooth_method: Optional[str] = 'gaussian',
) -> List[Spectra]:
    """Given a list of (r, g, b, o) tuples, read the corresponding power data if available.

    Returns a list of Spectra in the same order as the input RGBO list.
    If a file is missing, the corresponding entry is None and a warning is printed.

    For each RGBO, finds all timestamped files (e.g., r255g0b0o0_20251202_165603_050.csv),
    sorts them by timestamp, and uses the most recent measurement.

    Args:
        directory: Directory containing measurement CSVs.
        rgbo_list: List of (R, G, B, O) tuples to load.
        smooth_method: Interpolation method to smooth/upsample to 1nm resolution,
            matching the treatment applied to primaries in load_primaries_from_csv.
            Options: 'gaussian', 'asymmetric_gaussian', 'cubic', 'linear', None.
            Defaults to 'gaussian'.
    """
    raw_wavelengths = np.arange(380, 781, 4)  # PR-650 native resolution
    results = []

    for rgbo in rgbo_list:
        r, g, b, o = rgbo
        pattern = f"r{r}g{g}b{b}o{o}"

        # Find all files matching this RGBO pattern
        matching_files = []
        for filename in os.listdir(directory):
            if filename.startswith(pattern) and filename.endswith('.csv'):
                matching_files.append(filename)

        if not matching_files:
            print(f"Warning: No CSV files for {rgbo} found in {directory}")
            results.append(None)
            continue

        # Sort by timestamp (embedded in filename)
        matching_files.sort()

        # Use the most recent file
        filepath = os.path.join(directory, matching_files[-1])
        power_values = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header if present
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    power_values.append(float(row[1]))
                except ValueError:
                    continue  # Skip malformed rows

        if len(power_values) > len(raw_wavelengths):
            power_values = power_values[-len(raw_wavelengths):]
        spectra = Spectra(wavelengths=raw_wavelengths, data=np.array(power_values))

        if smooth_method is not None:
            spectra = spectra.interpolate(np.arange(380, 781, 1), method=smooth_method)

        results.append(spectra)

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
