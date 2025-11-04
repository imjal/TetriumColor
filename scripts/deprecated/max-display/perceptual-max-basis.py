

import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import combinations, product
from TetriumColor.Observer import *
from TetriumColor import *
import math
from tqdm import tqdm
import pandas as pd
from pandas.plotting import table
from joblib import Parallel, delayed
import pickle

AVG_FWHM = 22.4


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_pareto_front(volumes: np.ndarray, efficiencies: np.ndarray):
    """
    Efficiently identifies the Pareto front for maximizing both volume and efficiency.

    Returns:
        pareto_indices: Indices of the Pareto-optimal points
        pareto_volumes: Volumes of the Pareto-optimal points
        pareto_efficiencies: Efficiencies of the Pareto-optimal points
    """
    data = np.stack([volumes, efficiencies], axis=1)

    # Sort by volume descending, then efficiency descending
    sorted_idx = np.lexsort((-efficiencies, -volumes))
    sorted_data = data[sorted_idx]

    # Scan for Pareto-optimal points
    pareto_idx = []
    max_eff = -np.inf
    for i, (v, e) in tqdm(enumerate(sorted_data)):
        if e > max_eff:
            pareto_idx.append(sorted_idx[i])
            max_eff = e

    pareto_idx = np.array(pareto_idx)
    return pareto_idx


def compute_efficiency(color_space: ColorSpace, primary_candidates: npt.NDArray, spds: List[Spectra]) -> npt.NDArray:
    """Compute the efficiency of the primary candidates as inverse of power

    Args:
        color_space (ColorSpace): Color space object
        primary_candidates (npt.NDArray): primary candidates
        spds (List[Spectra]): spds of the primaries

    Returns:
        npt.NDArray: List of efficacies
    """
    # compute total power needed to reach luminance
    spd_powers = np.array([np.trapz(spd.data) for spd in spds])
    weights = []
    for p in primary_candidates:
        p = p.T
        try:
            w = np.linalg.solve(p, np.ones(color_space.dim))
            weights.append(w)
        except Exception as e:
            # Handle singular matrix case
            weights.append(np.ones(color_space.dim) * -1)
    efficacies = 1.0 / np.array([np.dot(w, spd_powers[idx])
                                 for w, idx in zip(weights, idxs)])  # power needed to reach luminance
    weights = np.array(weights)
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0
    return efficacies


def compute_max_chromatic_vol(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray) -> npt.NDArray:
    """Compute the maximum chromatic volume of the primary candidates

    Args:
        color_space (ColorSpace): Color space object
        chrom_basis (ColorSpaceType): chromatic basis to perform maximization in
        primary_candidates (npt.NDArray): primary candidates

    Returns:
        npt.NDArray: List of volumes
    """
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    # add a column of ones to use determinant to compute simplex volume
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)
    volumes[volumes < 0] = 0
    return volumes


def compute_CIELAB_volume(primaries: npt.NDArray, denom: float, hypercube_points: npt.NDArray, dimension: int):
    primaries = primaries.T
    w = np.linalg.solve(primaries, np.ones(dimension))
    if denom == 1.0:
        if np.any(w < 0):
            return 0
        return np.linalg.det(primaries@np.diag(w))
    try:
        cone_vals_inv = np.linalg.inv(primaries)
    except np.linalg.LinAlgError:
        return 0  # Singular matrix, volume is 0

    all_points = primaries @ np.diag(w) @ hypercube_points.T

    if np.isclose(denom, 3):
        transformed_points = np.cbrt(all_points)
    else:
        transformed_points = np.power(all_points, 1/denom)

    # TODO: Apply max basis here
    transformed_points = max_basis_mat @ transformed_points
    transformed_points[1, :] = transformed_points[1, :] * 1000

    min_coords = np.min(transformed_points, axis=1)
    max_coords = np.max(transformed_points, axis=1)

    num_samples = 1000000
    random_points = np.random.uniform(
        min_coords[:, np.newaxis],
        max_coords[:, np.newaxis],
        (dimension, num_samples)
    )

    inv_points = np.power(random_points, denom)
    coefficients = cone_vals_inv @ inv_points
    inside_mask = np.all((0 <= coefficients) & (coefficients <= 1), axis=0)
    points_inside = np.sum(inside_mask)

    bounding_box_volume = np.prod(max_coords - min_coords)
    volume = bounding_box_volume * (points_inside / num_samples)
    return volume


def compute_volume(primaries: npt.NDArray, denom: float, hypercube_points, max_basis_mat: npt.NDArray, dimension: int):
    primaries = primaries.T
    w = np.linalg.solve(primaries, np.ones(dimension))
    if denom == 1.0:
        if np.any(w < 0):
            return 0
        return np.linalg.det(primaries@np.diag(w))
    try:
        cone_vals_inv = np.linalg.inv(primaries)
    except np.linalg.LinAlgError:
        return 0  # Singular matrix, volume is 0

    all_points = primaries @ np.diag(w) @ hypercube_points.T

    if np.isclose(denom, 3):
        transformed_points = np.cbrt(all_points)
    else:
        transformed_points = np.power(all_points, 1/denom)

    # TODO: Apply max basis here
    transformed_points = max_basis_mat @ transformed_points
    transformed_points[1, :] = transformed_points[1, :] * 1000

    min_coords = np.min(transformed_points, axis=1)
    max_coords = np.max(transformed_points, axis=1)

    num_samples = 1000000
    random_points = np.random.uniform(
        min_coords[:, np.newaxis],
        max_coords[:, np.newaxis],
        (dimension, num_samples)
    )

    inv_points = np.power(random_points, denom)
    coefficients = cone_vals_inv @ inv_points
    inside_mask = np.all((0 <= coefficients) & (coefficients <= 1), axis=0)
    points_inside = np.sum(inside_mask)

    bounding_box_volume = np.prod(max_coords - min_coords)
    volume = bounding_box_volume * (points_inside / num_samples)
    return volume


def compute_perceptual_volume(color_space: ColorSpace, basis: ColorSpaceType,
                              denom: float, primary_candidates: npt.NDArray, idxs: npt.NDArray,
                              spds: List[Spectra]):
    grid_axes = [np.linspace(0, 1, 50) for _ in range(color_space.dim)]
    hypercube_points = np.array(list(product(*grid_axes)))
    idxs = idxs.reshape(-1, color_space.dim)
    max_basis_3 = MaxBasisFactory.get_object(observer, denom=denom, verbose=False)
    volumes = np.array(Parallel(n_jobs=-1)(delayed(compute_volume)(p, denom,
                       hypercube_points, max_basis_3.cone_to_maxbasis, color_space.dim) for p in tqdm(primary_candidates)))
    # volumes = np.array([compute_volume(p, denom, hypercube_points, color_space.dim) for p in tqdm(primary_candidates)])

    spd_powers = np.array([np.trapz(spd.data) for spd in spds])
    weights = []
    for p in primary_candidates:
        p = p.T
        try:
            w = np.linalg.solve(p, np.ones(color_space.dim))
            weights.append(w)
        except Exception as e:
            # Handle singular matrix case
            weights.append(np.ones(color_space.dim) * -1)
    weights = np.array(weights)
    efficacies = 1.0 / np.array([np.dot(w, spd_powers[idx])
                                 for w, idx in zip(weights, idxs)])  # power needed to reach luminance
    volumes[np.any(np.array(weights) < 0, axis=1)] = 0
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0
    volumes = np.nan_to_num(volumes, nan=0.0)
    efficacies = np.nan_to_num(efficacies, nan=0.0)

    return volumes, efficacies


def compute_max_perceptual_volume(color_space: ColorSpace, basis: ColorSpaceType,
                                  denom: float, primary_candidates: npt.NDArray, idxs: npt.NDArray,
                                  spds: List[Spectra]) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the maximum perceptual volume of the primary candidates

    Args:
        color_space (ColorSpace): _color space object
        chrom_basis (ColorSpaceType): _chromatic basis to perform maximization in
        denom (float): _denominator of the nonlinearity
        primary_candidates (npt.NDArray): _primary candidates
        spds (npt.NDArray): _spds of the primaries

    Returns:
        tuple[npt.NDArray, npt.NDArray]: volumes, efficacies
    """
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    idxs = idxs.reshape(-1, color_space.dim)
    # TODO: implement the rest of this function
    perceptual_points = cs.convert_to_perceptual(
        sets_of_primaries, ColorSpaceType.CONE, basis, denom_of_nonlin=denom).reshape(-1, color_space.dim, color_space.dim)
    perceptual_points = np.nan_to_num(perceptual_points, nan=0.0)

    # compute total power needed to reach luminance
    spd_powers = np.array([np.trapz(spd.data) for spd in spds])
    weights = []
    for p in perceptual_points:
        p = p.T
        try:
            w = np.linalg.solve(p, np.ones(color_space.dim))
            weights.append(w)
        except Exception as e:
            # Handle singular matrix case
            weights.append(np.ones(color_space.dim) * -1)
    weights = np.array(weights)
    efficacies = 1.0 / np.array([np.dot(w, spd_powers[idx])
                                 for w, idx in zip(weights, idxs)])  # power needed to reach luminance

    volumes = np.array([np.linalg.det(p.T@np.diag(w)) for w, p in zip(weights, perceptual_points)])
    volumes[np.any(np.array(weights) < 0, axis=1)] = 0
    efficacies[np.any(np.array(weights) < 0, axis=1)] = 0
    volumes = np.nan_to_num(volumes, nan=0.0)
    efficacies = np.nan_to_num(efficacies, nan=0.0)

    # print("valid number of weights: ", (~np.any(np.array(weights) < 0, axis=1)).sum() / len(weights))

    # # Plot histograms of volumes and efficacies
    # plt.figure(figsize=(12, 6))
    # # Histogram for volumes
    # plt.subplot(1, 2, 1)
    # plt.hist(volumes, bins=30, color='blue', alpha=0.7, label='Volumes')
    # plt.xlabel('Volume')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Volumes')
    # plt.grid(True)
    # plt.legend()

    # # Histogram for efficacies
    # plt.subplot(1, 2, 2)
    # plt.hist(efficacies, bins=30, color='green', alpha=0.7, label='Efficacies')
    # plt.xlabel('Efficacy')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Efficiencies')
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(save_dir, "volumes_efficacies_histogram.png"))
    # plt.close()
    return volumes, efficacies


def compute_max_parallelotope(primary_candidates: npt.NDArray) -> tuple[int, float]:
    """Old method - compute the max parallelotope of the primary candidates

    Args:
        primary_candidates (npt.NDArray): primary candidates

    Returns:
        tuple[int, float]: idx, volume of the optimal candidate
    """
    volumes = np.array([np.linalg.det(p) for p in primary_candidates])
    best_idx = np.argmax(volumes)
    return int(best_idx), volumes[best_idx]


def compute_max_pareto_vol_efficiency(volumes, efficacies, corresponding_peaks, paretoPlot: bool | str = False) -> tuple[int, float, float]:
    """Compute the maximum Pareto volume and efficiency of the primary candidates
    returns the closest point to (1, 1) line

    Args:
        volumes (npt.NDArray): volumes of each primary candidate
        efficacies (npt.NDArray): efficiacies of each primary candidate
        paretoPlot (bool | str, optional): plots the pareto frontier. Defaults to False.

    Returns:
        tuple[int, float, float]: idx, volume, efficiency returned of the optimal candidate
    """

    # Normalize volumes and efficacies to [0, 1]
    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    e_norm = (efficacies - efficacies.min()) / (efficacies.max() - efficacies.min())
    distances = np.sqrt((1 - v_norm) ** 2 + (1 - e_norm) ** 2)
    best_idx = np.argmin(distances)

    pareto_idxs = np.array(get_pareto_front(v_norm, e_norm)).astype(int)

    if paretoPlot:
        plt.figure(figsize=(10, 6))
        plt.scatter(v_norm, e_norm, c=distances, cmap='viridis', label='Candidates')
        if pareto_idxs.size > 0:
            plt.plot(v_norm[pareto_idxs], e_norm[pareto_idxs], color='red', label='Pareto Front')
        plt.colorbar(label='Scores')
        plt.xlabel('Volume')
        plt.ylabel('Efficacy')
        plt.title('Pareto Front: Volume vs Efficacy')
        plt.grid(True)
        plt.legend()
        if isinstance(paretoPlot, str):
            plt.savefig(paretoPlot)
        else:
            plt.show()
        plt.close()

    print(pareto_idxs)
    print(volumes[pareto_idxs])
    print(efficacies[pareto_idxs])
    print([corresponding_peaks[int(x)] for x in pareto_idxs])

    return int(best_idx), volumes[best_idx], efficacies[best_idx]


wavelengths = np.arange(400, 701, 10)
observer_wavelengths = np.arange(380, 781, 10)
observers = [
    # Observer.custom_observer(observer_wavelengths, dimension=3),  # standard LMS observer
    # Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=547),  # Cda29's kid
    # Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=551),  # ben-like observer
    # most likely functional tetrachromatic observer
    Observer.custom_observer(observer_wavelengths, dimension=4, template='govardovskii'),
    # Observer.custom_observer(observer_wavelengths, q_cone_peak=551, dimension=4),  # ben-like tetrachromatic observer
    # Observer.custom_observer(observer_wavelengths, q_cone_peak=555, dimension=4)
]  # ser180ala like observer

# set of primaries - monochromatic, gaussian, or discrete

fwhm = AVG_FWHM
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
peak_wavelengths = np.arange(400, 701, 20)  # Peaks every 10nm from 380nm to 720nm

gaussian_primaries = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                      for peak in peak_wavelengths]
monochromatic_lights = [Spectra(wavelengths=wavelengths, data=np.eye(1, len(wavelengths), np.abs(wavelengths - p).argmin()).flatten())
                        for p in peak_wavelengths]

led_spectrums_path = "../../measurements/2025-04-04/led-spectrums.csv"
led_spectrums_path = "../../data/vpixx/spds.csv"
primary_df = pd.read_csv(led_spectrums_path)
excluded = [5, 6, 7]
our_primaries = primary_df.iloc[:, 1:].to_numpy()
# our_primaries = our_primaries[:, [x for x in range(1, our_primaries.shape[1]) if x not in excluded]]
primary_wavelengths = primary_df["wavelength"].to_numpy()
# Normalize our primaries such that the peak of each spectrum is 1
our_primaries = (our_primaries / np.max(our_primaries, axis=(0, 1))).T
our_primaries = [Spectra(wavelengths=primary_wavelengths, data=spectrum) for spectrum in our_primaries]
our_primary_peaks = [primary_wavelengths[np.argmax(spectrum.data)] for spectrum in our_primaries]


# gaussian_primaries, our_primaries]  # [gaussian_primaries, our_primaries]  # gaussian_primaries,
primary_sets = [our_primaries]
corresponding_peaks = [our_primary_peaks]  # [peak_wavelengths, our_primary_peaks]  #
# set of bases to project into from chromaticity
# bases = [ColorSpaceType.CHROM, ColorSpaceType.HERING_CHROM, ColorSpaceType.CONE, ColorSpaceType.MAXBASIS]
# ColorSpaceType.MAXBASIS243_PERCEPTUAL_243,
# ColorSpaceType.MAXBASIS243_PERCEPTUAL_243,
bases = [ColorSpaceType.MAXBASIS300_PERCEPTUAL_300]

save_dir = "./results/"
os.makedirs(save_dir, exist_ok=True)

for spectras, spectra_name in zip(primary_sets, ['gaussians', "ours"]):
    plt.figure(figsize=(10, 6))
    for spectrum in spectras:
        plt.plot(spectrum.wavelengths, spectrum.data, label=f"Peak {spectrum.wavelengths[np.argmax(spectrum.data)]} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.title("Our Primaries")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{spectra_name}.png"))

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=pd.Index(["Observer", "Basis", "Denom", "Primary Set", "Max Volume",
                          "Efficacy", "Corresponding Max Peaks"]))

# For all possible combinations of observers, primary sets, and bases
for observer in observers:
    cs = ColorSpace(observer, generate_all_max_basis=True)
    for pset_idx, (spds, corresponding_peak_wavelengths) in enumerate(zip(primary_sets, corresponding_peaks)):
        corresponding_primaries = list(combinations(corresponding_peak_wavelengths, observer.dimension))
        for basis in bases:
            isPerceptual = basis in [ColorSpaceType.MAXBASIS243_PERCEPTUAL_243,
                                     ColorSpaceType.MAXBASIS300_PERCEPTUAL_300]
            if isPerceptual:
                denom = float(basis.name.split("_")[-1])/100
            else:
                denom = 1.0

            observed_primaries = np.array(
                [observer.observe(primary) for primary in spds])
            peak_combinations = np.array(list(combinations(corresponding_peak_wavelengths, observer.dimension)))

            primary_candidates = np.array(list(combinations(observed_primaries, observer.dimension)))
            idxs = np.array(list(combinations(range(len(observed_primaries)), observer.dimension)))
            # compute the chromaticity of the primary candidates
            if isPerceptual:
                volumes, efficacies = compute_max_perceptual_volume(
                    cs, basis, denom, primary_candidates, idxs, spds)
            else:
                efficacies = compute_efficiency(cs, primary_candidates, spds)
                volumes = compute_max_chromatic_vol(cs, basis, primary_candidates)

            # # Define file paths for pickled data
            # volumes_file = os.path.join(save_dir, f"volumes_{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.pkl")
            # efficacies_file = os.path.join(
            #     save_dir, f"efficacies_{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.pkl")

            # # Check if pickled data exists
            # if os.path.exists(volumes_file) and os.path.exists(efficacies_file):
            #     with open(volumes_file, 'rb') as vf, open(efficacies_file, 'rb') as ef:
            #         volumes = pickle.load(vf)
            #         efficacies = pickle.load(ef)
            # else:
            #     # Compute volumes and efficacies if not already pickled
            #     volumes, efficacies = compute_perceptual_volume(cs, basis, denom, primary_candidates, idxs, spds)
            #     # Save the computed data to pickle files
            #     with open(volumes_file, 'wb') as vf, open(efficacies_file, 'wb') as ef:
            #         pickle.dump(volumes, vf)
            #         pickle.dump(efficacies, ef)
            # efficacies = compute_efficiency(cs, primary_candidates, spds)

            max_vol_idx, volume = np.argmax(volumes), volumes.max()

            print("Max Volume: ", volume)
            print("Max Vol Corresponding Peaks: ", corresponding_primaries[max_vol_idx])

            idx, volume, efficacy = compute_max_pareto_vol_efficiency(
                volumes, efficacies, corresponding_primaries,
                paretoPlot=f"{save_dir}/{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.png")

            max_primaries = list(primary_candidates)[idx]
            corresponding_max_peaks = corresponding_primaries[idx]
            max_primaries = np.array(max_primaries)

            print("Pareto-Optimal Corresponding Peaks: ", corresponding_max_peaks)
            print("Optimal Volume: ", volume)
            print("Optimal Efficacy: ", efficacy)

            results_df = pd.concat([results_df, pd.DataFrame([{
                "Observer": f"peak_L{observer.sensors[-1].peak}" if observer.dimension == 3 else f"peak_Q_{observer.sensors[-2].peak}",
                "Primary Set": pset_idx,
                "Basis": str(basis),
                "Denom": denom,
                "Max Volume": volume,
                "Efficacy": efficacy,
                "Corresponding Max Peaks": list(corresponding_max_peaks)
            }])], ignore_index=True)


# Save the DataFrame as a CSV file
csv_output_file = os.path.join(save_dir, "all_results.csv")
results_df.to_csv(csv_output_file, index=False)

# Save a pretty table as a JPEG image
jpeg_output_file = os.path.join(save_dir, "all_results.jpeg")
fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
ax.axis('off')  # Turn off the axis
tbl = table(ax, results_df, loc='center', colWidths=[0.2] * len(results_df.columns))  # Show top 20 rows
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)  # Adjust scaling as needed
plt.savefig(jpeg_output_file, bbox_inches='tight', dpi=300)
plt.close(fig)
