

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd


from itertools import combinations

from scipy.spatial import ConvexHull
from TetriumColor.Observer import *
from TetriumColor import *
from tqdm import tqdm
from pandas.plotting import table

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


def compute_efficiency(color_space: ColorSpace, primary_candidates: npt.NDArray, spds: List[Spectra]):
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


def compute_max_chromatic_vol(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray):
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE,
                              chrom_basis).reshape(-1, color_space.dim, color_space.dim - 1)
    # chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    # chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    # volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim - 1) # this is not working
    volumes = np.array([ConvexHull(p).volume for p in tqdm(chrom_points)])

    volumes[volumes < 0] = 0
    idx = np.argmax(volumes)
    chrom_max = chrom_points[idx]
    max_k_simplex_vol = ConvexHull(chrom_max[:, :2]).volume

    sets_of_primaries = color_space.observer.normalized_sensor_matrix.T
    spectral_locus = color_space.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)

    if color_space.dim == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(spectral_locus[:, 0], spectral_locus[:, 1], spectral_locus[:, 2], c='blue', label='Spectral Locus')
        ax.scatter(chrom_max[:, 0], chrom_max[:, 1], chrom_max[:, 2], c='red', label='Max Chromaticity')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    elif color_space.dim == 3:
        plt.scatter(spectral_locus[:, 0], spectral_locus[:, 1], c='blue', label='spectral locus')
        plt.scatter(chrom_max[:, 0], chrom_max[:, 1], c='red', label='Max Chromaticity')
        plt.show()

    spectral_locus_vol = ConvexHull(spectral_locus).volume
    print("Max Chromaticity Volume: ", max_k_simplex_vol)
    print("Spectral Locus Volume: ", spectral_locus_vol)

    return volumes


def compute_spectral_vol(color_space: ColorSpace, chrom_basis: ColorSpaceType):
    sets_of_primaries = color_space.observer.normalized_sensor_matrix.T
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    hull = ConvexHull(chrom_points)
    return hull.volume


def compute_perceptual_volume(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray):
    """
    Computes the perceptual volume of a set of primary candidates in a given color space.

    Args:
        color_space (ColorSpace): The color space object.
        chrom_basis (ColorSpaceType): The chromaticity basis to project into.
        primary_candidates (npt.NDArray): The primary candidates.

    Returns:
        tuple: A tuple containing the index of the best candidate, the maximum volume, and the corresponding efficacy.
    """
    sets_of_primaries = primary_candidates.reshape(-1, color_space.dim)
    chrom_points = cs.convert(sets_of_primaries, ColorSpaceType.CONE, chrom_basis)
    chrom_points = np.hstack((chrom_points, np.ones((chrom_points.shape[0], 1))))
    chrom_points = chrom_points.reshape(-1, color_space.dim, color_space.dim)

    volumes = np.array([np.linalg.det(p) for p in chrom_points]) / math.factorial(color_space.dim)
    volumes[volumes < 0] = 0
    return volumes


def compute_max_pareto_vol_efficiency(color_space: ColorSpace, chrom_basis: ColorSpaceType, primary_candidates: npt.NDArray,
                                      idxs: npt.NDArray, spds: List[Spectra], paretoPlot: bool | str = False):
    efficacies = compute_efficiency(color_space, primary_candidates, spds)
    volumes = compute_max_chromatic_vol(color_space, chrom_basis, primary_candidates)

    # Normalize volumes and efficacies to [0, 1]
    v_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
    e_norm = (efficacies - efficacies.min()) / (efficacies.max() - efficacies.min())
    distances = np.sqrt((1 - v_norm) ** 2 + (1 - e_norm) ** 2)
    best_idx = np.argmin(distances)

    pareto_idxs = get_pareto_front(v_norm, e_norm)

    if paretoPlot:
        plt.figure(figsize=(10, 6))
        plt.scatter(v_norm, e_norm, c=distances, cmap='viridis', label='Candidates')
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

    return best_idx, volumes[best_idx], efficacies[best_idx], volumes, efficacies


def compute_max_parallelotope(primary_candidates: npt.NDArray):
    volumes = np.array([np.linalg.det(p) for p in primary_candidates])
    best_idx = np.argmax(volumes)
    return best_idx, volumes[best_idx]


wavelengths = np.arange(400, 701, 5)
observer_wavelengths = np.arange(380, 781, 5)
observers = [
    Observer.custom_observer(observer_wavelengths, dimension=3),  # standard LMS observer
    # Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=547),  # Cda29's kid
    # Observer.custom_observer(observer_wavelengths, dimension=3, l_cone_peak=551),  # ben-like observer
    # most likely functional tetrachromatic observer
    # Observer.custom_observer(observer_wavelengths, dimension=4, template='govardovskii'),
    # Observer.custom_observer(observer_wavelengths, q_cone_peak=551, dimension=4),  # ben-like tetrachromatic observer
    # Observer.custom_observer(observer_wavelengths, q_cone_peak=555, dimension=4)
]  # ser180ala like observer

# set of primaries - monochromatic, gaussian, or discrete

fwhm = AVG_FWHM
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
peak_wavelengths = np.arange(400, 701, 5)  # Peaks every 5m from 380nm to 720nm

gaussian_primaries = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                      for peak in peak_wavelengths]
monochromatic_lights = [Spectra(wavelengths=wavelengths, data=np.eye(1, len(wavelengths), np.abs(wavelengths - p).argmin()).flatten())
                        for p in peak_wavelengths]

led_spectrums_path = "../../measurements/2025-04-04/led-spectrums.csv"
primary_df = pd.read_csv(led_spectrums_path)
our_primaries = primary_df.iloc[:, 1:].to_numpy()
primary_wavelengths = primary_df["wavelength"].to_numpy()
# Normalize our primaries such that the peak of each spectrum is 1
our_primaries = (our_primaries / np.max(our_primaries, axis=0)).T
our_primaries = [Spectra(wavelengths=primary_wavelengths, data=spectrum) for spectrum in our_primaries]
our_primary_peaks = [primary_wavelengths[np.argmax(spectrum.data)] for spectrum in our_primaries]

# # Normalize each spectrum in our_primaries such that the maximum point is 1
# for spectrum in our_primaries:
#     spectrum.data = spectrum.data / np.max(spectrum.data)
#     # Save our_primaries as a CSV file
#     print(spectrum.to_hex())
# output_file = "normalized_primaries.csv"
# primary_data = np.column_stack([primary_wavelengths] + [spectrum.data for spectrum in our_primaries])
# primary_df = pd.DataFrame(primary_data, columns=["wavelength"] + [f"primary_{i+1}" for i in range(len(our_primaries))])
# primary_df.to_csv(output_file, index=False)

# exit()

primary_sets = [gaussian_primaries, our_primaries]
corresponding_peaks = [peak_wavelengths, our_primary_peaks]
# set of bases to project into from chromaticity
bases = [ColorSpaceType.CHROM, ColorSpaceType.MACLEOD_CHROM, ColorSpaceType.HERING_CHROM]

save_dir = "./results/"
os.makedirs(save_dir, exist_ok=True)

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=pd.Index(["Observer", "Basis", "Primary Set", "Max Volume",
                                            "Efficacy", "Corresponding Max Peaks"]))


# For all possible combinations of observers, primary sets, and bases
for observer in observers:
    for pset_idx, (spds, corresponding_peak_wavelengths) in enumerate(zip(primary_sets, corresponding_peaks)):
        corresponding_primaries = list(combinations(corresponding_peak_wavelengths, observer.dimension))
        for basis in bases:
            print(f"Observer: {observer}, Basis: {basis}, Primary Set: {pset_idx}")
            cs = ColorSpace(observer)

            observed_primaries = np.array(
                [observer.observe(primary) for primary in spds])
            peak_combinations = np.array(list(combinations(corresponding_peak_wavelengths, observer.dimension)))

            sets_of_observed = np.array(list(combinations(observed_primaries, observer.dimension)))
            idxs = np.array(list(combinations(range(len(observed_primaries)), observer.dimension)))

            idx, volume, efficacy, volumes, efficacies = compute_max_pareto_vol_efficiency(
                cs, basis, sets_of_observed, idxs, np.array(spds))  # , paretoPlot=f"{save_dir}/{str(observer)}_{basis}_primary_set_{pset_idx}.png")

            max_vol_idx, volume = np.argmax(volumes), volumes.max()
            spectral_vol = compute_spectral_vol(cs, basis)

            print("Max Chromatic Volume: ", volume)
            print("Max spectral vol: ", spectral_vol)
            print("k-Gamut vs Ideal: ", volume / spectral_vol)

            print("Max Vol Corresponding Peaks: ", corresponding_primaries[max_vol_idx])
            print("Max Volume: ", volume)

            max_primaries = list(sets_of_observed)[idx]
            corresponding_max_peaks = corresponding_primaries[idx]
            max_primaries = np.array(max_primaries)
            results_df = pd.concat([results_df, pd.DataFrame([{
                "Observer": f"peak_L{observer.sensors[-1].peak}" if observer.dimension == 3 else f"peak_Q_{observer.sensors[-2].peak}",
                "Basis": str(basis),
                "Primary Set": pset_idx,
                "Max Volume": volume,
                "Efficacy": efficacy,
                "Corresponding Max Peaks": list(corresponding_max_peaks)
            }])], ignore_index=True)


# Save the DataFrame as a CSV file
csv_output_file = os.path.join(save_dir, "all_results.csv")
results_df.to_csv(csv_output_file, index=False)

# Save a pretty table as a JPEG image
jpeg_output_file = os.path.join(save_dir, "all_results.jpeg")
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust size as needed
ax.axis('off')  # Turn off the axis
tbl = table(ax, results_df, loc='center', colWidths=[0.2] * len(results_df.columns))  # Show top 20 rows
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)  # Adjust scaling as needed
plt.savefig(jpeg_output_file, bbox_inches='tight', dpi=300)
plt.close(fig)
