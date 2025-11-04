

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import combinations, product

from pandas._libs.tslibs import normalize_i8_timestamps
from TetriumColor.ColorSpace import OKLAB_M1, IPT_M1
from TetriumColor.Observer import *
from TetriumColor import *
import math
from tqdm import tqdm
import pandas as pd
from pandas.plotting import table
from joblib import Parallel, delayed
import pickle
from colour import XYZ_to_Lab, Lab_to_XYZ

AVG_FWHM = 22.4


# Constants
epsilon = 216/24389  # ≈ 0.008856
kappa = 24389/27     # ≈ 903.3
delta = 6/29

# D65 white point
Xn, Yn, Zn = 0.95047, 1.00000, 1.08883


def f(t):
    delta3 = delta ** 3
    t = np.asarray(t)
    return np.where(t > delta3, np.cbrt(t), (t * (kappa / 116)) + (16 / 116))


def xyz_to_lab(XYZ):
    # Normalize
    X = XYZ[..., 0] / Xn
    Y = XYZ[..., 1] / Yn
    Z = XYZ[..., 2] / Zn

    fx = f(X)
    fy = f(Y)
    fz = f(Z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def f_inv(t):
    t = np.asarray(t)
    return np.where(t > delta, t**3, (116 * t - 16) / kappa)


def lab_to_xyz(Lab):
    L = Lab[..., 0]
    a = Lab[..., 1]
    b = Lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    X = f_inv(fx) * Xn
    Y = f_inv(fy) * Yn
    Z = f_inv(fz) * Zn

    return np.stack([X, Y, Z], axis=-1)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_primaries(wavelengths, nm_sampling=10):
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fwhm = AVG_FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    peak_wavelengths = np.arange(400, 701, nm_sampling)  # Peaks every 10nm from 380nm to 720nm

    gaussian_primaries = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                          for peak in peak_wavelengths]

    led_spectrums_path = "../../measurements/2025-04-04/led-spectrums.csv"
    primary_df = pd.read_csv(led_spectrums_path)
    excluded = [5, 6, 7]
    our_primaries = primary_df.iloc[:, 1:].to_numpy()
    our_primaries = our_primaries[:, [x for x in range(1, our_primaries.shape[1]) if x not in excluded]]
    primary_wavelengths = primary_df["wavelength"].to_numpy()
    # Normalize our primaries such that the peak of each spectrum is 1
    our_primaries = (our_primaries / np.max(our_primaries, axis=(0, 1))).T
    our_primaries = [Spectra(wavelengths=primary_wavelengths, data=spectrum) for spectrum in our_primaries]
    our_primary_peaks = [primary_wavelengths[np.argmax(spectrum.data)] for spectrum in our_primaries]

    primary_sets = [gaussian_primaries, our_primaries]
    corresponding_peaks = [peak_wavelengths, our_primary_peaks]
    return primary_sets, corresponding_peaks


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


def compute_volume(
    primaries: npt.NDArray, white_point: npt.NDArray,
    hypercube_points: npt.NDArray,
    dimension: int
) -> float:
    # Solve for white point mixing weights
    primaries = primaries.T  # shape: (3, N)

    try:
        w = np.linalg.solve(primaries, white_point)  # shape: (N,)
        if np.any(w < 0):
            return 0
        # import pdb
        # pdb.set_trace()
    except np.linalg.LinAlgError:
        return 0  # Can't solve system => degenerate
    try:
        cone_vals_inv = np.linalg.inv(primaries)  # shape: (N, 3)
    except np.linalg.LinAlgError:
        return 0  # Singular matrix

    # Step 1: Map all hypercube points to XYZ
    all_XYZ = (primaries @ np.diag(w) @ hypercube_points.T).T  # shape: (3, num_points)

    # Step 2: Convert all XYZ points to LAB
    # all_LAB = np.array([XYZ_to_Lab(xyz) for xyz in all_XYZ.T])
    all_LAB = np.power(all_XYZ @ IPT_M1.T, 0.43)
    # all_LAB = np.cbrt(all_XYZ @ OKLAB_M1.T)
    # all_LAB = f(all_XYZ)
    # all_LAB = xyz_to_lab(all_XYZ)  # shape: (num_points, 3
    # all_LAB = XYZ_to_Lab(all_XYZ)  # shape: (num_points, 3)

    # Plot all points in all_LAB
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_LAB[:, 0], all_LAB[:, 1], all_LAB[:, 2], c='b', marker='o', s=10, alpha=0.5)
    # ax.set_xlabel('L*')
    # ax.set_ylabel('a*')
    # ax.set_zlabel('b*')
    # plt.title('All Points in CIELAB Space')
    # plt.show()

    # Step 3: Bounding box in LAB space
    min_coords = np.min(all_LAB, axis=0)
    max_coords = np.max(all_LAB, axis=0)

    # Step 4: Monte Carlo sampling in LAB space
    num_samples = 1000000
    random_LAB = np.random.uniform(min_coords, max_coords, size=(num_samples, 3))

    # Step 5: Convert LAB back to XYZ
    # random_XYZ = Lab_to_XYZ(random_LAB.T)  # shape: (3, num_samples)
    # random_XYZ = lab_to_xyz(random_LAB)  # shape: (num_samples, 3)
    # random_XYZ = f_inv(random_LAB)  # shape: (num_samples, 3)
    # random_XYZ = np.power(random_LAB, 3) @ np.linalg.inv(OKLAB_M1).T
    random_XYZ = np.power(random_LAB, 1/0.43) @ np.linalg.inv(IPT_M1).T

    # Step 6: Convert XYZ to primary intensities
    coefficients = random_XYZ  @ cone_vals_inv.T  # shape: (N, num_samples)

    # Step 7: Check if points are inside the unit hypercube
    inside_mask = np.all((0 <= coefficients) & (coefficients <= 1), axis=1)
    fraction_inside = np.sum(inside_mask) / num_samples

    # Step 8: Volume of LAB bounding box * fraction inside
    lab_volume = np.prod(max_coords - min_coords)
    estimated_volume = lab_volume * fraction_inside
    return estimated_volume


def compute_CIELAB_volume(color_space: ColorSpace,
                          primary_candidates: npt.NDArray, idxs: npt.NDArray,
                          spds: List[Spectra]):
    grid_axes = [np.linspace(0, 1, 50) for _ in range(color_space.dim)]
    hypercube_points = np.array(list(product(*grid_axes)))
    idxs = idxs.reshape(-1, color_space.dim)
    d65 = Illuminant.get("D65").to_xyz_d65()
    # volumes = np.array(Parallel(n_jobs=-1)(delayed(compute_volume)(p, d65,
    #                                                                hypercube_points, color_space.dim) for p in tqdm(primary_candidates)))
    volumes = np.array([compute_volume(p, d65, hypercube_points, color_space.dim) for p in tqdm(primary_candidates)])

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


if __name__ == "__main__":
    wavelengths = np.arange(400, 701, 10)
    observer_wavelengths = np.arange(380, 781, 5)
    observer = Observer.custom_observer(observer_wavelengths, dimension=3)  # standard LMS observer

    primary_sets, corresponding_peaks = get_primaries(wavelengths, nm_sampling=10)

    save_dir = "./results/"
    os.makedirs(save_dir, exist_ok=True)

    results_df = pd.DataFrame(columns=pd.Index(["Observer", "Basis", "Denom", "Primary Set", "Max Volume",
                                                "Efficacy", "Corresponding Max Peaks"]))

    # For all possible combinations of observers, primary sets, and bases
    cs = ColorSpace(observer, generate_all_max_basis=True)
    for pset_idx, (spds, corresponding_peak_wavelengths) in enumerate(zip(primary_sets, corresponding_peaks)):
        corresponding_primaries = list(combinations(corresponding_peak_wavelengths, observer.dimension))

        xyz_primaries = [p.to_xyz_d65() for p in spds]

        xyz_val = Illuminant.get("D65").to_xyz_d65()
        peak_combinations = np.array(list(combinations(corresponding_peak_wavelengths, observer.dimension)))
        primary_candidates = np.array(list(combinations(xyz_primaries, observer.dimension)))
        idxs = np.array(list(combinations(range(len(xyz_primaries)), observer.dimension)))

        # volumes_file = os.path.join(
        #     save_dir, f"volumes_{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.pkl")
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

        volumes, efficacies = compute_CIELAB_volume(cs, primary_candidates, idxs, spds)

        max_vol_idx, volume = np.argmax(volumes), volumes.max()

        print("Max Vol Corresponding Peaks: ", corresponding_primaries[max_vol_idx])
        print("Max Volume: ", volume)

        max_primaries = list(primary_candidates)[max_vol_idx]
        corresponding_max_peaks = corresponding_primaries[max_vol_idx]
        max_primaries = np.array(max_primaries)

        results_df = pd.concat([results_df, pd.DataFrame([{
            "Observer": f"peak_L{observer.sensors[-1].peak}" if observer.dimension == 3 else f"peak_Q_{observer.sensors[-2].peak}",
            "Primary Set": pset_idx,
            "Max Volume": volume,
            "Corresponding Max Peaks": list(corresponding_max_peaks)
        }])], ignore_index=True)
