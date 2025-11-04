import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import combinations, product
from TetriumColor.Observer import *
from TetriumColor import *
from TetriumColor.Utils.ParserOptions import *
import pickle

AVG_FWHM = 22.4


def get_primaries(wavelengths):
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fwhm = AVG_FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    peak_wavelengths = np.arange(400, 701, 5)  # Peaks every 10nm from 380nm to 720nm

    gaussian_primaries = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                          for peak in peak_wavelengths]

    led_spectrums_path = "../../../measurements/2025-04-04/led-spectrums.csv"
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


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)

    parser.add_argument("--save_dir", default='', type=str)

    args = parser.parse_args()
    save_dir = args.save_dir

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 10)
    observer = Observer.custom_observer(observer_wavelengths, dimension=args.dimension, template='govardovskii')

    basis = ColorSpaceType.MAXBASIS300_PERCEPTUAL_300
    denom = 3.0
    wavelengths = np.arange(400, 701, 5)
    primary_sets, corresponding_peaks = get_primaries(wavelengths)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot with different transparency levels

    for pset_idx in range(2):
        # Define file paths for pickled data
        corresponding_peaks_list = np.array(list(combinations(corresponding_peaks[pset_idx], observer.dimension)))
        volumes_file = os.path.join(save_dir, f"volumes_{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.pkl")
        efficacies_file = os.path.join(
            save_dir, f"efficacies_{str(observer)}_{basis}_{denom}_primary_set_{pset_idx}.pkl")
        print(volumes_file)
        print(efficacies_file)

        # Check if pickled data exists
        if os.path.exists(volumes_file) and os.path.exists(efficacies_file):
            with open(volumes_file, 'rb') as vf, open(efficacies_file, 'rb') as ef:
                volumes = pickle.load(vf)
                efficacies = pickle.load(ef)
            alpha = 0.5 if pset_idx == 0 else 1.0
            plt.scatter(volumes, efficacies, alpha=alpha, label=f"Primary Set {pset_idx}")

            idx = np.argmax(volumes)
            max_vol = volumes[idx]
            print(corresponding_peaks_list[idx])
            print(max_vol)

        # # Find the index of specific peaks in the corresponding peaks list
        # target_peaks = np.array([450, 530, 590, 630])
        # target_idx = np.where((corresponding_peaks_list == target_peaks).all(axis=1))[0]
        # print(target_idx)
        # import pdb
        # pdb.set_trace()
        # print(f"Volume of chosen {volumes[target_idx]} with efficacies {efficacies[target_idx]}")
     # Add labels, legend, and title
    plt.xlabel("Volumes")
    plt.ylabel("Efficacies")
    plt.title("Scatter Plot of Volumes vs Efficacies")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


main()
