import csv
import math
from colorsys import rgb_to_hsv
import matplotlib.pyplot as plt


def save_top_inks_as_csv(top_volumes, filename):
    # Save top_volumes_all_fp_inks to a CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Volume", "Ink Combination"])  # Header
        for volume, inks in top_volumes:
            writer.writerow([volume, ", ".join(inks)])  # Write volume and ink combination


def load_top_inks(filename):
    top_volumes = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            volume = float(row[0])
            inks = row[1].split(", ")
            top_volumes.append((volume, inks))
    return top_volumes


def plot_inks_by_hue(ink_dataset, wavelengths):
    """
    Plots the inks in the dataset sorted by hue.

    Parameters:
    - ink_dataset: dict, a dictionary of ink names and their corresponding Spectra objects.
    - wavelengths: numpy.ndarray, array of wavelengths corresponding to the spectra data.
    """
    # Convert RGB to HSV and sort by hue
    def get_hue(spectra):
        r, g, b = spectra.to_rgb()
        h, _, _ = rgb_to_hsv(r, g, b)
        return h

    # Sort inks by hue
    sorted_inks = sorted(ink_dataset.items(), key=lambda item: get_hue(item[1]))

    # Plot sorted inks row by row by hue
    num_inks = len(sorted_inks)
    cols = math.ceil(math.sqrt(num_inks))
    rows = math.ceil(num_inks / cols)

    plt.figure(figsize=(15, 15))

    for idx, (name, spectra) in enumerate(sorted_inks):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(wavelengths, spectra.data, c=spectra.to_rgb())
        plt.title(name[:10], fontsize=8)  # Show only the first 10 characters of the name
        plt.xlabel("Wavelength (nm)", fontsize=6)
        plt.ylabel("Reflectance", fontsize=6)
        plt.grid(True)
        plt.xlim(wavelengths[0], wavelengths[-1])
        plt.ylim(0, 1)
        plt.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    plt.show()


def show_top_k_combinations(top_volumes, inkset,  k=10):
    """
    Displays the top k ink combinations with their volumes.

    Parameters:
    - top_volumes: list of tuples (volume, [ink names])
    - k: number of top combinations to display
    """
    # Plot the spectra of the top inks for the first k entries
    plt.figure(figsize=(10, 10))
    wavelengths = inkset.get_paper().wavelengths  # Get wavelengths from the paper spectra
    for idx, (volume, ink_names) in enumerate(top_volumes[:k]):
        plt.subplot(math.ceil(k / 4), 4, idx + 1)  # Create a subplot for each entry
        for ink_name in ink_names:  # Plot the spectra of the first 4 inks
            spectra = inkset[ink_name]
            # Show only the first 10 characters of the name
            plt.plot(wavelengths, spectra.data, label=ink_name[:10], c=spectra.to_rgb())
        plt.title(f"Volume: {volume:.2e}", fontsize=10)
        plt.xlabel("Wavelength (nm)", fontsize=8)
        plt.ylabel("Reflectance", fontsize=8)
        plt.grid(True)
        plt.xlim(wavelengths[0], wavelengths[-1])
        plt.ylim(0, 1)
        plt.legend(fontsize=6)
        plt.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    plt.show()
