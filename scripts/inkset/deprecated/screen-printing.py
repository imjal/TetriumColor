# %%
import itertools
from json import load
import math
from colorsys import rgb_to_hsv
from TetriumColor import ColorSpace, ColorSpaceType, PolyscopeDisplayType
import TetriumColor.Visualization as viz
from TetriumColor.Observer import Observer, Cone, Neugebauer, InkGamut, CellNeugebauer, Pigment, Spectra, Illuminant, InkLibrary, load_neugebauer
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
import csv
import tetrapolyscope as ps
import interactive_polyscope
from IPython.display import Image, display, HTML
from typing import Dict, Tuple

# %gui polyscope

# # %%
# %load_ext autoreload
# %autoreload 2

# %%
screenshot_count = 0
# ! mkdir - p screenshots


def save_ps_screenshot():
    global screenshot_count
    ps.show()  # renders window
    fname = f"screenshots/screenshot_{screenshot_count}.png"
    ps.screenshot(fname)
    # Display in notebook
    display(Image(filename=fname, width=400))  # need to use this for pdf export
    # display(HTML(f'<img src="screenshot_{screenshot_count}.png" style="width:50%;">'))

    screenshot_count += 1

# %%


def save_top_inks_as_csv(top_volumes, filename):
    import csv

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


# %%


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
        plt.ylim(0, 1.4)
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


def load_inkset(filepath) -> Tuple[Dict[str, Spectra], Spectra, npt.NDArray]:
    """
    Load an inkset from a CSV file.

    Parameters:
    - filepath: str, path to the CSV file containing the inkset data.

    Returns:
    - inks: dict, a dictionary of ink names and their corresponding Spectra objects.
    - paper: Spectra, the paper spectra.
    """
    df = pd.read_csv(filepath)
    spectras = df.iloc[:, 2:].to_numpy()  # Extract reflectance data
    # Extract wavelengths from column headers, keeping only numeric characters
    wavelengths = df.columns[2:].str.replace(r'[^0-9.]', '', regex=True).astype(float).to_numpy()
    # wavelengths = np.arange(400, 701, 10)  # Wavelengths from 400 to 700 nm in steps of 10 nm

    # Create Spectra objects for each ink
    inks = {}
    for i in range(spectras.shape[0]):
        name = df.iloc[i, 1]
        inks[name] = Spectra(data=spectras[i], wavelengths=wavelengths, normalized=False)

    paper = inks["paper"]
    del inks["paper"]  # remove paper from inks dictionary
    return inks, paper, wavelengths

# %% [markdown]
# ### Load Screen Printing Inkset


# %%
 ### Analyze our 100 ink gamut ###
    # Load the CSV data
data_path = "../../data/inksets/screenprinting/screenprinting-inks.csv"
all_inks, paper, wavelengths = load_inkset(data_path)

del_inks = []
for name, ink_spectra in all_inks.items():
    if 'fluorescent' in name.lower():
        del_inks.append(name)

for name in del_inks:
    del all_inks[name]

# %%
paper.plot()

# %%
plt.figure(figsize=(12, 8))

# Plot each ink's spectrum
for ink_name, ink_spectra in all_inks.items():
    plt.plot(wavelengths, ink_spectra.data, label=ink_name, linewidth=1.5, c=ink_spectra.to_rgb())

# Add the paper spectrum as a black dashed line
plt.plot(wavelengths, paper.data, label='paper', color='black', linestyle='--', linewidth=2)

# Add labels and legend
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Reflectance Spectra of Screen Printing Inks', fontsize=14)
plt.xlim(wavelengths[0], wavelengths[-1])
plt.ylim(0, 1.4)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

# %%
plot_inks_by_hue(all_inks, wavelengths)

# %% [markdown]
# ## Observer analysis (tetrachromat)

# %%
# Define observer and illuminant
d65 = Illuminant.get("d65")
tetrachromat = Observer.tetrachromat(illuminant=d65, wavelengths=wavelengths)

# %%
# Initialize the ink library|
ink_library = InkLibrary(all_inks, paper)

# %%
# Perform convex hull search
# top_width_all_inks = ink_library.distance_search(tetrachromat, d65)
# save_top_inks_as_csv(top_width_all_inks, "top_widths_all_screen_printing_inks.csv")


# -- currently not working but hacked a solution together either way using visual cues
# top_width_all_inks = load_top_inks("top_widths_all_screen_printing_inks.csv")


# %%
# Perform convex hull search
top_volumes_k4_all_inks = ink_library.convex_hull_search(tetrachromat, d65, k=4)
save_top_inks_as_csv(top_volumes_k4_all_inks, "./ink-combos/top_volumes_k4_all_screen_printing_inks.csv")


top_volumes_k5_all_inks = ink_library.convex_hull_search(tetrachromat, d65, k=5)
save_top_inks_as_csv(top_volumes_k5_all_inks, "./ink-combos/top_volumes_k5_all_screen_printing_inks.csv")

# %%
top_volumes_k4_all_inks = load_top_inks("./ink-combos/top_volumes_k4_all_screen_printing_inks.csv")
show_top_k_combinations(top_volumes_k4_all_inks, all_inks, k=16)

# %%
top_volumes_k5_all_inks = load_top_inks("./ink-combos/top_volumes_k5_all_screen_printing_inks.csv")
show_top_k_combinations(top_volumes_k5_all_inks, all_inks, k=16)

# %%
chosen_idx = 0
best4 = [all_inks[ink_name] for ink_name in top_volumes_k4_all_inks[chosen_idx][1]]
# best4_width = [all_inks[ink_name] for ink_name in top_width_all_inks[0][1]]

# %%
[all_inks[ink_name].plot() for ink_name in top_volumes_k4_all_inks[chosen_idx][1]]
[ink_name for ink_name in top_volumes_k4_all_inks[chosen_idx][1]]

# %%
[s.plot() for s in best4]
# %%
gamut = InkGamut(best4, paper, d65)
fp_point_cloud, fp_percentages = gamut.get_point_cloud(tetrachromat)

# %%
buckets = gamut.get_buckets(tetrachromat)

# %%
# Create a grid of plots with different ink combinations
concentrations = [0, 0.5, 1.0]  # Concentrations to try for each ink
plt.figure(figsize=(15, 15))

# Select a subset of interesting combinations to visualize
combinations = list(itertools.product([0, 1], repeat=4))  # Binary combinations (0% or 100%)
selected_combinations = combinations[:16]  # Take the first 16 combinations

# Plot each combination
for i, combo in enumerate(selected_combinations):
    plt.subplot(4, 4, i+1)

    # Get the spectrum for this combination

    # Plot the spectrum
    plt.plot(wavelengths, spectrum.data, color=spectrum.to_rgb())

    # Show the combination values
    plt.title(f"Combo: {combo}", fontsize=10)
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3)

    # Only show x-axis labels on bottom row
    if i >= 12:
        plt.xlabel('Wavelength (nm)', fontsize=8)
    else:
        plt.tick_params(axis='x', labelbottom=False)

    # Only show y-axis labels on left column
    if i % 4 == 0:
        plt.ylabel('Reflectance', fontsize=8)

plt.tight_layout()
plt.suptitle('Spectral Reflectance of Different Ink Combinations', fontsize=16, y=1.02)
plt.show()

# Now visualize some specific combinations with intermediate values
plt.figure(figsize=(12, 8))

# Selected interesting combinations with varying intermediate values
interesting_combos = list(itertools.product([0, 1], repeat=4))

# Plot these interesting combinations
for i, combo in enumerate(interesting_combos):
    spectrum = gamut.get_spectra(combo)
    plt.plot(wavelengths, spectrum.data, c=spectrum.to_rgb(), label=f"Combo {i}: {combo}", linewidth=2)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Selected Ink Combinations', fontsize=14)
plt.ylim(0, 1.2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


# %%
def get_spectras_from_metamer(metamer_idx):
    np.set_printoptions(precision=4, suppress=True)
    metamer1 = gamut.get_spectra(buckets[metamer_idx][1][0])

    print(buckets[metamer_idx][1][0])

    metamer2 = gamut.get_spectra(buckets[metamer_idx][1][1])
    cones = tetrachromat.observe_spectras([metamer1, metamer2])
    print("cone percentages", np.array([cone.data for cone in cones]))
    print("difference", np.abs(cones[0] - cones[1]))
    # Format and print the ink percentages for this metamer
    print("ink percentages", np.array(buckets[metamer_idx][1]))

    metamer1.plot()
    metamer2.plot()
    return [metamer1, metamer2]

# %%


def visualize_metamer_grid(gamut, buckets, img_size=50):
    # Create a 50x50 grid image
    grid = np.zeros((img_size, img_size, 3))

    # Calculate the total number of available metamers
    num_metamers = min(img_size * img_size, len(buckets))

    # Fill the grid with RGB values from different metamer pairs
    for i in range(img_size):
        for j in range(img_size):
            idx = i * img_size + j
            if idx < num_metamers:
                # Get the first spectrum from each metamer pair
                spectrum = gamut.get_spectra(buckets[idx][1][0])
                rgb = spectrum.to_rgb()

                # Clip values to be between 0 and 1
                grid[i, j] = np.clip(rgb, 0, 1)

    # Display the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.title(f"Grid of Metamers (First {num_metamers} pairs)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return grid


# %%
visualize_metamer_grid(gamut, buckets, img_size=50)

# %%
get_spectras_from_metamer(0)

# %%
cs = ColorSpace(tetrachromat)

# %%
all_inks_as_points = tetrachromat.observe_spectras(all_inks.values())
all_inks_point_cloud = cs.convert(all_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
all_inks_srgbs = cs.convert(all_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.SRGB)

# %%
ps.init()
ps.set_always_redraw(False)
ps.set_ground_plane_mode('shadow_only')
ps.set_SSAA_factor(2)
ps.set_window_size(720, 720)
factor = 0.1575  # 0.1/5.25
viz.ps.set_background_color((factor, factor, factor, 1))

viz.RenderOBS("observer", cs, PolyscopeDisplayType.HERING_MAXBASIS, num_samples=1000)
viz.ps.get_surface_mesh("observer").set_transparency(0.3)
viz.RenderPointCloud("fp_points", cs.convert(fp_point_cloud, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:])
viz.RenderPointCloud("all_fps", all_inks_point_cloud, all_inks_srgbs)

viz.RenderMetamericDirection("meta_dir", tetrachromat, PolyscopeDisplayType.HERING_MAXBASIS,
                             2, np.array([0, 0, 0]), radius=0.005, scale=1.2)
viz.ps.show()

# %%
save_ps_screenshot()

# %%
save_ps_screenshot()

# %%
viz.ps.unshow()

# %%
