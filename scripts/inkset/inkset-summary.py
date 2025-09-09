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


# %load_ext autoreload
# %autoreload 2
# %gui polyscope


# screenshot_count = 0
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
        inks[name] = Spectra(data=spectras[i], wavelengths=wavelengths)

    paper = inks["paper"]
    del inks["paper"]  # remove paper from inks dictionary
    return inks, paper, wavelengths


# Configuration - Change this to switch inksets
INKSET_CONFIG = {
    "name": "fp_inks",  # Change this: "fp_inks", "ansari", "screenprinting", "pantone"
    "data_path": "../../data/inksets/fp_inks/all_inks.csv"
}

# Then use it in your code
data_path = INKSET_CONFIG["data_path"]
all_inks, paper, wavelengths = load_inkset(data_path)

fp_paper.plot()

plot_inks_by_hue(all_fp_inks, wavelengths)

data_path = "../../data/inksets/ansari/ansari-inks.csv"
all_ansari_inks, ansari_paper, wavelengths = load_inkset(data_path)


ansari_paper.plot()
plot_inks_by_hue(all_ansari_inks, wavelengths)

# Define observer and illuminant
d65 = Illuminant.get("d65")
tetrachromat = Observer.tetrachromat(illuminant=d65, wavelengths=wavelengths)

# Initialize the ink library|
fp_library = InkLibrary(all_fp_inks, fp_paper)

# Perform convex hull search
top_volumes_all_fp_inks = fp_library.convex_hull_search(tetrachromat, d65)
save_top_inks_as_csv(top_volumes_all_fp_inks, "top_volumes_all_fp_inks.csv")

top_volumes_all_fp_inks = load_top_inks("top_volumes_all_fp_inks.csv")

# top_volumes_6_all_fp_inks = fp_library.cached_pca_search(tetrachromat, d65, k=5)
# save_top_inks_as_csv(top_volumes_6_all_fp_inks, "./ink-combos/top_volumes_6_all_fp_inks.csv")
# top_volumes_6_all_fp_inks = load_top_inks("./ink-combos/top_volumes_6_all_fp_inks.csv")

#  # Initialize the ink library
ansari_library = InkLibrary(all_ansari_inks, ansari_paper)

top_k4_ansari_inks = ansari_library.convex_hull_search(tetrachromat, d65, k=4)
save_top_inks_as_csv(top_k4_ansari_inks, "./ink-combos/top_volumes_k4_all_ansari_inks.csv")
top_k4_ansari_inks = load_top_inks("./ink-combos/top_volumes_k4_all_ansari_inks.csv")

top_k6_ansari_inks = ansari_library.convex_hull_search(tetrachromat, d65, k=6)
save_top_inks_as_csv(top_k6_ansari_inks, "./ink-combos/top_volumes_k6_all_ansari_inks.csv")
top_k6_ansari_inks = load_top_inks("./ink-combos/top_volumes_k6_all_ansari_inks.csv")

show_top_k_combinations(top_k4_ansari_inks, all_ansari_inks, k=16)
show_top_k_combinations(top_k6_ansari_inks, all_ansari_inks, k=16)

best4_fp = [all_fp_inks[ink_name] for ink_name in top_volumes_all_fp_inks[0][1]]
best4_ansari = [all_ansari_inks[ink_name] for ink_name in top_k4_ansari_inks[0][1]]
best6_ansari = [all_ansari_inks[ink_name] for ink_name in top_k6_ansari_inks[0][1]]


fp_gamut = InkGamut(best4_fp, fp_paper, d65)
fp_point_cloud, fp_percentages = fp_gamut.get_point_cloud(tetrachromat)

ansari_gamut_k4 = InkGamut(best4_ansari, ansari_paper, d65)
ansari_point_cloud_k4, ansari_percentages_k4 = ansari_gamut_k4.get_point_cloud(tetrachromat)

ansari_gamut_k6 = InkGamut(best6_ansari, ansari_paper, d65)
ansari_point_cloud_k6, ansari_percentages_k6 = ansari_gamut_k6.get_point_cloud(tetrachromat)


exit()

cs = ColorSpace(tetrachromat)


all_ansari_inks_as_points = tetrachromat.observe_spectras(all_ansari_inks.values())
all_ansari_inks_point_cloud = cs.convert(all_ansari_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
all_ansari_inks_srgbs = cs.convert(all_ansari_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.SRGB)


all_fp_inks_as_points = tetrachromat.observe_spectras(all_fp_inks.values())
all_fp_inks_point_cloud = cs.convert(all_fp_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
all_fp_inks_srgbs = cs.convert(all_fp_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.SRGB)


ps.init()
ps.set_always_redraw(False)
ps.set_ground_plane_mode('shadow_only')
ps.set_SSAA_factor(2)
ps.set_window_size(720, 720)
factor = 0.1575  # 0.1/5.25
viz.ps.set_background_color((factor, factor, factor, 1))

viz.RenderOBS("observer", cs, PolyscopeDisplayType.HERING_MAXBASIS, num_samples=1000)
viz.ps.get_surface_mesh("observer").set_transparency(0.3)
viz.RenderPointCloud("ansari_points", cs.convert(ansari_point_cloud_k4,
                     ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:], mode="sphere")
# viz.RenderMeshFromNonConvexPointCloud("ansari_gamut_k4", cs.convert(ansari_point_cloud_k4, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:])
viz.RenderPointCloud("all_ansari_inks", all_ansari_inks_point_cloud, all_ansari_inks_srgbs)
viz.RenderPointCloud("all_fp_inks", all_fp_inks_point_cloud, all_fp_inks_srgbs)
viz.RenderMetamericDirection("meta_dir", tetrachromat, PolyscopeDisplayType.HERING_MAXBASIS, 2,
                             np.array([0, 0, 0]), radius=0.005, scale=1.2)
viz.ps.show()


ps.init()
ps.set_always_redraw(False)
ps.set_ground_plane_mode('shadow_only')
ps.set_SSAA_factor(2)
ps.set_window_size(720, 720)
factor = 0.1575  # 0.1/5.25
viz.ps.set_background_color((factor, factor, factor, 1))

viz.RenderOBS("observer", cs, PolyscopeDisplayType.HERING_MAXBASIS, num_samples=1000)
viz.RenderPointCloud("fp_points", cs.convert(fp_point_cloud, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:])
viz.RenderPointCloud("all_fps", all_inks_point_cloud, all_inks_srgbs)
viz.RenderMetamericDirection("meta_dir", tetrachromat, PolyscopeDisplayType.HERING_MAXBASIS, 2,
                             np.array([0, 0, 0]), radius=0.005, scale=1.2)
viz.ps.show()


save_ps_screenshot()  # all possible inks + the best printer gamut (which would become completely stable)


save_ps_screenshot()


save_ps_screenshot()
