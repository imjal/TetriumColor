# %%
from json import load
import math
from colorsys import rgb_to_hsv
from TetriumColor import ColorSpace, ColorSpaceType, PolyscopeDisplayType
import TetriumColor.Visualization as viz
from TetriumColor.Observer import Observer, Cone, Neugebauer, InkGamut, CellNeugebauer, Pigment, Spectra, Illuminant, InkLibrary, load_neugebauer, load_inkset
from TetriumColor.Observer.InkUtils import load_top_inks
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
import csv
import tetrapolyscope as ps
from IPython.display import Image, display, HTML
from typing import Dict, Tuple, List
from TetriumColor.PsychoPhys.IshiharaPlate import generate_ishihara_plate

### Analyze our 100 ink gamut ###
# Load the CSV data
data_path = "../../data/inksets/fp_inks/all_inks.csv"
# all_fp_inks, fp_paper, wavelengths = load_inkset(data_path)
fp_ink_library = InkLibrary.load_ink_library(data_path)
observer = Observer.tetrachromat()

list_vols_subsets = load_top_inks("./ink-combos/top_ink_combinations_ours.csv")

# %%
# = fp_ink_library.convex_hull_search(observer, illuminant=Illuminant.get("D65"), k=4)
top_vol_ink_names = list_vols_subsets[0][1]
inks = [fp_ink_library.spectra_objs[i] for i, k in enumerate(fp_ink_library.names) if k in top_vol_ink_names]
# %%
ink_gamut = InkGamut(inks, fp_ink_library.get_paper())
color_space = ColorSpace(observer, ink_gamut=ink_gamut)
# %%
buckets = ink_gamut.get_buckets(observer, stepsize=0.1)

# %%
spectras: List[Spectra] = [ink_gamut.get_spectra(p) for p in buckets[0][1]]
cone_activations_metamers = observer.observe_spectras(spectras)
# %%
ink_percentages = color_space.convert(cone_activations_metamers,
                                      ColorSpaceType.CONE,
                                      ColorSpaceType.INK_PERCENTAGES)


redone_cone_activations = color_space.convert(ink_percentages,
                                              ColorSpaceType.INK_PERCENTAGES,
                                              ColorSpaceType.CONE)

# %%# Generate Ishihara plates with ink output
images = generate_ishihara_plate(
    cone_activations[:, 0], cone_activations[:, 1], color_space,
    output_space=ColorSpaceType.INK_PERCENTAGES
)
