from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import AddAnimationArgs, AddObserverArgs, AddVideoOutputArgs
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid, ExportPlates
from TetriumColor.Measurement import load_primaries_from_csv, compare_dataset_to_primaries, get_spectras_from_rgbo_list, export_metamer_difference, export_predicted_vs_measured_with_square_coords, get_spectras_from_rgbo_list, plot_measured_vs_predicted
from TetriumColor.Observer import Observer, Spectra
import pdb
from PIL import Image
import numpy as np
import numpy.typing as npt
import argparse

from typing import List

import tetrapolyscope as ps
import matplotlib.pyplot as plt


def renormalize_spectra(spectras: List[Spectra], observer, primaries: List[Spectra], scaling_factor: float = 10000) -> npt.NDArray:

    disp = observer.observe_spectras(primaries)  # each row is a cone_vec
    intensities = disp.T * scaling_factor  # each column is a cone_vec
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    return white_weights


# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
AddAnimationArgs(parser)
AddVideoOutputArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()


# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)
primaries: List[Spectra] = load_primaries_from_csv("../../measurements/2025-05-21/primaries")
# primaries = [primaries[i] for i in [2, 1, 0, 3]]  # ORDER primaries as RGBO

metameric_axis = 2

# colors = ['red', 'green', 'blue', 'black']
# for i,  primary in enumerate(primaries):
#     plt.plot(primary.wavelengths, primary.data, color=colors[i])

# plt.show()

cs_4d = ColorSpace(observer, cst_display_type='led',
                   display_primaries=primaries, metameric_axis=metameric_axis)

color_sampler = ColorSampler(cs_4d, cubemap_size=5)

# images = color_sampler.generate_cubemap(1.0, 0.4, ColorSpaceType.SRGB)
# for img in images[4:]:
#     img.save("./results/cubemap_" + str(images.index(img)) + ".png")
# image = color_sampler._concatenate_cubemap(images)
# # Apply gamma encoding to the image
# gamma = 2.2
# gamma_corrected_image = np.clip((np.array(image) / 255.0) ** (1 / gamma) * 255, 0, 255).astype(np.uint8)
# image = Image.fromarray(gamma_corrected_image)
# image.save("./results/cubemap.png")


# images = color_sampler.get_metameric_grid_plates(1.0, 0.4, 4, lum_noise=0.02)
# ExportPlates(images, "./results/avg-background")

lum_noise = 0.01
s_cone_noise = 0.04
images = color_sampler.get_metameric_grid_plates(
    1.0, 0.4, 4, background=np.zeros(4), lum_noise=lum_noise, s_cone_noise=s_cone_noise, isSRGB=True)
ExportPlates(images, f"./results/black-background_lum_noise_{lum_noise:.2f}_scone_noise_{s_cone_noise:.3f}")


# img = CreatePaddedGrid([i[0] for i in images], padding=0, canvas_size=(1280 * 8, 720 * 8))
# img.save("all.png")


# idxs = [(2, 3), (2, 4), (3, 3), (3, 4)]
# idxs = [j * 5 + i for i, j in idxs]
# for i, j in idxs:
#     images[i * 5 + j][0].save("./results/cubemap_" + str(i * 5 + j) + ".png")

# colors, cones = color_sampler.get_metameric_pairs(1.0, 0.4, 4)
# print(cones)
# colors = np.array([item for sublist in colors for item in sublist])
# # idxs = [6, 7, 9, 23, 40, 34, 35, 36, 40]
# # colors = colors[idxs]
# colors_8bit = (np.array(colors) * 255).astype(np.uint8)
# print(repr(colors_8bit))
# print("{" + ",\n ".join("{" + ", ".join(map(str, inner_list)) + "}" for inner_list in colors_8bit) + "}")

# measurements_dir = "../../measurements/2025-05-21/5x5-cubemap/"
# measured_spectras = get_spectras_from_rgbo_list(measurements_dir, colors_8bit.tolist())
# # results = compare_dataset_to_primaries(measurements_dir, colors_8bit.tolist(), primaries)
# export_predicted_vs_measured_with_square_coords(
#     measurements_dir, colors_8bit.tolist(), primaries, "./results/")
# # export_metamer_difference(observer, cs_4d, measurements_dir, colors_8bit.tolist(), primaries, "./results/")
