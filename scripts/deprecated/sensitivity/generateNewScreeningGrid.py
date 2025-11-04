"""
Generate a screening grid for the general experiment
"""
import numpy as np
import argparse
import os

from PIL import Image
from typing import List
import numpy.typing as npt

from TetriumColor.Observer.DisplayObserverSensitivity import GetCustomObserver
from TetriumColor.Observer import *
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticImages, CreatePaddedGrid
from TetriumColor.TetraColorPicker import NoiseGenerator
from TetriumColor.TetraPlate import GetControlTest, GetConeIdentifyingTest, GetObserverIdentifyingTest

# load parameters from the script
parser = argparse.ArgumentParser(description='Generate Screening Grid')
parser.add_argument('--factor', type=float, required=False, default=1, help='Scaling factor for noise generation')
parser.add_argument('--lum_noise', type=float, required=False, default=0.035, help='Luminance noise level')
parser.add_argument('--seed', type=int, required=False, default=12, help='Random seed for reproducibility')
parser.add_argument('--grid_indices', type=int, nargs=2, required=True,
                    help='Indices that we will take from the grid to choose the color')
parser.add_argument('--grid_size', type=int, required=False, default=5, help='Grid size as two integers')
parser.add_argument('--luminance', type=float, required=False, default=0.7, help='Luminance level')
parser.add_argument('--saturation', type=float, required=False, default=0.3, help='Saturation level')
parser.add_argument('--individual_grid', action='store_true', help='Generate individual grid')
args = parser.parse_args()

factor = args.factor
lum_noise = args.lum_noise
seed = args.seed
grid_indices = args.grid_indices
grid_size = args.grid_size
luminance = args.luminance
saturation = args.saturation
cube_idx = 4

# load primaries
wavelengths = np.arange(380, 781, 1)
# primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
primaries: List[Spectra] = LoadPrimaries("../../measurements/2025-01-16/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
metamer_dirs = [2, 3]

outputs: List[tuple[npt.NDArray, NoiseGenerator | None]] = []
names: List[str] = []

# Control Test
avg_observer = GetCustomObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559)
for met_axis in range(4):
    outputs.append(GetControlTest(
        avg_observer, gaussian_smooth_primaries, met_axis, luminance, saturation, lum_noise, grid_indices, grid_size, cube_idx))
    names.append(f"control_{met_axis}")

# test by hyperobserver, and each dimddension
# normal hyperobserver -> q = 547
normal_hyperobserver = GetCustomObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=547, l_cone_peak=559)
for met_axis in range(4):
    outputs.append(GetObserverIdentifyingTest(
        normal_hyperobserver, gaussian_smooth_primaries, met_axis, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
    names.append(f"standard_{530}_{547}_{559}_{met_axis}")

# q = 555 - common serala180
normal_hyperobserver = GetCustomObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=555, l_cone_peak=559)
for met_axis in range(4):
    outputs.append(GetObserverIdentifyingTest(
        normal_hyperobserver, gaussian_smooth_primaries, met_axis, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
    names.append(f"standard_{530}_{555}_{559}_{met_axis}")

# # q = 551 - ben genotype
normal_hyperobserver = GetCustomObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=551, l_cone_peak=559)
for met_axis in range(4):
    outputs.append(GetObserverIdentifyingTest(
        normal_hyperobserver, gaussian_smooth_primaries, met_axis, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
    names.append(f"standard_{530}_{551}_{559}_{met_axis}")

# Generate the Output Grid
dirname = "./screening_outputs/"
sub_image_dir = f"sub_images_{lum_noise}"
root = f"./{dirname}/{sub_image_dir}"
output_base = "screening"
rgb_image_files = [os.path.join(root, f"{output_base}_{base_name}_RGB.png") for base_name in names]
ocv_image_files = [os.path.join(root, f"{output_base}_{base_name}_OCV.png") for base_name in names]

center_pts, cone_diffs, noise_generators = zip(*[(output[0], output[1], output[2]) for output in outputs])
center_pts = np.array(center_pts)
CreatePseudoIsochromaticImages(center_pts, f"./{dirname}/", output_base, names,
                               noise_generator=noise_generators, sub_image_dir=sub_image_dir, seed=seed)
print(center_pts)
print(cone_diffs)

if args.individual_grid:
    for i in range(4):
        print(4 * i, 4 * i + 4)
        img_rgb = CreatePaddedGrid(rgb_image_files[4 * i: 4 * i + 4], grid_size=(2, 2))
        img_rgb = img_rgb.resize((800, 800), Image.Resampling.BOX)
        img_rgb.save(
            f"./{dirname}/{args.grid_indices[0]}_{args.grid_indices[1]}_{i}_grid_lum_{luminance}_{lum_noise}_RGB.png")

        img_ocv = CreatePaddedGrid(ocv_image_files[4 * i: 4 * i + 4], grid_size=(2, 2))
        img_ocv = img_ocv.resize((800, 800), Image.Resampling.BOX)
        img_ocv.save(
            f"./{dirname}/{args.grid_indices[0]}_{args.grid_indices[1]}_{i}_grid_lum_{luminance}_{lum_noise}_OCV.png")
else:
    img_rgb = CreatePaddedGrid(rgb_image_files, grid_size=(4, 4))
    img_rgb = img_rgb.resize((800, 800), Image.Resampling.BOX)
    img_rgb.save(
        f"./{dirname}/full_{args.grid_indices[0]}_{args.grid_indices[1]}_lum_{luminance}_{lum_noise}_RGB.png")

    img_ocv = CreatePaddedGrid(ocv_image_files, grid_size=(4, 4))
    img_ocv = img_ocv.resize((800, 800), Image.Resampling.BOX)
    img_ocv.save(
        f"./{dirname}/full_{args.grid_indices[0]}_{args.grid_indices[1]}_lum_{luminance}_{lum_noise}_OCV.png")
