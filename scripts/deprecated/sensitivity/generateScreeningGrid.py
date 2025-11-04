"""
Generate a screening grid for the general experiment
"""
import numpy as np
import argparse
import os

from PIL import Image
from typing import List
import numpy.typing as npt

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
outputs.append(GetControlTest(
    avg_observer, gaussian_smooth_primaries, 2, luminance, saturation, lum_noise, grid_indices, grid_size, cube_idx))
names.append("control")

outputs.append(GetObserverIdentifyingTest(
    avg_observer, gaussian_smooth_primaries, 0, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
names.append(f"observer_identifying_{530}_{559}_{0}")

outputs.append(GetObserverIdentifyingTest(
    avg_observer, gaussian_smooth_primaries, 1, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
names.append(f"observer_identifying_{530}_{559}_{1}")

# Observer Identifying Test
observer_identifying_observers, peaks = GetPeakPrevalentObservers()
for observer, peak in zip(observer_identifying_observers, peaks):
    for metamer_dir in metamer_dirs:
        # axis over Q cone
        outputs.append(GetObserverIdentifyingTest(
            observer, gaussian_smooth_primaries, metamer_dir, luminance, lum_noise, saturation, grid_indices, grid_size, cube_idx))
        names.append(f"observer_identifying_{peak[0]}_{peak[1]}_{metamer_dir}")

# Generate the Output Grid
dirname = "./screening_outputs/"
sub_image_dir = f"sub_images_{lum_noise}"
root = f"./{dirname}/{sub_image_dir}"
output_base = "screening"
rgb_image_files = [os.path.join(root, f"{output_base}_{base_name}_RGB.png") for base_name in names]
ocv_image_files = [os.path.join(root, f"{output_base}_{base_name}_OCV.png") for base_name in names]

center_pts, noise_generators = zip(*[(output[0], output[1]) for output in outputs])
center_pts = np.array(center_pts)
CreatePseudoIsochromaticImages(center_pts, f"./{dirname}/", output_base, names,
                               noise_generator=noise_generators, sub_image_dir=sub_image_dir, seed=seed)

img_rgb = CreatePaddedGrid(rgb_image_files, grid_size=(3, 4))
img_rgb = img_rgb.resize((1365, 1024), Image.Resampling.BOX)
img_rgb.save(f"./{dirname}/grid_lum_{factor}_{lum_noise}_RGB.png")

img_ocv = CreatePaddedGrid(ocv_image_files, grid_size=(3, 4))
img_ocv = img_ocv.resize((1365, 1024), Image.Resampling.BOX)
img_ocv.save(f"./{dirname}/grid_lum_{factor}_{lum_noise}_OCV.png")
