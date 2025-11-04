"""
Generate a screening grid for the general experiment
"""
import pdb
import numpy as np
import argparse
import tetrapolyscope as ps
import pickle

from PIL import Image
from typing import List
import numpy.typing as npt

from TetriumColor.ColorMath.Conversion import Convert6DArrayToPlateColors, ConvertPlateColorsToDisplayColors
from TetriumColor.Observer.DisplayObserverSensitivity import GetCustomObserver
from TetriumColor.Observer import *
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticImages, CreatePaddedGrid
from TetriumColor.TetraColorPicker import NoiseGenerator
from TetriumColor.TetraPlate import GetControlTest, GetConeIdentifyingTest, GetObserverIdentifyingTest

from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz

# load parameters from the script
parser = argparse.ArgumentParser(description='Generate Screening Grid')
parser.add_argument('--display_basis', type=lambda choice: DisplayBasisType[choice], choices=list(DisplayBasisType))

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
wavelengths = np.arange(380, 781, 10)
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)


outputs: List[tuple[npt.NDArray, NoiseGenerator | None]] = []
names: List[str] = []

observer = GetCustomObserver(
    wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=547, l_cone_peak=559, verbose=True)
observer = ObserverFactory.get_object(observer)

# # Cone Identifying Test -- we want to identify peaks S cone, M cone, and variants of all the L cones
# cone_identifying_observers = []
# cone_names = [419, 530, 559]
# q_peaks = [547, 552, 555]
# for peak in q_peaks:
#     cone_identifying_observers.append(GetCustomTetraObserver(
#         wavelengths=wavelengths, od=0.5, m_cone_peak=530, q_cone_peak=peak, l_cone_peak=559))

# for observer, peak in zip(cone_identifying_observers, q_peaks):
#     # axis over Q cone
#     outputs.append(GetConeIdentifyingTest(
#         observer, gaussian_smooth_primaries, 2, luminance, saturation, grid_indices, factor, grid_size, cube_idx))
#     names.append(f"cone_identifying_{peak}")

# noise_divisions = [1, 2, 4]
# for axis, peak, noise in zip([0, 1, 3], cone_names, noise_divisions):
#     outputs.append(GetConeIdentifyingTest(
#         cone_identifying_observers[0], gaussian_smooth_primaries, axis, luminance, saturation, grid_indices, noise, grid_size, cube_idx))
#     names.append(f"cone_identifying_{peak}_{noise}")

# Save outputs to a pickle file
# with open('outputs.pkl', 'wb') as f:
#     pickle.dump(outputs, f)

# Load outputs from the pickle file
with open('outputs.pkl', 'rb') as f:
    outputs = pickle.load(f)

center_pts, noise_generators = zip(*[(output[0], output[1]) for output in outputs])
center_pts = np.array(center_pts)

ps.init()
print(f"Rendering observer in the display basis {args.display_basis}")
viz.RenderOBS("tetra-custom-observer", observer, args.display_basis)
ps.get_surface_mesh("tetra-custom-observer").set_transparency(0.5)

for i in range(4):
    viz.RenderMetamericDirection(f"tetra-metameric-direction-{i}", observer, args.display_basis, i, np.array([0, 0, 0]))


for i in range(len(noise_generators)):
    cst = GetColorSpaceTransform(observer, gaussian_smooth_primaries, metameric_axis=2)
    # noise_generator_fn = noise_generators[i].GenerateNoiseFunction(center_pts[i])
    # points = np.array([noise_generator_fn() for _ in range(10)])
    # plate_colors = [Convert6DArrayToPlateColors(points[i]) for i in range(len(points))]
    plate_colors = [Convert6DArrayToPlateColors(center_pts[i])]
    disp_colors = ConvertPlateColorsToDisplayColors(plate_colors, cst)
    new_points = disp_colors@np.linalg.inv(cst.cone_to_disp).T

    viz.Render4DPointCloud(f"tetra-noise-{i}", new_points.reshape(-1, 4), observer, args.display_basis)

ps.show()
