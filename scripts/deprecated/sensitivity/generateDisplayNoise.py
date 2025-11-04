"""Fix a single observer, and peturb the measurement of the primaries to be as close to the real leds as possible
"""
import pdb
import enum
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetAllObservers, GetColorSpaceTransforms, Spectra, Observer, GetStockmanObserver
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries, PerturbSinglePrimary, PerturbWavelengthPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticGrid


# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observers: List[Observer] = [GetStockmanObserver(wavelengths)]
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
# perturbed_primary_orange: Spectra = PerturbWavelengthPrimaries(
# gaussian_smooth_primaries[3], wavelength_pertubation_range=[5])[0]  # 5 and 6 look good

# perturbed_wv_primaries: List[Spectra] = PerturbWavelengthPrimaries(
#     gaussian_smooth_primaries[0], wavelength_pertubation_range=[-2, -1, 0, 1, 2])  # 5 and 6 look good

orange_right_shifted_5 = [gaussian_smooth_primaries[0], gaussian_smooth_primaries[1],
                          gaussian_smooth_primaries[2], gaussian_smooth_primaries[3]]
all_perturbed_primaries: List[List[List[Spectra]]] = [PerturbSinglePrimary(
    i, gaussian_smooth_primaries, wavelength_pertubation=2) for i in [0, 1, 2, 3]]
# green_left_shifted_5 = [[gaussian_smooth_primaries[0], p, gaussian_smooth_primaries[2],
#                          perturbed_primary_orange] for p in perturbed_primaries]
for i, perturbed_primaries in enumerate(all_perturbed_primaries):
    all_primaries = [primaries, gaussian_smooth_primaries] + perturbed_primaries

    fig, ax = plt.subplots()
    for primary_set in all_primaries:
        for primary in primary_set:
            ax.plot(primary.wavelengths, primary.data, label='Primary')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('All Primaries')
    plt.show()

    # Perturb Observer with Fixed Primaries to see if the issues come from the observer side
    color_space_transforms: List[List[ColorSpaceTransform]] = GetColorSpaceTransforms(
        observers, all_primaries, scaling_factor=1000)

    # with open('./outputs/all_display_pertubations_color_space_transforms.pkl', 'wb') as f:
    #     pickle.dump(color_space_transforms, f)

    # # Load the color space transforms from the pickle file
    # with open('./outputs/all_display_pertubations_color_space_transforms.pkl', 'rb') as f:
    #     color_space_transforms = pickle.load(f)

    primary_types = ['Measured', 'Gaussian', 'Perturbed_Up', 'Perturbed_Down', 'Perturbed_Right', 'Perturbed_Left']
    # Display the center points of the metamer grids for each of the observers -- see how close we are

    center_points = []
    for j, primary_type in tqdm.tqdm(enumerate(primary_types)):
        color_space_transform = color_space_transforms[0][j]
        disp_points = GetMaximalMetamerPointsOnGrid(1, 0.3, 4, 5, color_space_transform)
        center_points += [disp_points[2][2]]

    closest_square = int(np.ceil(np.sqrt(len(center_points))))
    display_grid_points = np.array(center_points).reshape(3, 2, 2, 6)
    CreatePseudoIsochromaticGrid(
        display_grid_points, f"./outputs", f"{i}_perturbed_grid")
