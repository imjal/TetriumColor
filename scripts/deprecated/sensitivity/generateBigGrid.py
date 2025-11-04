"""Fix a single observer, and peturb the measurement of the primaries to be as close to the real leds as possible
"""
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetAllObservers, GetColorSpaceTransforms, Spectra, Observer, GetStockmanObserver
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries, PerturbPrimaries, SaveRGBOtoSixChannel
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticImages, CreatePseudoIsochromaticGrid


# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observers: List[Observer] = GetAllObservers()
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
# SaveRGBOtoSixChannel(gaussian_smooth_primaries, '../../measurements/2024-12-06/gaussian_smooth_primaries.csv')
# perturbed_primaries: List[List[Spectra]] = PerturbPrimaries(primaries)
# all_primaries = [primaries, gaussian_smooth_primaries]  # + perturbed_primaries

# print(len(observers))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i, observer in enumerate(observers):
#     ax.plot(observer.wavelengths, observer.sensor_matrix.T, c='r', label='All Observers')
# ax.plot(observers[0].wavelengths, observers[0].sensor_matrix.T, c='k', label='Stockman')
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# for primary in primaries:
#     primary.plot(ax=ax, color='k')
# for primary in gaussian_smooth_primaries:
#     primary.plot(ax=ax, color='r')

# for perturbed_primary in perturbed_primaries[2:]:
#     for primary in perturbed_primary:
#         primary.plot(ax=ax, color='b')
# plt.show()

# Perturb Observer with Fixed Primaries to see if the issues come from the observer side
color_space_transforms: List[List[ColorSpaceTransform]] = GetColorSpaceTransforms(
    observers, [gaussian_smooth_primaries], scaling_factor=1000)

primary_types = ['Gaussian']  # , 'Perterbed_Up', 'Perterbed_Down', 'Perterbed_Left', 'Perterbed_Right']

# subset = [19, 20, 36, 54, 55, 56, 61, 62, 72, 73, 74, 76, 77, 79, 80, 90, 91, 92, 94, 95, 96, 97, 107]
# subset = [44, 24, 42]

for i, primary_type in tqdm.tqdm(enumerate(primary_types)):
    # center_points = []
    for observer_type in range(len(observers)-1):
        # if observer_type in subset:
        color_space_transform = color_space_transforms[observer_type][i]
        disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, color_space_transform)
        print(disp_points)
        print(np.array(disp_points * 255).astype(np.uint8))

        center_points += [disp_points[2][2]]
        CreatePseudoIsochromaticGrid(disp_points, f"./all_observers/",
                                     f"observer_{observer_type}_{primary_type}_grid")
    # display_grid_points = np.array(center_points).reshape(-1, 2, 6)
    # CreatePseudoIsochromaticImages(
    #     display_grid_points, f"./outputs", f"observer_{primary_type}")
