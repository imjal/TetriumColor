"""Fix a single observer, and peturb the measurement of the primaries to be as close to the real leds as possible
"""
import pdb
from importlib.resources import contents
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import List

from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid
from TetriumColor.Observer.DisplayObserverSensitivity import GetCustomObserver
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.Observer import GetPrevalentObservers, GetColorSpaceTransform, Spectra, Observer, GetColorSpaceTransformTosRGB
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticImages, CreatePaddedGrid
from TetriumColor.TetraColorPicker import BackgroundNoiseGenerator, LuminanceNoiseGenerator
import pickle

# # Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
prevalent_observers, peaks = GetPrevalentObservers()
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)


for factor, lum_noise, seed in zip([1.2], [0.035], [72]):
    center_pt = []
    noise_generator = []
    cones = []
    # Tetrachromat Single Testing at 547 (Noise out all the peaks as well)
    avg_observer_overall: Observer = GetCustomObserver(
        wavelengths=wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559)
    avg_obs_cst = GetColorSpaceTransform(avg_observer_overall, gaussian_smooth_primaries, scaling_factor=1000)

    # all_csts: List[ColorSpaceTransform] = [x[0] for x in GetColorSpaceTransforms(
    #     [obs for observer_noise_list in prevalent_observers for obs in observer_noise_list], [gaussian_smooth_primaries], scaling_factor=1000)]

    # # Dump all_csts to a pickle file
    # with open('all_csts.pkl', 'wb') as f:
    #     pickle.dump(all_csts, f)

    # Reload all_csts from the pickle file
    with open('all_csts.pkl', 'rb') as f:
        all_csts = pickle.load(f)

    noise_object = BackgroundNoiseGenerator(all_csts, factor=factor)
    disp_points, _ = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, avg_obs_cst)

    center_pt += [disp_points[2][2]]
    cones += [disp_points[2][2][:, :4]@np.linalg.inv(avg_obs_cst.cone_to_disp).T]
    noise_generator += [noise_object]

    # Tetrachromat Single Testing at 555 vs 559
    # ser180: Observer = GetCustomTetraObserver(
    #     wavelengths=wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=559)
    # ala180: Observer = GetCustomTetraObserver(
    #     wavelengths=wavelengths, od=0.5, m_cone_peak=530, l_cone_peak=555)
    # observers = [ser180, ala180]
    # serala180 = [x[0] for x in GetColorSpaceTransforms(observers, [gaussian_smooth_primaries], scaling_factor=1000)]
    # disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, serala180[0])
    # center_pt += [disp_points[2][2]]
    # noise_generator += [BackgroundNoiseGenerator([serala180[0]])]
    # disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, serala180[1])
    # center_pt += [disp_points[2][2]]

    # Observer Targeting (Get rid of OD, Lens, and Macula Through Noise)
    for observer_noise_list in prevalent_observers:
        # Measure Observer Noise
        # color_space_transforms: List[ColorSpaceTransform] = [cst[0] for cst in GetColorSpaceTransforms(
        #     observer_noise_list, [gaussian_smooth_primaries], scaling_factor=1000)]
        color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
            observer_noise_list[0], gaussian_smooth_primaries)
        # avg_observer = color_space_transforms[0]
        # noise_object = BackgroundNoiseGenerator(color_space_transforms)
        noise_object = LuminanceNoiseGenerator(color_space_transform, lum_noise)
        disp_points, _ = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, color_space_transform)
        center_pt += [disp_points[2][2]]
        cones += [disp_points[2][2][:, :4]@np.linalg.inv(color_space_transform.cone_to_disp).T]
        noise_generator += [noise_object]

    center_pts = np.array(center_pt)
    rgb_image_files = [f"./civo_outputs/sub_images_{lum_noise}/observer_targeted_noise_{x}_RGB.png"
                       for x in ["all_547"] + [f"{peak[0]}_{peak[1]}" for peak in peaks]]
    ocv_image_files = [f"./civo_outputs/sub_images_{lum_noise}/observer_targeted_noise_{x}_OCV.png"
                       for x in ["all_547"] + [f"{peak[0]}_{peak[1]}" for peak in peaks]]

    # print("plate b", repr((center_pts[1] * 255).astype(int)))
    # print("plate c", repr((center_pts[2] * 255).astype(int)))
    # print("plate d", repr((center_pts[3] * 255).astype(int)))
    # print("plate e", repr((center_pts[4] * 255).astype(int)))
    # print("plate f", repr((center_pts[5] * 255).astype(int)))
    # print("plate g", repr((center_pts[6] * 255).astype(int)))
    # print("plate h", repr((center_pts[7] * 255).astype(int)))

    trichromat = GetCustomObserver(wavelengths=np.arange(360, 831, 1), od=0.5, dimension=3)
    cst_for_trichromat = GetColorSpaceTransformTosRGB(trichromat)
    cone_srgbs = np.array(cones)[:, :, [0, 1, 3]]@cst_for_trichromat.cone_to_disp.T
    colors = np.concatenate(
        (cone_srgbs.reshape(-1, 3), np.zeros((cone_srgbs.reshape(-1, 3).shape[0], 3))), axis=1).reshape(-1, 2, 6)

    CreatePseudoIsochromaticImages(center_pts, f"./civo_outputs/", "observer_targeted_noise",
                                   (["all_547"] + [f"{peak[0]}_{peak[1]}" for peak in peaks]),
                                   noise_generator=noise_generator, sub_image_dir=f"sub_images_{lum_noise}", seed=seed)

    img_rgb = CreatePaddedGrid(rgb_image_files[1:], grid_size=(3, 4))
    img_rgb = img_rgb.resize((1365, 1024), Image.Resampling.BOX)
    img_rgb.save(f"./civo_outputs/grid_lum_{factor}_{lum_noise}_RGB.png")

    img_ocv = CreatePaddedGrid(ocv_image_files[1:], grid_size=(3, 4))
    img_ocv = img_ocv.resize((1365, 1024), Image.Resampling.BOX)
    img_ocv.save(f"./civo_outputs/grid_lum_{factor}_{lum_noise}_OCV.png")
