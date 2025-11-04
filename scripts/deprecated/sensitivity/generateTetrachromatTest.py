import numpy as np

from typing import List

from TetriumColor.Observer import GetAllObservers, GetColorSpaceTransforms, Spectra, Observer
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor.TetraColorPicker import BackgroundNoiseGenerator
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform
from TetriumColor.PsychoPhys.HueSphere import CreatePseudoIsochromaticGrid
from TetriumColor.ColorMath.GamutMath import GetMaximalMetamerPointsOnGrid

# Load Observer and Measured Primaries
wavelengths = np.arange(380, 781, 1)
observers: List[Observer] = GetAllObservers()
primaries: List[Spectra] = LoadPrimaries("../../measurements/2024-12-06/primaries")
gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)
color_space_transforms: List[ColorSpaceTransform] = [x[0] for x in GetColorSpaceTransforms(
    observers, [gaussian_smooth_primaries], scaling_factor=1000)]

# Measure Observer Noise
color_space_transform = color_space_transforms[0]
disp_points = GetMaximalMetamerPointsOnGrid(0.7, 0.3, 4, 5, color_space_transform)
noise_object = BackgroundNoiseGenerator(color_space_transforms)
CreatePseudoIsochromaticGrid(disp_points, f"./outputs/", f"test_noise_grid", noise_generator=noise_object)
