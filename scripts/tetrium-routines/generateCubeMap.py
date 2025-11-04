import numpy as np
import argparse

from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer import Observer
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Utils.ParserOptions import AddObserverArgs
# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()


# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)
primaries = load_primaries_from_csv("../../measurements/2025-05-06/")

for metameric_axis in range(2, 3):
    cs_4d = ColorSpace(observer,
                       display_primaries=primaries, metameric_axis=metameric_axis)
    color_sampler = ColorSampler(cs_4d, cubemap_size=64)

    im = color_sampler.generate_concatenated_cubemap(1.4, 0.4, ColorSpaceType.SRGB)
    im.show()
