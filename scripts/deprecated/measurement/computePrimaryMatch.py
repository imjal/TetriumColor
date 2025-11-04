from typing import assert_type
import numpy as np

import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse

from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
from TetriumColor.Observer import Observer
from TetriumColor.Measurement import load_primaries_from_csv, compare_dataset_to_primaries, get_spectras_from_rgbo_list, plot_measured_vs_predicted
from TetriumColor.Utils.ParserOptions import AddObserverArgs
# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)

plt.plot(observer.wavelengths, observer.sensor_matrix.T)
plt.show()
primaries = load_primaries_from_csv("../../measurements/2025-05-09/primaries")
colors = np.array([[38, 78, 45, 255], [152, 116, 42, 103]]).astype(np.uint8)

print("{" + ",\n ".join("{" + ", ".join(map(str, inner_list)) + "}" for inner_list in colors) + "}")

measurements_dir = "../../measurements/2025-05-09/try1"
measured_spectras = get_spectras_from_rgbo_list(measurements_dir, colors.tolist())
results = compare_dataset_to_primaries(measurements_dir, colors.tolist(), primaries)

cs_4d = ColorSpace(observer, cst_display_type='led',
                   display_primaries=primaries, metameric_axis=2,
                   luminance_per_channel=[np.sqrt(1/3)] * 4, chromas_per_channel=[np.sqrt(2/3)] * 4)

cones = observer.observe_spectras(measured_spectras)
np.set_printoptions(precision=8, suppress=True)
print("Cone Values: ", cones)
