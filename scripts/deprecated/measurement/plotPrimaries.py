from typing import assert_type
import numpy as np

import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse

from TetriumColor import ColorSpace
from TetriumColor.Observer import Observer
from TetriumColor.Measurement import load_primaries_from_csv, save_primaries_into_csv
from TetriumColor.Utils.ParserOptions import AddObserverArgs
# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
args = parser.parse_args()

# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)

primaries = load_primaries_from_csv("../../measurements/2025-05-09/primaries")
save_primaries_into_csv("../../measurements/2025-05-09/primaries", "../../measurements/2025-05-09/primaries-all.csv")
