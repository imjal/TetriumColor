import sys
import os
import numpy as np

from TetriumColor.Measurement import MeasureMetamers


metamer_weights = np.array([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]])
save_directory = ""
primary_directory = ""

MeasureMetamers(metamer_weights, save_directory, primary_directory)