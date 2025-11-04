import sys
import os
import numpy as np
from datetime import datetime
import argparse

from TetriumColor.Measurement import MeasureMetamers


parser = argparse.ArgumentParser(description='Measure Metamers')
parser.add_argument('--directory_name', type=str, default='tmp', help='Directory to save measurements')
args = parser.parse_args()

metamer_weights = np.array([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]])
primary_directory = "../../measurements/2024-01-21/primaries"

current_date = datetime.now().strftime("%Y-%m-%d")
directory = f"../../measurements/{current_date}/{args.directory_name}"

MeasureMetamers(metamer_weights, directory, primary_directory)
