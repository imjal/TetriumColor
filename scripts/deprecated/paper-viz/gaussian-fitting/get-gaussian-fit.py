import argparse
import numpy as np
import tetrapolyscope as ps
from scipy.spatial import ConvexHull
from colour.notation import RGB_to_HEX

from TetriumColor.Measurement.MeasurementRoutines import GetGaussianFit
from TetriumColor.Observer import GetCustomObserver, GetsRGBfromWavelength, convert_refs_to_spectras
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
import pandas as pd


def main():
    # Load CSV file
    # need to remeasure primaries again -_- ==> the function only saves the first 4 primaries measured
    csv_path = "../../../measurements/2025-01-16/primaries/all_primaries.csv"
    data_frame = pd.read_csv(csv_path)

    # Extract wavelengths and spectras
    wavelengths = data_frame['# Wavelength'].values
    spectras = data_frame.drop(columns=['# Wavelength']).values
    primaries = convert_refs_to_spectras(np.array(spectras).T, np.array(wavelengths))
    # primaries = LoadPrimaries("../../../measurements/2025-01-16/primaries")
    data = GetGaussianFit(primaries)
    print(data)


if __name__ == "__main__":
    main()
