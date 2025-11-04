import argparse
from re import I
import numpy as np

from TetriumColor.Observer import Observer
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    parser.add_argument('--output_filename', type=str, default='observer_cone_fundamentals.csv')
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(400, 700, 1)

    observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)

    mat = np.concatenate([wavelengths[:, np.newaxis].T, observer.sensor_matrix]).T
    if args.dimension == 4:
        header = 'wavelength, S, M, Q, L'
    elif args.dimension == 3:
        header = 'wavelength, S, M, L'
    else:
        header = 'wavelength, S, M'
    np.savetxt(args.output_filename, mat, delimiter=',', header=header)


if __name__ == '__main__':
    main()
