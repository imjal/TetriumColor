import argparse
from os import close
import numpy as np
import tetrapolyscope as ps
import matplotlib.pyplot as plt
import csv

import numpy.typing as npt
from typing import List

from TetriumColor.Observer import Observer, ObserverFactory
import TetriumColor.Visualization as viz
from TetriumColor import ColorSpace, ColorSpaceType
from TetriumColor.Utils.ParserOptions import *


def save_ref_data(name, spectra):
    with open(f'{name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Wavelength', 'Reflectance'])
        for i, value in enumerate(spectra.data):
            writer.writerow([int(spectra.wavelengths[i]), value])


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=10, help='Step size for wavelengths')
    parser.add_argument('--ideal', action='store_true', default=False, help='Use ideal primaries')
    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(400, 721, 5)

    # observer = Observer.custom_observer(observer_wavelengths, args.od, 3, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
    #                                     args.l_cone_peak, args.macula, args.lens, args.template)
    observer = Observer.trichromat(observer_wavelengths)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    # viz.RenderOBSinCS("observer", observer, ColorSpaceType.CONE)

    viz.RenderOBSNew("observer", observer, args.display_basis)
    ps.get_surface_mesh("observer").set_transparency(0.5)

    T = viz.GetBasisConvert(observer, args.display_basis)
    basis = np.eye(3)@T.T

    spectra_color = np.array([0.05553725,  0.37131291,  0.39575001])
    spectra_srgb = [0.74985593, 0.66329803, 0.02981869]
    spectra_color_hering_arrow = spectra_color@T.T * 3

    observer = ObserverFactory.get_object(observer)
    points, srgbs = observer.get_optimal_colors()
    # Find the closest point to spectra_color * 3
    scaled_spectra_color = spectra_color * 2.3
    distances = np.linalg.norm(points - scaled_spectra_color, axis=1)
    closest_point_index = np.argmin(distances)
    closest_point = points[closest_point_index]
    closest_color = srgbs[closest_point_index]
    print(f"Closest point: {closest_point}, Closest color: {closest_color}")

    refs = observer.get_optimal_reflectances()
    closest_ref = refs[closest_point_index]
    save_ref_data("closest_ref", closest_ref)
    plt.plot(closest_ref.wavelengths, closest_ref.data, label="Closest Ref")
    plt.show()

    viz.RenderBasisArrows("arrows", basis, np.zeros((3, 3)), radius=0.01)

    viz.RenderMaxBasis("max_basis", observer, args.display_basis)

    # viz.RenderSetOfArrows("arrow_short", [([0, 0, 0], spectra_color@T.T.tolist())], radius=0.007)
    # viz.RenderSetOfArrows("arrow_long", [([0, 0, 0], spectra_color_hering_arrow.tolist())], radius=0.007)

    # viz.RenderPointCloud("spectra", np.array(
    #     [spectra_color@T.T]), np.array([spectra_srgb]), radius=0.015)

    # viz.RenderPointCloud("boundary", np.array(
    #     [closest_point@T.T]), np.array([closest_color]), radius=0.015)

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)
    viz.AnimationUtils.AddObject("arrows", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    viz.AnimationUtils.AddObject("max_basis", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    ps.set_automatically_compute_scene_extents(False)

    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps

        def callback():

            pass
            # viz.AnimationUtils.UpdateObjects(delta_time)

        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
