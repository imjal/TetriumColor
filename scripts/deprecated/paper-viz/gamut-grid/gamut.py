import argparse
from re import I
import numpy as np
from pandas._config import display
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory, GetParalleletopeBasis, convert_refs_to_spectras
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries


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
    observer_wavelengths = np.arange(380, 781, 5)

    observer = GetCustomObserver(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)
    wavelengths = np.arange(380, 780, 5)
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    if args.ideal:
        primary_spectra = np.zeros((len(args.primary_wavelengths), len(wavelengths)))
        for i, primary_wavelength in enumerate(args.primary_wavelengths):
            index = np.where(wavelengths == primary_wavelength)[0]
            if index.size > 0:
                primary_spectra[i, index[0]] = 1
        primary_spectra = convert_refs_to_spectras(primary_spectra, wavelengths)

    else:
        primaries = LoadPrimaries("../../../measurements/2024-12-06/primaries")
        primary_spectra = GaussianSmoothPrimaries(primaries)
        if args.dimension == 3:
            primary_spectra = [x for i, x in enumerate(primary_spectra) if i != 3]  # just use RGB

    basis_vectors = GetParalleletopeBasis(observer, primary_spectra)

    viz.RenderDisplayGamut("gamut", basis_vectors, viz.GetBasisConvert(observer, args.display_basis))
    ps.get_surface_mesh("gamut").set_transparency(0.5)

    viz.AnimationUtils.AddObject("gamut", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # Create Geometry & Register with Polyscope, and define the animation

    observer_wavelengths = np.arange(380, 781, args.step_size)
    display_observer = GetCustomObserver(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                         args.l_cone_peak, args.macula, args.lens, args.template)
    observer = ObserverFactory.get_object(display_observer)
    viz.RenderOBS("observer", display_observer, args.display_basis)
    ps.get_surface_mesh("observer").set_transparency(0.7)

    # # viz.AnimationUtils.AddObject("observer", "surface_mesh",
    # #                              args.position, args.veslocity, args.rotation_axis, args.rotation_speed)

    # basis_colors = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    # # Render Arrows for Basis
    # if args.dimension < 4:
    #     basis_pts = viz.ConvertPointsToBasis(np.eye(args.dimension), observer, args.display_basis)
    #     viz.RenderBasisArrows("arrows", basis_pts, basis_colors[args.subset], radius=0.025/3)
    #     viz.AnimationUtils.AddObject("arrows", "surface_mesh", args.position,
    #                                  args.velocity, args.rotation_axis, args.rotation_speed)
    # else:
    #     # Render Metameric Lines
    #     for i in range(4):
    #         viz.RenderMetamericDirection(
    #             f"tetra-metameric-direction-{i}", observer, args.display_basis, i, basis_colors[i])
    #         viz.AnimationUtils.AddObject(f"tetra-metameric-direction-{i}", "curve_network",
    #                                      args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # Need to call this after registering structures
    ps.set_automatically_compute_scene_extents(False)
    # Output Video to Screen or Save to File (based on options)
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
