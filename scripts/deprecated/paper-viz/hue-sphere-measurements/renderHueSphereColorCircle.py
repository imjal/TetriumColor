import argparse
import numpy as np
import tetrapolyscope as ps

import numpy.typing as npt
from typing import List

from TetriumColor.Observer import Observer, Spectra
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor import ColorSpace, PolyscopeDisplayType, ColorSpaceType
import TetriumColor.ColorMath.Geometry as Geometry


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=5, help='Step size for wavelengths')
    parser.add_argument('--transparency_ball', type=float, default=0.5, help='Transparency of the ball')
    parser.add_argument('--ball', action='store_true', help='Render Ball')
    parser.add_argument('--lattice', action='store_true', help='Render lattice')
    parser.add_argument('--metameric_dir', action='store_true', help='Render Metameric Direction')
    parser.add_argument('--which_dir', type=str, default='q', choices=['q', 'saq', 'none'])
    parser.add_argument('--sampling', type=int, default=100000, help='Render Sampling')
    parser.add_argument('--shadow', action='store_true', help='Render shadow on ground plane')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(360, 831, 1)

    # observer = Observer.custom_observer(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
    #                                     args.l_cone_peak, args.macula, args.lens, args.template)
    observer = Observer.old_tetrachromat(observer_wavelengths)
    cs = ColorSpace(observer)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(False)
    if args.shadow:
        ps.set_ground_plane_mode('shadow_only')
    else:
        ps.set_ground_plane_mode('none')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)
    factor = 0.1825
    viz.ps.set_background_color((factor, factor, factor, 1))

    # ps.set_up_dir('z_up')  # Set up direction to Z-axis
    # ps.set_front_dir('y_front')  # Set front direction to Y-axis

    if args.which_dir == 'q':
        metameric_axis = cs.get_metameric_axis_in(ColorSpaceType.HERING)
        rotation_mat = np.eye(4)
        rotation_mat[:3, :3] = np.linalg.inv(Geometry.RotateToZAxis(metameric_axis[1:]))
        rotation_mat = rotation_mat.T
    elif args.which_dir == 'saq':
        saq = np.array([[1, 0, 1, 0]])
        saq_in_hering = cs.convert(saq, ColorSpaceType.MAXBASIS, ColorSpaceType.HERING)[0, 1:]
        rotation_mat = np.eye(4)
        rotation_mat[:3, :3] = np.linalg.inv(Geometry.RotateToZAxis(saq_in_hering))
        rotation_mat = rotation_mat.T
        ps.set_ground_plane_height_factor(0.2, False)
    else:
        rotation_mat = np.eye(4)

    viz.RenderOBS("observer", cs, args.display_basis, num_samples=args.sampling)
    ps.get_surface_mesh("observer").set_transparency(args.transparency_ball)
    ps.get_surface_mesh("observer").set_transform(rotation_mat)
    ps.get_surface_mesh("observer").set_enabled(args.ball)
    # ps.get_surface_mesh("observer").set_material("flat")

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed, rotation_mat)

    viz.RenderMetamericDirection("meta_dir", observer, args.display_basis, 2,
                                 np.array([0, 0, 0]), radius=0.005, scale=1.2)
    ps.get_curve_network("meta_dir").set_transform(rotation_mat)
    ps.get_curve_network("meta_dir").set_enabled(args.metameric_dir)
    viz.AnimationUtils.AddObject("meta_dir", "curve_network",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed, rotation_mat)

    viz.RenderMaxBasis("max-basis", cs, args.display_basis)
    ps.get_surface_mesh("max-basis").set_transform(rotation_mat)
    ps.get_surface_mesh("max-basis").set_enabled(args.lattice)
    viz.AnimationUtils.AddObject("max-basis", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed, rotation_mat)

    ps.set_automatically_compute_scene_extents(False)
    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=60., aspect=1.)

    params = ps.CameraParameters(intrinsics, ps.get_view_camera_parameters().get_extrinsics())
    ps.set_view_camera_parameters(params)

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
