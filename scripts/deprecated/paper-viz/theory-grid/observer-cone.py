import argparse
from re import I
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory, GetsRGBfromWavelength
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)
    spectral_locus_colors = np.array([GetsRGBfromWavelength(wl) for wl in wavelengths])
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 550)

    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=30., aspect=1.)
    offset_up = 0.1 if args.dimension == 3 or args.dimension == 2 else 0
    extrinsics = ps.CameraExtrinsics(root=(0, offset_up, -1), look_dir=(0, 0, 1), up_dir=(0., 1., 0.))
    params = ps.CameraParameters(intrinsics, extrinsics)
    ps.set_view_camera_parameters(params)

    # Create Geometry & Register with Polyscope, and define the animation

    points = viz.ConvertPointsToBasis(observer.normalized_sensor_matrix.T, observer, args.display_basis)
    basis_points = viz.ConvertPointsToBasis(np.eye(args.dimension), observer, DisplayBasisType.ConeHering)

    viz.Render3DCone("observer-cone", points, line_colors=spectral_locus_colors,
                     mesh_color=np.array([0.25, 0, 1]) * 0.5, mesh_alpha=0.4, arrow_alpha=1)
    viz.RenderBasisArrows("basis", basis_points * 0.3, radius=0.025/10)

    viz.AnimationUtils.AddObject("observer-cone", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)
    viz.AnimationUtils.AddObject("observer-cone_arrows", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)
    viz.AnimationUtils.AddObject("observer-cone_curve", "curve_network",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    viz.AnimationUtils.AddObject("basis", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

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
            if args.dimension > 2:
                viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
