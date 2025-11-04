import argparse
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--subset', nargs='+', type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.subset, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    # Create Geometry & Register with Polyscope, and define the animation
    viz.RenderOBS("observer", observer, args.display_basis)
    ps.get_surface_mesh("observer").set_transparency(0.5)

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    basis_colors = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    # Render Arrows for Basis
    if args.dimension < 4:
        basis_pts = viz.ConvertPointsToBasis(np.eye(args.dimension), observer, args.display_basis)
        viz.RenderBasisArrows("arrows", basis_pts, basis_colors[args.subset], radius=0.025/3)
        viz.AnimationUtils.AddObject("arrows", "surface_mesh", args.position,
                                     args.velocity, args.rotation_axis, args.rotation_speed)
    else:
        # Render Metameric Lines
        for i in range(4):
            viz.RenderMetamericDirection(
                f"tetra-metameric-direction-{i}", observer, args.display_basis, i, basis_colors[i])
            viz.AnimationUtils.AddObject(f"tetra-metameric-direction-{i}", "curve_network",
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
            viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
