import argparse
from re import I
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import GetCustomObserver, ObserverFactory, Illuminant
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.LPMetamerMismatch import GetMetamerMismatchBodyAtColor


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Metamer Mismatch from Anomalous Trichromats in Standard Observer')
    parser.add_argument('--step_size', type=float, default=5)
    parser.add_argument('--display_basis', type=lambda choice: DisplayBasisType[choice], choices=list(DisplayBasisType))
    parser.add_argument('--mismatch', type=int, default=-1)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 780, args.step_size)

    # illuminant1 = Illuminant.get("E").interpolate_values(wavelengths)
    illuminant2 = Illuminant.get("D65").interpolate_values(wavelengths)  # np.ones(len(wavelengths))

    observer1 = GetCustomObserver(wavelengths=wavelengths, dimension=3, illuminant=illuminant2)
    observer2 = GetCustomObserver(wavelengths=wavelengths, dimension=3,
                                  s_cone_peak=547, m_cone_peak=551, l_cone_peak=555, illuminant=illuminant2)

    tiled_colors = [np.array([0.5, 0.5, 0.5]), np.array([0.25, 0.25, 0.25]), np.array([0.75, 0.75, 0.75])]
    converted_to_rgb = tiled_colors

    mismatch_vertices = []

    for color in tiled_colors:

        vertices, spectras = GetMetamerMismatchBodyAtColor(observer1.normalized_sensor_matrix,
                                                           observer2.normalized_sensor_matrix,
                                                           illuminant2.data, illuminant2.data, color)
        mismatch_vertices.append(vertices)

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    # Create Geometry & Register with Polyscope, and define the animation
    # viz.RenderOBS("observer1", observer1, args.display_basis)
    # ps.get_surface_mesh("observer1").set_transparency(0.5)

    if args.mismatch == -1:
        viz.RenderOBS("observer2", observer2, args.display_basis)
        ps.get_surface_mesh("observer2").set_transparency(0.5)

        viz.AnimationUtils.AddObject("observer2", "surface_mesh",
                                     args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    for i, vertices in enumerate(mismatch_vertices):
        if args.mismatch == i or args.mismatch == -1:
            new_points = viz.ConvertPointsToBasis(vertices, observer2, args.display_basis)
            viz.Render3DMesh("mismatch-{}".format(i), new_points, np.tile(converted_to_rgb[i], (vertices.shape[0], 1)))
            viz.AnimationUtils.AddObject(f"mismatch-{i}", "surface_mesh",
                                         args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # if args.mismatch == -1:
    #     basis_pts = viz.ConvertPointsToBasis(np.eye(3), observer2, args.display_basis)
    #     basis_pts = basis_pts / np.linalg.norm(basis_pts, axis=1)
    #     viz.RenderBasisArrows("arrows", basis_pts, np.eye(3), radius=0.025/3)
    #     viz.AnimationUtils.AddObject("arrows", "surface_mesh",
    #                                  args.position, args.velocity, args.rotation_axis, args.rotation_speed)

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
