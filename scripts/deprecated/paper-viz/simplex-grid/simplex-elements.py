import argparse
import numpy as np
import tetrapolyscope as ps
import numpy.typing as npt
from itertools import combinations

from TetriumColor.Observer import GetCustomObserver, Observer, ObserverFactory, GetsRGBfromWavelength
from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords, GetSimplexOrigin


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)

    parser.add_argument('--face_idx', type=int, default=0)

    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')

    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 780, 1)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    spectral_locus_colors = np.array([GetsRGBfromWavelength(wl) for wl in wavelengths])

    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    projection_idxs = list(range(1, observer.dimension))
    # Create Geometry & Register with Polyscope, and define the animation

    chromaticity_points = viz.ConvertPointsToChromaticity(
        observer.normalized_sensor_matrix.T, observer, projection_idxs)
    # get rid of zero points as they are not visible
    idxs = ~np.all(chromaticity_points == 0, axis=1)
    chromaticity_points = chromaticity_points[idxs]
    spectral_locus_colors = spectral_locus_colors[idxs]

    primary_indices = [np.argmin(np.abs(wavelengths - wl)) for wl in args.primary_wavelengths]
    primary_points = chromaticity_points[primary_indices]

    primaries_sRGB = np.array([GetsRGBfromWavelength(wl) for wl in args.primary_wavelengths])

    simplex_coords, points = GetSimplexBarycentricCoords(
        args.dimension, primary_points, chromaticity_points)

    if args.dimension < 4:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    simplex_names = viz.RenderSimplexElements(
        "simplex", args.dimension, simplex_coords, primaries_sRGB, isColored=False)

    viz.Render3DLine("spectral_locus", points, spectral_locus_colors)
    # Need to call this after registering structures
    ps.set_automatically_compute_scene_extents(False)

    # No animation needed for this script
    # simplex_names.append(("spectral_locus", "curve_network"))
    # for name, element_name in simplex_names:
    #     viz.AnimationUtils.AddObject(name, element_name, args.position, args.velocity,
    #                                  args.rotation_axis, args.rotation_speed)

    if args.dimension == 4:
        # ps.set_up_dir('z_up')
        # ps.set_front_dir('y_front')
        simplex_origin = GetSimplexOrigin(1, args.dimension)

        # Calculate the normal vector for each face in the simplex
        faces = list(combinations(range(len(simplex_coords)), 3))
        face = faces[args.face_idx]
        face_points = simplex_coords[list(face)]
        normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
        normal = normal / np.linalg.norm(normal)
        face_center = np.mean(face_points, axis=0)
        if np.dot(normal, face_center - simplex_origin) < 0:
            normal = -normal
        root = (simplex_origin + 3 * normal)
        root[1] = face_center[1]
        if args.face_idx == 1:
            ps.set_up_dir('z_up')
            ps.set_front_dir('neg_y_front')
            # ps.look_at(root, face_center - root)
        else:
            ps.look_at(root, face_center - root)

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
