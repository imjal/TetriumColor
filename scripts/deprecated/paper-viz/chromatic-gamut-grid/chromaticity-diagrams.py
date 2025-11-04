import argparse
import numpy as np
import tetrapolyscope as ps
from scipy.spatial import ConvexHull
from colour.notation import RGB_to_HEX

from TetriumColor.Observer import Observer, convert_refs_to_spectras
from TetriumColor.Observer.ColorSpaceTransform import GetsRGBfromWavelength
from TetriumColor import ColorSpace, ColorSpaceType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=5, help='Step size for wavelengths')
    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the ideal chromatic display')
    parser.add_argument('--viz_efficient_wavelengths', nargs='+', type=float, default=[445, 535, 590, 635],
                        help='Wavelengths for the visually efficient display')
    parser.add_argument('--chrom_color_space', type=lambda choice: ColorSpaceType[choice], choices=[
                        ColorSpaceType.CHROM, ColorSpaceType.HERING_CHROM], help='Chromaticity projection type')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(360, 831, 1)
    observer = Observer.custom_observer(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)
    cs = ColorSpace(observer)

    spectral_locus_colors = np.array([GetsRGBfromWavelength(wl) for wl in observer_wavelengths])
    chromaticity_points = cs.convert(observer.normalized_sensor_matrix.T,
                                     from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.CHROM)
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)
    ps.set_transparency_mode('pretty')
    ps.set_transparency_render_passes(4)

    # get rid of zero points as they are not visible
    idxs = ~np.all(chromaticity_points == 0, axis=1)
    chromaticity_points = chromaticity_points[idxs]
    spectral_locus_colors = spectral_locus_colors[idxs]

    cone_basis = np.eye(observer.dimension)
    chromaticity_cone_points = cs.convert(
        cone_basis, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.CHROM)
    simplex_coords, points = GetSimplexBarycentricCoords(
        args.dimension, chromaticity_cone_points, chromaticity_points)

    rgb_colors = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    names = viz.RenderSimplexElements("primary-gamut", args.dimension, simplex_coords, rgb_colors, isColored=False)

    # Render the chromaticity diagram geometry
    if args.dimension == 3:
        ideal_volume = viz.Render2DMesh("spectral-gamut", points, np.array([0.1, 0.1, 0.1]))
    else:
        ideal_volume = viz.Render3DMesh("spectral-gamut", points,
                                        rgbs=np.tile(np.array([0.1, 0.1, 0.1]), (points.shape[0], 1)))

    ps.get_surface_mesh("spectral-gamut").set_transparency(0.4)
    ps.get_surface_mesh("spectral-gamut").set_material("clay")
    names.append(("spectral-gamut", "surface_mesh"))

    # Render the spectral locus
    viz.Render3DLine("spectral_locus", points, spectral_locus_colors)
    ps.set_automatically_compute_scene_extents(False)

    # Output Video to Screen or Save to File (based on options)
    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps

        # def callback():
        #     viz.AnimationUtils.UpdateObjects(delta_time)
        # ps.set_user_callback(callback)
        ps.show()
        # ps.clear_user_callback()


if __name__ == "__main__":
    main()
