import argparse
from re import I
import numpy as np
import tetrapolyscope as ps
import numpy.typing as npt

from TetriumColor.Observer import Observer, ObserverFactory, getsRGBfromWavelength, Spectra
# from TetriumColor.Utils.CustomTypes import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords, GetSimplexOrigin
from TetriumColor import ColorSpace, ColorSpaceType


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)

    parser.add_argument('--max_simplex_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')

    parser.add_argument('--led_gamut_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')

    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 780, 1)

    observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)
    spectral_locus_colors = np.array([getsRGBfromWavelength(wl) for wl in wavelengths])
    cs = ColorSpace(observer)
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

    projection_idxs = list(range(1, observer.dimension))
    # Create Geometry & Register with Polyscope, and define the animation

    # chromaticity_points = viz.ConvertPointsToChromaticity(
    #     observer.normalized_sensor_matrix.T, observer, projection_idxs)
    # # get rid of zero points as they are not visible
    # idxs = ~np.all(chromaticity_points == 0, axis=1)
    # chromaticity_points = chromaticity_points[idxs]
    chromaticity_points = cs.convert(observer.normalized_sensor_matrix.T,
                                     ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    idxs = ~np.all(chromaticity_points == 0, axis=1)
    chromaticity_points = chromaticity_points[idxs]
    spectral_locus_colors = spectral_locus_colors[idxs]

    # viz.ConvertPointsToChromaticity(np.eye(args.dimension), observer, projection_idxs)
    basis_points = cs.convert(np.eye(args.dimension), ColorSpaceType.MAXBASIS, ColorSpaceType.HERING_CHROM)

    primary_indices = [np.argmin(np.abs(wavelengths - wl)) for wl in args.primary_wavelengths]
    primary_points = chromaticity_points[primary_indices]
    AVG_FWHM = 22.4
    sigma = AVG_FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation

    primaries = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                 for peak in args.led_gamut_wavelengths]
    primaries_sRGB = np.array([p.to_rgb() for p in primaries])

    simplex_coords, points = GetSimplexBarycentricCoords(
        args.dimension, primary_points, chromaticity_points)

    simplex_origin = GetSimplexOrigin(1, args.dimension)
    if args.dimension < 4:
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
        basis_points_3d = np.hstack((simplex_coords, np.zeros((basis_points.shape[0], 1))))
        simplex_origin = np.append(simplex_origin, 0)

        viz.RenderPointCloud("gamut-points", basis_points_3d, primaries_sRGB, radius=0.1)
        offset = np.array([0, 0, 0.01]) if args.dimension == 2 else np.array([0, 0, 0])

        if args.dimension > 2:
            viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors, radius=0.025/5)
            viz.RenderSetOfArrows("basis", [(simplex_origin + offset,
                                             x + offset) for x in basis_points_3d], radius=0.025/5)

            viz.Render2DMesh("gamut", simplex_coords, np.array([0.25, 0, 1]) * 0.5)
            ps.get_surface_mesh("gamut").set_transparency(0.4)
        else:
            viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors, radius=0.025/2)
            viz.RenderSetOfArrows("basis", [(simplex_origin + offset,
                                             x + offset) for x in basis_points_3d], radius=0.025/2)
    else:
        points_3d = points
        basis_points_3d = simplex_coords

        viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors)

        viz.RenderPointCloud("gamut-points", basis_points_3d, primaries_sRGB)
        viz.Render3DMesh("gamut", basis_points_3d, rgbs=np.tile(
            np.array([0.25, 0, 1]) * 0.5, (basis_points_3d.shape[0], 1)))
        ps.get_surface_mesh("gamut").set_transparency(0.4)
        viz.RenderSetOfArrows("basis", [(simplex_origin, x) for x in basis_points_3d], radius=0.025/10)

        viz.AnimationUtils.AddObject("basis", "surface_mesh", args.position, args.velocity,
                                     args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("spectral_locus", "curve_network", args.position,
                                     args.velocity, args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("gamut-points", "point_cloud", args.position,
                                     args.velocity, args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("gamut", "surface_mesh",
                                     args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # viz.RenderPointCloud("basis", basis_points_3d, np.eye(3))
    # viz.Render2DMesh("basis_mesh", basis_points, np.eye(3))
    # ps.get_surface_mesh("basis_mesh").set_transparency(0.5)

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
