import argparse
from re import I
import numpy as np
import tetrapolyscope as ps

from TetriumColor.Observer import Observer
from TetriumColor.Observer.ColorSpaceTransform import GetsRGBfromWavelength
from TetriumColor import ColorSpace, ColorSpaceType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *


def main():
    parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument(
        '--chrom_type', type=lambda choice: ColorSpaceType[choice],
        choices=[ColorSpaceType.CHROM, ColorSpaceType.HERING_CHROM],
        default=ColorSpaceType.CHROM, help='Step size for wavelengths')
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)

    observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)
    spectral_locus_colors = np.array([GetsRGBfromWavelength(wl) for wl in wavelengths])

    cs = ColorSpace(observer)
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
    # points_maybe_nan = viz.ConvertPointsToChromaticity(observer.normalized_sensor_matrix.T, observer, projection_idxs)
    points_maybe_nan = cs.convert(observer.normalized_sensor_matrix.T, ColorSpaceType.CONE, args.chrom_type)
    points = points_maybe_nan[~np.all(points_maybe_nan == 0, axis=1)]
    # points = points[:, [1, 0]]

    spectral_locus_colors = spectral_locus_colors[~np.all(points_maybe_nan == 0, axis=1), :]

    from_space = ColorSpaceType.MAXBASIS if args.chrom_type == ColorSpaceType.HERING_CHROM else ColorSpaceType.CONE
    basis_points = cs.convert(np.eye(args.dimension), from_space, args.chrom_type)
    basis_points = basis_points[~np.all(basis_points == 0, axis=1)]
    # basis_points = [[basis_points[0][0], -basis_points[0][0]]]

    # ideal_primary_spectra = np.zeros((len(args.viz_efficient_wavelengths), len(wavelengths)))
    AVG_FWHM = 22.4
    sigma = AVG_FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    ideal_primary_spectras = [Spectra(wavelengths=wavelengths, data=gaussian(wavelengths, peak, sigma))
                              for peak in args.max_perceptual_volume_wavelengths]
    # for i, primary_wavelength in enumerate(args.viz_efficient_wavelengths):
    #     index = np.where(wavelengths == primary_wavelength)[0]
    #     if index.size > 0:
    #         ideal_primary_spectra[i, index[0]] = 1
    ideal_primary_sRGB = np.array([s.to_rgb() for s in ideal_primary_spectras])
    ideal_primary_sRGB = ideal_primary_sRGB / np.max(ideal_primary_sRGB)

    # primaries = LoadPrimaries("../../../measurements/2025-05-20/primaries")
    # primary_spectra = GaussianSmoothPrimaries(primaries)
    our_primary_spectras = load_primaries_from_csv("../../../measurements/2025-05-20/primaries")
    if args.dimension == 3:
        primary_spectra = [x for i, x in enumerate(our_primary_spectras) if i != 3]  # just use RGB
    our_primaries_sRGB = np.array([s.to_rgb() for s in our_primary_spectras])
    our_primaries_sRGB = our_primaries_sRGB / np.max(our_primaries_sRGB)

    our_display_to_cone = observer.observe_spectras(our_primary_spectras)
    ideal_display_to_cone = observer.observe_spectras(ideal_primary_spectras)

    chromaticity_points = cs.convert(observer.normalized_sensor_matrix.T,
                                     ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    our_display_coords = cs.convert(our_display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    ideal_display_coords = cs.convert(ideal_display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    basis_points = cs.convert(np.eye(args.dimension), ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)

    if args.dimension < 4:
        points_3d = np.hstack((points, np.zeros((points.shape[0], 4 - args.dimension))))
        basis_points_3d = np.hstack((basis_points, np.zeros((basis_points.shape[0], 4 - args.dimension))))

        if args.dimension == 3:
            viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors)
            viz.Render2DMesh("gamut", points, np.array([0.25, 0, 1]) * 0.5)
            ps.get_surface_mesh("gamut").set_transparency(0.4)
            viz.RenderBasisArrows("basis", basis_points_3d, radius=0.025/3)
        else:
            viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors)
            offset = np.array([0, 0, 0.01]) if args.dimension == 2 else np.array([0, 0, 0])
            viz.RenderSetOfArrows("basis", [(np.zeros(3) + offset,
                                             x + offset) for x in basis_points_3d], radius=0.025/4)
            # ps.get_surface_mesh("basis").set_transparency(0.8)
    else:
        points_3d = points
        basis_points_3d = basis_points
        viz.Render3DLine("spectral_locus", points_3d, spectral_locus_colors)
        viz.Render3DMesh("gamut", points_3d, rgbs=np.tile(np.array([0.25, 0, 1]) * 0.5, (points_3d.shape[0], 1)))
        ps.get_surface_mesh("gamut").set_transparency(0.4)

        viz.RenderBasisArrows("basis", basis_points_3d, radius=0.025/3)
        viz.AnimationUtils.AddObject("basis", "surface_mesh",
                                     args.position, args.velocity, args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("spectral_locus", "curve_network", args.position,
                                     args.velocity, args.rotation_axis, args.rotation_speed)
        viz.AnimationUtils.AddObject("gamut", "surface_mesh",
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
            pass
            # viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
