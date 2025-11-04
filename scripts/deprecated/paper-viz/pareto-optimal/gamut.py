import argparse
import numpy as np
import tetrapolyscope as ps
from scipy.spatial import ConvexHull
from colour.notation import RGB_to_HEX

from TetriumColor.Observer import Observer, convert_refs_to_spectras
from TetriumColor.Observer.ColorSpaceTransform import GetsRGBfromWavelength
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries
from TetriumColor import ColorSpace, ColorSpaceType


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
    parser.add_argument('--display_type', choices=['ideal', 'viz-efficient',
                        'ours'], default='ours', help='Type of display to visualize')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 10)

    observer = Observer.custom_observer(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)
    cst = ColorSpace(observer)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)
    wavelengths = np.arange(380, 781, 5)
    spectral_locus_colors = np.array([GetsRGBfromWavelength(wl) for wl in observer_wavelengths])
    projection_idxs = list(range(1, observer.dimension))

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension <= 3:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(4)
    ps.set_window_size(720, 720)
    ps.set_transparency_mode('pretty')
    ps.set_transparency_render_passes(16)

    ideal_primary_spectra = np.zeros((len(args.viz_efficient_wavelengths), len(wavelengths)))
    for i, primary_wavelength in enumerate(args.viz_efficient_wavelengths):
        index = np.where(wavelengths == primary_wavelength)[0]
        if index.size > 0:
            ideal_primary_spectra[i, index[0]] = 1
    ideal_primary_spectra = convert_refs_to_spectras(ideal_primary_spectra, wavelengths)
    ideal_primary_sRGB = np.array([s.to_rgb() for s in ideal_primary_spectra])
    ideal_primary_sRGB = ideal_primary_sRGB / np.max(ideal_primary_sRGB)

    primaries = LoadPrimaries("../../../measurements/2024-12-06/primaries")
    primary_spectra = GaussianSmoothPrimaries(primaries)
    if args.dimension == 3:
        primary_spectra = [x for i, x in enumerate(primary_spectra) if i != 3]  # just use RGB
    our_primaries_sRGB = np.array([s.to_rgb() for s in primary_spectra])
    our_primaries_sRGB = our_primaries_sRGB / np.max(our_primaries_sRGB)

    display_to_cone = observer.observe_spectras(primary_spectra)
    ideal_display_to_cone = observer.observe_spectras(ideal_primary_spectra)

    chromaticity_points = cst.convert(observer.normalized_sensor_matrix.T,
                                      ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    display_coords = cst.convert(display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    ideal_display_coords = cst.convert(ideal_display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    basis_points = cst.convert(np.eye(args.dimension), ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)

    # chromaticity_points = viz.ConvertPointsToChromaticity(
    #     observer.normalized_sensor_matrix.T, observer, projection_idxs)
    # display_coords = viz.ConvertPointsToChromaticity(display_to_cone, observer, projection_idxs)
    # ideal_display_coords = viz.ConvertPointsToChromaticity(ideal_display_to_cone, observer, projection_idxs)
    #  basis_points = viz.ConvertPointsToChromaticity(np.eye(args.dimension), observer, projection_idxs)

    # get rid of zero points as they are not visible
    idxs = ~np.all(chromaticity_points == 0, axis=1)
    chromaticity_points = chromaticity_points[idxs]
    spectral_locus_colors = spectral_locus_colors[idxs]

    primary_indices = [np.argmin(np.abs(wavelengths - wl)) for wl in args.primary_wavelengths]
    primary_points = chromaticity_points[primary_indices]

    max_primaries_sRGB = np.array([GetsRGBfromWavelength(wl) for wl in args.primary_wavelengths])
    max_primaries_sRGB = max_primaries_sRGB / np.max(max_primaries_sRGB)

    simplex_coords, points = GetSimplexBarycentricCoords(
        args.dimension, primary_points, chromaticity_points)

    if args.display_type == 'ideal':
        bary_points = simplex_coords
        bary_primaries = max_primaries_sRGB
    elif args.display_type == 'viz-efficient':
        _, bary_points = GetSimplexBarycentricCoords(
            args.dimension, primary_points, ideal_display_coords)
        bary_primaries = ideal_primary_sRGB
    else:
        _, bary_points = GetSimplexBarycentricCoords(
            args.dimension, primary_points, display_coords)
        bary_primaries = our_primaries_sRGB

    actual_volume = ConvexHull(bary_points).volume
    names = viz.RenderSimplexElements("primary-gamut", args.dimension, bary_points,
                                      bary_primaries, isColored=False)
    if args.dimension == 3:
        ideal_volume = viz.Render2DMesh("spectral-gamut", points, np.array([0.1, 0.1, 0.1]))
    else:
        ideal_volume = viz.Render3DMesh("spectral-gamut", points,
                                        rgbs=np.tile(np.array([0.1, 0.1, 0.1]), (points.shape[0], 1)))
    print(f"Actual Volume: {actual_volume}, Ideal Volume: {ideal_volume}, Volume Ratio: {actual_volume / ideal_volume}")
    with open("./chromatic-gamut-grid-outputs/volumes.txt", "a") as file:
        file.write(
            f"Actual Volume: {actual_volume}, Ideal Volume: {ideal_volume}, Volume Ratio: {actual_volume / ideal_volume}\n")

    with open("./chromatic-gamut-grid-outputs/hex-colors.txt", "a") as file:
        file.write("".join([f"{RGB_to_HEX(srgb)}, " for srgb in bary_primaries]) + "\n")

    ps.get_surface_mesh("spectral-gamut").set_transparency(0.4)
    ps.get_surface_mesh("spectral-gamut").set_material("clay")
    names.append(("spectral-gamut", "surface_mesh"))

    viz.Render3DLine("spectral_locus", points, spectral_locus_colors)
    names.append(("spectral_locus", "curve_network"))
    ps.set_automatically_compute_scene_extents(False)

    for name, element_name in names:
        viz.AnimationUtils.AddObject(name, element_name, args.position, args.velocity,
                                     args.rotation_axis, args.rotation_speed)

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
