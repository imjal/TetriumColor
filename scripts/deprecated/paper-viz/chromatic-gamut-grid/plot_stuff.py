import argparse
import numpy as np
import tetrapolyscope as ps
from scipy.spatial import ConvexHull
from colour.notation import RGB_to_HEX

from TetriumColor.Measurement.TetriumMeasurementRoutines import load_primaries_from_csv
from TetriumColor.Observer import Observer, convert_refs_to_spectras, Spectra
from TetriumColor.Observer.ColorSpaceTransform import GetsRGBfromWavelength
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.ColorMath.Geometry import GetSimplexBarycentricCoords
from TetriumColor import ColorSpace, ColorSpaceType
from TetriumColor.Measurement import load_primaries_from_csv, compare_dataset_to_primaries, get_spectras_from_rgbo_list, export_metamer_difference, export_predicted_vs_measured_with_square_coords, get_spectras_from_rgbo_list, plot_measured_vs_predicted


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=5, help='Step size for wavelengths')
    parser.add_argument('--max_simplex_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the ideal chromatic display')
    parser.add_argument('--max_perceptual_volume_wavelengths', nargs='+', type=float, default=[435, 515, 610, 660],
                        help='Wavelengths for the visually efficient display')
    parser.add_argument('--display_type', choices=['ideal', 'max-perceptual',
                        'ours'], default='ours', help='Type of display to visualize')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 5)

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
        our_primary_spectras = [x for i, x in enumerate(our_primary_spectras) if i != 3]  # just use RGB
    our_primaries_sRGB = np.array([s.to_rgb() for s in our_primary_spectras])
    our_primaries_sRGB = our_primaries_sRGB / np.max(our_primaries_sRGB)

    our_display_to_cone = observer.observe_spectras(our_primary_spectras)
    ideal_display_to_cone = observer.observe_spectras(ideal_primary_spectras)

    chromaticity_points = cst.convert(observer.normalized_sensor_matrix.T,
                                      ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    our_display_coords = cst.convert(our_display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    ideal_display_coords = cst.convert(ideal_display_to_cone, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    basis_points = cst.convert(np.eye(args.dimension), ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)

    # chromaticity_points = viz.ConvertPointsToChromaticity(
    #     observer.normalized_sensor_matrix.T, observer, projection_idxs)
    # display_coords = viz.ConvertPointsToChromaticity(display_to_cone, observer, projection_idxs)
    # ideal_display_coords = viz.ConvertPointsToChromaticity(ideal_display_to_cone, observer, projection_idxs)
    #  basis_points = viz.ConvertPointsToChromaticity(np.eye(args.dimension), observer, projection_idxs)
    measurement_points = np.array([[108, 101,  88, 215],
                                   [89, 118,  84, 219],
                                   [69, 139,  82, 217],
                                   [55, 159,  85, 207],
                                   [48, 173,  91, 192],
                                   [101,  90, 109, 213],
                                   [77, 107, 107, 220],
                                   [53, 131, 106, 218],
                                   [37, 154, 109, 205],
                                   [33, 170, 112, 187],
                                   [95,  79, 135, 203],
                                   [68,  95, 138, 210],
                                   [41, 120, 140, 208],
                                   [26, 145, 140, 194],
                                   [24, 164, 139, 176],
                                   [92,  73, 161, 187],
                                   [67,  87, 168, 189],
                                   [42, 110, 172, 186],
                                   [27, 134, 170, 174],
                                   [24, 153, 165, 161],
                                   [92,  71, 181, 169],
                                   [71,  85, 188, 167],
                                   [51, 104, 192, 163],
                                   [37, 125, 190, 155],
                                   [32, 143, 184, 146],
                                   [206,  81, 163,  62],
                                   [199,  95, 169,  47],
                                   [185, 115, 172,  37],
                                   [165, 136, 170,  35],
                                   [146, 153, 166,  39],
                                   [221,  84, 142,  67],
                                   [217, 100, 145,  49],
                                   [201, 123, 148,  36],
                                   [177, 147, 147,  34],
                                   [153, 164, 145,  41],
                                   [230,  90, 115,  78],
                                   [228, 109, 114,  60],
                                   [213, 134, 114,  46],
                                   [186, 159, 116,  44],
                                   [159, 175, 119,  51],
                                   [230, 101,  89,  93],
                                   [227, 120,  84,  80],
                                   [212, 144,  82,  68],
                                   [187, 167,  86,  65],
                                   [162, 181,  93,  67],
                                   [222, 111,  70, 108],
                                   [217, 129,  64,  99],
                                   [203, 150,  62,  91],
                                   [183, 169,  66,  87],
                                   [162, 183,  73,  85]], dtype=np.uint8)
    measurements_dir = "../../../measurements/2025-05-21/5x5-cubemap/"
    measured_spectras = get_spectras_from_rgbo_list(measurements_dir, measurement_points.tolist())

    cones = observer.observe_spectras(measured_spectras)
    measured_pts = cst.convert(cones, ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)
    srgb_of_measured = cst.convert(cones * 1000, ColorSpaceType.CONE, ColorSpaceType.SRGB)

    # get rid of zero points as they are not visible
    idxs = ~np.all(chromaticity_points == 0, axis=1)
    chromaticity_points = chromaticity_points[idxs]
    spectral_locus_colors = spectral_locus_colors[idxs]

    primary_indices = [np.argmin(np.abs(wavelengths - wl)) for wl in args.max_simplex_wavelengths]
    primary_points = chromaticity_points[primary_indices]
    max_primaries_sRGB = np.array([GetsRGBfromWavelength(wl) for wl in args.max_simplex_wavelengths])
    max_primaries_sRGB = max_primaries_sRGB / np.max(max_primaries_sRGB)

    # simplex_coords, points = GetSimplexBarycentricCoords(
    #     args.dimension, primary_points, chromaticity_points)
    simplex_coords = np.array([np.zeros(args.dimension)])
    points = chromaticity_points

    if args.display_type == 'ideal':
        bary_points = simplex_coords
        bary_primaries = max_primaries_sRGB
    elif args.display_type == 'max-perceptual':
        bary_points = ideal_display_coords
        bary_primaries = ideal_primary_sRGB
    else:
        # _, bary_points = GetSimplexBarycentricCoords(
        #     args.dimension, primary_points, our_display_coords)
        bary_points = our_display_coords
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

    viz.RenderPointCloud("cube-map-predicted", measured_pts, srgb_of_measured)

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
