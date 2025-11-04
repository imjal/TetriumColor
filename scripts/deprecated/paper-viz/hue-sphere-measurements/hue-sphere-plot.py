import argparse
import numpy as np
import tetrapolyscope as ps
import tifffile

import numpy.typing as npt
from typing import List

from TetriumColor.Observer import GetCustomObserver, ObserverFactory, GetParalleletopeBasis, convert_refs_to_spectras, Spectra, GetHeringMatrix
from TetriumColor.Observer.DisplayObserverSensitivity import GetColorSpaceTransform, getDisplayedSpectraFromWeights
from TetriumColor.Utils.CustomTypes import DisplayBasisType, ColorSpaceTransform
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.PsychoPhys.HueSphere import GenerateEachFaceOfGrid, GenerateHueSpherePoints
from TetriumColor.Measurement import LoadPrimaries, GaussianSmoothPrimaries, LoadMetamers


def load_all_spectras():
    spectras = []
    for grid_idx in range(2):
        for row_idx in range(1, 6):
            spectras += LoadMetamers(f"../../../measurements/2025-01-21/grid_{grid_idx}_row_{row_idx}", "")
    return spectras


def plot_spectras_grid():
    import matplotlib.pyplot as plt

    spectras = load_all_spectras()
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i < len(spectras):
            ax.plot(spectras[i].wavelengths, spectras[i].values)
            ax.set_title(f'Spectra {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def load_image(image_path):
    # Load the TIFF image using tifffile
    img = tifffile.imread(image_path)

    # interpret byte as floats instead
    img = img.view(np.float32)

    return img


def load_measured_metamers(date: str, name: str):
    return LoadMetamers(f"../../../measurements/{date}/{name}", "")


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=10, help='Step size for wavelengths')
    parser.add_argument('--ideal', action='store_true', default=False, help='Use ideal primaries')
    parser.add_argument('--primary_wavelengths', nargs='+', type=float, default=[410, 510, 585, 695],
                        help='Wavelengths for the display')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(380, 781, 5)

    observer = GetCustomObserver(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    primaries: List[Spectra] = LoadPrimaries("../../../measurements/2025-01-16/primaries")
    gaussian_smooth_primaries: List[Spectra] = GaussianSmoothPrimaries(primaries)

    color_space_transform: ColorSpaceTransform = GetColorSpaceTransform(
        observer, gaussian_smooth_primaries, metameric_axis=2, scaling_factor=10000)

    face_colors = GenerateEachFaceOfGrid(1.3, 0.3, color_space_transform, 5, f'./outputs/cubemap_faces_5x5')
    true_cone_responses = GenerateHueSpherePoints(1.3, 0.3, color_space_transform, 5)
    colors = np.transpose(face_colors, axes=(2, 1, 0, 3))[:, :, :, [0, 1, 2, 3]].reshape(-1, 4)
    true_spectras = [getDisplayedSpectraFromWeights(color, gaussian_smooth_primaries) for color in colors]

    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    observer_wavelengths = np.arange(380, 781, args.step_size)
    display_observer = GetCustomObserver(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                         args.l_cone_peak, args.macula, args.lens, args.template)
    display_observer = ObserverFactory.get_object(display_observer)

    viz.RenderOBS("observer", display_observer, args.display_basis)
    ps.get_surface_mesh("observer").set_transparency(0.5)

    # GetColorSpaceTransform(observer, display_primari)

    viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                 args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # viz.RenderSphere("sub-sphere", 0.3)

    # # measured
    spectras = load_all_spectras()
    cone_responses = np.array([observer.observe_normalized(s * 100000) for s in spectras])
    print(repr(cone_responses[12]/cone_responses[12].max()))
    print(repr(cone_responses[12 + 25] / cone_responses[12 + 25].max()))

    cone_sRGBs = np.array([s.to_rgb() for s in spectras])
    normalized_sRGBs = np.array([rgb / rgb.max() for rgb in cone_sRGBs])
    basis_responses = viz.ConvertPointsToBasis(cone_responses, observer, args.display_basis)
    viz.RenderPointCloud("cube-map-spectras", basis_responses, normalized_sRGBs)

    # plot the measured metamers for the CIVO data
    # tests = "bcdefgh"
    # for i in range(len(tests)):
    #     test_char = tests[i]
    #     metamer_spectras = load_measured_metamers("2025-01-22", f"plate-{test_char}-CIVO")
    #     cone_responses = np.array([observer.observe_normalized(s * 100000) for s in metamer_spectras])
    #     print(repr(cone_responses))
    #     cone_sRGBs = np.array([s.to_rgb() for s in metamer_spectras])
    #     normalized_sRGBs = np.array([rgb / rgb.max() for rgb in cone_sRGBs])
    #     basis_responses = viz.ConvertPointsToBasis(cone_responses, observer, args.display_basis)
    #     viz.RenderPointCloud(f"metamer-spectra-{test_char}", basis_responses, normalized_sRGBs)

    RYGB_to_sRGB = np.array([[0.77920324,  0.90917001, -0.37526135,  0.04545921],
                             [-0.07345655,  0.28421096,  0.86285929, -0.07311962],
                             [-0.0127013, -0.053994,  0.03159406,  1.03177691]])
    # data = load_image(
    #     '../../../data/jennifer-drawing/apple/snapshot2025-01-21_19-45-15.tiff').reshape(-1, 4)[::1000, [3, 2, 1, 0]]
    # data = load_image('../../../data/jennifer-drawing/lime/snapshot2025-01-21_20-34-44.tiff').reshape(-1,
    #                                                                                                   4)[::1000, [3, 2, 1, 0]]
    data = load_image(
        '../../../data/jennifer-drawing/melon/snapshot2025-01-21_21-18-58.tiff').reshape(-1, 4)[::1000]
    sRGBs = data@RYGB_to_sRGB.T
    basis_responses = data[:, [3, 2, 1, 0]]@GetHeringMatrix(observer.dimension).T[:, 1:]
    # plot Jennifers RYGB data
    viz.RenderPointCloud(f"jennifer_data", basis_responses, sRGBs)

    # predicted
    # true_cone_responses = np.array([observer.observe_normalized(s * 100000/4) for s in true_spectras])
    # true_cone_sRGBs = np.array([s.to_rgb() for s in true_spectras])
    # normalized_true_sRGBs = np.array([rgb / rgb.max() for rgb in true_cone_sRGBs])
    # true_basis_responses = viz.ConvertPointsToBasis(true_cone_responses, observer, args.display_basis)
    # viz.RenderPointCloud("true-cube-map-spectras", true_basis_responses, None)

    # Render Metameric Lines
    basis_colors = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    # Render Arrows for Basis

    # Render Metameric Lines
    for i in range(2, 3):
        viz.RenderMetamericDirection(
            f"tetra-metameric-direction-{i}", observer, args.display_basis, i, np.zeros(3))
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
            pass
            # viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
