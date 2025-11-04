import argparse
import numpy as np
from scipy.sparse import construct
import tetrapolyscope as ps

from TetriumColor.Observer import Observer, MaxBasisFactory
import TetriumColor.Visualization as viz
from TetriumColor import ColorSpace, ColorSpaceType
from TetriumColor.Utils.ParserOptions import *
from TetriumColor.Utils.BasisMath import get_transform_to_angle_basis


def main():
    parser = argparse.ArgumentParser(description='Visualize Gamut from Observers')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)
    parser.add_argument('--step_size', type=float, default=10, help='Step size for wavelengths')
    parser.add_argument(
        '--polyscope_display_type', type=lambda choice: PolyscopeDisplayType[choice],
        choices=[x for x in PolyscopeDisplayType],
        default=PolyscopeDisplayType.HERING_CONE_PERCEPTUAL_300, help='Step size for wavelengths')
    args = parser.parse_args()

    # Observer attributes
    observer_wavelengths = np.arange(360, 830, 5)
    observer = Observer.custom_observer(observer_wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                        args.l_cone_peak, args.macula, args.lens, args.template)
    # Polyscope Animation Inits
    ps.init()
    ps.set_always_redraw(True)
    if args.dimension == 2:
        ps.set_ground_plane_mode('none')
    else:
        ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(2)
    ps.set_window_size(720, 720)

    cst = ColorSpace(observer)  # luminance_per_channel=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    # chromas_per_channel=[np.sqrt(2/3)/2, np.sqrt(2/3)/4, np.sqrt(2/3)/4])
    viz.RenderOBS("observer_hering", cst, args.polyscope_display_type)
    # viz.RenderMaxBasis("maxbasis_hering", cst, args.polyscope_display_type)
    ps.get_surface_mesh("observer_hering").set_transparency(0.8)

    # basis_in_space = cst.convert_to_polyscope(np.eye(args.dimension), ColorSpaceType.CONE, args.polyscope_display_type)
    # basis_in_space = basis_in_space / np.linalg.norm(basis_in_space, axis=1, keepdims=True)
    # viz.RenderBasisArrows("basis", basis_in_space, radius=0.01)

    # viz.AnimationUtils.AddObject("observer_hering", "surface_mesh",
    #                              args.position, args.velocity, args.rotation_axis, args.rotation_speed)
    # # viz.AnimationUtils.AddObject("maxbasis_hering", "surface_mesh",
    # #                              args.position, args.velocity, args.rotation_axis, args.rotation_speed)
    # viz.AnimationUtils.AddObject("basis", "surface_mesh",
    #                              args.position, args.velocity, args.rotation_axis, args.rotation_speed)

    # plot points from Munsell in the observer space

    # viz.RenderOBS("observer_cone_perceptual", cst, PolyscopeDisplayType.HERING_CONE_PERCEPTUAL_243)
    # viz.RenderMaxBasis("maxbasis_cone_perceptual", cst, PolyscopeDisplayType.HERING_CONE_PERCEPTUAL_243)
    # ps.get_surface_mesh("observer_cone_perceptual").set_transparency(0.8)

    # viz.RenderOBS("observer_maxbasis_perceptual", cst, PolyscopeDisplayType.HERING_MAXBASIS_PERCEPTUAL_243)
    # viz.RenderMaxBasis("maxbasis_maxbasis_perceptual", cst, PolyscopeDisplayType.HERING_MAXBASIS_PERCEPTUAL_243)
    # ps.get_surface_mesh("observer_maxbasis_perceptual").set_transparency(0.8)

    # viz.RenderOBS("observer_maxbasis243_perceptual243", cst, PolyscopeDisplayType.HERING_MAXBASIS243_PERCEPTUAL_243)
    # viz.RenderMaxBasis("maxbasis_maxbasis243_perceptual243", cst,
    #                    PolyscopeDisplayType.HERING_MAXBASIS243_PERCEPTUAL_243)
    # ps.get_surface_mesh("observer_maxbasis243_perceptual243").set_transparency(0.8)

    # maxbasis = MaxBasisFactory.get_object(observer, denom=2.43)
    # refs, _, rgbs, lines = maxbasis.GetDiscreteRepresentation()
    # cones = observer.observe_spectras(refs)
    # mat = get_transform_to_angle_basis(cones[1:4], [1/np.sqrt(3), 2/np.sqrt(3),
    #                                    3/(2 * np.sqrt(3))], [np.sqrt(2/3) / 2, np.sqrt(2/3) / 2, np.sqrt(2/3)])
    # vecs = cones[1:4]@mat.T
    # viz.RenderBasisArrows("basis_arrows", vecs.tolist() + [np.ones(3)], np.eye(3).tolist() + [np.zeros(3)], radius=0.01)
    # ps.set_automatically_compute_scene_extents(False)

    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps

        def callback():
            # pass
            viz.AnimationUtils.UpdateObjects(delta_time)
        ps.set_user_callback(callback)
        ps.show()
        ps.clear_user_callback()


if __name__ == "__main__":
    main()
