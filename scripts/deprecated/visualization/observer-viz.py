import argparse
import numpy as np
import tetrapolyscope as ps
import time

from TetriumColor.Observer import GetCustomObserver, ObserverFactory
from TetriumColor import DisplayBasisType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import AddObserverArgs, AddVideoOutputArgs


def main():
    parser = argparse.ArgumentParser(description='Visualize Custom Tetra Observer.')
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    args = parser.parse_args()

    # Observer attributes
    wavelengths = np.arange(380, 781, 10)

    observer = GetCustomObserver(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                 args.l_cone_peak, args.macula, args.lens, args.template)
    # load cached observer stuff if it exists, terrible design but whatever
    # observer = ObserverFactory.get_object(observer)

    # Polyscope Animation Inits
    ps.init()
    position = [0, 0, 0]
    velocity = [0, 0, 0]  # [-0.02, 0, 0]
    rotation_speed = 10
    rotation_axis = [0, 1, 0]

    # Create Geometry & Register with Polyscope, and define the animation

    viz.RenderOBS("tetra-custom-observer", observer, args.display_basis)
    ps.get_surface_mesh("tetra-custom-observer").set_transparency(0.5)

    viz.AnimationUtils.AddObject("tetra-custom-observer", "surface_mesh",
                                 position, velocity, rotation_axis, rotation_speed)

    viz.RenderMaxBasis("tetra-max-basis", observer, display_basis=args.display_basis)
    viz.AnimationUtils.AddObject("tetra-max-basis", "surface_mesh", position, velocity, rotation_axis, rotation_speed)

    # Output Video to Screen or Save to File (based on options)
    if args.output_filename:
        fd = viz.OpenVideo(args.output_filename)
        viz.RenderVideo(fd, args.total_frames, args.fps)
        viz.CloseVideo(fd)
    else:
        delta_time: float = 1 / args.fps
        while True:
            viz.AnimationUtils.UpdateObjects(delta_time)
            ps.frame_tick()


if __name__ == "__main__":
    main()
