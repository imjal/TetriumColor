#!/usr/bin/env python3
"""
Ink library analysis CLI tool.
Analyzes ink libraries for gamut optimization and visualization.
"""

from TetriumColor.Utils.ParserOptions import AddObserverArgs, AddVideoOutputArgs, AddAnimationArgs
from TetriumColor.Observer import Observer, Spectra, InkGamut, Illuminant, InkLibrary, load_top_inks, show_top_k_combinations, InkLibrary, plot_inks_by_hue, save_top_inks_as_csv, load_all_ink_libraries, combine_inksets, load_inkset
from TetriumColor import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from library_registry import registry
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, List
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))


# Import optional visualization dependencies
try:
    import tetrapolyscope as ps
    import TetriumColor.Visualization as viz
    from IPython.display import Image, display, HTML
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    ps = None
    viz = None
    Image = None
    display = None
    HTML = None
    print("Warning: tetrapolyscope not available. Visualization features will be disabled.")


screenshot_count = 0


def save_ps_screenshot(results_dir, ps, Image, display):
    """Save a polyscope screenshot to the results directory."""
    global screenshot_count
    ps.show()  # renders window
    fname = os.path.join(results_dir, f"screenshot_{screenshot_count}.png")
    ps.screenshot(fname)
    # Display in notebook
    display(Image(filename=fname, width=400))  # need to use this for pdf export
    screenshot_count += 1


def get_top_n_inks(top_volumes, top_d_inks=20):
    """
    Given the output of convex_hull_search (top_volumes), return a list of the top n unique ink names
    appearing in the top k combinations, ordered by frequency of appearance (descending).
    """
    from collections import Counter
    # Flatten all ink names in the top k combinations
    all_ink_names = []
    for _, ink_names in top_volumes:
        all_ink_names.extend(ink_names)
    # Count frequency
    counter = Counter(all_ink_names)
    # Get the n most common ink names
    top_inks = [ink for ink, _ in counter.most_common(top_d_inks)]
    import pdb
    pdb.set_trace()
    return top_inks


def analyze_ink_library(library_name: str, args):
    """Analyze a single ink library."""
    try:

        # Resolve library path
        library_path = registry.resolve_library_path(library_name)
        print(f"Loading ink library: {library_name}")
        print(f"Library path: {library_path}")

        # Load inks by library name via registry
        inks_dict, paper_spectra, wavelengths = load_inkset(library_path, filter_clogged=True)

        # Construct InkLibrary explicitly
        inkset_library = InkLibrary(inks_dict, paper_spectra)

        # Create results directory for this inkset
        results_dir = f"results/{library_name}_k{args.k}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

        # Set up observer and color space
        d65 = Illuminant.get("d65")
        tetrachromat = Observer.tetrachromat(wavelengths=np.arange(400, 710, 10))
        cs = ColorSpace(tetrachromat)

        # Save all ink spectra plot
        plot_inks_by_hue(inkset_library.library, np.arange(400, 710, 10),
                         filename=os.path.join(results_dir, f"all_ink_spectras_{library_name}.png"))

        # Perform convex hull search
        top_volumes_all_inks = inkset_library.convex_hull_search(tetrachromat, d65, k=args.k)

        # Save top volumes as CSV
        save_top_inks_as_csv(top_volumes_all_inks, os.path.join(
            results_dir, f"top_volumes_k{args.k}_{library_name}.csv"))

        # Show top k combinations (this creates plots)
        show_top_k_combinations(top_volumes_all_inks, inkset_library.library, k=16, filename=os.path.join(
            results_dir, f"top_k_combinations_k{args.k}_{library_name}.png"))

        best4_inks = [inkset_library.library[ink_name] for ink_name in top_volumes_all_inks[0][1]]

        import pdb
        pdb.set_trace()

        top_d_inks = get_top_n_inks(top_volumes_all_inks, top_d_inks=args.top_d_inks)

        # Plot all top 20 inks by hue
        plot_inks_by_hue(
            {ink_name: inkset_library.library[ink_name] for ink_name in top_d_inks},
            np.arange(400, 710, 10),
            filename=os.path.join(results_dir, f"top_{args.top_d_inks}_ink_spectras_{library_name}.png")
        )

        # Save top 20 inks list as text file
        with open(os.path.join(results_dir, f"top_{args.top_d_inks}_inks_{library_name}.txt"), 'w') as f:
            f.write(f"Top {args.top_d_inks} inks for {library_name} inkset:\n")
            f.write("=" * 50 + "\n")
            for i, ink_name in enumerate(top_d_inks, 1):
                f.write(f"{i:2d}. {ink_name}\n")

        gamut = InkGamut(best4_inks, inkset_library.get_paper(), d65)
        point_cloud_k4, percentages_k4 = gamut.get_point_cloud(tetrachromat, stepsize=0.25)

        cs = ColorSpace(tetrachromat)

        # Visualization (only if tetrapolyscope is available)
        if HAS_VISUALIZATION:
            all_inks_as_points = tetrachromat.observe_spectras(inkset_library.spectra_objs)
            all_inks_point_cloud = cs.convert(all_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
            all_inks_srgbs = cs.convert(all_inks_as_points, ColorSpaceType.CONE, ColorSpaceType.SRGB)

            ps.init()
            ps.set_always_redraw(False)
            ps.set_ground_plane_mode('shadow_only')
            ps.set_SSAA_factor(2)
            ps.set_window_size(720, 720)

            if args.which_dir == 'q':
                ps.set_up_dir('z_up')  # Set up direction to Z-axis
                ps.set_front_dir('y_front')  # Set front direction to Y-axis
                metameric_axis = cs.get_metameric_axis_in(ColorSpaceType.HERING)
                rotation_mat = np.eye(4)
                rotation_mat[:3, :3] = np.linalg.inv(viz.RotateToZAxis(metameric_axis[1:]))
                rotation_mat = rotation_mat.T
            elif args.which_dir == 'saq':
                ps.set_up_dir('z_up')  # Set up direction to Z-axis
                ps.set_front_dir('y_front')  # Set front direction to Y-axis
                saq = np.array([[1, 0, 1, 0]])
                saq_in_hering = cs.convert(saq, ColorSpaceType.MAXBASIS, ColorSpaceType.HERING)[0, 1:]
                rotation_mat = np.eye(4)
                rotation_mat[:3, :3] = np.linalg.inv(viz.RotateToZAxis(saq_in_hering))
                rotation_mat = rotation_mat.T
                ps.set_ground_plane_height_factor(0.2, False)
            else:
                rotation_mat = np.eye(4)

            factor = 0.1575  # 0.1/5.25
            viz.ps.set_background_color((factor, factor, factor, 1))

            viz.RenderOBS("observer", cs, PolyscopeDisplayType.HERING_MAXBASIS, num_samples=1000)
            viz.ps.get_surface_mesh("observer").set_transparency(0.3)
            viz.ps.get_surface_mesh("observer").set_transform(rotation_mat)
            viz.RenderPointCloud("points_k4", cs.convert(
                point_cloud_k4, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:], mode="sphere")
            viz.ps.get_point_cloud("points_k4").set_transform(rotation_mat)

            viz.RenderPointCloud("all_inks", all_inks_point_cloud, all_inks_srgbs)
            viz.ps.get_point_cloud("all_inks").set_transform(rotation_mat)
            viz.RenderMetamericDirection("meta_dir", tetrachromat, PolyscopeDisplayType.HERING_MAXBASIS, 2,
                                         np.array([0, 0, 0]), radius=0.005, scale=1.2)
            viz.ps.get_curve_network("meta_dir").set_transform(rotation_mat)

            viz.AnimationUtils.AddObject("observer", "surface_mesh",
                                         args.position, args.velocity, args.rotation_axis, args.rotation_speed)
            viz.AnimationUtils.AddObject("points_k4", "point_cloud",
                                         args.position, args.velocity, args.rotation_axis, args.rotation_speed)
            viz.AnimationUtils.AddObject("all_inks", "point_cloud",
                                         args.position, args.velocity, args.rotation_axis, args.rotation_speed)
            viz.AnimationUtils.AddObject("meta_dir", "curve_network",
                                         args.position, args.velocity, args.rotation_axis, args.rotation_speed)

            # Save polyscope screenshot to results directory
            # save_ps_screenshot(results_dir, ps, Image, display)

            # Save video if requested
            if hasattr(args, 'total_frames') and args.total_frames > 0:
                fd = viz.OpenVideo(os.path.join(results_dir, f"inkset_summary_{library_name}.mp4"))
                viz.RenderVideo(fd, args.total_frames, args.fps)
                viz.CloseVideo(fd)

            # viz.ps.show()
        else:
            print("Skipping 3D visualization (tetrapolyscope not available)")

        print(f"Analysis complete for {library_name}. Results saved in {results_dir}/")

    except Exception as e:
        print(f"Error analyzing library '{library_name}': {e}")
        sys.exit(1)


def main():

    parser = argparse.ArgumentParser(description='Analyze ink libraries for gamut optimization')
    parser.add_argument('library_name', help='Name of the ink library to analyze')
    parser.add_argument('--k', type=int, default=4, help='Number of inks to show')
    parser.add_argument('--top_d_inks', type=int, default=20, help='Number of top inks to show')
    parser.add_argument('--output_dir', help='Output directory for results (default: results/{library_name}_k{k})')

    # Add observer and visualization arguments
    AddObserverArgs(parser)
    AddVideoOutputArgs(parser)
    AddAnimationArgs(parser)

    args = parser.parse_args()

    # Auto-discover libraries if registry is empty
    if not registry.list_libraries():
        print("Auto-discovering ink libraries...")
        registry.auto_discover_libraries()

    # Check if library exists
    if not registry.library_exists(args.library_name):
        available = registry.list_libraries()
        print(f"Error: Library '{args.library_name}' not found.")
        if available:
            print(f"Available libraries: {', '.join(available)}")
        else:
            print("No libraries found. Use 'ink_processing create' to create a new library.")
        sys.exit(1)

    # Override output directory if specified
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = f"results/{args.library_name}_k{args.k}"

    # Analyze the library
    analyze_ink_library(args.library_name, args)


if __name__ == "__main__":
    main()
