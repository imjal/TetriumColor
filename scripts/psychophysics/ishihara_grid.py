import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse
import os

from TetriumColor import ColorSpace, ColorSampler
from TetriumColor.Observer import Observer
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Utils.ParserOptions import AddObserverArgs
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid


def generate_grid_from_color_space(filename, output_dir, color_space, grid_size, num_plates, luminance, chroma):
    """
    Generate Ishihara-style metameric grid plates from a color space.
    """
    color_sampler = ColorSampler(color_space, cubemap_size=grid_size)

    images = color_sampler.get_metameric_grid_plates(luminance, chroma, num_plates)

    foreground_images = [x for x, _ in images]
    background_images = [y for _, y in images]

    # bg_color = color_space.get_background(1)
    foreground = CreatePaddedGrid(foreground_images, padding=0)
    background = CreatePaddedGrid(background_images, padding=0)

    os.makedirs(output_dir, exist_ok=True)

    foreground.save(os.path.join(output_dir, f"{filename}_RGB.png"))
    background.save(os.path.join(output_dir, f"{filename}_OCV.png"))


def generate_ishihara_grid(output_dir, measurement_dir, grid_size, metameric_axis,
                           display_type, num_plates, luminance, chroma, scramble_prob,
                           od, dimension, s_cone_peak, m_cone_peak, q_cone_peak,
                           l_cone_peak, macula, lens, template):
    """
    Generate Ishihara-style metameric grid plates for a single observer.

    Args:
        output_dir: Output directory for generated plates
        measurement_dir: Directory containing primary measurements
        grid_size: Size of the grid (e.g., 5 for 5x5 grid)
        metameric_axis: Metameric axis for colorspace
        display_type: Display type for colorspace
        num_plates: Number of plates to generate
        luminance: Luminance value
        chroma: Chroma value
        scramble_prob: Probability of scrambling the color
        od, dimension, s_cone_peak, m_cone_peak, q_cone_peak, l_cone_peak, macula, lens, template: Observer parameters
    """
    # Load Observer and Measured Primaries
    wavelengths = np.arange(360, 831, 1)
    observer = Observer.custom_observer(wavelengths, od, dimension, s_cone_peak, m_cone_peak, q_cone_peak,
                                        l_cone_peak, macula, lens, template)

    primaries = load_primaries_from_csv(measurement_dir)  # RGBO

    print(primaries)

    cs_4d = ColorSpace(observer, cst_display_type=display_type,
                       display_primaries=primaries, metameric_axis=metameric_axis)

    filename = f"{od}_{dimension}_{s_cone_peak}_{m_cone_peak}_{q_cone_peak}_{l_cone_peak}_{macula}_{lens}_{template}"

    generate_grid_from_color_space(filename, output_dir, cs_4d, grid_size, num_plates, luminance, chroma)


def generate_ishihara_grid_genotypes(output_dir, measurement_dir, grid_size, metameric_axis,
                                     display_type, num_plates, luminance, chroma,
                                     top_percentage, peak_to_test, sex='male'):
    """
    Generate Ishihara-style metameric grid plates for multiple observer genotypes.

    Args:
        output_dir: Output directory for generated plates
        measurement_dir: Directory containing primary measurements
        grid_size: Size of the grid (e.g., 5 for 5x5 grid)
        metameric_axis: Metameric axis for colorspace
        display_type: Display type for colorspace
        num_plates: Number of plates to generate
        luminance: Luminance value
        chroma: Chroma value
        scramble_prob: Probability of scrambling the color
        top_percentage: Percentage of population to cover (e.g., 0.9 for 90%)
        sex: 'male', 'female', or 'both'
    """
    # Initialize ObserverGenotypes
    og = ObserverGenotypes(dimensions=[2])

    # Get genotypes covering the specified percentage
    genotypes = og.get_genotypes_covering_probability(
        target_probability=top_percentage, sex=sex)

    print("Genotypes: ", genotypes)

    primaries = load_primaries_from_csv(measurement_dir)
    print(f"Loaded primaries: {primaries}")

    color_spaces = [og.get_color_space_for_peaks(
        genotype + (peak_to_test,), cst_display_type='led', display_primaries=primaries) for genotype in genotypes if peak_to_test not in genotype]

    print(f"Generating plates for {len(genotypes)} genotypes covering {top_percentage*100:.1f}% of {sex} population")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plates for each genotype
    for i, (genotype, color_space) in enumerate(zip(genotypes, color_spaces)):
        print(f"Processing genotype {i+1}/{len(genotypes)}: {genotype}")

        # Create genotype-specific filename
        genotype_str = "_".join(map(str, genotype))
        genotype_filename = f"genotype_{genotype_str}"

        generate_grid_from_color_space(genotype_filename, output_dir, color_space,
                                       grid_size, num_plates, luminance, chroma)

        print(f"Saved plates for genotype {genotype} to {output_dir}")

    print(f"Completed generation for {len(genotypes)} genotypes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Ishihara-style metameric grid plates')

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single observer command
    single_parser = subparsers.add_parser('single', help='Generate plates for a single observer')
    AddObserverArgs(single_parser)

    # Genotypes command
    genotypes_parser = subparsers.add_parser('genotypes', help='Generate plates for multiple observer genotypes')

    # Common arguments for both commands
    for subparser in [single_parser, genotypes_parser]:
        subparser.add_argument('--output_dir', type=str, default='./assets',
                               help='Output directory for generated plates')
        subparser.add_argument('--measurement_dir', type=str, default='../../measurements/2025-10-10/primaries/',
                               help='Directory containing primary measurements')
        subparser.add_argument('--grid_size', type=int, default=5, help='Size of the grid (e.g., 5 for 5x5 grid)')
        subparser.add_argument('--metameric_axis', type=int, default=2, help='Metameric axis for colorspace')
        subparser.add_argument('--display_type', type=str, default='led', help='Display type for colorspace')
        subparser.add_argument('--num_plates', type=int, default=4, help='Number of plates to generate')
        subparser.add_argument('--luminance', type=float, default=1.0, help='Luminance value')
        subparser.add_argument('--chroma', type=float, default=0.5, help='Chroma value')

    # Genotypes-specific arguments
    genotypes_parser.add_argument('--top_percentage', type=float, default=0.9,
                                  help='Percentage of population to cover (e.g., 0.9 for 90%)')
    genotypes_parser.add_argument('--sex', type=str, default='male',
                                  choices=['male', 'female', 'both'], help='Sex for genotype selection')
    genotypes_parser.add_argument('--peak_to_test', type=float, default=547, help='Peak to test for')

    args = parser.parse_args()

    if args.command == 'single':
        generate_ishihara_grid(
            output_dir=args.output_dir,
            measurement_dir=args.measurement_dir,
            grid_size=args.grid_size,
            metameric_axis=args.metameric_axis,
            display_type=args.display_type,
            num_plates=args.num_plates,
            luminance=args.luminance,
            chroma=args.chroma,
            scramble_prob=args.scrambleProb,
            od=args.od,
            dimension=args.dimension,
            s_cone_peak=args.s_cone_peak,
            m_cone_peak=args.m_cone_peak,
            q_cone_peak=args.q_cone_peak,
            l_cone_peak=args.l_cone_peak,
            macula=args.macula,
            lens=args.lens,
            template=args.template
        )
    elif args.command == 'genotypes':
        generate_ishihara_grid_genotypes(
            output_dir=args.output_dir,
            measurement_dir=args.measurement_dir,
            grid_size=args.grid_size,
            metameric_axis=args.metameric_axis,
            display_type=args.display_type,
            num_plates=args.num_plates,
            luminance=args.luminance,
            chroma=args.chroma,
            top_percentage=args.top_percentage,
            sex=args.sex,
            peak_to_test=args.peak_to_test
        )
    else:
        parser.print_help()
