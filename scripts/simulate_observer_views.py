#!/usr/bin/env python3
"""
Simulate what different observer genotypes see when viewing DISP_6P images in sRGB.

This script takes an image encoded in DISP_6P format and converts it to sRGB for various
observer genotypes (trichromats or dichromats), showing how different genetic variants of color vision
perceive the same stimulus.

The workflow is:
1. Load DISP_6P image (6-channel display weights)
2. Load display primaries (spectral power distributions of the display used)
3. For each observer genotype:
   a. Compute spectral radiances from display weights
   b. Observer views radiances → cone responses
   c. Convert cone responses → sRGB for that observer
   d. Save resulting sRGB image

Usage:
    python simulate_observer_views.py <input_base_filename> --primaries-dir <dir> [options]

The input should be specified without the _RGB.png and _OCV.png suffixes.
For example, if you have:
    - image_RGB.png
    - image_OCV.png
Then use: python simulate_observer_views.py image --primaries-dir measurements/2025-05-21/primaries

Note: The primaries directory should contain measurements in the format expected by
load_primaries_from_csv (typically RGBO measurements at full intensity).
"""

import numpy as np
import argparse
import os
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt

from TetriumColor.Observer import Observer, Spectra
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Measurement import load_primaries_from_csv


def load_disp_6p_image(base_filename: str, mode: str = 'rgb-ocv') -> np.ndarray:
    """
    Load a DISP_6P image from RGB and OCV files (or RGB RGB mode).

    Args:
        base_filename: Base filename without _RGB.png or _OCV.png suffix
        mode: 'rgb-ocv' for RGB-OCV pattern, 'rgb-rgb' for RGB RGB pattern

    Returns:
        numpy array of shape (height, width, 6) with values in [0, 1]
    """
    rgb_path = f"{base_filename}_RGB.png"

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")

    # Load RGB image
    rgb_img = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0

    # Handle grayscale images
    if len(rgb_img.shape) == 2:
        rgb_img = np.stack([rgb_img] * 3, axis=-1)

    if mode == 'rgb-rgb':
        # For RGB RGB mode, use RGB image for both frames
        disp_6p = np.concatenate([rgb_img, rgb_img], axis=-1)
    else:
        # For RGB-OCV mode, load OCV file
        ocv_path = f"{base_filename}_OCV.png"
        if not os.path.exists(ocv_path):
            raise FileNotFoundError(f"OCV file not found: {ocv_path}")
        ocv_img = np.array(Image.open(ocv_path)).astype(np.float32) / 255.0
        if len(ocv_img.shape) == 2:
            ocv_img = np.stack([ocv_img] * 3, axis=-1)
        disp_6p = np.concatenate([rgb_img, ocv_img], axis=-1)

    return disp_6p


def disp_6p_to_nchannel(weights_6channel: np.ndarray,
                        led_mapping: List[int]) -> np.ndarray:
    """
    Convert 6-channel display weights to N-channel DISP weights.

    The 6-channel format represents alternating frames (RGB-OCV or RGB RGB pattern).
    We average them to get N-channel weights.

    Args:
        weights_6channel: Display weights (N, 6) in [0, 1]
        led_mapping: How 6 channels map to N LEDs (e.g., [0,1,3,2,1,3] for RGBO or [0,1,2,0,1,2] for RGB)

    Returns:
        Array of shape (N, num_leds) with N-channel weights
    """
    n_pixels = weights_6channel.shape[0]
    num_leds = max(led_mapping) + 1
    weights_nchannel = np.zeros((n_pixels, num_leds))

    for led_idx in range(num_leds):
        # Find which 6D indices correspond to this LED
        indices = [i for i, x in enumerate(led_mapping) if x == led_idx]
        if len(indices) > 0:
            weights_nchannel[:, led_idx] = np.mean(weights_6channel[:, indices], axis=1)

    return weights_nchannel


def compute_spectral_radiances(weights_nchannel: np.ndarray,
                               primaries: List[Spectra]) -> np.ndarray:
    """
    Compute spectral radiances from display weights (vectorized).

    Args:
        weights_nchannel: Display weights (N, num_leds) in [0, 1]
        primaries: List of Spectra objects for primaries (RGBO or RGB)

    Returns:
        Array of shape (N, num_wavelengths) with spectral radiances
    """
    # Stack primary spectra into a matrix: (num_leds, num_wavelengths)
    primary_matrix = np.array([p.data for p in primaries])

    # Compute radiances: (N, num_leds) @ (num_leds, num_wavelengths) = (N, num_wavelengths)
    radiances = weights_nchannel @ primary_matrix

    return radiances


def convert_disp_6p_to_srgb_for_observer(disp_6p_image: np.ndarray,
                                         observer: Observer,
                                         display_primaries: List[Spectra],
                                         led_mapping: List[int] = [0, 1, 3, 2, 1, 3],
                                         scaling_factor: float = 1.0) -> np.ndarray:
    """
    Convert a DISP_6P image to sRGB for a specific observer (vectorized, fast).

    The process is:
    1. Convert DISP_6P weights → N-channel DISP weights (averaging frames)
    2. Compute spectral radiances from DISP weights and primaries (matrix multiply)
    3. Compute observer cone responses from radiances (matrix multiply)
    4. Normalize cone responses to [0, 1] range
    5. Convert cone responses → sRGB for this observer

    For dichromats (2D cone responses):
    - The conversion uses a 3x2 projection matrix (GetConeToXYZPrimaries) that maps
      the 2D cone space onto a 2D plane in 3D XYZ space, then to sRGB.
    - This is mathematically valid but results in colors constrained to a 2D manifold
      in sRGB space, representing what the dichromat actually perceives.
    - This is different from typical "color blindness simulation" approaches (e.g., 
      Brettel/Machado) which start from normal 3D vision and project down.

    Args:
        disp_6p_image: Image in DISP_6P format (H, W, 6) with values in [0, 1]
        observer: Observer object (can be dichromat with 2 cones or trichromat with 3)
        display_primaries: List of Spectra objects (RGBO or RGB primaries)
        led_mapping: How 6 channels map to N LEDs
        scaling_factor: Additional scaling factor for cone responses

    Returns:
        sRGB image as numpy array (H, W, 3) in [0, 1]
        Note: For dichromats, colors will be constrained to a 2D manifold in sRGB space.
    """
    # Get image shape
    height, width = disp_6p_image.shape[:2]

    # Reshape to (N, 6) for batch processing
    pixels_6p = disp_6p_image.reshape(-1, 6)

    # Convert 6-channel to N-channel (fast, no observer needed)
    pixels_nchannel = disp_6p_to_nchannel(pixels_6p, led_mapping)

    # Compute spectral radiances (vectorized matrix multiply)
    radiances = compute_spectral_radiances(pixels_nchannel, display_primaries)

    # Observer views radiances → cone responses (vectorized matrix multiply)
    # observer.sensor_matrix is shape (num_cones, num_wavelengths)
    # radiances is shape (N, num_wavelengths)
    # Result: (N, num_cones)
    cone_responses = radiances @ observer.sensor_matrix.T

    # Scale cone responses to a reasonable range
    if scaling_factor == 1.0:
        # Compute what white looks like (all display weights at 1.0)
        num_leds = len(display_primaries)
        white_nchannel = np.ones((1, num_leds))
        white_radiance = compute_spectral_radiances(white_nchannel, display_primaries)
        white_cone = white_radiance @ observer.sensor_matrix.T

        # Normalize each cone channel independently so white = [1, 1, 1, ...]
        # This preserves the white balance
        white_cone = white_cone[0]  # Shape: (num_cones,)
        if np.all(white_cone > 0):
            # Divide each cone channel by its white point value
            cone_responses = cone_responses / white_cone
        else:
            # Fallback: normalize by max if any channel is zero
            max_response = np.max(white_cone)
            if max_response > 0:
                cone_responses = cone_responses / max_response
    else:
        cone_responses = cone_responses * scaling_factor

    # Create a color space for this observer (no display primaries needed for CONE→sRGB)
    color_space = ColorSpace(observer)

    # Convert cone responses to sRGB
    pixels_srgb = color_space.convert(cone_responses,
                                      from_space=ColorSpaceType.CONE,
                                      to_space=ColorSpaceType.SRGB)

    # Reshape back to image
    srgb_image = pixels_srgb.reshape(height, width, 3)

    # Clip to valid range
    srgb_image = np.clip(srgb_image, 0, 1)

    return srgb_image


def format_genotype_name(genotype: Tuple[float, ...]) -> str:
    """Format genotype tuple for display."""
    peaks_str = '_'.join([f"{int(peak)}" for peak in genotype])
    return f"observer_{peaks_str}nm"


def save_srgb_image(srgb_image: np.ndarray, output_path: str):
    """Save an sRGB image."""
    # Convert to 8-bit
    img_8bit = (srgb_image * 255).astype(np.uint8)

    # Save
    Image.fromarray(img_8bit).save(output_path)
    print(f"Saved: {output_path}")


def create_comparison_figure(images: List[Tuple[str, np.ndarray]],
                             output_path: str,
                             title: str = "Observer Comparison"):
    """
    Create a comparison figure showing all observer views.

    Args:
        images: List of (label, image) tuples
        output_path: Path to save the figure
        title: Figure title
    """
    n_images = len(images)

    # Calculate grid dimensions
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(title, fontsize=16)

    # Handle single image case
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (label, image) in enumerate(images):
        axes[idx].imshow(image)
        axes[idx].set_title(label, fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate observer views of DISP_6P images in sRGB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python simulate_observer_views.py test_outputs/my_image \\
      --primaries-dir measurements/2025-05-21/primaries \\
      --num-observers 8 \\
      --create-comparison
  
This will load:
  - test_outputs/my_image_RGB.png
  - test_outputs/my_image_OCV.png
  - Display primaries from measurements/2025-05-21/primaries
  
And create sRGB versions for 8 different observers (trichromats for RGB-OCV mode, dichromats for RGB RGB mode).
        """
    )

    parser.add_argument('input_base', type=str,
                        help='Base filename (without _RGB.png or _OCV.png suffix)')
    parser.add_argument('--primaries-dir', type=str, required=True,
                        help='Directory containing display primary measurements')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <input_base>_observer_views)')
    parser.add_argument('--num-observers', type=int, default=8,
                        help='Number of most common observers to simulate (default: 8)')
    parser.add_argument('--wavelengths', type=str, default='360:831:1',
                        help='Wavelength range as start:stop:step (default: 360:831:1)')
    parser.add_argument('--create-comparison', action='store_true',
                        help='Create a comparison figure with all observers')
    parser.add_argument('--sex', type=str, default='male', choices=['male', 'female', 'both'],
                        help='Sex for observer genotype distribution (default: male)')
    parser.add_argument('--mode', type=str, default='rgb-ocv', choices=['rgb-ocv', 'rgb-rgb'],
                        help='Display mode: rgb-ocv for RGB-OCV pattern (4 LEDs), rgb-rgb for RGB RGB pattern (3 LEDs) (default: rgb-ocv)')
    parser.add_argument('--led-mapping', type=str, default=None,
                        help='LED mapping for 6P display as comma-separated values (default: auto-based on mode)')
    parser.add_argument('--scaling-factor', type=float, default=1.0,
                        help='Manual scaling factor for cone responses (default: 1.0 = auto-scale to white point)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information about cone responses and scaling')

    args = parser.parse_args()

    # Parse wavelengths
    wl_parts = args.wavelengths.split(':')
    wavelengths = np.arange(int(wl_parts[0]), int(wl_parts[1]), int(wl_parts[2]))

    # Set LED mapping based on mode if not provided
    if args.led_mapping is None:
        if args.mode == 'rgb-rgb':
            led_mapping = [0, 1, 2, 0, 1, 2]  # RGB RGB mode
        else:
            led_mapping = [0, 1, 3, 2, 1, 3]  # RGB-OCV mode (default)
    else:
        # Parse LED mapping
        led_mapping = [int(x) for x in args.led_mapping.split(',')]
        if len(led_mapping) != 6:
            raise ValueError("LED mapping must have exactly 6 values")

    # Set up output directory
    if args.output_dir is None:
        input_path = Path(args.input_base)
        args.output_dir = str(input_path.parent / f"{input_path.name}_observer_views")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load DISP_6P image
    print(f"\nLoading DISP_6P image from: {args.input_base} (mode: {args.mode})")
    disp_6p_image = load_disp_6p_image(args.input_base, mode=args.mode)
    print(f"Image shape: {disp_6p_image.shape}")

    # Load display primaries
    print(f"\nLoading display primaries from: {args.primaries_dir}")
    try:
        display_primaries = load_primaries_from_csv(args.primaries_dir)
        print(f"Loaded {len(display_primaries)} primaries")
        print(f"Wavelength range: {display_primaries[0].wavelengths[0]}-{display_primaries[0].wavelengths[-1]} nm")

        # For RGB RGB mode, only use first 3 primaries (RGB)
        if args.mode == 'rgb-rgb':
            if len(display_primaries) < 3:
                raise ValueError(f"RGB RGB mode requires at least 3 primaries, but only {len(display_primaries)} found")
            display_primaries = display_primaries[:3]
            print(f"Using first 3 primaries (RGB) for RGB RGB mode")

        # Resample primaries to match observer wavelengths if needed
        if not np.array_equal(display_primaries[0].wavelengths, wavelengths):
            print("Resampling primaries to match observer wavelengths...")
            from scipy.interpolate import interp1d
            resampled_primaries = []
            for primary in display_primaries:
                f = interp1d(primary.wavelengths, primary.data,
                             kind='linear', bounds_error=False, fill_value=0)
                resampled_data = f(wavelengths)
                resampled_primaries.append(Spectra(wavelengths=wavelengths, data=resampled_data))
            display_primaries = resampled_primaries

    except Exception as e:
        print(f"Error loading primaries: {e}")
        print("Make sure the primaries directory contains properly formatted measurement files")
        return

    # Initialize observer genotypes based on mode
    if args.mode == 'rgb-rgb':
        # For RGB RGB mode, use dichromats (dimension=1, 0-indexed)
        print(f"\nInitializing observer genotypes for dichromats (sex: {args.sex})...")
        observer_gen = ObserverGenotypes(wavelengths=wavelengths, dimensions=[2])
        observer_type = "dichromat"
    else:
        # For RGB-OCV mode, use trichromats (dimension=2, 0-indexed)
        print(f"\nInitializing observer genotypes for trichromats (sex: {args.sex})...")
        observer_gen = ObserverGenotypes(wavelengths=wavelengths, dimensions=[3])
        observer_type = "trichromat"

    # Get the most common genotypes
    genotypes = observer_gen.get_genotypes_by_probability(sex=args.sex)[:args.num_observers]
    probabilities = observer_gen.get_probabilities_by_genotype(sex=args.sex)[:args.num_observers]

    print(f"\nSimulating views for top {len(genotypes)} {observer_type} genotypes:")
    for i, (genotype, prob) in enumerate(zip(genotypes, probabilities)):
        print(f"  {i+1}. {genotype} nm - probability: {prob:.4f} ({prob*100:.2f}%)")

    # Process each observer genotype
    observer_images = []

    for i, genotype in enumerate(genotypes):
        print(f"\nProcessing observer {i+1}/{len(genotypes)}: {genotype} nm...")

        try:
            # Create observer for this genotype
            observer = observer_gen.get_observer_for_peaks(genotype)

            if args.debug and i == 0:
                # Debug first observer to check scaling
                print(f"  Observer dimension: {observer.dimension}")
                # Test white point
                num_leds = len(display_primaries)
                white_nchannel = np.ones((1, num_leds))
                white_radiance = compute_spectral_radiances(white_nchannel, display_primaries)
                white_cone = white_radiance @ observer.sensor_matrix.T
                print(f"  White point cone response (raw): {white_cone[0]}")
                # Should be [1, 1, 1, ...]
                print(f"  White point (after normalization): {white_cone[0] / white_cone[0]}")

            # Convert DISP_6P → sRGB for this observer
            srgb_image = convert_disp_6p_to_srgb_for_observer(
                disp_6p_image,
                observer,
                display_primaries,
                led_mapping,
                scaling_factor=args.scaling_factor
            )

            if args.debug and i == 0:
                print(f"  sRGB range: [{np.min(srgb_image):.4f}, {np.max(srgb_image):.4f}]")
                print(f"  sRGB mean: {np.mean(srgb_image):.4f}")

            # Save the image
            output_filename = format_genotype_name(genotype) + ".png"
            output_path = os.path.join(args.output_dir, output_filename)
            save_srgb_image(srgb_image, output_path)

            # Store for comparison figure
            label = f"{', '.join([str(int(p)) for p in genotype])}nm\n({probabilities[i]*100:.1f}%)"
            observer_images.append((label, srgb_image))

        except Exception as e:
            print(f"Error processing observer {genotype}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comparison figure if requested
    if args.create_comparison and observer_images:
        comparison_path = os.path.join(args.output_dir, "comparison.png")
        create_comparison_figure(
            observer_images,
            comparison_path,
            title=f"Observer Comparison: {Path(args.input_base).name}"
        )

    print(f"\nDone! Generated {len(observer_images)} observer views in: {args.output_dir}")


if __name__ == "__main__":
    main()
