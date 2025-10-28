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
from itertools import product, repeat

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
    return top_inks


def analyze_metamer_capability(point_cloud, cs, num_samples=1000):
    """
    Analyze how well an ink gamut can produce metamers (same LMS, different Q).

    Args:
        point_cloud: Point cloud in CONE (LMSQ) space
        cs: ColorSpace object
        num_samples: Number of LMS points to sample

    Returns:
        Dictionary with metamer statistics
    """
    # Extract LMS and Q from point cloud
    lms_points = point_cloud[:, [0, 1, 3]]  # LMS
    q_points = point_cloud[:, 2]      # 3rd channel is Q

    # Sample random LMS points from the gamut
    indices = np.random.choice(len(lms_points), min(num_samples, len(lms_points)), replace=False)
    sampled_lms = lms_points[indices]

    # For each sampled LMS, find all points with similar LMS but different Q
    metamer_ranges = []
    lms_threshold = 0.05  # Threshold for LMS similarity

    for target_lms in sampled_lms:
        # Find points with similar LMS values
        distances = np.linalg.norm(lms_points - target_lms, axis=1)
        similar_lms_mask = distances < lms_threshold

        if np.sum(similar_lms_mask) > 1:
            # Get Q range for these similar LMS points
            q_values = q_points[similar_lms_mask]
            q_range = np.max(q_values) - np.min(q_values)
            metamer_ranges.append(q_range)

    if len(metamer_ranges) == 0:
        return {
            'mean_q_range': 0.0,
            'median_q_range': 0.0,
            'max_q_range': 0.0,
            'samples_with_metamers': 0,
            'total_samples': len(sampled_lms)
        }

    return {
        'mean_q_range': np.mean(metamer_ranges),
        'median_q_range': np.median(metamer_ranges),
        'max_q_range': np.max(metamer_ranges),
        'samples_with_metamers': len(metamer_ranges),
        'total_samples': len(sampled_lms)
    }


def analyze_hue_coverage(
    point_cloud,
    cs,
    luminance_levels=[0.7, 1.0, 1.3],
    saturation_levels=[0.1, 0.2, 0.3],
    num_hue_bins=18
):
    """
    Analyze hue sphere coverage at different luminance and saturation levels,
    using two spherical hue angles.

    Args:
        point_cloud: Point cloud in CONE space
        cs: ColorSpace object
        luminance_levels: List of luminance values to test
        saturation_levels: List of saturation values to test
        num_hue_bins: Number of bins per hue angle (θ, φ), so total bins = num_hue_bins**2

    Returns:
        Dictionary with hue sphere coverage statistics and VSH points
    """

    # Convert point cloud to VSH (Value-Saturation-Hue1-Hue2) space
    vsh_points = cs.convert(point_cloud, ColorSpaceType.CONE, ColorSpaceType.VSH)

    # vsh_points shape: (N, 4) columns: [V, r, theta, phi]
    # V = luminance
    # r = saturation (radius)
    # theta = azimuthal angle in RADIANS (-π to π)
    # phi = polar angle in RADIANS (0 to π)
    values = vsh_points[:, 0]
    saturations = vsh_points[:, 1]
    theta_rad = vsh_points[:, 2]  # azimuthal angle in radians
    phi_rad = vsh_points[:, 3]    # polar angle in radians

    # Convert to degrees and normalize theta to [0, 360)
    theta_deg = np.rad2deg(theta_rad) % 360  # 0 to 360 degrees
    phi_deg = np.rad2deg(phi_rad)             # 0 to 180 degrees

    coverage_results = {}

    for lum in luminance_levels:
        for sat in saturation_levels:
            # Find points near this luminance and saturation
            lum_threshold = 0.1
            sat_threshold = 0.1

            mask = (np.abs(values - lum) < lum_threshold) & (np.abs(saturations - sat) < sat_threshold)

            if np.sum(mask) > 0:
                # Get hue angles at this luminance/saturation
                theta_at_ls = theta_deg[mask]
                phi_at_ls = phi_deg[mask]

                # 2D histogram: [theta, phi], both in degrees
                theta_bins = np.linspace(0, 360, num_hue_bins + 1)
                phi_bins = np.linspace(0, 180, num_hue_bins + 1)
                hue_counts, _, _ = np.histogram2d(theta_at_ls, phi_at_ls, bins=[theta_bins, phi_bins])

                # Each bin: has at least one color?
                occupied_bins = np.sum(hue_counts > 0)
                total_bins = num_hue_bins * num_hue_bins
                coverage_percent = (occupied_bins / total_bins) * 100

                coverage_results[f'L{lum:.1f}_S{sat:.1f}'] = {
                    'coverage_percent': coverage_percent,
                    'occupied_bins': occupied_bins,
                    'total_bins': total_bins,
                    'num_colors': np.sum(mask),
                    'hue_counts': hue_counts,
                    'h1_bins': theta_bins,
                    'h2_bins': phi_bins
                }
            else:
                total_bins = num_hue_bins * num_hue_bins
                coverage_results[f'L{lum:.1f}_S{sat:.1f}'] = {
                    'coverage_percent': 0.0,
                    'occupied_bins': 0,
                    'total_bins': total_bins,
                    'num_colors': 0,
                    'hue_counts': np.zeros((num_hue_bins, num_hue_bins)),
                    'h1_bins': np.linspace(0, 360, num_hue_bins + 1),
                    'h2_bins': np.linspace(0, 180, num_hue_bins + 1)
                }

    return coverage_results, vsh_points


def plot_hue_coverage(hue_coverage, results_dir, library_name, vsh_points, cs):
    """
    Create visualization plots for hue coverage analysis.
    This version visualizes hue coverage as occupied points on a sphere surface (using 2D histogram H1/H2 as longitude/latitude).
    Uses actual sRGB colors from the color space conversion.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import numpy as np
    import os

    # Extract VSH components
    values = vsh_points[:, 0]
    saturations = vsh_points[:, 1]
    theta_rad = vsh_points[:, 2]
    phi_rad = vsh_points[:, 3]

    # Convert angles to degrees
    theta_deg = np.rad2deg(theta_rad) % 360
    phi_deg = np.rad2deg(phi_rad)

    # Convert all VSH points to sRGB for color lookup
    srgb_colors = cs.convert(vsh_points, ColorSpaceType.VSH, ColorSpaceType.SRGB)
    srgb_colors = np.clip(srgb_colors, 0, 1)  # Clip to valid sRGB range

    # Set up figure: one 3D subplot per luminance/saturation combo, up to 9
    num_conditions = len(hue_coverage)
    fig = plt.figure(figsize=(18, 18))
    nrows, ncols = 3, 3

    for idx, (key, stats) in enumerate(sorted(hue_coverage.items())):
        if idx >= nrows * ncols:
            break
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')

        # Extract luminance and saturation from key
        parts = key.split('_')
        lum = float(parts[0][1:])
        sat = float(parts[1][1:])

        # Find points that match this luminance and saturation
        lum_threshold = 0.1
        sat_threshold = 0.1
        mask = (np.abs(values - lum) < lum_threshold) & (np.abs(saturations - sat) < sat_threshold)

        if np.sum(mask) == 0:
            # No points in this region, skip
            ax.text(0, 0, 0, 'No colors\nin this region', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            lum_sat = key.replace('L', 'L=').replace('_S', ', S=')
            ax.set_title(f"{lum_sat}\nCoverage: 0.0%", fontsize=11)
            continue

        # Prepare bins and counts
        h1_bins = stats['h1_bins']  # 0-360 deg (longitude)
        h2_bins = stats['h2_bins']  # 0-180 deg (latitude)
        hue_counts = stats['hue_counts']

        # Get the center value of each bin for plotting
        h1_centers = 0.5 * (h1_bins[:-1] + h1_bins[1:])
        h2_centers = 0.5 * (h2_bins[:-1] + h2_bins[1:])
        H1, H2 = np.meshgrid(h1_centers, h2_centers, indexing='ij')

        # Mask for bins with at least one color
        occupied_mask = (hue_counts > 0)

        # For each occupied bin, find representative color
        colors_for_bins = []
        positions = []

        for i in range(len(h1_centers)):
            for j in range(len(h2_centers)):
                if hue_counts[i, j] > 0:
                    # Find points in this bin
                    bin_mask = mask & \
                        (theta_deg >= h1_bins[i]) & (theta_deg < h1_bins[i+1]) & \
                        (phi_deg >= h2_bins[j]) & (phi_deg < h2_bins[j+1])

                    if np.sum(bin_mask) > 0:
                        # Use the mean color of points in this bin
                        mean_color = np.mean(srgb_colors[bin_mask], axis=0)
                        colors_for_bins.append(mean_color)

                        # Convert bin center to Cartesian coordinates
                        h1_rad = np.deg2rad(h1_centers[i])
                        h2_rad = np.deg2rad(h2_centers[j])
                        x = np.sin(h2_rad) * np.cos(h1_rad)
                        y = np.sin(h2_rad) * np.sin(h1_rad)
                        z = np.cos(h2_rad)
                        positions.append([x, y, z])

        if len(positions) > 0:
            positions = np.array(positions)
            colors_for_bins = np.array(colors_for_bins)

            # Plot occupied bins with actual colors
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c=colors_for_bins, s=100, edgecolor='k', linewidth=0.5, alpha=0.9)

        # Draw sphere wireframe for context
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color='lightgray', linewidth=0.3, alpha=0.35)

        lum_sat = key.replace('L', 'L=').replace('_S', ', S=')
        ax.set_title(f"{lum_sat}\nCoverage: {stats['coverage_percent']:.1f}%", fontsize=11)
        # Remove axes for visual clarity, but keep aspect ratio
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    # Hide unused subplots
    for unused in range(num_conditions, nrows * ncols):
        fig.add_subplot(nrows, ncols, unused+1, projection='3d').axis('off')

    plt.suptitle(f"Hue Sphere Coverage Analysis - {library_name}", fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(os.path.join(
        results_dir, f"hue_sphere_coverage_visualization_{library_name}.png"), dpi=180, bbox_inches='tight')
    plt.close()


def plot_aggregated_hue_coverage_at_ls(all_gamut_data, results_dir, library_name, target_lum=0.7, target_sat=0.1):
    """
    Create an aggregated plot showing hue sphere coverage for top 10 gamuts at a specific L/S condition.

    Args:
        all_gamut_data: List of dicts containing point_cloud, vsh_points, rank, volume, ink_names for each gamut
        results_dir: Output directory
        library_name: Name of the library
        target_lum: Target luminance value
        target_sat: Target saturation value
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(20, 8))
    nrows, ncols = 2, 5

    for idx, gamut_data in enumerate(all_gamut_data[:10]):  # Top 10
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

        vsh_points = gamut_data['vsh_points']
        cs = gamut_data['cs']
        rank = gamut_data['rank']
        volume = gamut_data['volume']
        ink_names = gamut_data['ink_names']

        # Extract VSH components
        values = vsh_points[:, 0]
        saturations = vsh_points[:, 1]
        theta_rad = vsh_points[:, 2]
        phi_rad = vsh_points[:, 3]

        # Convert angles to degrees
        theta_deg = np.rad2deg(theta_rad) % 360
        phi_deg = np.rad2deg(phi_rad)

        # Find points near target luminance and saturation
        lum_threshold = 0.1
        sat_threshold = 0.05
        mask = (np.abs(values - target_lum) < lum_threshold) & (np.abs(saturations - target_sat) < sat_threshold)

        if np.sum(mask) == 0:
            ax.text(0, 0, 0, 'No colors\nat this L/S', ha='center', va='center', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"Rank {rank}\n{', '.join(ink_names[:2])}...\nVol: {volume:.3f}", fontsize=8)
            continue

        # Get points at this L/S
        theta_at_ls = theta_deg[mask]
        phi_at_ls = phi_deg[mask]

        # Convert all VSH points to sRGB for coloring
        srgb_colors = cs.convert(vsh_points, ColorSpaceType.VSH, ColorSpaceType.SRGB)
        srgb_colors = np.clip(srgb_colors, 0, 1)
        colors_at_ls = srgb_colors[mask]

        # Convert angles to Cartesian coordinates on unit sphere
        theta_rad_ls = np.deg2rad(theta_at_ls)
        phi_rad_ls = np.deg2rad(phi_at_ls)

        x = np.sin(phi_rad_ls) * np.cos(theta_rad_ls)
        y = np.sin(phi_rad_ls) * np.sin(theta_rad_ls)
        z = np.cos(phi_rad_ls)

        # Plot points with actual colors
        ax.scatter(x, y, z, c=colors_at_ls, s=20, alpha=0.8, edgecolor='none')

        # Draw sphere wireframe for context
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color='lightgray', linewidth=0.2, alpha=0.2)

        # Calculate coverage percentage
        num_hue_bins = 36
        theta_bins = np.linspace(0, 360, num_hue_bins + 1)
        phi_bins = np.linspace(0, 180, num_hue_bins + 1)
        hue_counts, _, _ = np.histogram2d(theta_at_ls, phi_at_ls, bins=[theta_bins, phi_bins])
        occupied_bins = np.sum(hue_counts > 0)
        total_bins = num_hue_bins * num_hue_bins
        coverage_percent = (occupied_bins / total_bins) * 100

        # Title with rank, volume, and coverage
        ax.set_title(f"Rank {rank}: {coverage_percent:.1f}%\nVol: {volume:.3f}\n{', '.join(ink_names[:2])}...",
                     fontsize=8, pad=5)

        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    plt.suptitle(f'Hue Sphere Coverage at L={target_lum}, S={target_sat} - Top 10 Gamuts\n{library_name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(results_dir, f"top10_hue_coverage_L{target_lum}_S{target_sat}_{library_name}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created aggregated hue coverage plot at L={target_lum}, S={target_sat}")


def create_comparison_plots(all_metamer_stats, all_hue_coverage_summary, results_dir, library_name):
    """
    Create comparison plots for the top 10 gamuts.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ranks = [s['rank'] for s in all_metamer_stats]

    # Plot 1: Metamer capability metrics
    ax = axes[0, 0]
    ax.plot(ranks, [s['mean_q_range'] for s in all_metamer_stats], 'o-', label='Mean Q Range', linewidth=2)
    ax.plot(ranks, [s['median_q_range'] for s in all_metamer_stats], 's-', label='Median Q Range', linewidth=2)
    ax.plot(ranks, [s['max_q_range'] for s in all_metamer_stats], '^-', label='Max Q Range', linewidth=2)
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Q Range', fontsize=12)
    ax.set_title('Metamer Capability by Rank', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    # Plot 2: Metamer success rate
    ax = axes[0, 1]
    success_rates = [100 * s['samples_with_metamers'] / max(1, s['total_samples']) for s in all_metamer_stats]
    bars = ax.bar(ranks, success_rates, alpha=0.7, edgecolor='black')
    # Color bars by gradient
    for i, bar in enumerate(bars):
        bar.set_facecolor(plt.cm.viridis(i / len(bars)))
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Metamer Success Rate by Rank', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(ranks)
    ax.set_ylim([0, 100])

    # Plot 3: Hue coverage metrics
    ax = axes[1, 0]
    ax.plot(ranks, [s['avg_coverage'] for s in all_hue_coverage_summary], 'o-', label='Average', linewidth=2)
    ax.plot(ranks, [s['min_coverage'] for s in all_hue_coverage_summary], 's-', label='Minimum', linewidth=2)
    ax.plot(ranks, [s['max_coverage'] for s in all_hue_coverage_summary], '^-', label='Maximum', linewidth=2)
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Hue Coverage by Rank', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)
    ax.set_ylim([0, 100])

    # Plot 4: Volume vs Metamer Capability
    ax = axes[1, 1]
    volumes = [s['volume'] for s in all_metamer_stats]
    mean_q_ranges = [s['mean_q_range'] for s in all_metamer_stats]
    scatter = ax.scatter(volumes, mean_q_ranges, c=ranks, s=100, cmap='viridis', edgecolor='black', linewidth=1.5)

    # Add rank labels to points
    for i, (v, q, r) in enumerate(zip(volumes, mean_q_ranges, ranks)):
        ax.annotate(f'{r}', (v, q), fontsize=9, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel('Gamut Volume', fontsize=12)
    ax.set_ylabel('Mean Q Range', fontsize=12)
    ax.set_title('Volume vs Metamer Capability', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank', fontsize=10)

    plt.suptitle(f'Top 10 Gamuts Comparison - {library_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"top10_comparison_{library_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Create a second figure for detailed hue coverage comparison
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, summary in enumerate(all_hue_coverage_summary):
        ax = axes[i]

        # Extract coverage data for each L/S combination
        conditions = sorted(summary['hue_coverage'].keys())
        coverages = [summary['hue_coverage'][cond]['coverage_percent'] for cond in conditions]

        # Create bar plot
        x_pos = np.arange(len(conditions))
        bars = ax.bar(x_pos, coverages, alpha=0.7, edgecolor='black')

        # Color bars by coverage level
        for bar, cov in zip(bars, coverages):
            bar.set_facecolor(plt.cm.RdYlGn(cov / 100))

        ax.set_ylim([0, 100])
        ax.set_ylabel('Coverage (%)', fontsize=9)
        ax.set_title(f"Rank {summary['rank']}\nVol: {summary['volume']:.3f}", fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace('L', '').replace('_S', '/') for c in conditions],
                           rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Hue Coverage Across L/S Conditions - {library_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir, f"top10_hue_coverage_detailed_{library_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_metamer_distribution(point_cloud, metamer_stats, results_dir, library_name):
    """
    Create visualization for metamer capability.
    """
    import matplotlib.pyplot as plt

    # Extract LMS and Q from point cloud
    lms_points = point_cloud[:, [0, 1, 3]]
    q_points = point_cloud[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Q vs L
    axes[0, 0].scatter(lms_points[:, 0], q_points, alpha=0.1, s=1)
    axes[0, 0].set_xlabel('L (Long wavelength)')
    axes[0, 0].set_ylabel('Q (Tetrachromatic)')
    axes[0, 0].set_title('Q vs L')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Q vs M
    axes[0, 1].scatter(lms_points[:, 1], q_points, alpha=0.1, s=1)
    axes[0, 1].set_xlabel('M (Medium wavelength)')
    axes[0, 1].set_ylabel('Q (Tetrachromatic)')
    axes[0, 1].set_title('Q vs M')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Q vs S
    axes[1, 0].scatter(lms_points[:, 2], q_points, alpha=0.1, s=1)
    axes[1, 0].set_xlabel('S (Short wavelength)')
    axes[1, 0].set_ylabel('Q (Tetrachromatic)')
    axes[1, 0].set_title('Q vs S')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    axes[1, 1].axis('off')
    stats_text = f"""Metamer Printing Capability
    
Mean Q Range:    {metamer_stats['mean_q_range']:.4f}
Median Q Range:  {metamer_stats['median_q_range']:.4f}
Max Q Range:     {metamer_stats['max_q_range']:.4f}

Samples with Metamers: {metamer_stats['samples_with_metamers']}
Total Samples:         {metamer_stats['total_samples']}
Success Rate:          {100 * metamer_stats['samples_with_metamers'] / max(1, metamer_stats['total_samples']):.1f}%

Higher Q range indicates better
metamer printing capability."""

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f"Metamer Analysis - {library_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"metamer_visualization_{library_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def analyze_single_inkset(ink_names, inkset_library, paper_spectra, d65, tetrachromat, cs,
                          stepsize=0.1, num_samples=1000, save_visualizations=False,
                          results_dir=None, rank_suffix=""):
    """
    Analyze a single ink combination for metamer capability and hue coverage.

    Args:
        ink_names: List of ink names to analyze
        inkset_library: InkLibrary object containing all inks
        paper_spectra: Paper spectra
        d65: D65 illuminant
        tetrachromat: Tetrachromat observer
        cs: ColorSpace object
        stepsize: Grid resolution for point cloud generation
        num_samples: Number of samples for metamer analysis
        save_visualizations: Whether to save detailed visualization plots
        results_dir: Directory to save results (required if save_visualizations=True)
        rank_suffix: Suffix for filenames (e.g., "_rank1")

    Returns:
        Dictionary with keys: metamer_stats, hue_coverage, vsh_points, point_cloud
    """
    print(f"Analyzing: {', '.join(ink_names)}")

    # Create gamut
    inks = [inkset_library.library[ink_name] for ink_name in ink_names]
    gamut = InkGamut(inks, paper_spectra, d65)

    # Generate point cloud
    values = np.arange(0, 1 + stepsize, stepsize)
    grid = np.array(list(product(values, repeat=len(ink_names))))
    point_cloud, percentages = gamut.get_point_cloud(tetrachromat, grid=grid)

    print(f"  Generated {len(point_cloud)} points")

    # Analyze metamer capability
    metamer_stats = analyze_metamer_capability(point_cloud, cs, num_samples=num_samples)
    print(f"  Metamer - Mean Q Range: {metamer_stats['mean_q_range']:.4f}, "
          f"Success Rate: {100 * metamer_stats['samples_with_metamers'] / max(1, metamer_stats['total_samples']):.1f}%")

    # Analyze hue coverage
    hue_coverage, vsh_points = analyze_hue_coverage(point_cloud, cs,
                                                    luminance_levels=[0.7, 1.0, 1.3],
                                                    saturation_levels=[0.1, 0.2, 0.3],
                                                    num_hue_bins=36)

    # Calculate summary statistics
    coverages = [stats['coverage_percent'] for stats in hue_coverage.values() if stats['num_colors'] > 0]
    if coverages:
        avg_coverage = np.mean(coverages)
        min_coverage = np.min(coverages)
        max_coverage = np.max(coverages)
    else:
        avg_coverage = min_coverage = max_coverage = 0.0

    print(f"  Hue Coverage - Average: {avg_coverage:.1f}%, Min: {min_coverage:.1f}%, Max: {max_coverage:.1f}%")

    # Save visualizations if requested
    if save_visualizations and results_dir:
        library_name = rank_suffix.replace("_rank", "")
        plot_metamer_distribution(point_cloud, metamer_stats, results_dir,
                                  f"{library_name}{rank_suffix}")
        plot_hue_coverage(hue_coverage, results_dir, f"{library_name}{rank_suffix}",
                          vsh_points, cs)

    return {
        'metamer_stats': metamer_stats,
        'hue_coverage': hue_coverage,
        'vsh_points': vsh_points,
        'point_cloud': point_cloud,
        'avg_coverage': avg_coverage,
        'min_coverage': min_coverage,
        'max_coverage': max_coverage
    }


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

        # Set up observer and color space (needed for all modes)
        d65 = Illuminant.get("d65")
        tetrachromat = Observer.tetrachromat(wavelengths=np.arange(400, 710, 10))
        cs = ColorSpace(tetrachromat)

        # Check if user wants to analyze specific inks directly
        analyze_inks = args.analyze_inks if hasattr(args, 'analyze_inks') and args.analyze_inks else []

        if analyze_inks:
            # Direct analysis mode: analyze specified inks without search
            print(f"\n{'='*60}")
            print(f"Direct Analysis Mode")
            print(f"{'='*60}\n")

            # Validate inks exist
            missing_inks = [ink for ink in analyze_inks if ink not in inkset_library.library]
            if missing_inks:
                raise ValueError(f"Inks not found in library: {', '.join(missing_inks)}")

            # Create results directory
            analyze_inks_nospaces = [ink.replace(' ', '') for ink in analyze_inks]
            results_dir = f"results/{library_name}_analyze_{'_'.join(analyze_inks_nospaces)}"
            os.makedirs(results_dir, exist_ok=True)
            print(f"Created results directory: {results_dir}")

            # Analyze the specified inkset
            results = analyze_single_inkset(
                ink_names=analyze_inks,
                inkset_library=inkset_library,
                paper_spectra=paper_spectra,
                d65=d65,
                tetrachromat=tetrachromat,
                cs=cs,
                stepsize=args.stepsize,
                num_samples=1000,
                save_visualizations=True,
                results_dir=results_dir,
                rank_suffix=""
            )

            # Save summary report
            with open(os.path.join(results_dir, f"analysis_report_{'_'.join(analyze_inks_nospaces)}.txt"), 'w') as f:
                f.write(f"Direct Analysis Report: {library_name}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Analyzed Inks ({len(analyze_inks)}):\n")
                for i, ink_name in enumerate(analyze_inks, 1):
                    f.write(f"  {i}. {ink_name}\n")
                f.write("\n")

                metamer_stats = results['metamer_stats']
                f.write("Metamer Printing Capability:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Mean Q Range:        {metamer_stats['mean_q_range']:.4f}\n")
                f.write(f"  Median Q Range:      {metamer_stats['median_q_range']:.4f}\n")
                f.write(f"  Max Q Range:         {metamer_stats['max_q_range']:.4f}\n")
                f.write(
                    f"  Samples w/ Metamers: {metamer_stats['samples_with_metamers']}/{metamer_stats['total_samples']}\n")
                f.write(
                    f"  Success Rate:        {100 * metamer_stats['samples_with_metamers'] / max(1, metamer_stats['total_samples']):.1f}%\n\n")

                f.write("Hue Sphere Coverage:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Average Coverage:    {results['avg_coverage']:.2f}%\n")
                f.write(f"  Min Coverage:        {results['min_coverage']:.2f}%\n")
                f.write(f"  Max Coverage:        {results['max_coverage']:.2f}%\n\n")

                f.write("Coverage by Luminance/Saturation:\n")
                for key, stats in sorted(results['hue_coverage'].items()):
                    lum_sat = key.replace('L', 'L=').replace('_S', ', S=')
                    f.write(f"  {lum_sat}: {stats['coverage_percent']:.1f}% ({stats['num_colors']} colors)\n")

            print(f"\n{'='*60}")
            print(f"Direct analysis complete!")
            print(f"Results saved in {results_dir}/")
            print(f"{'='*60}\n")
            return

        # Standard search mode
        # Handle fixed inks
        fixed_inks = args.fixed_inks if hasattr(args, 'fixed_inks') else []
        if fixed_inks:
            print(f"\n{'='*60}")
            print(f"Fixed inks: {', '.join(fixed_inks)}")
            print(f"Searching for {args.k - len(fixed_inks)} complementary inks")
            print(f"{'='*60}\n")

        # Create results directory for this inkset
        fixed_inks_nospaces = [ink.replace(' ', '') for ink in fixed_inks]
        fixed_suffix = f"_fixed{'_'.join(fixed_inks_nospaces)}" if fixed_inks_nospaces else ""
        results_dir = f"results/{library_name}_k{args.k}{fixed_suffix}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

        # Save all ink spectra plot
        plot_inks_by_hue(inkset_library.library, np.arange(400, 710, 10),
                         filename=os.path.join(results_dir, f"all_ink_spectras_{library_name}.png"))

        # Perform convex hull search with optional fixed inks
        top_volumes_all_inks = inkset_library.convex_hull_search(
            tetrachromat, d65, top=args.top_volume_k, k=args.k, fixed_inks=fixed_inks or None
        )

        # Save top volumes as CSV
        save_top_inks_as_csv(top_volumes_all_inks, os.path.join(
            results_dir, f"top_volumes_k{args.k}_{library_name}.csv"))

        # Show top k combinations (this creates plots)
        show_top_k_combinations(top_volumes_all_inks, inkset_library.library, k=16, filename=os.path.join(
            results_dir, f"top_k_combinations_k{args.k}_{library_name}.png"))

        best4_inks = [inkset_library.library[ink_name] for ink_name in top_volumes_all_inks[0][1]]

        top_d_inks = get_top_n_inks(top_volumes_all_inks, top_d_inks=args.top_d_inks)

        # Plot all top 20 inks by hue
        plot_inks_by_hue(
            {ink_name: inkset_library.library[ink_name] for ink_name in top_d_inks},
            np.arange(400, 710, 10),
            filename=os.path.join(results_dir, f"top_{args.top_d_inks}_ink_spectras_{library_name}.png")
        )

        # Save top 20 inks list as text file
        with open(os.path.join(results_dir, f"top_{args.top_d_inks}_inks_{library_name}.txt"), 'w') as f:
            if fixed_inks:
                f.write(f"Fixed inks: {', '.join(fixed_inks)}\n")
                f.write(f"Searching for {args.k - len(fixed_inks)} complementary inks\n\n")
            f.write(f"Top {args.top_d_inks} inks for {library_name} inkset:\n")
            f.write("=" * 50 + "\n")
            for i, ink_name in enumerate(top_d_inks, 1):
                marker = " [FIXED]" if ink_name in fixed_inks else ""
                f.write(f"{i:2d}. {ink_name}{marker}\n")

        cs = ColorSpace(tetrachromat)

        # Analyze top 10 gamuts
        print(f"\n{'='*60}")
        print(f"Analyzing Top 10 Gamuts for Metamer and Hue Coverage")
        print(f"{'='*60}\n")

        num_gamuts_to_analyze = min(10, len(top_volumes_all_inks))
        all_metamer_stats = []
        all_hue_coverage_summary = []
        all_gamut_data = []  # Store data for aggregated visualization

        for rank in range(num_gamuts_to_analyze):
            volume, ink_names = top_volumes_all_inks[rank]
            print(f"\n--- Rank {rank + 1} (Volume: {volume:.4f}) ---")

            # Mark fixed vs optimized inks
            if fixed_inks:
                optimized_inks = [ink for ink in ink_names if ink not in fixed_inks]
                print(f"Fixed inks: {', '.join(fixed_inks)}")
                print(f"Optimized inks: {', '.join(optimized_inks)}")
            else:
                print(f"Inks: {', '.join(ink_names)}")

            # Analyze this inkset
            results = analyze_single_inkset(
                ink_names=ink_names,
                inkset_library=inkset_library,
                paper_spectra=inkset_library.get_paper(),
                d65=d65,
                tetrachromat=tetrachromat,
                cs=cs,
                stepsize=args.stepsize,
                num_samples=1000,
                save_visualizations=(rank < 3),  # Save detailed visualizations for top 3
                results_dir=results_dir,
                rank_suffix=f"_rank{rank + 1}"
            )

            # Extract results
            metamer_stats = results['metamer_stats']
            hue_coverage = results['hue_coverage']
            vsh_points = results['vsh_points']
            point_cloud = results['point_cloud']

            # Add rank and volume to metamer stats
            metamer_stats['rank'] = rank + 1
            metamer_stats['volume'] = volume
            metamer_stats['ink_names'] = ink_names
            all_metamer_stats.append(metamer_stats)

            # Store data for aggregated visualization
            all_gamut_data.append({
                'point_cloud': point_cloud,
                'vsh_points': vsh_points,
                'cs': cs,
                'rank': rank + 1,
                'volume': volume,
                'ink_names': ink_names
            })

            # Store hue coverage summary
            all_hue_coverage_summary.append({
                'rank': rank + 1,
                'volume': volume,
                'ink_names': ink_names,
                'avg_coverage': results['avg_coverage'],
                'min_coverage': results['min_coverage'],
                'max_coverage': results['max_coverage'],
                'hue_coverage': hue_coverage
            })

        # Save comprehensive comparison report
        print(f"\nGenerating comparison reports...")

        with open(os.path.join(results_dir, f"top10_metamer_comparison_{library_name}.txt"), 'w') as f:
            f.write(f"Metamer Printing Capability - Top 10 Gamuts Comparison\n")
            f.write(f"Library: {library_name}\n")
            if fixed_inks:
                f.write(f"Fixed inks: {', '.join(fixed_inks)}\n")
                f.write(f"Optimizing for {args.k - len(fixed_inks)} complementary inks\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Rank':<6} {'Volume':<10} {'Mean Q':<10} {'Median Q':<10} {'Max Q':<10} {'Success':<10} {'Inks'}\n")
            f.write("-" * 80 + "\n")

            for stats in all_metamer_stats:
                success_rate = 100 * stats['samples_with_metamers'] / max(1, stats['total_samples'])
                ink_str = ', '.join(stats['ink_names'])
                f.write(f"{stats['rank']:<6} {stats['volume']:<10.4f} {stats['mean_q_range']:<10.4f} "
                        f"{stats['median_q_range']:<10.4f} {stats['max_q_range']:<10.4f} {success_rate:<10.1f} {ink_str}\n")

            f.write("\n\nDetailed Statistics:\n")
            f.write("=" * 80 + "\n\n")

            for stats in all_metamer_stats:
                f.write(f"Rank {stats['rank']}: {', '.join(stats['ink_names'])}\n")
                f.write(f"  Volume:              {stats['volume']:.4f}\n")
                f.write(f"  Mean Q Range:        {stats['mean_q_range']:.4f}\n")
                f.write(f"  Median Q Range:      {stats['median_q_range']:.4f}\n")
                f.write(f"  Max Q Range:         {stats['max_q_range']:.4f}\n")
                f.write(f"  Samples w/ Metamers: {stats['samples_with_metamers']}/{stats['total_samples']}\n")
                f.write(
                    f"  Success Rate:        {100 * stats['samples_with_metamers'] / max(1, stats['total_samples']):.1f}%\n\n")

        with open(os.path.join(results_dir, f"top10_hue_coverage_comparison_{library_name}.txt"), 'w') as f:
            f.write(f"Hue Sphere Coverage - Top 10 Gamuts Comparison\n")
            f.write(f"Library: {library_name}\n")
            if fixed_inks:
                f.write(f"Fixed inks: {', '.join(fixed_inks)}\n")
                f.write(f"Optimizing for {args.k - len(fixed_inks)} complementary inks\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Rank':<6} {'Volume':<10} {'Avg Cov':<10} {'Min Cov':<10} {'Max Cov':<10} {'Inks'}\n")
            f.write("-" * 80 + "\n")

            for summary in all_hue_coverage_summary:
                ink_str = ', '.join(summary['ink_names'])
                f.write(f"{summary['rank']:<6} {summary['volume']:<10.4f} {summary['avg_coverage']:<10.1f} "
                        f"{summary['min_coverage']:<10.1f} {summary['max_coverage']:<10.1f} {ink_str}\n")

            f.write("\n\nDetailed Coverage by Luminance/Saturation:\n")
            f.write("=" * 80 + "\n\n")

            for summary in all_hue_coverage_summary:
                f.write(f"Rank {summary['rank']}: {', '.join(summary['ink_names'])}\n")
                f.write(f"  Volume: {summary['volume']:.4f}\n\n")

                for key, stats in sorted(summary['hue_coverage'].items()):
                    lum_sat = key.replace('L', 'Luminance ').replace('_S', ', Saturation ')
                    f.write(f"  {lum_sat}:\n")
                    f.write(f"    Hue Coverage:     {stats['coverage_percent']:.1f}%\n")
                    f.write(f"    Occupied Bins:    {stats['occupied_bins']}/{stats['total_bins']}\n")
                    f.write(f"    Colors in Region: {stats['num_colors']}\n")
                f.write("\n")

        # Create comparison plots
        create_comparison_plots(all_metamer_stats, all_hue_coverage_summary, results_dir, library_name)

        # Create aggregated hue coverage plot at L=0.7, S=0.1
        print(f"\nGenerating aggregated hue coverage visualization...")
        plot_aggregated_hue_coverage_at_ls(all_gamut_data, results_dir, library_name,
                                           target_lum=0.7, target_sat=0.1)

        print(f"\n{'='*60}")
        print(f"Top 10 gamut analysis complete!")
        print(f"{'='*60}\n")

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
                point_cloud[::100], ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:], mode="sphere")
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

        print(f"\nAnalysis complete for {library_name}!")
        print(f"Results saved in {results_dir}/")
        print(f"\nGenerated files:")
        print(f"  - Top ink combinations and spectra")
        print(f"  - Top 10 gamut metamer comparison (text + plots)")
        print(f"  - Top 10 gamut hue coverage comparison (text + plots)")
        print(f"  - Aggregated hue coverage at L=0.7, S=0.1 for all 10 gamuts")
        print(f"  - Detailed visualizations for top 3 gamuts")
        print(f"  - Comparison plots across all 10 gamuts")

    except Exception as e:
        print(f"Error analyzing library '{library_name}': {e}")
        sys.exit(1)


def main():

    parser = argparse.ArgumentParser(description='Analyze ink libraries for gamut optimization')
    parser.add_argument('library_name', help='Name of the ink library to analyze')
    parser.add_argument('--k', type=int, default=4, help='Number of inks to show')
    parser.add_argument('--stepsize', type=float, default=0.1,
                        help='Gamut sampling stepsize (smaller = finer, default: 0.1)')
    parser.add_argument('--top_d_inks', type=int, default=20, help='Number of top inks to show')
    parser.add_argument('--top_volume_k', type=int, default=1000, help='Number of top volume inks to search within')
    parser.add_argument('--output_dir', help='Output directory for results (default: results/{library_name}_k{k})')
    parser.add_argument('--fixed_inks', nargs='+', default=[],
                        help='Ink names to keep fixed in all combinations (e.g., --fixed_inks C255 Y255)')
    parser.add_argument('--analyze_inks', nargs='+', default=[],
                        help='Analyze specific inks directly without search (e.g., --analyze_inks C255 Y255 M255 K255)')

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
