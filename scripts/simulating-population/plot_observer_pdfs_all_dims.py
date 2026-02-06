#!/usr/bin/env python3
"""
Plot male and female observer genotype PDFs for all dimensions using seaborn with Linux Biolinum font.
"""

from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes

# ===== CONFIGURATION =====
# Set to True for combined plot (50% male, 50% female), False for side-by-side
COMBINED_PLOT = True  # Change to True for combined plot
TRICHROMATS_ONLY = True  # Set to True to show only trichromats in combined plot
top_n = 20  # Number of top genotypes to plot
# =========================

# Set the style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

# Set Linux Biolinum font
plt.rcParams['font.family'] = 'Linux Biolinum'
print("Using Linux Biolinum font")

# Initialize ObserverGenotypes for all dimensions (1-5 total cones including implicit S)
print("Initializing ObserverGenotypes for all dimensions...")
og = ObserverGenotypes(dimensions=[1, 2, 3, 4, 5])

# Create color palette based on dimension (0-4 M/L cones)
dim_colors = {
    0: '#d62728',  # red - monochromat
    1: '#ff7f0e',  # orange - dichromat
    2: '#2ca02c',  # green - trichromat
    3: '#1f77b4',  # blue - tetrachromat
    4: '#9467bd',  # purple - pentachromat
}

# Add legend for dimensions (including implicit S cone)
legend_elements = [
    Patch(facecolor=dim_colors[0], edgecolor='black', label='Monochromat (1)'),
    Patch(facecolor=dim_colors[1], edgecolor='black', label='Dichromat (2)'),
    Patch(facecolor=dim_colors[2], edgecolor='black', label='Trichromat (3)'),
    Patch(facecolor=dim_colors[3], edgecolor='black', label='Tetrachromat (4)'),
    Patch(facecolor=dim_colors[4], edgecolor='black', label='Pentachromat (5)'),
]

if COMBINED_PLOT:
    # ===== COMBINED PLOT (50% male, 50% female) =====
    print("Creating combined plot (50% male, 50% female)...")
    both_pdf = og.get_pdf('both')

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    # Plot Combined PDF
    if TRICHROMATS_ONLY:
        print("Plotting combined PDF (trichromats only)...")
        # Filter for trichromats only (2 M/L cones = dimension 3 total with S cone)
        all_genotypes = list(both_pdf.keys())
        trichromat_data = [(g, both_pdf[g]) for g in all_genotypes if len(g) == 2]
        # Sort by probability
        trichromat_data.sort(key=lambda x: x[1], reverse=True)
        genotypes = [g for g, _ in trichromat_data[:top_n]]
        probs = [p for _, p in trichromat_data[:top_n]]
    else:
        print("Plotting combined PDF...")
        genotypes = list(both_pdf.keys())[:top_n]
        probs = list(both_pdf.values())[:top_n]

    labels = ['{' + ', '.join(map(str, peaks)) + '}' for peaks in genotypes]
    dims = [len(peaks) for peaks in genotypes]

    colors = [dim_colors.get(d, '#7f7f7f') for d in dims]

    bars1 = ax1.bar(range(len(labels)), probs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Genotype M/L Peaks (nm)\n+ implicit S cone at 420 nm', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=18, fontweight='bold')
    if TRICHROMATS_ONLY:
        ax1.set_title('Trichromat Observer Genotypes (50% Male, 50% Female)', fontsize=20, fontweight='bold', pad=15)
    else:
        ax1.set_title('Observer Genotypes (50% Male, 50% Female)', fontsize=20, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)

    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars1, probs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{prob:.3f}', ha='center', va='bottom', fontsize=7)

    # Add CDF on secondary axis
    ax1_cdf = ax1.twinx()
    cdf_values = np.cumsum(probs)
    max_bar_height = max(probs)
    # Scale CDF to match bar height visually
    scaled_cdf = cdf_values / cdf_values[-1] * max_bar_height
    ax1_cdf.plot(range(len(labels)), scaled_cdf, color='black', linewidth=2,
                 marker='o', markersize=3, linestyle='-', alpha=0.7, label='Cumulative')
    ax1_cdf.set_ylabel('Cumulative Probability (%)', fontsize=18, fontweight='bold')
    ax1_cdf.set_ylim(0, max_bar_height)
    # Set ticks to show 0-100%
    ax1_cdf.set_yticks(np.linspace(0, max_bar_height, 6))
    ax1_cdf.set_yticklabels([f'{int(p)}%' for p in np.linspace(0, 100, 6)])
    ax1_cdf.spines['top'].set_visible(False)
    ax1_cdf.grid(False)
    ax1_cdf.tick_params(axis='y', labelsize=12)

    # Add legend for dimensions (only if showing all dimensions)
    if not TRICHROMATS_ONLY:
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if TRICHROMATS_ONLY:
        filename = 'observer_genotype_pdfs_trichromats.png'
    else:
        filename = 'observer_genotype_pdfs_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to '{filename}'")

    # Print summary statistics
    if TRICHROMATS_ONLY:
        print("\n=== Summary Statistics (Trichromats, 50% Male, 50% Female) ===")
        total_trichromat_prob = sum([both_pdf[g] for g in all_genotypes if len(g) == 2])
        print(f"  Total trichromat probability: {total_trichromat_prob:.4f}")
        print(f"  Showing top {len(genotypes)} trichromat genotypes")
        print(f"  Top {len(genotypes)} genotypes cover: {sum(probs):.4f} of trichromat probability")
    else:
        print("\n=== Summary Statistics (50% Male, 50% Female) ===")
        for dim in [0, 1, 2, 3, 4]:
            dim_genotypes = [g for g in genotypes if len(g) == dim]
            dim_probs = [both_pdf[g] for g in dim_genotypes]
            if dim_probs:
                total_prob = sum(dim_probs)
                actual_dim = dim + 1
                print(
                    f"  Dimension {actual_dim} ({dim} M/L + 1 S): {len(dim_genotypes)} genotypes, {total_prob:.4f} total probability")

else:
    # ===== SIDE-BY-SIDE PLOTS (Male and Female separate) =====
    print("Creating side-by-side plots (male and female)...")
    male_pdf = og.get_pdf('male')
    female_pdf = og.get_pdf('female')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot Male PDF
    print("Plotting male PDF...")
    male_genotypes = list(male_pdf.keys())[:top_n]
    male_probs = list(male_pdf.values())[:top_n]
    male_labels = ['{' + ', '.join(map(str, peaks)) + '}' for peaks in male_genotypes]
    male_dims = [len(peaks) for peaks in male_genotypes]

    colors_male = [dim_colors.get(d, '#7f7f7f') for d in male_dims]

    bars1 = ax1.bar(range(len(male_labels)), male_probs, color=colors_male, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Genotype M/L Peaks (nm)\n+ implicit S cone at 420 nm', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=18, fontweight='bold')
    ax1.set_title('Male Observer Genotypes', fontsize=20, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(male_labels)))
    ax1.set_xticklabels(male_labels, rotation=45, ha='right', fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)

    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars1, male_probs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{prob:.3f}', ha='center', va='bottom', fontsize=7)

    # Add CDF on secondary axis
    ax1_cdf = ax1.twinx()
    male_cdf_values = np.cumsum(male_probs)
    max_bar_height = max(male_probs)
    # Scale CDF to match bar height visually
    scaled_cdf = male_cdf_values / male_cdf_values[-1] * max_bar_height
    ax1_cdf.plot(range(len(male_labels)), scaled_cdf, color='black', linewidth=2,
                 marker='o', markersize=3, linestyle='-', alpha=0.7, label='Cumulative')
    ax1_cdf.set_ylabel('Cumulative Probability (%)', fontsize=18, fontweight='bold')
    ax1_cdf.set_ylim(0, max_bar_height)
    # Set ticks to show 0-100%
    ax1_cdf.set_yticks(np.linspace(0, max_bar_height, 6))
    ax1_cdf.set_yticklabels([f'{int(p)}%' for p in np.linspace(0, 100, 6)])
    ax1_cdf.spines['top'].set_visible(False)
    ax1_cdf.grid(False)
    ax1_cdf.tick_params(axis='y', labelsize=12)

    # Add legend for dimensions
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Plot Female PDF
    print("Plotting female PDF...")
    female_genotypes = list(female_pdf.keys())[:top_n]
    female_probs = list(female_pdf.values())[:top_n]
    female_labels = ['{' + ', '.join(map(str, peaks)) + '}' for peaks in female_genotypes]
    female_dims = [len(peaks) for peaks in female_genotypes]

    colors_female = [dim_colors.get(d, '#7f7f7f') for d in female_dims]

    bars2 = ax2.bar(range(len(female_labels)), female_probs, color=colors_female, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Genotype M/L Peaks (nm)\n+ implicit S cone at 420 nm', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=18, fontweight='bold')
    ax2.set_title('Female Observer Genotypes', fontsize=20, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(female_labels)))
    ax2.set_xticklabels(female_labels, rotation=45, ha='right', fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)

    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars2, female_probs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{prob:.3f}', ha='center', va='bottom', fontsize=7)

    # Add CDF on secondary axis
    ax2_cdf = ax2.twinx()
    female_cdf_values = np.cumsum(female_probs)
    max_bar_height = max(female_probs)
    # Scale CDF to match bar height visually
    scaled_cdf = female_cdf_values / female_cdf_values[-1] * max_bar_height
    ax2_cdf.plot(range(len(female_labels)), scaled_cdf, color='black', linewidth=2,
                 marker='o', markersize=3, linestyle='-', alpha=0.7, label='Cumulative')
    ax2_cdf.set_ylabel('Cumulative Probability (%)', fontsize=18, fontweight='bold')
    ax2_cdf.set_ylim(0, max_bar_height)
    # Set ticks to show 0-100%
    ax2_cdf.set_yticks(np.linspace(0, max_bar_height, 6))
    ax2_cdf.set_yticklabels([f'{int(p)}%' for p in np.linspace(0, 100, 6)])
    ax2_cdf.spines['top'].set_visible(False)
    ax2_cdf.grid(False)
    ax2_cdf.tick_params(axis='y', labelsize=12)

    # Add legend for dimensions
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Adjust layout for side-by-side presentation
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25)

    plt.savefig('observer_genotype_pdfs_all_dims.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'observer_genotype_pdfs_all_dims.png'")

    # Print summary statistics (including implicit S cone)
    print("\n=== Summary Statistics (including implicit S cone) ===")
    print(f"\nMale distribution:")
    for dim in [0, 1, 2, 3, 4]:
        dim_genotypes = [g for g in male_genotypes if len(g) == dim]
        dim_probs = [male_pdf[g] for g in dim_genotypes]
        if dim_probs:
            total_prob = sum(dim_probs)
            actual_dim = dim + 1  # Add implicit S cone
            print(
                f"  Dimension {actual_dim} ({dim} M/L + 1 S): {len(dim_genotypes)} genotypes, {total_prob:.4f} total probability")

    print(f"\nFemale distribution:")
    for dim in [0, 1, 2, 3, 4]:
        dim_genotypes = [g for g in female_genotypes if len(g) == dim]
        dim_probs = [female_pdf[g] for g in dim_genotypes]
        if dim_probs:
            total_prob = sum(dim_probs)
            actual_dim = dim + 1  # Add implicit S cone
            print(
                f"  Dimension {actual_dim} ({dim} M/L + 1 S): {len(dim_genotypes)} genotypes, {total_prob:.4f} total probability")

plt.show()
