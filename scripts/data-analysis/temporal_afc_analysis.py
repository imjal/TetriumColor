import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def load_data(input_path):
    """Load data from a single file or multiple files in a folder.

    Args:
        input_path: Path to a CSV file or folder containing CSV files

    Returns:
        DataFrame with all data, and a label for the output file
    """
    input_path = Path(input_path)

    if input_path.is_file():
        # Single file
        df = pd.read_csv(input_path)
        subject_id = df['subject_id'].iloc[0]
        label = subject_id
        print(f"Loaded data for subject: {subject_id} ({len(df)} trials)")
        return df, label

    elif input_path.is_dir():
        # Multiple files in a folder
        csv_files = list(input_path.glob('*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in {input_path}")
            sys.exit(1)

        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"  Loaded {csv_file.name}: {len(df)} trials")
            dfs.append(df)

        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal: {len(csv_files)} participants, {len(combined_df)} trials")
        label = 'aggregated'
        return combined_df, label

    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


def create_plots(df, output_label, output_dir):
    """Create 4 plots and save to file.

    Args:
        df: DataFrame with experimental data
        output_label: Label for output filename
        output_dir: Directory to save output
    """
    # Calculate errors (1 - correct)
    df['error'] = 1 - df['correct']

    # Group by rg_ratio and luminance
    grouped = df.groupby(['rg_ratio', 'luminance']).agg({
        'error': 'sum',  # Total number of errors
        'reaction_time_ms': 'mean'  # Average response time
    }).reset_index()

    # Create pivot tables for heatmaps
    error_heatmap = grouped.pivot(index='luminance', columns='rg_ratio', values='error')
    rt_heatmap = grouped.pivot(index='luminance', columns='rg_ratio', values='reaction_time_ms')

    # Calculate averages across luminance for line plots
    rg_ratio_avg = df.groupby('rg_ratio').agg({
        'error': 'mean',  # Average error rate
        'reaction_time_ms': 'mean'  # Average response time
    }).reset_index()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Error heatmap
    im1 = axes[0, 0].imshow(error_heatmap, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('Number of Errors by RG Ratio and Luminance', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('RG Ratio', fontsize=10)
    axes[0, 0].set_ylabel('Luminance', fontsize=10)
    axes[0, 0].set_xticks(np.arange(len(error_heatmap.columns)))
    axes[0, 0].set_yticks(np.arange(len(error_heatmap.index)))
    axes[0, 0].set_xticklabels(error_heatmap.columns)
    axes[0, 0].set_yticklabels(error_heatmap.index)
    plt.colorbar(im1, ax=axes[0, 0], label='Number of Errors')
    # Add text annotations
    for i in range(len(error_heatmap.index)):
        for j in range(len(error_heatmap.columns)):
            text = axes[0, 0].text(j, i, f'{error_heatmap.iloc[i, j]:.0f}',
                                   ha="center", va="center", color="black", fontsize=8)

    # Plot 2: Response time heatmap
    im2 = axes[0, 1].imshow(rt_heatmap, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_title('Average Response Time by RG Ratio and Luminance', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('RG Ratio', fontsize=10)
    axes[0, 1].set_ylabel('Luminance', fontsize=10)
    axes[0, 1].set_xticks(np.arange(len(rt_heatmap.columns)))
    axes[0, 1].set_yticks(np.arange(len(rt_heatmap.index)))
    axes[0, 1].set_xticklabels(rt_heatmap.columns)
    axes[0, 1].set_yticklabels(rt_heatmap.index)
    plt.colorbar(im2, ax=axes[0, 1], label='Response Time (ms)')
    # Add text annotations
    for i in range(len(rt_heatmap.index)):
        for j in range(len(rt_heatmap.columns)):
            text = axes[0, 1].text(j, i, f'{rt_heatmap.iloc[i, j]:.0f}',
                                   ha="center", va="center", color="white", fontsize=8)

    # Plot 3: Error rate vs RG ratio
    axes[1, 0].plot(rg_ratio_avg['rg_ratio'], rg_ratio_avg['error'],
                    marker='o', linewidth=2, markersize=6, color='crimson')
    axes[1, 0].set_title('Error Rate vs RG Ratio (Averaged Across Luminance)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('RG Ratio', fontsize=10)
    axes[1, 0].set_ylabel('Error Rate', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)

    # Plot 4: Response time vs RG ratio
    axes[1, 1].plot(rg_ratio_avg['rg_ratio'], rg_ratio_avg['reaction_time_ms'],
                    marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[1, 1].set_title('Response Time vs RG Ratio (Averaged Across Luminance)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('RG Ratio', fontsize=10)
    axes[1, 1].set_ylabel('Response Time (ms)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{output_label}_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nSummary plot saved to: {output_path}')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze temporal 3-AFC psychophysics data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python temporal_afc_analysis.py data/subject1.csv
  
  # Aggregate multiple participants in a folder
  python temporal_afc_analysis.py data/
  
  # Specify custom output directory
  python temporal_afc_analysis.py data/subject1.csv -o custom_results/
        """
    )

    parser.add_argument('input', type=str,
                        help='Path to CSV file or folder containing CSV files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for results (default: results/)')

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to results/ in the data-analysis folder
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'results'

    # Load data
    df, label = load_data(args.input)

    # Create plots
    create_plots(df, label, output_dir)


if __name__ == '__main__':
    main()
