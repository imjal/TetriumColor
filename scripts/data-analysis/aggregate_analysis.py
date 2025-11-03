import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import re
from datetime import datetime


def parse_filename(filepath):
    """Parse subject_id and timestamp from filename.

    Format: {subject_id}_{timestamp}.csv
    Returns: (subject_id, timestamp_str)
    """
    filename = Path(filepath).stem
    parts = filename.split('_')

    if len(parts) >= 2:
        subject_id = parts[0]
        timestamp_str = '_'.join(parts[1:])
        return subject_id, timestamp_str
    return None, None


def get_latest_files_per_subject(data_dir):
    """Get the latest file for each subject for each task.

    Returns:
        dict: {subject_id: {task_name: filepath}}
    """
    data_dir = Path(data_dir)
    subject_data = defaultdict(lambda: defaultdict(list))

    # Iterate through task folders
    for task_folder in data_dir.iterdir():
        if not task_folder.is_dir():
            continue

        task_name = task_folder.name

        # Get all CSV files in this task folder
        for csv_file in task_folder.glob('*.csv'):
            subject_id, timestamp_str = parse_filename(csv_file)
            if subject_id:
                subject_data[subject_id][task_name].append((timestamp_str, csv_file))

    # Select the latest file for each subject/task combination
    latest_files = {}
    for subject_id, tasks in subject_data.items():
        latest_files[subject_id] = {}
        for task_name, files in tasks.items():
            # Sort by timestamp and get the latest
            files.sort(key=lambda x: x[0], reverse=True)
            latest_files[subject_id][task_name] = files[0][1]

    return latest_files


def analyze_temporal_afc(df, output_dir, subject_id):
    """Analyze AppTemporalAFC data (reuse existing logic)."""
    # Check if required columns exist
    if 'rg_ratio' not in df.columns or 'luminance' not in df.columns:
        print(f"    Warning: Missing required columns (rg_ratio or luminance), skipping analysis")
        return

    # Calculate errors (1 - correct)
    df['error'] = 1 - df['correct']

    # Group by rg_ratio and luminance
    grouped = df.groupby(['rg_ratio', 'luminance']).agg({
        'error': 'sum',
        'reaction_time_ms': 'mean'
    }).reset_index()

    # Create pivot tables for heatmaps
    error_heatmap = grouped.pivot(index='luminance', columns='rg_ratio', values='error')
    rt_heatmap = grouped.pivot(index='luminance', columns='rg_ratio', values='reaction_time_ms')

    # Calculate averages across luminance
    rg_ratio_avg = df.groupby('rg_ratio').agg({
        'error': 'mean',
        'reaction_time_ms': 'mean'
    }).reset_index()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Error heatmap
    im1 = axes[0, 0].imshow(error_heatmap, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title(f'Number of Errors - {subject_id}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('RG Ratio', fontsize=10)
    axes[0, 0].set_ylabel('Luminance', fontsize=10)
    axes[0, 0].set_xticks(np.arange(len(error_heatmap.columns)))
    axes[0, 0].set_yticks(np.arange(len(error_heatmap.index)))
    axes[0, 0].set_xticklabels(error_heatmap.columns)
    axes[0, 0].set_yticklabels(error_heatmap.index)
    plt.colorbar(im1, ax=axes[0, 0], label='Number of Errors')
    for i in range(len(error_heatmap.index)):
        for j in range(len(error_heatmap.columns)):
            axes[0, 0].text(j, i, f'{error_heatmap.iloc[i, j]:.0f}',
                            ha="center", va="center", color="black", fontsize=8)

    # Plot 2: Response time heatmap
    im2 = axes[0, 1].imshow(rt_heatmap, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_title(f'Avg Response Time - {subject_id}', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('RG Ratio', fontsize=10)
    axes[0, 1].set_ylabel('Luminance', fontsize=10)
    axes[0, 1].set_xticks(np.arange(len(rt_heatmap.columns)))
    axes[0, 1].set_yticks(np.arange(len(rt_heatmap.index)))
    axes[0, 1].set_xticklabels(rt_heatmap.columns)
    axes[0, 1].set_yticklabels(rt_heatmap.index)
    plt.colorbar(im2, ax=axes[0, 1], label='Response Time (ms)')
    for i in range(len(rt_heatmap.index)):
        for j in range(len(rt_heatmap.columns)):
            axes[0, 1].text(j, i, f'{rt_heatmap.iloc[i, j]:.0f}',
                            ha="center", va="center", color="white", fontsize=8)

    # Plot 3: Error rate vs RG ratio
    axes[1, 0].plot(rg_ratio_avg['rg_ratio'], rg_ratio_avg['error'],
                    marker='o', linewidth=2, markersize=6, color='crimson')
    axes[1, 0].set_title(f'Error Rate vs RG Ratio - {subject_id}', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('RG Ratio', fontsize=10)
    axes[1, 0].set_ylabel('Error Rate', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)

    # Plot 4: Response time vs RG ratio
    axes[1, 1].plot(rg_ratio_avg['rg_ratio'], rg_ratio_avg['reaction_time_ms'],
                    marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[1, 1].set_title(f'Response Time vs RG Ratio - {subject_id}', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('RG Ratio', fontsize=10)
    axes[1, 1].set_ylabel('Response Time (ms)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / 'AppTemporalAFC_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def analyze_pseudoisochromatic(df, output_dir, subject_id):
    """Analyze AppPseudoIsochromaticTest data (4AFC, 25% guessing rate)."""
    # Check if metameric_axis exists
    if 'metameric_axis' not in df.columns:
        print(f"    Warning: No 'metameric_axis' column found, skipping analysis")
        return

    # Create genotype tuple (if genotype columns exist)
    if 'genotype_1' in df.columns and 'genotype_2' in df.columns:
        df['genotype'] = df.apply(lambda row: f"({row['genotype_1']}, {row['genotype_2']})", axis=1)
    else:
        df['genotype'] = 'Unknown'

    # Calculate accuracy by genotype and metameric_axis
    grouped = df.groupby(['genotype', 'metameric_axis']).agg({
        'correct': 'mean',
    }).reset_index()
    grouped.columns = ['genotype', 'metameric_axis', 'accuracy']

    # Calculate response time (if available)
    has_rt = 'response_time' in df.columns
    if has_rt:
        rt_grouped = df.groupby(['genotype', 'metameric_axis']).agg({
            'response_time': 'mean'
        }).reset_index()

    # Create figure - arrange genotypes in 2-column grid
    genotypes = sorted(grouped['genotype'].unique())
    metameric_axes = sorted(grouped['metameric_axis'].unique())

    n_genotypes = len(genotypes)
    # Arrange in 2 columns (accuracy + response time for each genotype)
    n_cols = 4 if has_rt else 2  # 2 genotypes per row when has_rt, 2 when not
    n_rows = int(np.ceil(n_genotypes / 2))

    fig = plt.figure(figsize=(14, 4 * n_rows))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metameric_axes)))

    plot_idx = 1
    for i, genotype in enumerate(genotypes):
        data = grouped[grouped['genotype'] == genotype]
        data = data.sort_values('metameric_axis')

        # Plot 1: Accuracy
        ax_acc = plt.subplot(n_rows, n_cols, plot_idx)
        x = np.arange(len(data))
        accuracies = data['accuracy'].values
        bars = ax_acc.bar(x, accuracies, color=colors[:len(data)], alpha=0.8, edgecolor='black')

        # Add guessing rate line
        ax_acc.axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Chance (25%)')

        ax_acc.set_xlabel('Metameric Axis', fontsize=9)
        ax_acc.set_ylabel('Accuracy', fontsize=9)
        ax_acc.set_title(f'Accuracy - Genotype {genotype}', fontsize=10, fontweight='bold')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(data['metameric_axis'].values)
        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(True, alpha=0.3, axis='y')
        ax_acc.legend(fontsize=8)

        # Add value labels on bars
        for j, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

        plot_idx += 1

        # Plot 2: Response time (if available)
        if has_rt:
            ax_rt = plt.subplot(n_rows, n_cols, plot_idx)
            if 'response_time' in rt_grouped.columns:
                data_rt = rt_grouped[rt_grouped['genotype'] == genotype]
                data_rt = data_rt.sort_values('metameric_axis')
                rts = data_rt['response_time'].values
            else:
                rts = [0] * len(data)

            bars_rt = ax_rt.bar(x, rts, color=colors[:len(data)], alpha=0.8, edgecolor='black')

            ax_rt.set_xlabel('Metameric Axis', fontsize=9)
            ax_rt.set_ylabel('Response Time (s)', fontsize=9)
            ax_rt.set_title(f'Response Time - Genotype {genotype}', fontsize=10, fontweight='bold')
            ax_rt.set_xticks(x)
            ax_rt.set_xticklabels(data['metameric_axis'].values)
            ax_rt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, rt in zip(bars_rt, rts):
                height = bar.get_height()
                ax_rt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{rt:.2f}', ha='center', va='bottom', fontsize=8)

            plot_idx += 1

    fig.suptitle(f'AppPseudoIsochromaticTest - {subject_id}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'AppPseudoIsochromaticTest_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def analyze_scrambled_face(df, output_dir, subject_id):
    """Analyze AppScrambledFaceTest data (3AFC, 33% guessing rate)."""
    # Check if metameric_axis exists
    if 'metameric_axis' not in df.columns:
        print(f"    Warning: No 'metameric_axis' column found, skipping analysis")
        return

    # Create genotype tuple (if genotype columns exist)
    if 'genotype_1' in df.columns and 'genotype_2' in df.columns:
        df['genotype'] = df.apply(lambda row: f"({row['genotype_1']}, {row['genotype_2']})", axis=1)
    else:
        df['genotype'] = 'Unknown'

    # Calculate accuracy by genotype and metameric_axis
    agg_dict = {'correct': 'mean'}
    if 'response_time' in df.columns:
        agg_dict['response_time'] = 'mean'

    grouped = df.groupby(['genotype', 'metameric_axis']).agg(agg_dict).reset_index()

    if 'response_time' in agg_dict:
        grouped.columns = ['genotype', 'metameric_axis', 'accuracy', 'response_time']
    else:
        grouped.columns = ['genotype', 'metameric_axis', 'accuracy']

    # Create figure - arrange genotypes in 2-column grid
    has_rt = 'response_time' in grouped.columns
    genotypes = sorted(grouped['genotype'].unique())
    metameric_axes = sorted(grouped['metameric_axis'].unique())

    n_genotypes = len(genotypes)
    # Arrange in 2 columns (accuracy + response time for each genotype)
    n_cols = 4 if has_rt else 2  # 2 genotypes per row when has_rt, 2 when not
    n_rows = int(np.ceil(n_genotypes / 2))

    fig = plt.figure(figsize=(14, 4 * n_rows))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metameric_axes)))

    plot_idx = 1
    for i, genotype in enumerate(genotypes):
        data = grouped[grouped['genotype'] == genotype]
        data = data.sort_values('metameric_axis')

        # Plot 1: Accuracy
        ax_acc = plt.subplot(n_rows, n_cols, plot_idx)
        x = np.arange(len(data))
        accuracies = data['accuracy'].values
        bars = ax_acc.bar(x, accuracies, color=colors[:len(data)], alpha=0.8, edgecolor='black')

        # Add guessing rate line
        ax_acc.axhline(y=0.333, color='red', linestyle='--', linewidth=2, label='Chance (33%)')

        ax_acc.set_xlabel('Metameric Axis', fontsize=9)
        ax_acc.set_ylabel('Accuracy', fontsize=9)
        ax_acc.set_title(f'Accuracy - Genotype {genotype}', fontsize=10, fontweight='bold')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(data['metameric_axis'].values)
        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(True, alpha=0.3, axis='y')
        ax_acc.legend(fontsize=8)

        # Add value labels on bars
        for j, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

        plot_idx += 1

        # Plot 2: Response time (if available)
        if has_rt:
            ax_rt = plt.subplot(n_rows, n_cols, plot_idx)
            rts = data['response_time'].values
            bars_rt = ax_rt.bar(x, rts, color=colors[:len(data)], alpha=0.8, edgecolor='black')

            ax_rt.set_xlabel('Metameric Axis', fontsize=9)
            ax_rt.set_ylabel('Response Time (s)', fontsize=9)
            ax_rt.set_title(f'Response Time - Genotype {genotype}', fontsize=10, fontweight='bold')
            ax_rt.set_xticks(x)
            ax_rt.set_xticklabels(data['metameric_axis'].values)
            ax_rt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, rt in zip(bars_rt, rts):
                height = bar.get_height()
                ax_rt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{rt:.2f}', ha='center', va='bottom', fontsize=8)

            plot_idx += 1

    fig.suptitle(f'AppScrambledFaceTest - {subject_id}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'AppScrambledFaceTest_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def analyze_subject(subject_id, task_files, results_dir):
    """Analyze all tasks for a single subject."""
    print(f"\nAnalyzing subject: {subject_id}")

    # Create output directory
    subject_dir = results_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for task_name, filepath in task_files.items():
        print(f"  Processing: {task_name}")
        df = pd.read_csv(filepath)

        if task_name == 'AppTemporalAFC':
            analyze_temporal_afc(df, subject_dir, subject_id)
        elif task_name == 'AppPseudoIsochromaticTest':
            analyze_pseudoisochromatic(df, subject_dir, subject_id)
        elif task_name == 'AppScrambledFaceTest':
            analyze_scrambled_face(df, subject_dir, subject_id)
        else:
            print(f"    Warning: Unknown task type '{task_name}', skipping")


def aggregate_across_subjects(latest_files, results_dir):
    """Create aggregate analysis across all subjects."""
    print("\n" + "="*60)
    print("Creating aggregate analysis across all subjects")
    print("="*60)

    aggregate_dir = results_dir / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Group data by task
    task_data = defaultdict(list)
    for subject_id, tasks in latest_files.items():
        for task_name, filepath in tasks.items():
            df = pd.read_csv(filepath)
            df['subject_id_analysis'] = subject_id  # Add subject ID for grouping
            task_data[task_name].append(df)

    # Analyze each task type
    for task_name, dfs in task_data.items():
        print(f"\nAggregating {task_name} (n={len(dfs)} subjects)")
        combined_df = pd.concat(dfs, ignore_index=True)

        if task_name == 'AppTemporalAFC':
            analyze_temporal_afc(combined_df, aggregate_dir, 'All Subjects')
        elif task_name == 'AppPseudoIsochromaticTest':
            analyze_pseudoisochromatic(combined_df, aggregate_dir, 'All Subjects')
        elif task_name == 'AppScrambledFaceTest':
            analyze_scrambled_face(combined_df, aggregate_dir, 'All Subjects')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate analysis of psychophysics data across multiple tasks and subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific subject
  python aggregate_analysis.py --subject 012
  
  # Analyze all subjects
  python aggregate_analysis.py --all
  
  # Analyze all and create aggregate
  python aggregate_analysis.py --all --aggregate
        """
    )

    parser.add_argument('--subject', type=str, default=None,
                        help='Analyze a specific subject ID')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all subjects')
    parser.add_argument('--aggregate', action='store_true',
                        help='Create aggregate analysis across all subjects')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: ./data/)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: ./results/)')

    args = parser.parse_args()

    # Determine directories
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / 'data'
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / 'results'

    # Get latest files per subject
    latest_files = get_latest_files_per_subject(data_dir)

    if not latest_files:
        print("No data files found!")
        return

    print("Found subjects and their latest data:")
    for subject_id, tasks in latest_files.items():
        print(f"  {subject_id}:")
        for task_name, filepath in tasks.items():
            print(f"    - {task_name}: {filepath.name}")

    # Analyze specific subject
    if args.subject:
        if args.subject in latest_files:
            analyze_subject(args.subject, latest_files[args.subject], results_dir)
        else:
            print(f"Error: Subject '{args.subject}' not found in data")
            print(f"Available subjects: {', '.join(latest_files.keys())}")
            return

    # Analyze all subjects
    elif args.all:
        for subject_id, task_files in latest_files.items():
            analyze_subject(subject_id, task_files, results_dir)

    # Create aggregate analysis
    if args.aggregate or (args.all and args.aggregate):
        aggregate_across_subjects(latest_files, results_dir)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {results_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
