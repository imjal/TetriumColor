#!/usr/bin/env python3
"""
Batch analysis script for multiple inksets
"""

import subprocess
import sys
from pathlib import Path
from config import INKSET_CONFIGS


def run_batch_analysis():
    """Run analysis on all inksets"""

    # Create summary report
    summary = []

    for inkset_name, config in INKSET_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {config.name}")
        print(f"{'='*50}")

        # Run analysis
        cmd = [
            sys.executable, "analyze_inkset.py",
            "--inkset", inkset_name,
            "--k-values", "4", "5", "6"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully analyzed {config.name}")
                summary.append(f"✓ {config.name}: SUCCESS")
            else:
                print(f"✗ Failed to analyze {config.name}")
                print(f"Error: {result.stderr}")
                summary.append(f"✗ {config.name}: FAILED")
        except Exception as e:
            print(f"✗ Exception analyzing {config.name}: {e}")
            summary.append(f"✗ {config.name}: EXCEPTION")

    # Print summary
    print(f"\n{'='*50}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*50}")
    for status in summary:
        print(status)


if __name__ == "__main__":
    run_batch_analysis()
