import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from TetriumColor.Observer import Spectra, Illuminant, Observer, InkLibrary
from TetriumColor.Observer.Inks import load_inkset
from config import INKSET_CONFIGS, InksetConfig


class InksetAnalyzer:
    def __init__(self, config: InksetConfig):
        self.config = config
        self.inks, self.paper, self.wavelengths = load_inkset(config.data_path)
        self.library = InkLibrary(self.inks, self.paper)

        # Setup observer and illuminant
        self.d65 = Illuminant.get("d65")
        self.observer = Observer.tetrachromat(illuminant=self.d65, wavelengths=self.wavelengths)

        # Create output directory
        self.output_dir = Path(f"outputs/{config.name.lower().replace(' ', '_')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_convex_hull(self, k_values: List[int] = None) -> Dict[int, List]:
        """Analyze convex hull for different k values"""
        if k_values is None:
            k_values = self.config.analysis_params.get("k_values", [4])

        results = {}
        for k in k_values:
            print(f"Analyzing {self.config.name} with k={k}")
            top_combinations = self.library.convex_hull_search(self.observer, self.d65, k=k)
            results[k] = top_combinations

            # Save results
            self.save_top_combinations(top_combinations, f"top_combinations_k{k}.csv")

        return results

    def save_top_combinations(self, combinations: List, filename: str):
        """Save top combinations to CSV"""
        df = pd.DataFrame(combinations, columns=['Volume', 'Ink Combination'])
        df.to_csv(self.output_dir / filename, index=False)

    def generate_plots(self, results: Dict[int, List]):
        """Generate visualization plots"""
        # Plot ink spectra by hue
        self.plot_inks_by_hue()

        # Plot top combinations
        for k, combinations in results.items():
            self.plot_top_combinations(combinations, k)

    def plot_inks_by_hue(self):
        """Plot all inks sorted by hue"""
        # Implementation here
        pass

    def plot_top_combinations(self, combinations: List, k: int):
        """Plot top ink combinations"""
        # Implementation here
        pass


def main():
    parser = argparse.ArgumentParser(description="Analyze inksets")
    parser.add_argument("--inkset", choices=list(INKSET_CONFIGS.keys()),
                        help="Inkset to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all inksets")
    parser.add_argument("--k-values", type=int, nargs="+",
                        help="K values to analyze")

    args = parser.parse_args()

    if args.all:
        # Analyze all inksets
        for inkset_name, config in INKSET_CONFIGS.items():
            print(f"\n=== Analyzing {config.name} ===")
            analyzer = InksetAnalyzer(config)
            results = analyzer.analyze_convex_hull(args.k_values)
            analyzer.generate_plots(results)
    else:
        # Analyze single inkset
        if not args.inkset:
            print("Please specify --inkset or --all")
            return

        config = INKSET_CONFIGS[args.inkset]
        analyzer = InksetAnalyzer(config)
        results = analyzer.analyze_convex_hull(args.k_values)
        analyzer.generate_plots(results)


if __name__ == "__main__":
    main()
