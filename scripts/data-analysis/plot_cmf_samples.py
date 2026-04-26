#!/usr/bin/env python3
"""
Generate and plot sampled CMFs to a PNG image.

Samples observers with physiological variation using CMFSampler and renders
all their spectral sensitivities (sensor matrices) to a single image showing
variation across the population.
"""

import numpy as np
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.CMFSampler import CMFSampler


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and plot sampled CMFs to an image"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of observers to sample (default 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sampled_cmfs.png",
        help="Output image path (default sampled_cmfs.png)",
    )
    parser.add_argument(
        "--macular_std",
        type=float,
        default=0.25,
        help="Std dev of macular pigment density",
    )
    parser.add_argument(
        "--od_lm_std",
        type=float,
        default=0.05,
        help="Std dev of L/M photopigment OD",
    )
    parser.add_argument(
        "--od_s_std",
        type=float,
        default=0.04,
        help="Std dev of S photopigment OD",
    )
    parser.add_argument(
        "--lens_std",
        type=float,
        default=0.1,
        help="Std dev of lens density",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Optional dict of parameter overrides as JSON, e.g. '{\"macular_std\": 0.0}'",
    )
    parser.add_argument(
        "--subplots",
        action="store_true",
        help="Use subplots (one per cone position) instead of single plot (default: single plot)",
    )
    parser.add_argument(
        "--top_n_genotypes",
        type=int,
        default=None,
        help="Only sample from top N most probable genotypes (default: all)",
    )

    args = parser.parse_args()

    # Parse params dict if provided
    params_dict = None
    if args.params:
        import json
        params_dict = json.loads(args.params)

    print(f"Generating {args.n_samples} CMF samples...")

    # Create sampler
    observer_wavelengths = np.arange(380, 781, 5)
    observer_genotypes = ObserverGenotypes(
        wavelengths=observer_wavelengths,
        dimensions=[3],
        seed=42
    )

    if params_dict:
        sampler = CMFSampler(observer_genotypes, params=params_dict, top_n_genotypes=args.top_n_genotypes)
    else:
        sampler = CMFSampler(
            observer_genotypes,
            macular_std=args.macular_std,
            od_lm_std=args.od_lm_std,
            od_s_std=args.od_s_std,
            lens_std=args.lens_std,
            top_n_genotypes=args.top_n_genotypes,
        )

    # Sample observers
    samples = sampler.sample(args.n_samples, sex='both')
    print(f"Sampled {len(samples)} observers")

    # Plot and save
    sampler.plot_sampled_cmfs(samples, output_path=args.output, single_plot=not args.subplots)


if __name__ == "__main__":
    main()
