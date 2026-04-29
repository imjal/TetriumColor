"""
Power analysis for 4-AFC psychophysics experiments.

Computes the minimum number of trials needed per intensity bucket to reliably
detect above-chance performance in a 4-AFC task (chance = 25%).

Uses a one-sided binomial test:
    H0: p = 0.25  (chance)
    H1: p > 0.25  (above chance)

Usage:
    python power_analysis_4afc.py
    python power_analysis_4afc.py --alpha 0.01 --power 0.90
    python power_analysis_4afc.py --true-acc 0.35 0.50 0.75
"""

import argparse
import numpy as np
from scipy.stats import binom


CHANCE = 0.25  # 4-AFC: up/down/left/right


def min_trials_for_power(p_true: float, alpha: float = 0.05, power_target: float = 0.80) -> tuple:
    """Find minimum n for given power to detect p_true > CHANCE.

    Returns (n, achieved_power, k_crit).
    """
    for n in range(5, 5000):
        # Smallest k s.t. P(X >= k | n, CHANCE) <= alpha
        k_crit = int(binom.ppf(1 - alpha, n, CHANCE)) + 1
        if binom.sf(k_crit - 1, n, CHANCE) > alpha:
            k_crit += 1
        power = binom.sf(k_crit - 1, n, p_true)
        if power >= power_target:
            return n, power, k_crit
    return None, None, None


def print_table(true_accuracies, alpha, power_target):
    print(f"\n4-AFC power analysis  |  chance={CHANCE:.0%}  alpha={alpha}  target power={power_target:.0%}")
    print(f"One-sided binomial test: H0: p = {CHANCE:.0%},  H1: p > {CHANCE:.0%}\n")
    print(f"{'True acc':>9}  {'Min n':>6}  {'Power':>7}  {'SE':>6}  {'95% CI (±)':>11}  {'k_crit':>7}")
    print("-" * 56)

    for p_true in true_accuracies:
        n, power, k_crit = min_trials_for_power(p_true, alpha, power_target)
        if n is None:
            print(f"  {p_true:.0%}       >5000  (unreachable)")
            continue
        se = np.sqrt(p_true * (1 - p_true) / n)
        ci_half = 1.96 * se
        print(f"  {p_true:.0%}       {n:>5d}   {power:.0%}   {se:.1%}     ±{ci_half:.1%}      {k_crit:>5d}")

    print()
    print("SE = standard error of the accuracy estimate at n trials.")
    print("95% CI (±) = half-width of normal-approx confidence interval.")
    print("k_crit = minimum correct answers to reject H0 at this n.")


def main():
    parser = argparse.ArgumentParser(description="4-AFC trial count power analysis")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Type-I error rate (default: 0.05)")
    parser.add_argument("--power", type=float, default=0.80,
                        help="Target power 1-beta (default: 0.80)")
    parser.add_argument("--true-acc", type=float, nargs="+",
                        default=[0.35, 0.40, 0.50, 0.60, 0.75],
                        help="True accuracy levels to evaluate (default: 0.35 0.40 0.50 0.60 0.75)")
    args = parser.parse_args()

    print_table(args.true_acc, args.alpha, args.power)


if __name__ == "__main__":
    main()
