#!/usr/bin/env python3
"""
Compute expected number of isolation tests to classify a trichromat observer
using a P-primary display, comparing random vs. sequential genetic prior strategies.

The hyperobserver has N=12 cone types: S(420) + 11 M/L peaks.
A trichromat has d=3 cones: S + 2 M/L peaks.

Random baseline:
    Each round picks a random P-subset and runs P isolation tests.
    E[tests] = P * C(N,P) / C(N-d, P-d)

Sequential with genetic prior (Algorithm 1):
    Ordered list of (R, B) pairs where |R|=1 (one decoy cone) and |B|=P.
    For each pair:
      - Test decoy d ∈ R within display B: 1 test
      - PASS (observer has d): move on, cost 1
      - FAIL (observer lacks d, cones ⊆ B): test B \ R to identify, cost P total

    B must contain the observer's cones for correct identification. With P
    primaries, B has P-2 M/L slots (after S and decoy), so each pair covers
    genotypes whose M/L peaks fit in those slots AND lack the decoy.

    E[tests] = (# passes before trigger) + P
             = P - 1 + Σ_{i=0}^{K-1} (1 - c_i)

    Pairs are chosen greedily to maximize correctly-identifiable probability
    each round. Larger P → more genotypes per pair → fewer passes, but each
    trigger costs P.
"""

import sys
from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes

# Hyperobserver cone peaks (N=12)
S_PEAK = 420
ML_PEAKS = [530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
ALL_PEAKS = [S_PEAK] + ML_PEAKS
N = len(ALL_PEAKS)  # 12
D = 3  # trichromat: S + 2 M/L


def random_expected_tests(N, d, P):
    """E[tests] = P * C(N,P) / C(N-d, P-d)"""
    return P * comb(N, P) / comb(N - d, P - d)


def sequential_decoy_expected_tests(P, genotype_probs, ml_peaks):
    """
    Sequential strategy with |R|=1 decoy per pair.

    Each pair (d, B) with |B|=P:
      - Covers genotypes that (1) lack decoy d and (2) have M/L cones ⊆ B
      - PASS costs 1, trigger costs P
      - B has P-2 M/L slots (after S and decoy)

    Greedily choose (d, B) each round to maximize covered probability.

    Returns (E[tests], K rounds, details per round).
    """
    remaining = dict(genotype_probs)
    cumulative = 0.0
    tail_sum = 0.0
    ml_slots = P - 2  # M/L slots in B (excluding S and decoy)
    rounds = []

    while remaining:
        best_prob = -1
        best_covered = None
        best_decoy = None
        best_ml = None

        for d in ml_peaks:
            available = [m for m in ml_peaks if m != d]

            if ml_slots >= len(available):
                combos = [tuple(available)]
            else:
                combos = combinations(available, ml_slots)

            for ml_combo in combos:
                ml_set = set(ml_combo)
                covered = set()
                covered_prob = 0.0
                for g, p in remaining.items():
                    if d not in g and set(g).issubset(ml_set):
                        covered.add(g)
                        covered_prob += p

                if covered_prob > best_prob:
                    best_prob = covered_prob
                    best_covered = covered
                    best_decoy = d
                    best_ml = ml_set

        if not best_covered:
            break

        tail_sum += (1 - cumulative)

        for g in best_covered:
            cumulative += remaining.pop(g)

        rounds.append({
            'decoy': best_decoy,
            'ml_peaks': sorted(best_ml),
            'covered': best_covered,
            'prob': best_prob,
            'cumulative': cumulative,
        })

    expected_tests = P - 1 + tail_sum
    return expected_tests, len(rounds), rounds


def main():
    og = ObserverGenotypes(dimensions=[3])
    both_pdf = og.get_pdf('both')

    total = sum(both_pdf.values())
    genotype_probs = {g: p / total for g, p in both_pdf.items()}

    print(f"N = {N} cone types, d = {D} (trichromat)")
    print(f"Number of trichromat genotypes: {len(genotype_probs)}")
    print(f"\nTop 10 genotypes:")
    for g, p in list(genotype_probs.items())[:10]:
        print(f"  {g}: {p:.4f}")

    # Compute for each P
    print(f"\n{'P':>3}  {'Random':>10}  {'Genetic':>10}  {'Rounds':>6}")
    print("-" * 36)
    rows = []
    all_details = {}
    for P in range(4, 13):
        e_random = random_expected_tests(N, D, P)
        e_seq, K, details = sequential_decoy_expected_tests(P, genotype_probs, ML_PEAKS)
        rows.append((P, e_random, e_seq, K))
        all_details[P] = details
        print(f"{P:>3}  {e_random:>10.1f}  {e_seq:>10.1f}  {K:>6}")

    # Print round details for a few interesting P values
    for P in [4, 6, 8]:
        details = all_details[P]
        print(f"\n--- P={P}: Round details ---")
        for i, r in enumerate(details):
            genotypes_str = ', '.join(str(g) for g in sorted(r['covered']))
            print(f"  Round {i+1}: decoy={r['decoy']}, "
                  f"B_ML={r['ml_peaks']}, "
                  f"covers {r['prob']:.4f} (cum={r['cumulative']:.4f})")

    # LaTeX table
    print("\n")
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \begin{tabular}{@{}cccc@{}}")
    print(r"        \toprule")
    print(r"        $P$ & \textbf{Exp.\ Tests (Random)} & "
          r"\textbf{Exp.\ Tests (Genetic Prior)} & Rounds\\")
    print(r"        \midrule")
    for P, e_random, e_seq, K in rows:
        r_str = (f"{e_random:.0f}" if e_random == int(e_random)
                 else f"$\\sim$ {e_random:.0f}")
        g_str = f"{e_seq:.1f}"
        print(f"        {P} & {r_str} & {g_str} & {K} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(f"    \\caption{{Expected tests for $N={N}$, $d={D}$ by number of "
          f"primaries. Genetic prior uses greedy decoy selection with "
          f"$|R|=1$.}}")
    print(r"    \label{tab:expected}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()
