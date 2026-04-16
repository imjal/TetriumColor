"""
Shared plot style for publication figures (ACM SIGGRAPH / acmart.cls).

Usage:
    from TetriumColor.Visualization.PlotStyle import apply_style, COLORS, WAVELENGTHS, build_observers

    apply_style()  # call once at module level
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from TetriumColor.Observer.Observer import Observer

# ============================================================
# Constants
# ============================================================
WAVELENGTHS = np.arange(400, 701, 1)

# Nominal opsin peaks (before prereceptoral filtering)
S_PEAK = 420
M_PEAKS = [530, 533, 536]
L_PEAKS = [547, 551, 552, 553, 555, 556, 556.5, 559]
ALL_PEAKS = [S_PEAK] + M_PEAKS + L_PEAKS  # 12 total

# One color per cone, keyed by nominal peak
# S = blue, M variants = greens, L variants = yellow→red gradient
COLORS = {
    S_PEAK: '#2166ac',       # S cone — blue
    530: '#1b7837',          # M cones — greens
    533: '#4d9221',
    536: '#7fbc41',
    547: '#d4b800',          # L cones — yellow → red
    551: '#e6a000',
    552: '#f08c00',
    553: '#f57600',
    555: '#e85a00',
    556: '#d94701',
    556.5: '#c62d04',
    559: '#b2182b',
}

# SIGGRAPH column widths (inches)
SINGLE_COL = 3.33
DOUBLE_COL = 7.0
FULL_PAGE = 7.5  # full text width including margins for wide figures


def apply_style():
    """Apply the shared publication plot style.

    Uses Linux Biolinum (sans-serif) via LaTeX for consistent rendering.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': (
            r'\usepackage[tt=false]{libertine}'
            r'\usepackage[libertine]{newtxmath}'
            r'\renewcommand{\familydefault}{\sfdefault}'
        ),
        'font.family': 'sans-serif',
    })


def build_observers(wavelengths=None) -> dict:
    """Construct standard observer set using Observer class methods.

    Returns dict mapping name -> (Observer, nominal_peaks).
    """
    wl = wavelengths if wavelengths is not None else WAVELENGTHS
    return {
        'Dichromat': (
            Observer.custom_observer(wavelengths=wl, dimension=2),
            [420, 530]),
        'Trichromat': (
            Observer.custom_observer(wavelengths=wl, dimension=3),
            [420, 530, 559]),
        'Tetrachromat': (
            Observer.custom_observer(wavelengths=wl, dimension=4, q_cone_peak=547),
            [420, 530, 547, 559]),
        'Hyperobserver': (
            Observer.hyperobserver(wavelengths=wl),
            ALL_PEAKS),
    }
