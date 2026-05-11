"""
Shared plot style for publication figures (ACM SIGGRAPH / acmart.cls).

Usage:
    from TetriumColor.Plotting.PlotStyle import apply_style, COLORS, WAVELENGTHS, build_observers

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

# Shared saturated paper palette. The core colors match the fresh blue/yellow/red
# used by the Quest/MOCS summary figures; use alpha in individual plots when a
# softer appearance is needed.
PAPER_BLUE = COLORS[S_PEAK]
PAPER_YELLOW = COLORS[551]
PAPER_RED = '#c62828'
PAPER_NEUTRAL = '#5c5c5c'
PAPER_LIGHT_GRAY = '#f7f7f7'

PAPER_DIMENSION_COLORS = {
    0: PAPER_RED,
    1: COLORS[552],
    2: COLORS[530],
    3: PAPER_BLUE,
    4: '#7b3294',
}

PAPER_CONE_FAMILY_COLORS = {
    "S": [PAPER_BLUE],
    "M": [COLORS[530], COLORS[533], COLORS[536]],
    "L": [COLORS[547], COLORS[551], COLORS[552], COLORS[553], COLORS[555], COLORS[556], COLORS[556.5], COLORS[559]],
}

# SIGGRAPH column widths (inches)
SINGLE_COL = 3.33
DOUBLE_COL = 7.0

def apply_style():
    """Apply the shared publication plot style.

    Uses explicit point sizes so figure text matches the LaTeX document when
    the generated figure width is included at the same width in the paper.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': (
            r'\usepackage[tt=false]{libertine}'
            r'\usepackage[libertine]{newtxmath}'
            r'\renewcommand{\familydefault}{\sfdefault}'
        ),
        'font.family': 'sans-serif',
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 8,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.0,
        'lines.markersize': 3.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'savefig.dpi': 300,
        'figure.dpi': 150,
    })

    sns.set_style("ticks")   


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
