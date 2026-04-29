"""
Compare MOCS (Genetic) vs Quest trial data for a given subject.

Automatically finds the latest Genetic and Quest trial files for the subject,
then for each genotype × axis plots:
  1. MOCS: mean accuracy per fixed intensity bucket
  2. Quest: raw 0/1 scatter + fitted Weibull psychometric function
  3. Both overlaid

Filename convention: {subject_id}_{Genetic|Quest}_{...}_{YYYYMMDD_HHMMSS_mmm}.csv

Usage:
    python compare_mocs_quest.py --subject jessica-4-18
    python compare_mocs_quest.py --subject jessica-4-18 --data-dir /path/to/AppPseudoIsochromaticTest
    python compare_mocs_quest.py --list   # show all detected subjects
"""

import argparse
import re
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict
from pathlib import Path
from scipy.optimize import curve_fit

TETRIUM_COLOR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TETRIUM_COLOR_ROOT))

from TetriumColor.Plotting.PlotStyle import apply_style, COLORS, DOUBLE_COL

FULL_PAGE = DOUBLE_COL

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data" / "AppPseudoIsochromaticTest"
)

CHANCE    = 1 / 4   # 4-AFC (up/down/left/right)
CRITERION = 0.50    # Quest uses pThreshold=0.5 (TetraColorPicker.py)
AXIS_LABEL = {1: "Axis 1 (S)", 2: "Axis 2 (Q)", 3: "Axis 3 (L)"}
MOCS_COLOR = COLORS[420]
QUEST_COLOR = COLORS[551]
NEUTRAL_COLOR = "#5c5c5c"

# Matches: {subject_id}_{Genetic|Quest}_{anything}_{YYYYMMDD_HHMMSS_mmm}.csv
# Does NOT match *_thresholds.csv
_FILE_RE = re.compile(
    r'^(?P<subject>.+?)_(?P<method>Genetic|Quest)_(?P<rest>.+?)_(?P<ts>\d{8}_\d{6}_\d{3})$'
)


# ── file discovery ─────────────────────────────────────────────────────────────
def discover_files(data_dir: Path):
    """
    Returns {subject_id: {'Genetic': latest_path, 'Quest': latest_path}}.
    Follows the same latest-file-per-subject pattern as aggregate_analysis.py.
    """
    # subject → method → [(timestamp, path)]
    index = defaultdict(lambda: defaultdict(list))

    for csv in data_dir.glob("*.csv"):
        m = _FILE_RE.match(csv.stem)
        if m is None:
            continue
        subject = m.group("subject")
        method  = m.group("method")
        ts      = m.group("ts")
        index[subject][method].append((ts, csv))

    # keep only the latest file per subject/method
    latest = {}
    for subject, methods in index.items():
        latest[subject] = {}
        for method, files in methods.items():
            files.sort(key=lambda x: x[0], reverse=True)
            latest[subject][method] = files[0][1]

    return latest


# ── Weibull ───────────────────────────────────────────────────────────────────
def weibull(x, alpha, beta):
    return CHANCE + (1 - CHANCE) * (1 - np.exp(-((x / alpha) ** beta)))


def fit_weibull(intensities, corrects):
    """Returns (alpha, beta, threshold_at_criterion) or None."""
    try:
        popt, _ = curve_fit(
            weibull, intensities, corrects,
            p0=[0.5, 2.0], bounds=([1e-3, 0.1], [10.0, 20.0]),
            maxfev=5000,
        )
        alpha, beta = popt
        thresh = alpha * (-np.log((1 - CRITERION) / (1 - CHANCE))) ** (1 / beta)
        return alpha, beta, thresh
    except Exception:
        return None


# ── load ───────────────────────────────────────────────────────────────────────
def load_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genotype_key"] = df.apply(
        lambda r: format_genotype_key(row_genotype_peaks(r)), axis=1
    )
    df["axis_peak_nm"] = df.apply(axis_peak_from_row, axis=1)
    return df


def _format_peak(peak) -> str:
    peak = float(peak)
    return f"{peak:.0f}" if peak.is_integer() else f"{peak:g}"


def row_genotype_peaks(row) -> tuple:
    peaks = []
    for col in ("genotype_1", "genotype_2", "genotype_3"):
        if col in row and pd.notna(row[col]):
            peaks.append(float(row[col]))
    return tuple(peaks)


def format_genotype_key(peaks: tuple) -> str:
    return ",".join(_format_peak(p) for p in peaks)


def parse_genotype_key(genotype_key: str) -> tuple:
    return tuple(float(part) for part in genotype_key.split(",") if part)


def axis_peak_from_row(row):
    peaks = sorted({420.0, *row_genotype_peaks(row)})
    axis = int(row["metameric_axis"])
    if 0 <= axis < len(peaks):
        return peaks[axis]
    return np.nan


def axis_label(genotype_key: str, axis: int, axis_peak) -> str:
    if pd.notna(axis_peak):
        peak_text = _format_peak(axis_peak)
        if abs(float(axis_peak) - 547.0) < 1e-6:
            return f"Axis {axis} (Q {peak_text} nm)"
        return f"Axis {axis} ({peak_text} nm)"

    peaks = sorted({420.0, *parse_genotype_key(genotype_key)})
    if 0 <= axis < len(peaks):
        return axis_label(genotype_key, axis, peaks[axis])
    return AXIS_LABEL.get(axis, f"Axis {axis}")


def ml_genotype_label(genotype_key: str) -> str:
    peaks = [p for p in parse_genotype_key(genotype_key) if abs(p - 547.0) > 1e-6]
    return ",".join(_format_peak(p) for p in peaks)


# ── one panel ─────────────────────────────────────────────────────────────────
def plot_panel(ax, mocs_group, quest_group, n_quest_bins=5):
    x_range = np.linspace(0, 1.05, 400)

    # ── Quest: binned bar chart ───────────────────────────────────────────────
    if quest_group is not None and len(quest_group) > 0:
        intens  = quest_group["intensity"].values
        correct = quest_group["correct"].values

        bins       = np.linspace(0, 1, n_quest_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width   = bins[1] - bins[0]
        labels      = pd.cut(intens, bins=bins, labels=bin_centers, include_lowest=True)
        quest_agg   = (
            pd.DataFrame({"bin": labels.astype(float), "correct": correct})
            .groupby("bin")["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        ax.bar(
            quest_agg["bin"], quest_agg["mean"],
            width=bin_width * 0.4, align="center",
            color=QUEST_COLOR, alpha=0.35, zorder=2,
            edgecolor=QUEST_COLOR, linewidth=0.5,
            label=f"Quest binned (n~{int(quest_agg['count'].median())}/bin)",
        )
        for _, row in quest_agg.iterrows():
            ax.text(row["bin"], row["mean"] + 0.03, f"{row['count']:.0f}",
                    ha="center", fontsize=6, color=QUEST_COLOR)

    # ── Quest: scatter + Weibull ──────────────────────────────────────────────
    if quest_group is not None and len(quest_group) > 0:
        intens  = quest_group["intensity"].values
        correct = quest_group["correct"].values

        rng    = np.random.default_rng(42)
        jitter = rng.uniform(-0.03, 0.03, size=len(correct))
        ax.scatter(intens, correct + jitter, s=10, color=QUEST_COLOR,
                   alpha=0.3, zorder=3, label=f"Quest trials (n={len(correct)})")

        fit = fit_weibull(intens, correct)
        if fit is not None:
            alpha, beta, thresh = fit
            ax.plot(x_range, weibull(x_range, alpha, beta),
                    color=QUEST_COLOR, lw=1.8, zorder=4,
                    label=f"Quest Weibull (thr={thresh:.2f})")
            if thresh < 1.1:
                ax.axvline(thresh, color=QUEST_COLOR, ls="--", lw=1.0, alpha=0.8, zorder=4)

    # ── MOCS: accuracy per bucket ─────────────────────────────────────────────
    if mocs_group is not None and len(mocs_group) > 0:
        agg = (
            mocs_group.groupby("intensity")["correct"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("intensity")
        )
        intens = agg["intensity"].values
        accs   = agg["mean"].values
        ns     = agg["count"].values

        bar_width = np.diff(intens).min() * 0.4 if len(intens) > 1 else 0.08
        ax.bar(intens, accs, width=bar_width, align="center",
               color=MOCS_COLOR, alpha=0.35, zorder=5,
               edgecolor=MOCS_COLOR, linewidth=0.5,
               label=f"MOCS binned (n~{int(np.median(ns))}/bucket)")
        ax.plot(intens, accs, "o-", color=MOCS_COLOR, lw=1.6, ms=4, zorder=6)
        for xi, yi, ni in zip(intens, accs, ns):
            ax.text(xi, yi + 0.05, f"{ni:.0f}", ha="center",
                    fontsize=7, color=MOCS_COLOR)

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(CHANCE, color=NEUTRAL_COLOR, ls=":", lw=1.0, alpha=0.7,
               label=f"Chance ({CHANCE:.2f})")
    ax.axhline(CRITERION, color=NEUTRAL_COLOR, ls="-.", lw=0.8, alpha=0.6,
               label=f"Criterion ({CRITERION:.2f})")

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.12, 1.18)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.85)


def legend_handles():
    return [
        Patch(
            facecolor=MOCS_COLOR,
            edgecolor=MOCS_COLOR,
            alpha=0.35,
            label="MOCS binned",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color=MOCS_COLOR,
            lw=1.6,
            markersize=4,
            label="MOCS mean",
        ),
        Patch(
            facecolor=QUEST_COLOR,
            edgecolor=QUEST_COLOR,
            alpha=0.35,
            label="Quest binned",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color=QUEST_COLOR,
            linestyle="None",
            alpha=0.3,
            markersize=4,
            label="Quest trials",
        ),
        Line2D(
            [0], [0],
            color=QUEST_COLOR,
            lw=1.8,
            label="Quest Weibull",
        ),
        Line2D(
            [0], [0],
            color=QUEST_COLOR,
            linestyle="--",
            lw=1.0,
            alpha=0.8,
            label="Quest threshold",
        ),
        Line2D(
            [0], [0],
            color=NEUTRAL_COLOR,
            linestyle=":",
            lw=1.0,
            alpha=0.7,
            label=f"Chance ({CHANCE:.2f})",
        ),
        Line2D(
            [0], [0],
            color=NEUTRAL_COLOR,
            linestyle="-.",
            lw=0.8,
            alpha=0.6,
            label=f"Criterion ({CRITERION:.2f})",
        ),
    ]


# ── main figure ───────────────────────────────────────────────────────────────
def make_figure(subject, mocs_df, quest_df, output_dir):
    panel_keys = sorted(
        set(zip(quest_df["genotype_key"], quest_df["metameric_axis"]))
        | set(zip(mocs_df["genotype_key"], mocs_df["metameric_axis"]))
    )
    if not panel_keys:
        print("No genotype/axis data to plot.")
        return

    n_panels = len(panel_keys)
    n_slots = n_panels + 1
    n_cols = min(4, n_slots)
    n_rows = int(np.ceil(n_slots / n_cols))

    fig_width = FULL_PAGE
    fig_height = max(3.0, 2.0 * n_rows)
    fig, grid = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    # fig.suptitle(
    #     f"MOCS vs Quest: {subject}",
    #     fontweight="bold",
    # )

    for panel_idx, (geno, axis) in enumerate(panel_keys):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = grid[row][col]

        q_grp = quest_df[
            (quest_df["genotype_key"] == geno) &
            (quest_df["metameric_axis"] == axis)
        ]
        m_grp = mocs_df[
            (mocs_df["genotype_key"] == geno) &
            (mocs_df["metameric_axis"] == axis)
        ]

        plot_panel(
            ax,
            m_grp if len(m_grp) > 0 else None,
            q_grp if len(q_grp) > 0 else None,
        )

        axis_peaks = pd.concat([
            m_grp["axis_peak_nm"] if len(m_grp) > 0 else pd.Series(dtype=float),
            q_grp["axis_peak_nm"] if len(q_grp) > 0 else pd.Series(dtype=float),
        ]).dropna().unique()
        axis_peak = axis_peaks[0] if len(axis_peaks) > 0 else np.nan
        title = f"({ml_genotype_label(geno)})"
        ax.set_title(title, fontsize=7, fontweight="bold")

    legend_idx = n_rows * n_cols - 1
    legend_row = legend_idx // n_cols
    legend_col = legend_idx % n_cols
    legend_ax = grid[legend_row][legend_col]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_xlabel("")
    legend_ax.set_ylabel("")
    legend_ax.grid(False)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    legend_ax.legend(
        handles=legend_handles(),
        loc="center",
        frameon=True,
        framealpha=0.95,
        borderpad=0.7,
        labelspacing=0.6,
        handlelength=2.0,
    )

    for empty_idx in range(n_panels, n_rows * n_cols):
        if empty_idx == legend_idx:
            continue
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        grid[row][col].set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = output_dir / f"{subject}_mocs_vs_quest.png"
    out_pdf = output_dir / f"{subject}_mocs_vs_quest.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    apply_style()
    if shutil.which("latex") is None:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
        })

    parser = argparse.ArgumentParser(description="Compare MOCS vs Quest for a subject")
    parser.add_argument("--subject",  type=str, default=None,
                        help="Subject ID to analyse (e.g. jessica-4-18)")
    parser.add_argument("--list",     action="store_true",
                        help="List all detected subjects and exit")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir \
                 else data_dir.parent / "mocs_quest_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = discover_files(data_dir)

    if args.list:
        print("Detected subjects:")
        for subj, methods in sorted(all_files.items()):
            print(f"  {subj}:")
            for method, path in sorted(methods.items()):
                print(f"    {method}: {path.name}")
        return

    if args.subject is None:
        parser.error("Provide --subject <id> or use --list to see available subjects.")

    subject = args.subject
    if subject not in all_files:
        print(f"Subject '{subject}' not found. Available: {sorted(all_files.keys())}")
        return

    methods = all_files[subject]
    if "Genetic" not in methods:
        print(f"No Genetic (MOCS) file found for subject '{subject}'.")
        return
    if "Quest" not in methods:
        print(f"No Quest file found for subject '{subject}'.")
        return

    print(f"Subject:      {subject}")
    print(f"MOCS file:    {methods['Genetic'].name}")
    print(f"Quest file:   {methods['Quest'].name}")

    mocs_df  = load_trials(methods["Genetic"])
    quest_df = load_trials(methods["Quest"])

    make_figure(subject, mocs_df, quest_df, output_dir)


if __name__ == "__main__":
    main()
