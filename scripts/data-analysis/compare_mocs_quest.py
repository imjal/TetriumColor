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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy.optimize import curve_fit

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data" / "AppPseudoIsochromaticTest"
)

CHANCE    = 1 / 4   # 4-AFC (up/down/left/right)
CRITERION = 0.50    # Quest uses pThreshold=0.5 (TetraColorPicker.py)
AXIS_LABEL = {1: "Axis 1 (S)", 2: "Axis 2 (Q)", 3: "Axis 3 (L)"}

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
        lambda r: f"{r.genotype_1:.0f},{r.genotype_2:.0f}", axis=1
    )
    return df


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
            color="darkorange", alpha=0.5, zorder=2,
            label=f"Quest binned (n≈{int(quest_agg['count'].median())}/bin)",
        )
        for _, row in quest_agg.iterrows():
            ax.text(row["bin"], row["mean"] + 0.03, f"{row['count']:.0f}",
                    ha="center", fontsize=6, color="darkorange")

    # ── Quest: scatter + Weibull ──────────────────────────────────────────────
    if quest_group is not None and len(quest_group) > 0:
        intens  = quest_group["intensity"].values
        correct = quest_group["correct"].values

        rng    = np.random.default_rng(42)
        jitter = rng.uniform(-0.03, 0.03, size=len(correct))
        ax.scatter(intens, correct + jitter, s=10, color="darkorange",
                   alpha=0.3, zorder=3, label=f"Quest trials (n={len(correct)})")

        fit = fit_weibull(intens, correct)
        if fit is not None:
            alpha, beta, thresh = fit
            ax.plot(x_range, weibull(x_range, alpha, beta),
                    color="darkorange", lw=2.5, zorder=4,
                    label=f"Quest Weibull (θ={thresh:.2f})")
            if thresh < 1.1:
                ax.axvline(thresh, color="darkorange", ls="--", lw=1.5, alpha=0.8, zorder=4)

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
               color="steelblue", alpha=0.5, zorder=5,
               label=f"MOCS binned (n≈{int(np.median(ns))}/bucket)")
        ax.plot(intens, accs, "o-", color="steelblue", lw=2, ms=7, zorder=6)
        for xi, yi, ni in zip(intens, accs, ns):
            ax.text(xi, yi + 0.05, f"{ni:.0f}", ha="center",
                    fontsize=7, color="steelblue")

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(CHANCE,    color="gray", ls=":",  lw=1.5, alpha=0.7, label=f"Chance ({CHANCE:.0%})")
    ax.axhline(CRITERION, color="gray", ls="-.", lw=1.0, alpha=0.5, label=f"Criterion ({CRITERION:.0%})")

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.12, 1.18)
    ax.set_xlabel("Intensity (proportion)", fontsize=10)
    ax.set_ylabel("Accuracy (p correct)",   fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")


# ── main figure ───────────────────────────────────────────────────────────────
def make_figure(subject, mocs_df, quest_df, output_dir):
    genotypes = sorted(
        set(quest_df["genotype_key"].unique()) | set(mocs_df["genotype_key"].unique())
    )
    axes = sorted(
        set(quest_df["metameric_axis"].unique()) | set(mocs_df["metameric_axis"].unique())
    )

    n_geno = len(genotypes)
    n_axes = len(axes)

    fig, grid = plt.subplots(
        n_axes, n_geno,
        figsize=(3.8 * n_geno, 3.8 * n_axes),
        squeeze=False,
    )
    fig.suptitle(
        f"MOCS vs Quest — subject: {subject}\n"
        "Blue = MOCS accuracy per bucket  |  Orange = Quest scatter + Weibull",
        fontsize=13, fontweight="bold",
    )

    for row, axis in enumerate(axes):
        for col, geno in enumerate(genotypes):
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

            has_both = len(m_grp) > 0 and len(q_grp) > 0
            title = f"({geno})  {AXIS_LABEL.get(axis, f'Axis {axis}')}"
            if has_both:
                title += "  ★"
            ax.set_title(title, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = output_dir / f"{subject}_mocs_vs_quest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
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
