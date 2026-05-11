"""
Summarize MOCS (Genetic) pseudo-isochromatic data for multiple subjects.

Given a list of subject IDs, this script finds the latest Genetic trial CSV for
each subject, computes accuracy for each genotype/metameric-axis group, writes a
summary CSV, and plots one bar-chart row per subject.

Usage:
    python summarize_mocs_subjects.py --subjects hannah-5-6 jess-5-6 lauren-5-6
    python summarize_mocs_subjects.py --subjects-file subjects.txt
    python summarize_mocs_subjects.py --list
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

TETRIUM_COLOR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TETRIUM_COLOR_ROOT))

from TetriumColor.Plotting.PlotStyle import apply_style, COLORS, DOUBLE_COL


DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data"
    / "AppPseudoIsochromaticTest"
)

CHANCE = 1 / 4
MOCS_COLOR = COLORS[420]
NEUTRAL_COLOR = "#5c5c5c"
BOTTOM_TICK_FONT = "Linux Biolinum O"
FULL_PAGE = DOUBLE_COL

_FILE_RE = re.compile(
    r"^(?P<subject>.+?)_(?P<method>Genetic|Quest)_(?P<rest>.+?)_(?P<ts>\d{8}_\d{6}_\d{3})$"
)


def discover_files(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Return latest Genetic/Quest trial CSVs by subject."""
    index = defaultdict(lambda: defaultdict(list))

    for csv in data_dir.glob("*.csv"):
        if csv.name.lower().endswith("_thresholds.csv"):
            continue
        match = _FILE_RE.match(csv.stem)
        if match is None:
            continue
        subject = match.group("subject")
        method = match.group("method")
        timestamp = match.group("ts")
        index[subject][method].append((timestamp, csv))

    latest = {}
    for subject, methods in index.items():
        latest[subject] = {}
        for method, files in methods.items():
            files.sort(key=lambda item: item[0], reverse=True)
            latest[subject][method] = files[0][1]
    return latest


def read_trial_scalar_columns(path: Path) -> pd.DataFrame:
    """Read stable scalar columns from trial CSVs.

    Recent logs include list-valued metadata fields that may be unquoted. The
    scalar analysis columns all appear before color_picking_space, so read that
    stable prefix directly.
    """
    with path.open("r", newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        if "color_picking_space" in header:
            prefix_len = header.index("color_picking_space") + 1
        else:
            prefix_len = len(header)
        prefix_header = header[:prefix_len]

        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", maxsplit=prefix_len - 1)
            if len(parts) < prefix_len:
                parts.extend([""] * (prefix_len - len(parts)))
            rows.append(parts[:prefix_len])

    df = pd.DataFrame(rows, columns=prefix_header)
    numeric_cols = (
        "genotype_1",
        "genotype_2",
        "genotype_3",
        "metameric_axis",
        "correct",
        "intensity",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_peak(peak) -> str:
    peak = float(peak)
    return f"{peak:.0f}" if peak.is_integer() else f"{peak:g}"


def row_genotype_peaks(row) -> tuple[float, ...]:
    peaks = []
    for col in ("genotype_1", "genotype_2", "genotype_3"):
        if col in row and pd.notna(row[col]):
            peaks.append(float(row[col]))
    return tuple(peaks)


def format_genotype_key(peaks: tuple[float, ...]) -> str:
    return ",".join(_format_peak(p) for p in peaks)


def parse_genotype_key(genotype_key: str) -> tuple[float, ...]:
    return tuple(float(part) for part in genotype_key.split(",") if part)


def ml_genotype_label(genotype_key: str) -> str:
    """Label genotype by M/L peaks, omitting the fixed Q peak when present."""
    peaks = [p for p in parse_genotype_key(genotype_key) if abs(p - 547.0) > 1e-6]
    return ",".join(_format_peak(p) for p in peaks)


def axis_peak_from_row(row) -> float:
    peaks = sorted({420.0, *row_genotype_peaks(row)})
    axis = int(row["metameric_axis"])
    if 0 <= axis < len(peaks):
        return float(peaks[axis])
    return np.nan


def load_mocs_trials(path: Path) -> pd.DataFrame:
    df = read_trial_scalar_columns(path)
    required = {"genotype_1", "genotype_2", "genotype_3", "metameric_axis", "correct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df["genotype_key"] = df.apply(
        lambda row: format_genotype_key(row_genotype_peaks(row)),
        axis=1,
    )
    df["axis_peak_nm"] = df.apply(axis_peak_from_row, axis=1)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    return df.dropna(subset=["genotype_key", "metameric_axis", "correct"])


def compile_summary(
    subjects: list[str],
    all_files: dict[str, dict[str, Path]],
    intensity: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    records = []
    missing = []
    subject_numbers = {subject: idx + 1 for idx, subject in enumerate(subjects)}

    for subject in subjects:
        methods = all_files.get(subject)
        if not methods or "Genetic" not in methods:
            missing.append(subject)
            continue

        path = methods["Genetic"]
        df = load_mocs_trials(path)
        if intensity is not None:
            if "intensity" not in df.columns:
                raise ValueError(f"{path.name} has no intensity column for --intensity filtering")
            df = df[np.isclose(df["intensity"], intensity)]

        grouped = (
            df.groupby(["genotype_key", "metameric_axis", "axis_peak_nm"], dropna=False)["correct"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "sum": "n_correct", "count": "n_total"})
        )
        grouped["subject"] = subject
        grouped["subject_number"] = subject_numbers[subject]
        grouped["source_file"] = path.name
        records.append(grouped)

    if not records:
        return pd.DataFrame(), missing

    summary = pd.concat(records, ignore_index=True)
    summary["metameric_axis"] = summary["metameric_axis"].astype(int)
    summary["genotype_label"] = summary["genotype_key"].map(ml_genotype_label)
    return summary[
        [
            "subject",
            "subject_number",
            "genotype_key",
            "genotype_label",
            "metameric_axis",
            "axis_peak_nm",
            "accuracy",
            "n_correct",
            "n_total",
            "source_file",
        ]
    ], missing


def _genotype_sort_key(genotype_key: str) -> tuple[float, ...]:
    return parse_genotype_key(genotype_key)


def ordered_metamer_keys(summary: pd.DataFrame) -> list[tuple[str, int]]:
    keys = set(zip(summary["genotype_key"], summary["metameric_axis"]))
    return sorted(keys, key=lambda item: (_genotype_sort_key(item[0]), int(item[1])))


def x_tick_labels(keys: list[tuple[str, int]]) -> list[str]:
    base_labels = [ml_genotype_label(genotype_key) for genotype_key, _ in keys]
    duplicated = {label for label in base_labels if base_labels.count(label) > 1}
    labels = []
    for (genotype_key, axis), label in zip(keys, base_labels):
        if label in duplicated:
            labels.append(f"{label}\naxis {axis}")
        else:
            labels.append(label)
    return labels


def plot_subject_rows(
    summary: pd.DataFrame,
    subjects: list[str],
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    keys = ordered_metamer_keys(summary)
    labels = x_tick_labels(keys)
    x = np.arange(len(keys))

    fig_width = max(FULL_PAGE, 0.42 * max(len(keys), 1) + 1.8)
    fig_height = max(1.12, 0.7 * (1.05 * len(subjects) + 0.65))
    fig, axes = plt.subplots(
        len(subjects),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for row_idx, subject in enumerate(subjects):
        ax = axes[row_idx]
        subject_summary = summary[summary["subject"] == subject]
        subject_number = int(subject_summary["subject_number"].iloc[0])
        by_key = {
            (record.genotype_key, int(record.metameric_axis)): record
            for record in subject_summary.itertuples(index=False)
        }
        heights = [
            float(by_key[key].accuracy) if key in by_key else np.nan
            for key in keys
        ]
        counts = [
            int(by_key[key].n_total) if key in by_key else 0
            for key in keys
        ]

        bars = ax.bar(
            x,
            np.nan_to_num(heights, nan=0.0),
            width=0.72,
            color=MOCS_COLOR,
            edgecolor=MOCS_COLOR,
            alpha=0.45,
            linewidth=0.6,
        )
        for bar, height, count in zip(bars, heights, counts):
            if np.isnan(height) or count == 0:
                bar.set_alpha(0.08)
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 0.045, 1.08),
                f"{height:.2f}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=5,
                color=NEUTRAL_COLOR,
            )

        ax.axhline(CHANCE, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"Subject {subject_number}", rotation=90, ha="center", va="center", labelpad=12)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row_idx != len(subjects) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(labels, rotation=0, ha="center")
    for tick in bottom_ax.get_xticklabels():
        tick.set_fontname(BOTTOM_TICK_FONT)
    bottom_ax.set_xlabel("Genotype")
    fig.supylabel("Accuracy")
    fig.legend(
        handles=[
            Patch(facecolor=MOCS_COLOR, edgecolor=MOCS_COLOR, alpha=0.45, label="MOCS accuracy"),
        ],
        loc="upper right",
        frameon=False,
    )

    plt.tight_layout()
    png_path = output_dir / f"{output_prefix}.png"
    pdf_path = output_dir / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def read_subjects_file(path: Path) -> list[str]:
    subjects = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        subjects.extend(part.strip() for part in line.split(",") if part.strip())
    return subjects


def main() -> None:
    apply_style()
    if shutil.which("latex") is None:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
        })

    parser = argparse.ArgumentParser(
        description="Compile and plot MOCS accuracy by genotype for selected subjects"
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs to include")
    parser.add_argument("--subjects-file", type=str, default=None, help="Text file of subject IDs")
    parser.add_argument("--list", action="store_true", help="List detected subjects and exit")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="mocs_subject_summary")
    parser.add_argument("--intensity", type=float, default=None, help="Optional MOCS intensity filter")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / "mocs_subject_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = discover_files(data_dir)

    if args.list:
        print("Detected subjects with MOCS files:")
        for subject, methods in sorted(all_files.items()):
            if "Genetic" in methods:
                print(f"  {subject}: {methods['Genetic'].name}")
        return

    subjects = []
    if args.subjects:
        subjects.extend(args.subjects)
    if args.subjects_file:
        subjects.extend(read_subjects_file(Path(args.subjects_file)))
    subjects = list(dict.fromkeys(subjects))

    if not subjects:
        parser.error("Provide --subjects and/or --subjects-file, or use --list.")

    summary, missing = compile_summary(subjects, all_files, intensity=args.intensity)
    if missing:
        print("Subjects without latest Genetic/MOCS file:")
        for subject in missing:
            print(f"  {subject}")

    if summary.empty:
        raise SystemExit("No MOCS data found for requested subjects.")

    plotted_subjects = [subject for subject in subjects if subject in set(summary["subject"])]
    csv_path = output_dir / f"{args.output_prefix}.csv"
    summary.to_csv(csv_path, index=False)
    png_path, pdf_path = plot_subject_rows(summary, plotted_subjects, output_dir, args.output_prefix)

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved plot PNG:    {png_path}")
    print(f"Saved plot PDF:    {pdf_path}")


if __name__ == "__main__":
    main()
