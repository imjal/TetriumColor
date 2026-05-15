"""
Summarize Quest and MOCS pseudo-isochromatic data for multiple subjects.

This is the multi-subject companion to summarize_mocs_subjects.py. It finds the
latest Genetic/MOCS and Quest files for each requested subject, summarizes
thresholds by genotype/metameric axis, writes summary CSVs, plots subject
columns by observer/genotype rows, and emits a genotype conclusion table.

Usage:
    python summarize_quest_mocs_subjects.py --subjects hannah-5-6 jess-5-6 lauren-5-6
    python summarize_quest_mocs_subjects.py --subjects-file subjects.txt --plot-type threshold-lines
    python summarize_quest_mocs_subjects.py --subjects-file subjects.txt --plot-type compare-grid --plot-mode both
    python summarize_quest_mocs_subjects.py --all --plot-mode both
    python summarize_quest_mocs_subjects.py --list
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

TETRIUM_COLOR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TETRIUM_COLOR_ROOT))

from TetriumColor.Plotting.PlotStyle import apply_style, COLORS, DOUBLE_COL


DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data"
    / "AppPseudoIsochromaticTest"
)

CHANCE = 1 / 4
DEFAULT_CRITERION = 0.625
MOCS_COLOR = COLORS[420]
QUEST_COLOR = COLORS[551]
NEUTRAL_COLOR = "#5c5c5c"
FULL_PAGE = DOUBLE_COL
MAX_FULL_PAGE_FIG_HEIGHT = 9.0
LABEL_FONT_SIZE = 6
X_TICK_FONT_SIZE = 8
Y_TICK_FONT_SIZE = 6
AXIS_LABEL_FONT_SIZE = 8
SUBJECT_ID_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
GAMUT_MAX = 1.0
CENSORED_COLOR = "#c62828"

_FILE_RE = re.compile(
    r"^(?P<subject>.+?)_(?P<method>Genetic|Quest)_(?P<rest>.+?)_(?P<ts>\d{8}_\d{6}_\d{3})(?P<thresholds>_thresholds)?$"
)


def discover_files(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Return latest trial and Quest threshold CSVs by subject."""
    index = defaultdict(lambda: defaultdict(list))

    for csv in data_dir.glob("*.csv"):
        match = _FILE_RE.match(csv.stem)
        if match is None:
            continue
        subject = match.group("subject")
        method = match.group("method")
        timestamp = match.group("ts")
        key = f"{method}Thresholds" if match.group("thresholds") else method
        index[subject][key].append((timestamp, csv))

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
    return tuple(float(part) for part in str(genotype_key).split(",") if part)


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


def axis_peak_from_key(genotype_key: str, axis: int) -> float:
    peaks = sorted({420.0, *parse_genotype_key(genotype_key)})
    if 0 <= axis < len(peaks):
        return float(peaks[axis])
    return np.nan


def observer_label(genotype_key: str, axis: int, axis_peak_nm: float | None = None) -> str:
    peak = axis_peak_nm
    if peak is None or pd.isna(peak):
        peak = axis_peak_from_key(genotype_key, axis)
    peak_text = f" axis {axis}"
    if pd.notna(peak):
        peak_text = f" {int(axis)}:{_format_peak(peak)}"
    return f"{ml_genotype_label(genotype_key)}{peak_text}"


def _genotype_sort_key(genotype_key: str) -> tuple[float, ...]:
    return parse_genotype_key(genotype_key)


def observer_sort_key(item: tuple[str, int]) -> tuple[tuple[float, ...], int]:
    genotype_key, axis = item
    return _genotype_sort_key(genotype_key), int(axis)


def weibull(x, alpha, beta):
    return CHANCE + (1 - CHANCE) * (1 - np.exp(-((x / alpha) ** beta)))


def wilson_ci(n_correct: float, n_total: float, z: float = 1.96) -> tuple[float, float]:
    if not np.isfinite(n_correct) or not np.isfinite(n_total) or n_total <= 0:
        return np.nan, np.nan
    p = np.clip(float(n_correct) / float(n_total), 0.0, 1.0)
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half_width = z * np.sqrt((p * (1 - p) / n_total) + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def fit_weibull(intensities, corrects, criterion: float = DEFAULT_CRITERION):
    """Return alpha, beta, threshold at criterion, or None."""
    intensities = np.asarray(intensities, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    mask = np.isfinite(intensities) & np.isfinite(corrects)
    intensities = intensities[mask]
    corrects = corrects[mask]
    if len(intensities) < 5 or len(np.unique(intensities)) < 2:
        return None
    try:
        popt, pcov = curve_fit(
            weibull,
            intensities,
            corrects,
            p0=[0.5, 2.0],
            bounds=([1e-3, 0.1], [10.0, 20.0]),
            maxfev=5000,
        )
        alpha, beta = popt
        const = -np.log((1 - criterion) / (1 - CHANCE))
        threshold = alpha * const ** (1 / beta)
        threshold_se = np.nan
        if np.all(np.isfinite(pcov)):
            grad_alpha = const ** (1 / beta)
            grad_beta = threshold * (-np.log(const) / (beta ** 2))
            grad = np.array([grad_alpha, grad_beta], dtype=float)
            threshold_var = float(grad @ pcov @ grad.T)
            if threshold_var >= 0:
                threshold_se = float(np.sqrt(threshold_var))
        return float(alpha), float(beta), float(threshold), threshold_se
    except Exception:
        return None


def load_trials(path: Path) -> pd.DataFrame:
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
    return df.dropna(subset=["genotype_key", "metameric_axis", "correct"])


def summarize_trials(
    df: pd.DataFrame,
    subject: str,
    method: str,
    source_file: str,
    intensity: float | None = None,
    criterion: float = DEFAULT_CRITERION,
) -> pd.DataFrame:
    if intensity is not None:
        if "intensity" not in df.columns:
            raise ValueError(f"{source_file} has no intensity column for --intensity filtering")
        df = df[np.isclose(df["intensity"], intensity)]

    records = []
    group_cols = ["genotype_key", "metameric_axis", "axis_peak_nm"]
    for (genotype_key, axis, axis_peak_nm), group in df.groupby(group_cols, dropna=False):
        max_intensity = np.nan
        max_accuracy = np.nan
        max_n_correct = np.nan
        max_n_total = np.nan
        max_ci_low = np.nan
        max_ci_high = np.nan
        if "intensity" in group.columns and group["intensity"].notna().any():
            max_intensity = float(group["intensity"].max())
            max_group = group[np.isclose(group["intensity"], max_intensity)]
            max_n_correct = int(max_group["correct"].sum())
            max_n_total = int(max_group["correct"].count())
            max_accuracy = float(max_group["correct"].mean())
            max_ci_low, max_ci_high = wilson_ci(max_n_correct, max_n_total)
        record = {
            "subject": subject,
            "method": method,
            "genotype_key": genotype_key,
            "genotype_label": ml_genotype_label(genotype_key),
            "metameric_axis": int(axis),
            "axis_peak_nm": axis_peak_nm,
            "accuracy": float(group["correct"].mean()),
            "n_correct": int(group["correct"].sum()),
            "n_total": int(group["correct"].count()),
            "max_intensity": max_intensity,
            "max_accuracy": max_accuracy,
            "max_n_correct": max_n_correct,
            "max_n_total": max_n_total,
            "max_accuracy_ci_low": max_ci_low,
            "max_accuracy_ci_high": max_ci_high,
            "threshold_raw": np.nan,
            "threshold_censored": False,
            "threshold_report": "",
            "threshold": np.nan,
            "threshold_ci_low": np.nan,
            "threshold_ci_high": np.nan,
            "threshold_criterion": criterion,
            "threshold_source": "unavailable",
            "source_file": source_file,
        }
        if "intensity" in group.columns:
            fit = fit_weibull(
                group["intensity"].to_numpy(),
                group["correct"].to_numpy(),
                criterion=criterion,
            )
            if fit is not None:
                _alpha, _beta, threshold, threshold_se = fit
                record["threshold_raw"] = threshold
                record["threshold_censored"] = bool(threshold > GAMUT_MAX)
                record["threshold"] = min(threshold, GAMUT_MAX)
                record["threshold_report"] = (
                    f"threshold &gt gamut max ({GAMUT_MAX:g})"
                    if threshold > GAMUT_MAX
                    else f"{threshold:.6g}"
                )
                if np.isfinite(threshold_se):
                    record["threshold_ci_low"] = max(0.0, record["threshold"] - 1.96 * threshold_se)
                    record["threshold_ci_high"] = min(GAMUT_MAX, record["threshold"] + 1.96 * threshold_se)
                record["threshold_source"] = "trial_weibull"
            elif method == "Quest":
                record["threshold_raw"] = np.nan
                record["threshold_censored"] = True
                record["threshold"] = GAMUT_MAX
                record["threshold_report"] = f"threshold not estimable; censored at gamut max ({GAMUT_MAX:g})"
                record["threshold_source"] = "trial_weibull_unavailable"
        records.append(record)

    return pd.DataFrame.from_records(records)


def load_quest_thresholds(path: Path, subject: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"genotype", "metameric_axis"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    threshold_col = None
    for col in ("threshold_proportion", "threshold_distance"):
        if col in df.columns:
            threshold_col = col
            break
    if threshold_col is None:
        raise ValueError(f"{path.name} has no threshold_proportion or threshold_distance column")

    out = df.copy()
    out["genotype_key"] = out["genotype"].astype(str).str.replace(" ", "", regex=False)
    out["metameric_axis"] = pd.to_numeric(out["metameric_axis"], errors="coerce")
    out["threshold"] = pd.to_numeric(out[threshold_col], errors="coerce")
    out["axis_peak_nm"] = out.apply(
        lambda row: axis_peak_from_key(row["genotype_key"], int(row["metameric_axis"])),
        axis=1,
    )
    out["subject"] = subject
    out["method"] = "Quest"
    out["genotype_label"] = out["genotype_key"].map(ml_genotype_label)
    out["accuracy"] = np.nan
    out["n_correct"] = np.nan
    out["n_total"] = np.nan
    out["max_intensity"] = np.nan
    out["max_accuracy"] = np.nan
    out["max_n_correct"] = np.nan
    out["max_n_total"] = np.nan
    out["max_accuracy_ci_low"] = np.nan
    out["max_accuracy_ci_high"] = np.nan
    out["threshold_raw"] = out["threshold"]
    out["threshold_censored"] = out["threshold_raw"] > GAMUT_MAX
    out["threshold"] = out["threshold_raw"].clip(upper=GAMUT_MAX)
    out["threshold_report"] = out.apply(
        lambda row: (
            f"threshold > gamut max ({GAMUT_MAX:g})"
            if bool(row["threshold_censored"])
            else f"{float(row['threshold_raw']):.6g}"
        ),
        axis=1,
    )
    out["threshold_ci_low"] = np.nan
    out["threshold_ci_high"] = np.nan
    out["threshold_criterion"] = np.nan
    out["threshold_source"] = threshold_col
    out["source_file"] = path.name
    return out[
        [
            "subject",
            "method",
            "genotype_key",
            "genotype_label",
            "metameric_axis",
            "axis_peak_nm",
            "accuracy",
            "n_correct",
            "n_total",
            "max_intensity",
            "max_accuracy",
            "max_n_correct",
            "max_n_total",
            "max_accuracy_ci_low",
            "max_accuracy_ci_high",
            "threshold_raw",
            "threshold_censored",
            "threshold_report",
            "threshold",
            "threshold_ci_low",
            "threshold_ci_high",
            "threshold_criterion",
            "threshold_source",
            "source_file",
        ]
    ].dropna(subset=["metameric_axis"])


def attach_quest_trial_ci(thresholds: pd.DataFrame, quest_trials: pd.DataFrame) -> pd.DataFrame:
    """Attach trial-fit CIs to exported Quest threshold rows when possible."""
    trial_summary = summarize_trials(quest_trials, "", "Quest", "").copy()
    if trial_summary.empty:
        return thresholds
    ci_cols = [
        "genotype_key",
        "metameric_axis",
        "threshold_ci_low",
        "threshold_ci_high",
        "threshold_source",
    ]
    trial_summary = trial_summary[ci_cols].rename(
        columns={
            "threshold_ci_low": "trial_threshold_ci_low",
            "threshold_ci_high": "trial_threshold_ci_high",
            "threshold_source": "trial_threshold_source",
        }
    )
    out = thresholds.merge(
        trial_summary,
        on=["genotype_key", "metameric_axis"],
        how="left",
    )
    has_ci = (
        np.isfinite(out["trial_threshold_ci_low"])
        & np.isfinite(out["trial_threshold_ci_high"])
        & ~out["threshold_censored"].astype(bool)
    )
    half_width = (
        out.loc[has_ci, "trial_threshold_ci_high"]
        - out.loc[has_ci, "trial_threshold_ci_low"]
    ) / 2
    out.loc[has_ci, "threshold_ci_low"] = np.maximum(0.0, out.loc[has_ci, "threshold"] - half_width)
    out.loc[has_ci, "threshold_ci_high"] = np.minimum(
        GAMUT_MAX,
        out.loc[has_ci, "threshold"] + half_width,
    )
    out.loc[has_ci, "threshold_source"] = out.loc[has_ci, "threshold_source"] + "+trial_ci"
    return out.drop(columns=["trial_threshold_ci_low", "trial_threshold_ci_high", "trial_threshold_source"])


def attach_mocs_support_to_censored_quest(summary: pd.DataFrame) -> pd.DataFrame:
    """For censored Quest thresholds, attach matching max-contrast MOCS evidence."""
    out = summary.copy()
    support_cols = [
        "censor_support_accuracy",
        "censor_support_ci_low",
        "censor_support_ci_high",
        "censor_support_n_correct",
        "censor_support_n_total",
        "censor_support_intensity",
    ]
    for col in support_cols:
        if col not in out.columns:
            out[col] = np.nan

    mocs = out[out["method"] == "MOCS"]
    if mocs.empty:
        return out

    mocs_support = {
        (row.subject, row.genotype_key, int(row.metameric_axis)): row
        for row in mocs.itertuples(index=False)
    }
    quest_mask = (out["method"] == "Quest") & out["threshold_censored"].astype(bool)
    for idx, row in out[quest_mask].iterrows():
        support = mocs_support.get((row["subject"], row["genotype_key"], int(row["metameric_axis"])))
        if support is None:
            continue
        out.at[idx, "censor_support_accuracy"] = support.max_accuracy
        out.at[idx, "censor_support_ci_low"] = support.max_accuracy_ci_low
        out.at[idx, "censor_support_ci_high"] = support.max_accuracy_ci_high
        out.at[idx, "censor_support_n_correct"] = support.max_n_correct
        out.at[idx, "censor_support_n_total"] = support.max_n_total
        out.at[idx, "censor_support_intensity"] = support.max_intensity
    return out


def compile_summary(
    subjects: list[str],
    all_files: dict[str, dict[str, Path]],
    intensity: float | None = None,
    criterion: float = DEFAULT_CRITERION,
) -> tuple[pd.DataFrame, list[str]]:
    records = []
    missing = []

    for subject in subjects:
        methods = all_files.get(subject)
        if not methods:
            missing.append(subject)
            continue

        found_any = False
        if "Genetic" in methods:
            mocs_df = load_trials(methods["Genetic"])
            records.append(
                summarize_trials(
                    mocs_df,
                    subject,
                    "MOCS",
                    methods["Genetic"].name,
                    intensity=intensity,
                    criterion=criterion,
                )
            )
            found_any = True

        if "Quest" in methods:
            quest_df = load_trials(methods["Quest"])
            records.append(
                summarize_trials(
                    quest_df,
                    subject,
                    "Quest",
                    methods["Quest"].name,
                    intensity=None,
                    criterion=criterion,
                )
            )
            found_any = True
        elif "QuestThresholds" in methods:
            quest_thresholds = load_quest_thresholds(methods["QuestThresholds"], subject)
            records.append(quest_thresholds)
            found_any = True

        if not found_any:
            missing.append(subject)

    if not records:
        return pd.DataFrame(), missing

    summary = pd.concat(records, ignore_index=True)
    summary = attach_mocs_support_to_censored_quest(summary)
    summary["metameric_axis"] = summary["metameric_axis"].astype(int)
    summary["observer"] = summary.apply(
        lambda row: observer_label(row["genotype_key"], int(row["metameric_axis"]), row["axis_peak_nm"]),
        axis=1,
    )
    return summary[
        [
            "subject",
            "method",
            "observer",
            "genotype_key",
            "genotype_label",
            "metameric_axis",
            "axis_peak_nm",
            "accuracy",
            "n_correct",
            "n_total",
            "max_intensity",
            "max_accuracy",
            "max_n_correct",
            "max_n_total",
            "max_accuracy_ci_low",
            "max_accuracy_ci_high",
            "threshold_raw",
            "threshold_censored",
            "threshold_report",
            "threshold",
            "threshold_ci_low",
            "threshold_ci_high",
            "threshold_criterion",
            "censor_support_accuracy",
            "censor_support_ci_low",
            "censor_support_ci_high",
            "censor_support_n_correct",
            "censor_support_n_total",
            "censor_support_intensity",
            "threshold_source",
            "source_file",
        ]
    ], missing


def filter_plot_mode(summary: pd.DataFrame, plot_mode: str) -> pd.DataFrame:
    if plot_mode == "mocs":
        return summary[summary["method"] == "MOCS"].copy()
    if plot_mode == "quest":
        return summary[summary["method"] == "Quest"].copy()
    return summary[summary["method"].isin(["MOCS", "Quest"])].copy()


def load_subject_trial_data(
    subjects: list[str],
    all_files: dict[str, dict[str, Path]],
    intensity: float | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load raw trial data for compare_mocs_quest-style panels."""
    subject_data: dict[str, dict[str, pd.DataFrame]] = {}
    for subject in subjects:
        methods = all_files.get(subject, {})
        loaded: dict[str, pd.DataFrame] = {}
        if "Genetic" in methods:
            mocs_df = load_trials(methods["Genetic"])
            if intensity is not None:
                if "intensity" not in mocs_df.columns:
                    raise ValueError(f"{methods['Genetic'].name} has no intensity column")
                mocs_df = mocs_df[np.isclose(mocs_df["intensity"], intensity)]
            loaded["MOCS"] = mocs_df
        if "Quest" in methods:
            loaded["Quest"] = load_trials(methods["Quest"])
        if loaded:
            subject_data[subject] = loaded
    return subject_data


def panel_keys_for_subject_data(
    subject_data: dict[str, dict[str, pd.DataFrame]],
    plot_mode: str,
) -> list[tuple[str, int]]:
    key_sets = []
    for methods in subject_data.values():
        if plot_mode in ("mocs", "both") and "MOCS" in methods:
            key_sets.append(set(zip(methods["MOCS"]["genotype_key"], methods["MOCS"]["metameric_axis"])))
        if plot_mode in ("quest", "both") and "Quest" in methods:
            key_sets.append(set(zip(methods["Quest"]["genotype_key"], methods["Quest"]["metameric_axis"])))
    keys = set().union(*key_sets) if key_sets else set()
    return sorted(((g, int(a)) for g, a in keys), key=observer_sort_key)


def plot_compare_panel(
    ax,
    mocs_group: pd.DataFrame | None,
    quest_group: pd.DataFrame | None,
    n_quest_bins: int = 5,
    show_x_label: bool = False,
    show_y_label: bool = False,
    criterion: float = DEFAULT_CRITERION,
) -> None:
    """Compact version of compare_mocs_quest.plot_panel using 6pt labels."""
    x_range = np.linspace(0, 1.05, 300)

    if quest_group is not None and len(quest_group) > 0:
        intens = quest_group["intensity"].values
        correct = quest_group["correct"].values
        bins = np.linspace(0, 1, n_quest_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        labels = pd.cut(intens, bins=bins, labels=bin_centers, include_lowest=True)
        quest_agg = (
            pd.DataFrame({"bin": labels.astype(float), "correct": correct})
            .groupby("bin")["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        ax.bar(
            quest_agg["bin"],
            quest_agg["mean"],
            width=bin_width * 0.4,
            align="center",
            color=QUEST_COLOR,
            alpha=0.35,
            zorder=2,
            edgecolor=QUEST_COLOR,
            linewidth=0.35,
        )

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.025, 0.025, size=len(correct))
        ax.scatter(
            intens,
            correct + jitter,
            s=4,
            color=QUEST_COLOR,
            alpha=0.28,
            zorder=3,
        )

        fit = fit_weibull(intens, correct, criterion=criterion)
        if fit is not None:
            alpha, beta, thresh, _alpha_se = fit
            ax.plot(x_range, weibull(x_range, alpha, beta), color=QUEST_COLOR, lw=0.8, zorder=4)
            if thresh < 1.1:
                ax.axvline(thresh, color=QUEST_COLOR, ls="--", lw=0.55, alpha=0.8, zorder=4)
                ax.text(
                    thresh,
                    1.08,
                    f"{thresh:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_FONT_SIZE,
                    color=QUEST_COLOR,
                )

    if mocs_group is not None and len(mocs_group) > 0:
        agg = (
            mocs_group.groupby("intensity")["correct"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("intensity")
        )
        intens = agg["intensity"].values
        accs = agg["mean"].values
        ns = agg["count"].values
        bar_width = np.diff(intens).min() * 0.4 if len(intens) > 1 else 0.08
        ax.bar(
            intens,
            accs,
            width=bar_width,
            align="center",
            color=MOCS_COLOR,
            alpha=0.35,
            zorder=5,
            edgecolor=MOCS_COLOR,
            linewidth=0.35,
        )
        ax.plot(intens, accs, "o-", color=MOCS_COLOR, lw=0.75, ms=2.2, zorder=6)
        for xi, yi, ni in zip(intens, accs, ns):
            ax.text(
                xi,
                min(yi + 0.045, 1.1),
                f"{int(ni)}",
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                color=MOCS_COLOR,
            )

    add_accuracy_reference_lines(ax, criterion=criterion)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.12, 1.18)
    ax.grid(True, alpha=0.18, linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=LABEL_FONT_SIZE, pad=5.0, length=2)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, pad=1.5, length=2)
    if show_x_label:
        ax.set_xlabel("Intensity", fontsize=LABEL_FONT_SIZE, labelpad=1)
    else:
        ax.set_xlabel("")
    if show_y_label:
        ax.set_ylabel("Accuracy", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=2)
    else:
        ax.set_ylabel("")


def compare_legend_handles(
    include_mocs: bool,
    include_quest: bool,
    criterion: float = DEFAULT_CRITERION,
) -> list:
    handles = []
    if include_mocs:
        handles.extend([
            Patch(facecolor=MOCS_COLOR, edgecolor=MOCS_COLOR, alpha=0.35, label="MOCS binned"),
            Line2D([0], [0], marker="o", color=MOCS_COLOR, lw=0.75, markersize=3, label="MOCS mean"),
        ])
    if include_quest:
        handles.extend([
            Patch(facecolor=QUEST_COLOR, edgecolor=QUEST_COLOR, alpha=0.35, label="Quest binned"),
            Line2D([0], [0], marker="o", color=QUEST_COLOR, linestyle="None", alpha=0.3, markersize=3, label="Quest trials"),
            Line2D([0], [0], color=QUEST_COLOR, lw=0.8, label="Quest Weibull"),
            Line2D([0], [0], color=QUEST_COLOR, linestyle="--", lw=0.55, alpha=0.8, label="Quest threshold"),
        ])
    handles.extend([
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle=":", lw=0.55, alpha=0.7, label=f"Chance ({CHANCE:.2f})"),
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle="-.", lw=0.45, alpha=0.6, label=f"Criterion ({criterion:.3g})"),
    ])
    return handles


def subject_number_map(subjects: list[str]) -> dict[str, int]:
    return {subject: idx + 1 for idx, subject in enumerate(subjects)}


def anonymized_subject_label(subject: str, subjects: list[str]) -> str:
    return f"Subject {subject_number_map(subjects)[subject]}"


def add_accuracy_reference_lines(ax, criterion: float = DEFAULT_CRITERION) -> None:
    ax.axhline(CHANCE, color=NEUTRAL_COLOR, ls=":", lw=0.55, alpha=0.7, zorder=1)
    ax.axhline(criterion, color=NEUTRAL_COLOR, ls="-.", lw=0.45, alpha=0.6, zorder=1)


def capped_figure_height(preferred_height: float, max_height: float) -> float:
    """Keep full-page plots short enough for page margins and caption."""
    return min(float(preferred_height), float(max_height))


def condition_label(genotype_key: str, axis: int, all_keys: list[tuple[str, int]]) -> str:
    labels = [ml_genotype_label(key[0]) for key in all_keys]
    label = ml_genotype_label(genotype_key)
    if labels.count(label) > 1:
        return f"{label}\naxis {axis}"
    return label


def plot_threshold_lines(
    summary: pd.DataFrame,
    subjects: list[str],
    output_dir: Path,
    output_prefix: str,
    max_figure_height: float = MAX_FULL_PAGE_FIG_HEIGHT,
    criterion: float = DEFAULT_CRITERION,
) -> tuple[Path, Path]:
    """Readable Quest threshold plot: one row per anonymized subject."""
    plot_df = summary[
        (summary["method"] == "Quest")
        & np.isfinite(summary["threshold"])
    ].copy()
    if plot_df.empty:
        raise ValueError("No finite Quest thresholds available to plot.")

    plotted_subjects = [subject for subject in subjects if subject in set(plot_df["subject"])]
    condition_keys = sorted(
        set(zip(plot_df["genotype_key"], plot_df["metameric_axis"])),
        key=observer_sort_key,
    )
    x = np.arange(len(condition_keys))
    x_labels = [condition_label(key[0], int(key[1]), condition_keys) for key in condition_keys]

    fig_height = capped_figure_height(
        max(2.9, 0.68 * len(plotted_subjects) + 0.9),
        max_figure_height,
    )
    fig, axes = plt.subplots(
        len(plotted_subjects),
        1,
        figsize=(FULL_PAGE, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for row_idx, subject in enumerate(plotted_subjects):
        ax = axes[row_idx]
        add_accuracy_reference_lines(ax, criterion=criterion)
        subject_df = plot_df[plot_df["subject"] == subject]
        values = []
        ci_low = []
        ci_high = []
        censored = []
        support_acc = []
        support_ci_low = []
        support_ci_high = []
        support_n_correct = []
        support_n_total = []
        for genotype_key, axis in condition_keys:
            cell = subject_df[
                (subject_df["genotype_key"] == genotype_key)
                & (subject_df["metameric_axis"] == int(axis))
            ].sort_values("threshold")
            if cell.empty:
                values.append(np.nan)
                ci_low.append(np.nan)
                ci_high.append(np.nan)
                censored.append(False)
                support_acc.append(np.nan)
                support_ci_low.append(np.nan)
                support_ci_high.append(np.nan)
                support_n_correct.append(np.nan)
                support_n_total.append(np.nan)
                continue
            rec = cell.iloc[0]
            values.append(float(rec["threshold"]))
            ci_low.append(float(rec["threshold_ci_low"]) if np.isfinite(rec["threshold_ci_low"]) else np.nan)
            ci_high.append(float(rec["threshold_ci_high"]) if np.isfinite(rec["threshold_ci_high"]) else np.nan)
            censored.append(bool(rec["threshold_censored"]))
            support_acc.append(float(rec["censor_support_accuracy"]) if np.isfinite(rec["censor_support_accuracy"]) else np.nan)
            support_ci_low.append(float(rec["censor_support_ci_low"]) if np.isfinite(rec["censor_support_ci_low"]) else np.nan)
            support_ci_high.append(float(rec["censor_support_ci_high"]) if np.isfinite(rec["censor_support_ci_high"]) else np.nan)
            support_n_correct.append(float(rec["censor_support_n_correct"]) if np.isfinite(rec["censor_support_n_correct"]) else np.nan)
            support_n_total.append(float(rec["censor_support_n_total"]) if np.isfinite(rec["censor_support_n_total"]) else np.nan)

        values = np.asarray(values, dtype=float)
        ci_low = np.asarray(ci_low, dtype=float)
        ci_high = np.asarray(ci_high, dtype=float)
        censored = np.asarray(censored, dtype=bool)
        support_acc = np.asarray(support_acc, dtype=float)
        support_ci_low = np.asarray(support_ci_low, dtype=float)
        support_ci_high = np.asarray(support_ci_high, dtype=float)
        support_n_correct = np.asarray(support_n_correct, dtype=float)
        support_n_total = np.asarray(support_n_total, dtype=float)
        finite = np.isfinite(values)
        measured = finite & ~censored
        ci_finite = measured & np.isfinite(ci_low) & np.isfinite(ci_high)
        support_finite = censored & np.isfinite(support_acc)

        yerr = None
        if ci_finite.any():
            yerr = np.vstack([
                np.maximum(0.0, np.where(ci_finite, values - ci_low, 0.0)),
                np.maximum(0.0, np.where(ci_finite, ci_high - values, 0.0)),
            ])
        line_values = np.where(finite, values, np.nan)
        ax.plot(
            x,
            line_values,
            color=QUEST_COLOR,
            linewidth=0.85,
            alpha=0.65,
        )
        if ci_finite.any():
            ax.fill_between(
                x,
                np.where(ci_finite, ci_low, np.nan),
                np.where(ci_finite, ci_high, np.nan),
                color=QUEST_COLOR,
                alpha=0.08,
                linewidth=0,
            )
            ax.plot(
                x,
                np.where(ci_finite, ci_low, np.nan),
                color=QUEST_COLOR,
                linewidth=0.45,
                alpha=0.45,
                linestyle=":",
            )
            ax.plot(
                x,
                np.where(ci_finite, ci_high, np.nan),
                color=QUEST_COLOR,
                linewidth=0.45,
                alpha=0.45,
                linestyle=":",
            )
        ax.errorbar(
            x[measured],
            values[measured],
            yerr=yerr[:, measured] if yerr is not None else None,
            color=QUEST_COLOR,
            marker="o",
            markersize=3.0,
            linestyle="None",
            capsize=1.8,
            elinewidth=0.6,
            label="Quest threshold" if row_idx == 0 else None,
        )
        if censored.any():
            for xi, yi in zip(x[censored], values[censored]):
                ax.annotate(
                    "",
                    xy=(xi, GAMUT_MAX + 0.105),
                    xytext=(xi, GAMUT_MAX),
                    arrowprops=dict(arrowstyle="-|>", color=CENSORED_COLOR, lw=0.9),
                    annotation_clip=False,
                )
        if support_finite.any():
            support_yerr = None
            support_ci_finite = support_finite & np.isfinite(support_ci_low) & np.isfinite(support_ci_high)
            if support_ci_finite.any():
                support_yerr = np.vstack([
                    np.maximum(0.0, np.where(support_ci_finite, support_acc - support_ci_low, 0.0)),
                    np.maximum(0.0, np.where(support_ci_finite, support_ci_high - support_acc, 0.0)),
                ])[:, support_finite]
            support_idxs = np.flatnonzero(support_finite)
            for left, right in zip(support_idxs[:-1], support_idxs[1:]):
                if right == left + 1:
                    ax.plot(
                        x[[left, right]],
                        support_acc[[left, right]],
                        color=MOCS_COLOR,
                        linewidth=0.65,
                        alpha=0.55,
                        zorder=3,
                    )
            ax.errorbar(
                x[support_finite],
                support_acc[support_finite],
                yerr=support_yerr,
                color=MOCS_COLOR,
                marker="s",
                markersize=2.6,
                linestyle="None",
                capsize=1.8,
                elinewidth=0.6,
                label="MOCS at gamut max" if row_idx == 0 else None,
            )
            for xi, acc, n_corr, n_total in zip(
                x[support_finite],
                support_acc[support_finite],
                support_ci_low[support_finite],
                support_ci_high[support_finite],
            ):
                ax.text(
                    xi,
                    max(0.06, acc - 0.15),
                    f"{acc:.2f} [{n_corr:.2f}, {n_total:.2f}]",
                    ha="center",
                    va="top",
                    alpha=0.8,
                    fontsize=LABEL_FONT_SIZE,
                    color=MOCS_COLOR,
                )
        ax.set_ylabel(
            str(row_idx + 1),
            rotation=0,
            ha="right",
            va="center",
            fontsize=SUBJECT_ID_FONT_SIZE,
            labelpad=5,
        )
        if row_idx == 0:
            ax.text(
                -0.025,
                1.12,
                "ID",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=SUBJECT_ID_FONT_SIZE,
                fontweight="bold",
                clip_on=False,
            )
        ax.grid(axis="y", alpha=0.25, linewidth=0.45)
        ax.grid(axis="x", alpha=0.16, linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE, pad=5.0)
        ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, pad=1.5)
        if row_idx != len(plotted_subjects) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylim(0, GAMUT_MAX + 0.14)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=0, ha="center", fontsize=X_TICK_FONT_SIZE)
    fig.supylabel("Threshold (Yellow) / Proportion Correct (Blue)", x=0.012, fontsize=AXIS_LABEL_FONT_SIZE)
    handles = [
        Line2D([0], [0], color=QUEST_COLOR, marker="o", lw=0.85, markersize=3, label=f"Quest criterion for {100 * criterion:.1f}\\% correct"),
        Line2D([0], [0], color=QUEST_COLOR, lw=0.45, alpha=0.45, linestyle=":", label="95\% CI"),
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle="-.", lw=0.45, alpha=0.6, label=f"Criterion ({criterion:.3g})"),
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle=":", lw=0.55, alpha=0.7, label=f"Chance ({CHANCE:.2f})"),
    ]
    if plot_df["threshold_censored"].astype(bool).any():
        handles.append(
            Line2D(
                [0],
                [0],
                color=CENSORED_COLOR,
                marker="^",
                linestyle="None",
                markersize=5,
                label=f"Censored: Threshold Exceeds Gamut Boundary)",
            )
        )
    if np.isfinite(plot_df["censor_support_accuracy"]).any():
        handles.append(
            Line2D([0], [0], color=MOCS_COLOR, marker="s", linestyle="None", markersize=3, label="Proportion Correct at Gamut Boundary")
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    plt.tight_layout(rect=(0.04, 0.02, 0.995, 0.94), h_pad=0.16)
    fig.subplots_adjust(left=0.075, right=0.995)

    suffix = f"{output_prefix}_threshold_lines"
    png_path = output_dir / f"{suffix}.png"
    pdf_path = output_dir / f"{suffix}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_compare_grid(
    subject_data: dict[str, dict[str, pd.DataFrame]],
    subjects: list[str],
    output_dir: Path,
    output_prefix: str,
    plot_mode: str,
    max_figure_height: float = MAX_FULL_PAGE_FIG_HEIGHT,
    criterion: float = DEFAULT_CRITERION,
) -> tuple[Path, Path]:
    panel_keys = panel_keys_for_subject_data(subject_data, plot_mode)
    plotted_subjects = [subject for subject in subjects if subject in subject_data]
    if not panel_keys or not plotted_subjects:
        raise ValueError("No raw Quest/MOCS trial panels available to plot.")

    n_rows = len(panel_keys)
    n_cols = len(plotted_subjects)
    fig_width = FULL_PAGE
    fig_height = capped_figure_height(
        max(3.0, 0.46 * n_rows + 0.75),
        max_figure_height,
    )
    fig, grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for col_idx, subject in enumerate(plotted_subjects):
        grid[0][col_idx].set_title(
            anonymized_subject_label(subject, plotted_subjects),
            fontsize=LABEL_FONT_SIZE,
            pad=2,
        )

    for row_idx, (genotype_key, axis) in enumerate(panel_keys):
        row_label = observer_label(genotype_key, int(axis))
        for col_idx, subject in enumerate(plotted_subjects):
            ax = grid[row_idx][col_idx]
            methods = subject_data.get(subject, {})
            mocs_group = None
            quest_group = None
            if plot_mode in ("mocs", "both") and "MOCS" in methods:
                mocs = methods["MOCS"]
                mocs_group = mocs[
                    (mocs["genotype_key"] == genotype_key)
                    & (mocs["metameric_axis"] == int(axis))
                ]
            if plot_mode in ("quest", "both") and "Quest" in methods:
                quest = methods["Quest"]
                quest_group = quest[
                    (quest["genotype_key"] == genotype_key)
                    & (quest["metameric_axis"] == int(axis))
                ]

            has_mocs = mocs_group is not None and len(mocs_group) > 0
            has_quest = quest_group is not None and len(quest_group) > 0
            if has_mocs or has_quest:
                plot_compare_panel(
                    ax,
                    mocs_group if has_mocs else None,
                    quest_group if has_quest else None,
                    show_x_label=row_idx == n_rows - 1,
                    show_y_label=col_idx == 0,
                    criterion=criterion,
                )
            else:
                ax.set_xlim(-0.02, 1.08)
                ax.set_ylim(-0.12, 1.18)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_color("#dddddd")
                    spine.set_linewidth(0.35)

            if col_idx == 0:
                ax.set_ylabel(
                    row_label,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=AXIS_LABEL_FONT_SIZE,
                    labelpad=14,
                )
            if row_idx != n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col_idx != 0:
                ax.tick_params(axis="y", labelleft=False)

    include_mocs = plot_mode in ("mocs", "both")
    include_quest = plot_mode in ("quest", "both")
    fig.legend(
        handles=compare_legend_handles(include_mocs, include_quest, criterion=criterion),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.996),
        ncol=min(7, len(compare_legend_handles(include_mocs, include_quest, criterion=criterion))),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.08, w_pad=0.08)

    suffix = f"{output_prefix}_{plot_mode}_compare_grid"
    png_path = output_dir / f"{suffix}.png"
    pdf_path = output_dir / f"{suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def add_plot_values(summary: pd.DataFrame) -> pd.DataFrame:
    """Add a single plotting value per row.

    Quest rows use threshold. MOCS rows use fitted threshold when available,
    otherwise accuracy, because current Genetic/MOCS blocks are fixed-intensity
    and often cannot identify a psychometric threshold.
    """
    out = summary.copy()
    out["plot_value"] = out["threshold"]
    out["plot_value_kind"] = np.where(np.isfinite(out["threshold"]), "threshold", "")
    mocs_accuracy = (
        (out["method"] == "MOCS")
        & ~np.isfinite(out["plot_value"])
        & np.isfinite(out["accuracy"])
    )
    out.loc[mocs_accuracy, "plot_value"] = out.loc[mocs_accuracy, "accuracy"]
    out.loc[mocs_accuracy, "plot_value_kind"] = "accuracy"
    return out


def conclude_genotypes(summary: pd.DataFrame, rule: str = "within-ci") -> pd.DataFrame:
    """Choose a genotype per subject/method.

    For within-ci, include all rows with threshold <= the upper confidence bound
    of the minimum-threshold row. If no CI is available, this falls back to the
    single minimum threshold. MOCS falls back to minimum accuracy when no
    threshold is available, because lower fixed-intensity accuracy means the
    plate was more confusable for that observer genotype.
    """
    usable = add_plot_values(summary)
    usable = usable[np.isfinite(usable["plot_value"])].copy()
    records = []
    if usable.empty:
        return pd.DataFrame()

    for (subject, method), group in usable.groupby(["subject", "method"], dropna=False):
        group = group.sort_values(["plot_value", "genotype_key", "metameric_axis"])
        best = group.iloc[0]
        cutoff = float(best["plot_value"])
        rule_used = f"min_{best['plot_value_kind']}"
        if (
            best["plot_value_kind"] == "threshold"
            and rule == "within-ci"
            and np.isfinite(best["threshold_ci_high"])
        ):
            cutoff = float(best["threshold_ci_high"])
            rule_used = "within_min_ci"
        selected = group[group["plot_value"] <= cutoff]
        records.append(
            {
                "subject": subject,
                "method": method,
                "conclusion_rule": rule_used,
                "value_kind": best["plot_value_kind"],
                "value_cutoff": cutoff,
                "best_value": float(best["plot_value"]),
                "selected_observers": ";".join(selected["observer"].astype(str)),
                "selected_genotypes": ";".join(selected["genotype_key"].astype(str)),
                "selected_axes": ";".join(selected["metameric_axis"].astype(str)),
                "n_selected": len(selected),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_threshold_matrix(
    summary: pd.DataFrame,
    subjects: list[str],
    output_dir: Path,
    output_prefix: str,
    plot_mode: str,
    max_figure_height: float = MAX_FULL_PAGE_FIG_HEIGHT,
    criterion: float = DEFAULT_CRITERION,
) -> tuple[Path, Path]:
    plot_df = filter_plot_mode(summary, plot_mode)
    plot_df = add_plot_values(plot_df)
    plot_df = plot_df[np.isfinite(plot_df["plot_value"])].copy()
    if plot_df.empty:
        raise ValueError("No finite thresholds or MOCS accuracies available to plot.")

    observer_keys = sorted(
        set(zip(plot_df["genotype_key"], plot_df["metameric_axis"])),
        key=observer_sort_key,
    )
    observers = {
        key: observer_label(key[0], int(key[1]))
        for key in observer_keys
    }
    plotted_subjects = [subject for subject in subjects if subject in set(plot_df["subject"])]
    x = np.arange(len(plotted_subjects))

    fig_width = max(FULL_PAGE, 0.55 * max(len(plotted_subjects), 1) + 2.2)
    fig_height = capped_figure_height(
        max(1.8, 0.34 * len(observer_keys) + 0.82),
        max_figure_height,
    )
    fig, axes = plt.subplots(
        len(observer_keys),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    offsets = {"MOCS": -0.09, "Quest": 0.09} if plot_mode == "both" else {"MOCS": 0.0, "Quest": 0.0}
    colors = {"MOCS": MOCS_COLOR, "Quest": QUEST_COLOR}
    markers = {"MOCS": "o", "Quest": "s"}

    for row_idx, key in enumerate(observer_keys):
        ax = axes[row_idx]
        add_accuracy_reference_lines(ax, criterion=criterion)
        genotype_key, axis = key
        row_df = plot_df[
            (plot_df["genotype_key"] == genotype_key)
            & (plot_df["metameric_axis"] == int(axis))
        ]
        for method in ("MOCS", "Quest"):
            method_df = row_df[row_df["method"] == method]
            if method_df.empty:
                continue
            values = []
            ci_low = []
            ci_high = []
            for subject in plotted_subjects:
                cell = method_df[method_df["subject"] == subject].sort_values("plot_value")
                if cell.empty:
                    values.append(np.nan)
                    ci_low.append(np.nan)
                    ci_high.append(np.nan)
                    continue
                rec = cell.iloc[0]
                value = float(rec["plot_value"])
                values.append(value)
                has_threshold_ci = rec["plot_value_kind"] == "threshold"
                ci_low.append(float(rec["threshold_ci_low"]) if has_threshold_ci and np.isfinite(rec["threshold_ci_low"]) else np.nan)
                ci_high.append(float(rec["threshold_ci_high"]) if has_threshold_ci and np.isfinite(rec["threshold_ci_high"]) else np.nan)

            values = np.asarray(values, dtype=float)
            yerr = None
            ci_low = np.asarray(ci_low, dtype=float)
            ci_high = np.asarray(ci_high, dtype=float)
            if np.isfinite(ci_low).any() and np.isfinite(ci_high).any():
                yerr = np.vstack([
                    np.maximum(0.0, np.where(np.isfinite(ci_low), values - ci_low, 0.0)),
                    np.maximum(0.0, np.where(np.isfinite(ci_high), ci_high - values, 0.0)),
                ])
            ax.errorbar(
                x + offsets.get(method, 0.0),
                values,
                yerr=yerr,
                linestyle="None",
                marker=markers[method],
                markersize=3.2,
                capsize=1.8,
                color=colors[method],
                label=method,
                alpha=0.9,
            )
            for xi, value in zip(x + offsets.get(method, 0.0), values):
                if np.isfinite(value):
                    ax.text(
                        xi,
                        value,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=LABEL_FONT_SIZE,
                        color=colors[method],
                    )

        ax.set_ylabel(
            observers[key],
            rotation=90,
            ha="center",
            va="center",
            labelpad=18,
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.grid(axis="x", alpha=0.15, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
        if row_idx != len(observer_keys) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [anonymized_subject_label(subject, plotted_subjects) for subject in plotted_subjects],
        rotation=90,
        ha="center",
        fontsize=LABEL_FONT_SIZE,
    )
    if plot_mode == "mocs":
        y_label = "MOCS accuracy"
    elif plot_mode == "quest":
        y_label = "Quest threshold proportion"
    else:
        y_label = "Value (MOCS accuracy, Quest threshold)"
    fig.supylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    handles = []
    if plot_mode in ("mocs", "both"):
        handles.append(Line2D([0], [0], color=MOCS_COLOR, marker="o", linestyle="None", label="MOCS"))
    if plot_mode in ("quest", "both"):
        handles.append(Line2D([0], [0], color=QUEST_COLOR, marker="s", linestyle="None", label="Quest"))
    handles.extend([
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle="-.", lw=0.45, alpha=0.6, label=f"Criterion ({criterion:.3g})"),
        Line2D([0], [0], color=NEUTRAL_COLOR, linestyle=":", lw=0.55, alpha=0.7, label=f"Chance ({CHANCE:.2f})"),
    ])
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    suffix = f"{output_prefix}_{plot_mode}"
    png_path = output_dir / f"{suffix}.png"
    pdf_path = output_dir / f"{suffix}.pdf"
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
        description="Compile and plot Quest/MOCS thresholds by observer genotype for selected subjects"
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs to include")
    parser.add_argument("--subjects-file", type=str, default=None, help="Text file of subject IDs")
    parser.add_argument("--all", action="store_true", help="Include every detected subject")
    parser.add_argument("--list", action="store_true", help="List detected subjects and exit")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="quest_mocs_subject_summary")
    parser.add_argument("--plot-mode", choices=("mocs", "quest", "both"), default="both")
    parser.add_argument(
        "--plot-type",
        choices=("threshold-lines", "compare-grid", "threshold-matrix"),
        default="threshold-lines",
        help="Main plot type. Use compare-grid for the dense appendix figure.",
    )
    parser.add_argument("--intensity", type=float, default=None, help="Optional MOCS intensity filter")
    parser.add_argument(
        "--criterion",
        type=float,
        default=DEFAULT_CRITERION,
        help=f"Weibull threshold criterion as proportion correct (default: {DEFAULT_CRITERION}).",
    )
    parser.add_argument(
        "--max-figure-height",
        type=float,
        default=MAX_FULL_PAGE_FIG_HEIGHT,
        help=(
            "Maximum plot height in inches before tight bounding-box export "
            f"(default: {MAX_FULL_PAGE_FIG_HEIGHT})."
        ),
    )
    parser.add_argument(
        "--conclusion-rule",
        choices=("min", "within-ci"),
        default="within-ci",
        help="Genotype conclusion rule for each subject/method",
    )
    args = parser.parse_args()
    if not (CHANCE < args.criterion < 1.0):
        parser.error(f"--criterion must be greater than chance ({CHANCE:g}) and less than 1.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / "quest_mocs_subject_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = discover_files(data_dir)

    if args.list:
        print("Detected subjects:")
        for subject, methods in sorted(all_files.items()):
            print(f"  {subject}:")
            for method, path in sorted(methods.items()):
                print(f"    {method}: {path.name}")
        return

    subjects = []
    if args.all:
        subjects.extend(sorted(all_files.keys()))
    if args.subjects:
        subjects.extend(args.subjects)
    if args.subjects_file:
        subjects.extend(read_subjects_file(Path(args.subjects_file)))
    subjects = list(dict.fromkeys(subjects))

    if not subjects:
        parser.error("Provide --subjects, --subjects-file, --all, or use --list.")

    summary, missing = compile_summary(
        subjects,
        all_files,
        intensity=args.intensity,
        criterion=args.criterion,
    )
    if missing:
        print("Subjects without latest Quest or MOCS files:")
        for subject in missing:
            print(f"  {subject}")

    if summary.empty:
        raise SystemExit("No Quest or MOCS data found for requested subjects.")

    plotted_subjects = [subject for subject in subjects if subject in set(summary["subject"])]
    csv_path = output_dir / f"{args.output_prefix}.csv"
    conclusion_path = output_dir / f"{args.output_prefix}_conclusions.csv"
    summary.to_csv(csv_path, index=False)

    conclusions = conclude_genotypes(summary, rule=args.conclusion_rule)
    conclusions.to_csv(conclusion_path, index=False)

    png_path = pdf_path = None
    try:
        if args.plot_type == "threshold-lines":
            png_path, pdf_path = plot_threshold_lines(
                summary,
                plotted_subjects,
                output_dir,
                args.output_prefix,
                max_figure_height=args.max_figure_height,
                criterion=args.criterion,
            )
        elif args.plot_type == "compare-grid":
            subject_data = load_subject_trial_data(
                plotted_subjects,
                all_files,
                intensity=args.intensity,
            )
            png_path, pdf_path = plot_compare_grid(
                subject_data,
                plotted_subjects,
                output_dir,
                args.output_prefix,
                args.plot_mode,
                max_figure_height=args.max_figure_height,
                criterion=args.criterion,
            )
        else:
            png_path, pdf_path = plot_threshold_matrix(
                summary,
                plotted_subjects,
                output_dir,
                args.output_prefix,
                args.plot_mode,
                max_figure_height=args.max_figure_height,
                criterion=args.criterion,
            )
    except ValueError as exc:
        print(f"Skipped plot: {exc}")

    print(f"Saved summary CSV:     {csv_path}")
    print(f"Saved conclusions CSV: {conclusion_path}")
    if png_path is not None and pdf_path is not None:
        print(f"Saved plot PNG:        {png_path}")
        print(f"Saved plot PDF:        {pdf_path}")


if __name__ == "__main__":
    main()
