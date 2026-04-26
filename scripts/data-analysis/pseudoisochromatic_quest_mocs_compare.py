#!/usr/bin/env python3
"""
Compare Quest vs Method-of-Constant-Stimuli (MOCS) for pseudo-isochromatic data.

With ``--subject``, loads the latest per-subject files from ``aggregate_analysis``:
trial CSVs whose names contain ``QUEST`` vs ``GENETIC`` (MOCS); ``*_thresholds.csv``
exports are ignored.

Alternatively: a single CSV with a paradigm column, or ``--quest-csv`` / ``--mocs-csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from aggregate_analysis import get_latest_files_per_subject, _coerce_correct_column


# 4-AFC chance rate; Pelli-style threshold midway between chance and ceiling
DEFAULT_GAMMA = 0.25
DEFAULT_DELTA = 0.01
DEFAULT_P_THRESHOLD = 0.625

INTENSITY_CANDIDATES = (
    'intensity', 'stimulus_intensity', 'signal_intensity', 'level',
    'contrast', 'stimulus_level',
)
PARADIGM_CANDIDATES = (
    'paradigm', 'method', 'staircase', 'staircase_type', 'block_type',
    'session_type', 'mode', 'task_mode', 'procedure',
)


def psychometric_weibull_log(x_log: np.ndarray, t: float, beta: float,
                             gamma: float = DEFAULT_GAMMA,
                             delta: float = DEFAULT_DELTA) -> np.ndarray:
    """Quest / Psychtoolbox-style Weibull on log10 intensity."""
    return delta * gamma + (1 - delta) * (
        1 - (1 - gamma) * np.exp(-(10.0 ** (beta * (x_log - t))))
    )


def _resolve_intensity_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for name in INTENSITY_CANDIDATES:
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce')
    return None


def _to_log_intensity(intensity: np.ndarray, scale: str) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=float)
    if scale == 'log10':
        return intensity
    return np.log10(np.clip(intensity, 1e-20, None))


def _log_to_display(x_log: np.ndarray, scale: str) -> np.ndarray:
    if scale == 'log10':
        return x_log
    return 10.0 ** x_log


def fit_weibull_threshold(
    intensity: np.ndarray,
    y: np.ndarray,
    *,
    scale: str,
    gamma: float,
    delta: float,
    p_threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Fit (t, beta) on log-intensity grid; return threshold intensity, t, beta, nll.
    """
    mask = np.isfinite(intensity) & np.isfinite(y)
    intensity = intensity[mask]
    y = y[mask].astype(int)
    if len(intensity) < 5 or len(np.unique(intensity)) < 2:
        return None, None, None, None

    x_log = _to_log_intensity(intensity, scale)

    t0 = float(np.median(x_log))
    beta0 = 3.5

    def nll(theta: np.ndarray) -> float:
        t, beta = float(theta[0]), float(theta[1])
        if beta <= 0.05:
            return 1e12
        p = psychometric_weibull_log(x_log, t, beta, gamma, delta)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize(
        nll,
        x0=np.array([t0, beta0]),
        method='L-BFGS-B',
        bounds=[(float(np.min(x_log) - 3), float(np.max(x_log) + 3)), (0.2, 25.0)],
    )
    if not res.success:
        return None, None, None, None

    t_hat, beta_hat = float(res.x[0]), float(res.x[1])

    def err(xl: float) -> float:
        return psychometric_weibull_log(np.array([xl]), t_hat, beta_hat, gamma, delta)[0] - p_threshold

    lo, hi = float(np.min(x_log) - 4), float(np.max(x_log) + 4)
    try:
        v_lo, v_hi = err(lo), err(hi)
        spread = 0
        while v_lo * v_hi > 0 and spread < 40:
            spread += 2
            lo -= 1
            hi += 1
            v_lo, v_hi = err(lo), err(hi)
        xl_thr = brentq(err, lo, hi, maxiter=200)
    except ValueError:
        xl_thr = float('nan')

    thr_display = _log_to_display(np.array([xl_thr]), scale)[0]
    if not np.isfinite(thr_display):
        thr_display = None

    return thr_display, t_hat, beta_hat, float(res.fun)


def fit_weibull_mocs_binned(
    intensity_levels: np.ndarray,
    n_correct: np.ndarray,
    n_total: np.ndarray,
    *,
    scale: str,
    gamma: float,
    delta: float,
    p_threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Binomial log-likelihood fit at discrete intensity levels."""
    mask = n_total > 0
    intensity_levels = intensity_levels[mask]
    n_correct = n_correct[mask]
    n_total = n_total[mask]
    if len(intensity_levels) < 2:
        return None, None, None, None

    x_log = _to_log_intensity(intensity_levels, scale)
    t0 = float(np.average(x_log, weights=n_total))
    beta0 = 3.5

    def nll(theta: np.ndarray) -> float:
        t, beta = float(theta[0]), float(theta[1])
        if beta <= 0.05:
            return 1e12
        p = psychometric_weibull_log(x_log, t, beta, gamma, delta)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.sum(n_correct * np.log(p) + (n_total - n_correct) * np.log(1 - p)))

    res = minimize(
        nll,
        x0=np.array([t0, beta0]),
        method='L-BFGS-B',
        bounds=[(float(np.min(x_log) - 3), float(np.max(x_log) + 3)), (0.2, 25.0)],
    )
    if not res.success:
        return None, None, None, None

    t_hat, beta_hat = float(res.x[0]), float(res.x[1])

    def err(xl: float) -> float:
        return psychometric_weibull_log(np.array([xl]), t_hat, beta_hat, gamma, delta)[0] - p_threshold

    lo, hi = float(np.min(x_log) - 4), float(np.max(x_log) + 4)
    try:
        xl_thr = brentq(err, lo, hi, maxiter=200)
    except ValueError:
        xl_thr = float('nan')

    thr_display = _log_to_display(np.array([xl_thr]), scale)[0]
    if not np.isfinite(thr_display):
        thr_display = None

    return thr_display, t_hat, beta_hat, float(res.fun)


def _add_genotype_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'genotype_1' in out.columns and 'genotype_2' in out.columns:
        out['genotype'] = out.apply(
            lambda row: f"({row['genotype_1']}, {row['genotype_2']})", axis=1
        )
    else:
        out['genotype'] = 'Unknown'
    return out


def _split_quest_mocs(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    for col in PARADIGM_CANDIDATES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.lower()
        quest_m = s.str.contains('quest') | s.str.contains('adaptive')
        mocs_m = (
            s.str.contains('mocs')
            | s.str.contains('constant')
            | s.str.contains('method of constant')
            | s.str.contains('fixed')
        )
        if quest_m.any() and mocs_m.any():
            return df[quest_m].copy(), df[mocs_m].copy(), col
    return None, None, None


def _prepare_condition_plot_data(
    quest_df: pd.DataFrame,
    mocs_df: pd.DataFrame,
    *,
    scale: str,
    gamma: float,
    delta: float,
    p_threshold: float,
) -> Optional[Dict[str, Any]]:
    int_q = _resolve_intensity_series(quest_df)
    int_m = _resolve_intensity_series(mocs_df)
    if int_q is None or int_m is None:
        return None

    cq = _coerce_correct_column(quest_df)
    cm = _coerce_correct_column(mocs_df)
    if cq is None or cm is None:
        return None

    iq = int_q.to_numpy(dtype=float)
    yq = cq.to_numpy(dtype=int)
    im = int_m.to_numpy(dtype=float)
    ym = cm.to_numpy(dtype=int)

    keys = np.round(im, 12)
    uniq = np.unique(keys[np.isfinite(keys)])
    levels_list: List[float] = []
    n_corr_list: List[int] = []
    n_tot_list: List[int] = []
    for u in uniq:
        sel = keys == u
        levels_list.append(float(u))
        n_tot_list.append(int(np.sum(sel)))
        n_corr_list.append(int(np.sum(ym[sel])))
    levels = np.array(levels_list, dtype=float)
    n_corr = np.array(n_corr_list)
    n_tot = np.array(n_tot_list)
    pct = np.divide(n_corr, np.maximum(n_tot, 1))

    thr_m, t_m, beta_m, _ = fit_weibull_mocs_binned(
        levels, n_corr, n_tot, scale=scale, gamma=gamma, delta=delta, p_threshold=p_threshold
    )
    thr_q, t_q, beta_q, _ = fit_weibull_threshold(
        iq, yq, scale=scale, gamma=gamma, delta=delta, p_threshold=p_threshold
    )

    x_min = float(np.nanmin([np.min(iq), np.min(im)]))
    x_max = float(np.nanmax([np.max(iq), np.max(im)]))
    if scale == 'linear':
        xs = np.linspace(max(x_min, 1e-12), x_max, 200)
    else:
        xs = np.linspace(x_min - 0.05 * abs(x_min + 1e-6), x_max + 0.05 * abs(x_max + 1e-6), 200)
    x_log_curve = _to_log_intensity(xs, scale)

    return {
        'iq': iq,
        'yq': yq,
        'levels': levels,
        'pct': pct,
        'n_tot': n_tot,
        'xs': xs,
        'x_log_curve': x_log_curve,
        't_m': t_m,
        'beta_m': beta_m,
        't_q': t_q,
        'beta_q': beta_q,
        'thr_m': thr_m,
        'thr_q': thr_q,
    }


def _draw_combined_psychometric_panel(
    ax,
    data: Dict[str, Any],
    *,
    title: str,
    scale: str,
    gamma: float,
    delta: float,
    rng: np.random.Generator,
    title_fontsize: float,
    tick_fontsize: float,
) -> None:
    """Single axes: MOCS binned P(correct), Quest trials, both Weibull fits, thresholds."""
    iq, yq = data['iq'], data['yq']
    levels, pct, n_tot = data['levels'], data['pct'], data['n_tot']
    xs, x_log_curve = data['xs'], data['x_log_curve']
    t_m, beta_m = data.get('t_m'), data.get('beta_m')
    t_q, beta_q = data.get('t_q'), data.get('beta_q')
    thr_m, thr_q = data.get('thr_m'), data.get('thr_q')

    yerr = np.sqrt(np.clip(pct * (1 - pct) / np.maximum(n_tot, 1), 0, 0.25))
    ax.errorbar(
        levels,
        pct,
        yerr=yerr,
        fmt='o',
        color='tab:blue',
        capsize=2,
        markersize=4,
        label='MOCS (binned)',
    )

    jitter = (rng.random(len(iq)) - 0.5) * 0.04
    ax.scatter(iq, yq.astype(float) + jitter, alpha=0.25, s=12, c='tab:orange', label='Quest trials')

    if t_m is not None and beta_m is not None:
        curve_m = psychometric_weibull_log(x_log_curve, t_m, beta_m, gamma, delta)
        ax.plot(xs, curve_m, color='tab:blue', linewidth=2, label='MOCS fit')
    if t_q is not None and beta_q is not None:
        curve_q = psychometric_weibull_log(x_log_curve, t_q, beta_q, gamma, delta)
        ax.plot(xs, curve_q, color='tab:red', linewidth=2, label='Quest fit')

    if thr_m is not None:
        ax.axvline(thr_m, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.9)
    if thr_q is not None:
        ax.axvline(thr_q, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.9)
    ax.axhline(gamma, color='gray', linestyle=':', alpha=0.55, linewidth=1)

    ax.set_xlabel('Intensity', fontsize=tick_fontsize)
    ax.set_ylabel('P(correct)', fontsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=tick_fontsize)


def plot_combined_figure(
    panels: List[Tuple[str, Tuple, Dict[str, Any]]],
    *,
    suptitle: str,
    out_path: Path,
    scale: str,
    gamma: float,
    delta: float,
) -> None:
    """One figure: grid of combined psychometric panels (MOCS + Quest + fits on same axes)."""
    n = len(panels)
    if n == 0:
        return

    ncols = int(min(4, max(1, int(np.ceil(np.sqrt(n))))))
    nrows = int(np.ceil(n / ncols))

    title_fs = max(7.0, 11.0 - 0.45 * max(nrows, ncols))
    tick_fs = max(6.0, 9.0 - 0.35 * max(nrows, ncols))
    fig_w = min(22, 3.8 * ncols + 1.5)
    fig_h = min(28, 3.4 * nrows + 1.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    for idx, (panel_title, _pair_key, pdata) in enumerate(panels):
        ax = axes_flat[idx]
        rng = np.random.default_rng(42 + idx)
        _draw_combined_psychometric_panel(
            ax,
            pdata,
            title=panel_title,
            scale=scale,
            gamma=gamma,
            delta=delta,
            rng=rng,
            title_fontsize=title_fs,
            tick_fontsize=tick_fs,
        )

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(5, len(labels)),
        fontsize=max(7, 10 - ncols),
        frameon=True,
    )

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=str, default=None, help='Subject id (uses latest CSV per aggregate_analysis)')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--csv', type=str, default=None, help='Explicit trial CSV (overrides --subject)')
    parser.add_argument('--quest-csv', type=str, default=None)
    parser.add_argument('--mocs-csv', type=str, default=None)
    parser.add_argument('--metameric-axis', type=float, default=None, help='Restrict to one metameric_axis value')
    parser.add_argument('--genotype', type=str, default=None, help='Restrict to genotype label string')
    parser.add_argument(
        '--intensity-scale', choices=('linear', 'log10'), default='linear',
        help='Whether CSV intensity is linear contrast or already log10.',
    )
    parser.add_argument('--p-threshold', type=float, default=DEFAULT_P_THRESHOLD,
                        help='Performance level for reported threshold (default 0.625 for 4AFC).')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / 'data'
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / 'results'

    quest_df: Optional[pd.DataFrame] = None
    mocs_df: Optional[pd.DataFrame] = None
    subject_tag = 'custom'

    if args.quest_csv and args.mocs_csv:
        quest_df = pd.read_csv(args.quest_csv)
        mocs_df = pd.read_csv(args.mocs_csv)
        subject_tag = Path(args.quest_csv).stem
    elif args.csv:
        df = pd.read_csv(args.csv)
        subject_tag = Path(args.csv).stem
        q, m, col = _split_quest_mocs(df)
        if q is None:
            raise SystemExit(
                'Could not split Quest vs MOCS from a single CSV. '
                f'Tried columns {PARADIGM_CANDIDATES}. '
                'Use --quest-csv and --mocs-csv, or add a paradigm/method column.'
            )
        print(f"Split using column {col!r}: Quest n={len(q)}, MOCS n={len(m)}")
        quest_df, mocs_df = q, m
    elif args.subject:
        latest = get_latest_files_per_subject(data_dir)
        if args.subject not in latest:
            raise SystemExit(f"Subject {args.subject!r} not found under {data_dir}")
        tasks = latest[args.subject]
        pseudo = tasks.get('AppPseudoIsochromaticTest')
        subject_tag = args.subject
        if isinstance(pseudo, dict):
            q_path, m_path = pseudo.get('QUEST'), pseudo.get('GENETIC')
            if q_path is None or m_path is None:
                raise SystemExit(
                    f"Need both a QUEST and a GENETIC trial CSV for {args.subject!r} "
                    f"(filenames must contain those substrings). Got QUEST={q_path!s}, "
                    f"GENETIC={m_path!s}. Use --quest-csv / --mocs-csv to override."
                )
            print(f"Quest trials: {q_path.name}")
            print(f"MOCS (GENETIC) trials: {m_path.name}")
            quest_df = pd.read_csv(q_path)
            mocs_df = pd.read_csv(m_path)
        else:
            raise SystemExit(
                'No AppPseudoIsochromaticTest QUEST/GENETIC pair for this subject '
                '(expected filenames containing QUEST and GENETIC under the task folder).'
            )
    else:
        parser.error('Provide --subject, --csv, or both --quest-csv and --mocs-csv')

    assert quest_df is not None and mocs_df is not None

    quest_df = _add_genotype_column(quest_df)
    mocs_df = _add_genotype_column(mocs_df)

    if args.metameric_axis is not None:
        quest_df = quest_df[quest_df['metameric_axis'] == args.metameric_axis]
        mocs_df = mocs_df[mocs_df['metameric_axis'] == args.metameric_axis]
    if args.genotype is not None:
        quest_df = quest_df[quest_df['genotype'] == args.genotype]
        mocs_df = mocs_df[mocs_df['genotype'] == args.genotype]

    if 'metameric_axis' not in quest_df.columns or 'metameric_axis' not in mocs_df.columns:
        raise SystemExit("Expected 'metameric_axis' column (same as aggregate_analysis).")

    pairs = sorted(
        set(zip(quest_df['genotype'], quest_df['metameric_axis']))
        & set(zip(mocs_df['genotype'], mocs_df['metameric_axis']))
    )
    if not pairs:
        raise SystemExit('No overlapping (genotype, metameric_axis) between Quest and MOCS after filters.')

    out_root = results_dir / subject_tag / 'quest_mocs_compare'
    out_path = out_root / 'quest_mocs_combined.png'
    panels: List[Tuple[str, Tuple, Dict[str, Any]]] = []

    for genotype, axis in pairs:
        q_sub = quest_df[(quest_df['genotype'] == genotype) & (quest_df['metameric_axis'] == axis)]
        m_sub = mocs_df[(mocs_df['genotype'] == genotype) & (mocs_df['metameric_axis'] == axis)]
        if len(q_sub) < 3 or len(m_sub) < 3:
            print(f"Skip {genotype} axis={axis}: insufficient trials.")
            continue
        pdata = _prepare_condition_plot_data(
            q_sub,
            m_sub,
            scale=args.intensity_scale,
            gamma=DEFAULT_GAMMA,
            delta=DEFAULT_DELTA,
            p_threshold=args.p_threshold,
        )
        if pdata is None:
            print(f"Skip {genotype} axis={axis}: missing intensity or outcome columns.")
            continue
        panel_title = f'{genotype}\naxis={axis}'
        panels.append((panel_title, (genotype, axis), pdata))

    if not panels:
        raise SystemExit('No panels to plot after filtering (check trial counts and columns).')

    suptitle = f'{subject_tag} — Quest vs MOCS (GENETIC), combined'
    print(f"Writing {out_path} ({len(panels)} conditions) …")
    plot_combined_figure(
        panels,
        suptitle=suptitle,
        out_path=out_path,
        scale=args.intensity_scale,
        gamma=DEFAULT_GAMMA,
        delta=DEFAULT_DELTA,
    )


if __name__ == '__main__':
    main()
