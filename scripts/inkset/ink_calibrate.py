#!/usr/bin/env python3
"""
Inkset checking and calibration CLI.

Subcommands:
- register-primaries: Register a measured primaries CSV as an ink library (nix_printer_primaries_*).
- evaluate: Compare Neugebauer predictions vs measured multi-channel patches.
- calibrate: Fit Yule–Nielsen n and per-ink TRC gammas to minimize spectral error.

Notes:
- Prediction/export functionality is intentionally omitted for now.
"""

from TetriumColor.Observer import Observer, Illuminant
from TetriumColor.Observer.Inks import InkGamut, Neugebauer, km_mix, k_s_from_data, data_from_k_s
from TetriumColor.Observer.Inks import load_inkset as load_inkset_csv
from TetriumColor.Observer.Spectra import Spectra
from TetriumColor.Measurement.Nix import read_nix_csv

from library_registry import registry

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# -------- Naming and parsing utilities --------

PRIMARY_RE = re.compile(r"^([A-Z])([0-9]{1,3})$")


def discover_channels_from_names(names: List[str]) -> List[str]:
    channels = []
    seen = set()
    for name in names:
        m = PRIMARY_RE.match(name.strip())
        if not m:
            continue
        ch = m.group(1)
        if ch not in seen:
            seen.add(ch)
            channels.append(ch)
    return channels


def parse_combo_name(name: str, channels: List[str]) -> np.ndarray:
    """Parse a combination like 'C255M128' or 'C255 M128' into a vector aligned to channels order."""
    s = name.strip().replace(" ", "")
    levels = {ch: 0 for ch in channels}
    # Greedy scan: letter followed by up to 3 digits
    i = 0
    while i < len(s):
        ch = s[i]
        if ch not in levels:
            raise ValueError(f"Unknown channel '{ch}' in name '{name}'")
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        if j == i:
            raise ValueError(f"Missing numeric level after '{ch}' in '{name}'")
        lvl = int(s[i:j])
        if lvl < 0 or lvl > 255:
            raise ValueError(f"Level out of range 0..255 for '{ch}{lvl}' in '{name}'")
        levels[ch] = lvl
        i = j
    return np.array([levels[ch] for ch in channels], dtype=float)


# -------- Helpers for new calibration pipeline --------

def build_complete_neugebauer_primaries(measurements: Dict[str, Spectra], paper: Spectra,
                                        channels: List[str], prefer_overprints: bool = True) -> Tuple[Dict[str, Spectra], np.ndarray]:
    """Build complete Neugebauer primaries dictionary from measurements.

    Args:
        measurements: Dictionary mapping measurement names to Spectra
        paper: Paper spectra
        channels: List of channel names (e.g., ['C', 'M', 'Y', 'K'])
        prefer_overprints: Whether to prefer measured overprints over computed ones

    Returns:
        Tuple of (primaries_dict, wavelengths) where primaries_dict maps binary keys to Spectra
    """
    wavelengths = paper.wavelengths

    # Start with paper (all channels off)
    primaries_dict: Dict[str, Spectra] = {}
    primaries_dict['0' * len(channels)] = paper.interpolate_values(wavelengths)

    # Add single-channel primaries (one channel on, others off)
    for i, ch in enumerate(channels):
        # Find the highest level measurement for this channel
        best_spectra = None
        best_level = -1

        for name, sp in measurements.items():
            m = PRIMARY_RE.match(name.strip())
            if m and m.group(1) == ch:
                level = int(m.group(2))
                if level > best_level:
                    best_level = level
                    best_spectra = sp

        if best_spectra is None:
            raise ValueError(f"No measurements found for channel {ch}")

        # Create binary key for this single-channel primary
        key_vec = ['0'] * len(channels)
        key_vec[i] = '1'
        key = ''.join(key_vec)

        primaries_dict[key] = best_spectra.interpolate_values(wavelengths)

    # Add multi-channel primaries (overprints) if available
    if prefer_overprints:
        for name, sp in measurements.items():
            try:
                levels = parse_combo_name(name, channels)
            except Exception:
                continue

            # Only consider exact 0/255 combinations (binary solids)
            if not np.all((levels == 0) | (levels == 255)):
                continue

            # Skip paper (already added)
            if not np.any(levels == 255):
                continue

            # Create binary key
            key = combo_to_binary_key(levels)

            # Skip if already present (single-channel primaries)
            if key in primaries_dict and key.count('1') <= 1:
                continue

            primaries_dict[key] = sp.interpolate_values(wavelengths)

    return primaries_dict, wavelengths


def save_primaries_as_csv(primaries_dict: Dict[str, Spectra], wavelengths: np.ndarray, output_path: str):
    """Save primaries dictionary as a standard CSV format."""
    # Convert to DataFrame format
    rows = []
    for key, spectra in primaries_dict.items():
        rows.append((key, spectra.data))

    df = pd.DataFrame([r[1] for r in rows],
                      index=[r[0] for r in rows],
                      columns=wavelengths)
    df.index.name = "Name"
    df = df.reset_index()
    df.insert(0, 'ID', range(1, len(df) + 1))
    df['clogged'] = 0
    df.to_csv(output_path, index=False)


def save_primaries_dict(primaries_dict: Dict[str, Spectra], output_path: str):
    """Save primaries dictionary as a pickle file for easy loading."""
    import pickle

    # Convert Spectra objects to a serializable format
    serializable_dict = {}
    for key, spectra in primaries_dict.items():
        serializable_dict[key] = {
            'wavelengths': spectra.wavelengths,
            'data': spectra.data
        }

    with open(output_path, 'wb') as f:
        pickle.dump(serializable_dict, f)


def load_primaries_dict(input_path: str) -> Dict[str, Spectra]:
    """Load primaries dictionary from pickle file."""
    import pickle

    with open(input_path, 'rb') as f:
        serializable_dict = pickle.load(f)

    # Convert back to Spectra objects
    primaries_dict = {}
    for key, data in serializable_dict.items():
        primaries_dict[key] = Spectra(
            wavelengths=data['wavelengths'],
            data=data['data']
        )

    return primaries_dict


def is_single_channel_name(name: str, channels: List[str]) -> bool:
    try:
        v = parse_combo_name(name, channels)
    except Exception:
        return False
    nonzero = np.count_nonzero(v)
    return nonzero > 0 and np.count_nonzero(v > 0) == 1


def is_binary_solid(name: str, channels: List[str]) -> bool:
    try:
        v = parse_combo_name(name, channels)
    except Exception:
        return False
    # All channels either 0 or 255, and not all zeros
    if np.any((v != 0) & (v != 255)):
        return False
    return np.any(v == 255)


def combo_to_binary_key(levels_vec_0_255: np.ndarray) -> str:
    return ''.join('1' if x >= 255 else '0' for x in levels_vec_0_255.astype(int))


def estimate_area_from_murray_davies(R_meas: np.ndarray, R_paper: np.ndarray, R_solid: np.ndarray, n: float) -> float:
    # Work in Yule–Nielsen domain; average across wavelengths
    Rp = np.power(np.clip(R_paper, 1e-6, 1.0), 1.0 / n)
    Rs = np.power(np.clip(R_solid, 1e-6, 1.0), 1.0 / n)
    Rm = np.power(np.clip(R_meas, 1e-6, 1.0), 1.0 / n)
    denom = (Rs - Rp)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    a_wl = (Rm - Rp) / denom
    a = float(np.clip(np.nanmedian(a_wl), 0.0, 1.0))
    return a


def fit_tone_to_area_gammas(channels: List[str], combos: Dict[str, Spectra], paper: Spectra,
                            n_assumed: float = 2.0) -> Dict[str, float]:
    """Fit per-ink gamma for tone->area using single-ink ramps.

    For each ink channel, use single-channel patches (varying that ink only) to
    estimate dot area via Murray–Davies with an assumed n, then fit a gamma a=t^g.
    """
    gammas: Dict[str, float] = {}
    # Find solid spectra per channel (255)
    solid_by_ch: Dict[str, Spectra] = {}
    for ch in channels:
        best = None
        best_lvl = -1
        for name, sp in combos.items():
            m = PRIMARY_RE.match(name.strip())
            if m and m.group(1) == ch:
                lvl = int(m.group(2))
                if lvl > best_lvl:
                    best_lvl = lvl
                    best = sp
        if best is None or best_lvl <= 0:
            raise ValueError(f"Missing solid for channel {ch}")
        solid_by_ch[ch] = best

    wavelengths = paper.wavelengths
    Rp = paper.interpolate_values(wavelengths).data

    from scipy.optimize import minimize_scalar

    for ch in channels:
        # Collect (t, a_est)
        samples_t: List[float] = []
        samples_a: List[float] = []
        Rs = solid_by_ch[ch].interpolate_values(wavelengths).data

        for name, sp in combos.items():
            if not is_single_channel_name(name, channels):
                continue
            levels = parse_combo_name(name, channels)
            # Only this channel nonzero
            idx = channels.index(ch)
            if np.count_nonzero(levels) == 0 or np.count_nonzero(levels > 0) != 1 or levels[idx] == 0:
                continue
            t = float(np.clip(levels[idx] / 255.0, 0.0, 1.0))
            Rm = sp.interpolate_values(wavelengths).data
            a_hat = estimate_area_from_murray_davies(Rm, Rp, Rs, n_assumed)
            samples_t.append(t)
            samples_a.append(a_hat)

        if len(samples_t) < 3:
            # fallback gamma 1.0
            gammas[ch] = 1.0
            continue

        t_arr = np.array(samples_t)
        a_arr = np.array(samples_a)

        def loss(g):
            g = np.clip(g, 0.1, 5.0)
            pred = np.power(t_arr, g)
            return float(np.mean((pred - a_arr) ** 2))

        res = minimize_scalar(loss, bounds=(0.1, 5.0), method='bounded')
        gammas[ch] = float(np.clip(res.x if res.success else 1.0, 0.1, 5.0))

    return gammas


def build_neugebauer_from_measurements(channels: List[str], combos: Dict[str, Spectra], paper: Spectra,
                                       prefer_overprints: bool = True) -> Tuple[Neugebauer, np.ndarray]:
    """Construct a Neugebauer from measured paper/single solids and any measured overprint solids.

    Returns the Neugebauer and the wavelengths used.
    """
    wavelengths = paper.wavelengths

    # Start with paper and single solids
    primaries_dict: Dict[str, Spectra] = {}
    primaries_dict['0' * len(channels)] = paper

    # Singles
    for ch in channels:
        # Choose max available level for this channel as solid
        best = None
        best_lvl = -1
        for name, sp in combos.items():
            m = PRIMARY_RE.match(name.strip())
            if m and m.group(1) == ch:
                lvl = int(m.group(2))
                if lvl > best_lvl:
                    best_lvl = lvl
                    best = sp
        if best is None or best_lvl <= 0:
            raise ValueError(f"Missing solid for channel {ch}")
        key_vec = np.zeros(len(channels), dtype=int)
        key_vec[channels.index(ch)] = 1
        primaries_dict[''.join(map(str, key_vec.tolist()))] = best.interpolate_values(wavelengths)

    # Overprint solids (exact 0/255 per channel)
    if prefer_overprints:
        for name, sp in combos.items():
            try:
                levels = parse_combo_name(name, channels)
            except Exception:
                continue
            if not np.all((levels == 0) | (levels == 255)):
                continue
            if not np.any(levels == 255):
                continue
            key = combo_to_binary_key(levels)
            # Skip singles and paper (already present)
            if key in primaries_dict and key.count('1') <= 1:
                continue
            primaries_dict[key] = sp.interpolate_values(wavelengths)

    neug = Neugebauer(primaries_dict, n=2.0)
    return neug, wavelengths


def predict_with_model(neug: Neugebauer, levels_vec_0_255: np.ndarray,
                       channels: List[str], gammas_by_ch: Dict[str, float],
                       residual_scale: Optional[np.ndarray] = None) -> np.ndarray:
    # tone->area per ink using fitted gamma
    a = []
    for i, ch in enumerate(channels):
        t = np.clip(levels_vec_0_255[i] / 255.0, 0.0, 1.0)
        g = float(gammas_by_ch.get(ch, 1.0))
        a.append(np.power(t, np.clip(g, 0.1, 5.0)))
    a = np.array(a)
    pred = neug.mix(a).reshape(-1)
    if residual_scale is not None:
        pred = np.clip(pred * residual_scale, 0.0, 1.0)
    return pred


def fit_global_n(neug: Neugebauer, train_X: np.ndarray, train_Y: np.ndarray, channels: List[str],
                 gammas_by_ch: Dict[str, float], wavelengths: np.ndarray,
                 n_bounds: Tuple[float, float] = (1.0, 3.0), n_reg: float = 0.0, n_init: float = 2.0) -> float:
    from scipy.optimize import minimize_scalar

    def objective(n):
        old_n = neug.n
        neug.n = float(n)
        errs = []
        for i in range(train_X.shape[0]):
            pred = predict_with_model(neug, train_X[i], channels, gammas_by_ch)
            errs.append(np.mean((pred - train_Y[i]) ** 2))
        neug.n = old_n
        base = float(np.mean(errs))
        if n_reg > 0:
            base += float(n_reg) * (float(n) - 2.0) ** 2
        return base

    res = minimize_scalar(objective, bounds=n_bounds, method='bounded', options={'xatol': 1e-3})
    n_opt = float(np.clip(res.x if res.success else n_init, n_bounds[0], n_bounds[1]))
    neug.n = n_opt
    return n_opt


def fit_residual_scale(neug: Neugebauer, train_X: np.ndarray, train_Y: np.ndarray, channels: List[str],
                       gammas_by_ch: Dict[str, float], wavelengths: np.ndarray,
                       ridge: float = 1e-3) -> np.ndarray:
    # Solve per-wavelength multiplicative scale s minimizing sum ||s*pred - y||^2 + ridge||s-1||^2
    P = []
    for i in range(train_X.shape[0]):
        P.append(predict_with_model(neug, train_X[i], channels, gammas_by_ch))
    P = np.vstack(P)  # N x W
    Y = train_Y  # N x W
    num_w = P.shape[1]
    s = np.ones(num_w)
    for w in range(num_w):
        p = P[:, w]
        y = Y[:, w]
        num = np.dot(p, y) + ridge * 1.0
        den = np.dot(p, p) + ridge
        if den <= 1e-9:
            s[w] = 1.0
        else:
            s[w] = float(np.clip(num / den, 0.8, 1.2))
    return s


def split_train_holdout(indices: List[int], holdout_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.array(indices)
    rng.shuffle(idx)
    n_hold = int(round(len(idx) * holdout_frac))
    hold_idx = idx[:n_hold].tolist()
    train_idx = idx[n_hold:].tolist()
    if len(train_idx) == 0:  # ensure non-empty
        train_idx, hold_idx = hold_idx, []
    return train_idx, hold_idx


def calibrate_from_measurements(measured: Dict[str, Spectra], paper: Spectra,
                                holdout_frac: float = 0.2, seed: int = 42,
                                n_min: float = 1.0, n_max: float = 3.0, n_reg: float = 0.0,
                                enable_residual: bool = True) -> Tuple[Neugebauer, Dict[str, any]]:
    # Determine channels
    channels = discover_channels_from_names(list(measured.keys()))
    if not channels:
        raise ValueError("Could not infer channels from measurement names")

    # 1) Tone->area: fit per-ink gamma from single-ink ramps
    gammas_by_ch = fit_tone_to_area_gammas(channels, measured, paper, n_assumed=2.0)

    # 2) Fix primaries from measured W/singles/overprints
    neug, wavelengths = build_neugebauer_from_measurements(channels, measured, paper, prefer_overprints=True)

    # 3) Prepare datasets: mixed halftones only for fitting n/residual
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    names: List[str] = []
    for name, sp in measured.items():
        try:
            lv = parse_combo_name(name, channels)
        except Exception:
            continue
        # skip pure paper and pure solids (used as primaries), and single-channel-only
        if np.all(lv == 0):
            continue
        if np.all((lv == 0) | (lv == 255)):
            continue
        if np.count_nonzero(lv > 0) == 1:
            continue
        X_list.append(lv)
        Y_list.append(sp.interpolate_values(wavelengths).data)
        names.append(name)
    if not X_list:
        raise ValueError("No mixed halftone samples found for fitting")
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    # Train/holdout split
    train_idx, hold_idx = split_train_holdout(list(range(X.shape[0])), holdout_frac, seed)
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xho, Yho = (X[hold_idx], Y[hold_idx]) if len(hold_idx) else (None, None)

    # 4) Fit n
    n_opt = fit_global_n(neug, Xtr, Ytr, channels, gammas_by_ch, wavelengths, (n_min, n_max), n_reg, n_init=2.0)

    # 5) Optional residual model
    residual_scale = fit_residual_scale(neug, Xtr, Ytr, channels, gammas_by_ch,
                                        wavelengths) if enable_residual else None

    # Metrics
    def rmse(pred, gt):
        return float(np.sqrt(np.mean((pred - gt) ** 2)))

    train_errs = []
    for i in range(Xtr.shape[0]):
        pr = predict_with_model(neug, Xtr[i], channels, gammas_by_ch, residual_scale)
        train_errs.append(rmse(pr, Ytr[i]))
    hold_errs = []
    if Xho is not None:
        for i in range(Xho.shape[0]):
            pr = predict_with_model(neug, Xho[i], channels, gammas_by_ch, residual_scale)
            hold_errs.append(rmse(pr, Yho[i]))

    model = {
        "channels": channels,
        "gammas_by_ch": gammas_by_ch,
        "n": float(neug.n),
        "residual_scale": residual_scale.tolist() if residual_scale is not None else None,
        "wavelengths": wavelengths.tolist(),
        "train_mean_rmse": float(np.mean(train_errs)) if train_errs else None,
        "holdout_mean_rmse": float(np.mean(hold_errs)) if hold_errs else None,
        "num_train": len(train_idx),
        "num_holdout": len(hold_idx),
        "names_train": [names[i] for i in train_idx],
        "names_holdout": [names[i] for i in hold_idx]
    }

    return neug, model

# -------- Neugebauer wrapper using full-tone primaries --------


def select_fulltone_inks(ink_library: Dict[str, Spectra], channels: List[str]) -> List[Spectra]:
    """Pick the highest measured level (prefer 255) per channel to act as Neugebauer inks."""
    selected = []
    for ch in channels:
        candidates: List[Tuple[int, Spectra]] = []
        for name, sp in ink_library.items():
            m = PRIMARY_RE.match(name.strip())
            if m and m.group(1) == ch:
                lvl = int(m.group(2))
                candidates.append((lvl, sp))
        if not candidates:
            raise ValueError(f"No primaries found for channel '{ch}'")
        # Prefer exact 255, else max available
        candidates.sort(key=lambda x: x[0])
        best = None
        for lvl, sp in candidates:
            if lvl == 255:
                best = sp
        if best is None:
            best = candidates[-1][1]
        selected.append(best)
    return selected


def build_gamut_from_library(library_path: str) -> Tuple[InkGamut, List[str]]:
    inks_dict, paper, wavelengths = load_inkset_csv(library_path, filter_clogged=True)
    channels = discover_channels_from_names(list(inks_dict.keys()))
    fulltone = select_fulltone_inks(inks_dict, channels)
    gamut = InkGamut(fulltone, paper)
    return gamut, channels


# -------- TRC model (per-ink gamma) --------

def apply_trc_gamma(levels_0_255: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    t = np.clip(levels_0_255 / 255.0, 0.0, 1.0)
    g = np.clip(gammas, 0.1, 5.0)
    return np.power(t, g)


# -------- HTML Report Generation --------

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64


def generate_calibration_report(library_name: str, original_params: dict, calibrated_params: dict,
                                measured_combos: dict, gamut: InkGamut, channels: List[str],
                                wavelengths: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                original_errors: List[float], calibrated_errors: List[float],
                                combo_names: List[str]):
    """Generate comprehensive HTML calibration report."""

    report_dir = f"results/{library_name}_calibration_report"
    os.makedirs(report_dir, exist_ok=True)

    # Calculate metrics
    original_mean_rmse = np.mean(original_errors)
    calibrated_mean_rmse = np.mean(calibrated_errors)
    improvement = original_mean_rmse - calibrated_mean_rmse
    improvement_pct = (improvement / original_mean_rmse) * 100

    # Generate plots
    spectral_plots = generate_spectral_comparison_plots(measured_combos, gamut, channels, wavelengths,
                                                        original_params, calibrated_params, combo_names)
    gamut_plots = generate_gamut_visualization(measured_combos, gamut, channels, wavelengths,
                                               original_params, calibrated_params, combo_names)
    error_plots = generate_error_analysis_plots(original_errors, calibrated_errors, combo_names)

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calibration Report: {library_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .improvement {{ color: green; font-weight: bold; }}
            .parameter-comparison {{ display: flex; justify-content: space-around; }}
            .parameter-box {{ border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Calibration Report: {library_name}</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>Summary Metrics</h2>
            <table class="metrics-table">
                <tr><th>Metric</th><th>Original</th><th>Calibrated</th><th>Improvement</th></tr>
                <tr>
                    <td>Mean Spectral RMSE</td>
                    <td>{original_mean_rmse:.5f}</td>
                    <td>{calibrated_mean_rmse:.5f}</td>
                    <td class="improvement">{improvement:.5f} ({improvement_pct:.1f}%)</td>
                </tr>
                <tr>
                    <td>Median Spectral RMSE</td>
                    <td>{np.median(original_errors):.5f}</td>
                    <td>{np.median(calibrated_errors):.5f}</td>
                    <td class="improvement">{np.median(original_errors) - np.median(calibrated_errors):.5f}</td>
                </tr>
                <tr>
                    <td>95th Percentile RMSE</td>
                    <td>{np.percentile(original_errors, 95):.5f}</td>
                    <td>{np.percentile(calibrated_errors, 95):.5f}</td>
                    <td class="improvement">{np.percentile(original_errors, 95) - np.percentile(calibrated_errors, 95):.5f}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Parameter Comparison</h2>
            <div class="parameter-comparison">
                <div class="parameter-box">
                    <h3>Original Parameters</h3>
                    <p><strong>Yule-Nielsen n:</strong> {original_params.get('n', 'N/A')}</p>
                    <p><strong>TRC Gammas:</strong> {original_params.get('gammas', 'N/A')}</p>
                </div>
                <div class="parameter-box">
                    <h3>Calibrated Parameters</h3>
                    <p><strong>Yule-Nielsen n:</strong> {calibrated_params.get('n', 'N/A')}</p>
                    <p><strong>TRC Gammas:</strong> {calibrated_params.get('gammas', 'N/A')}</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Spectral Comparison Plots</h2>
            <p>Blue: Measured, Red: Original Prediction, Green: Calibrated Prediction</p>
            {spectral_plots}
        </div>

        <div class="section">
            <h2>Gamut Visualization</h2>
            <p>Lab color space projections showing measured vs predicted points</p>
            {gamut_plots}
        </div>

        <div class="section">
            <h2>Error Analysis</h2>
            {error_plots}
        </div>
    </body>
    </html>
    """

    # Save HTML report
    report_path = os.path.join(report_dir, "calibration_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Calibration report saved to: {report_path}")
    return report_path


def generate_spectral_comparison_plots(measured_combos: dict, gamut: InkGamut, channels: List[str],
                                       wavelengths: np.ndarray, original_params: dict, calibrated_params: dict,
                                       combo_names: List[str]) -> str:
    """Generate spectral comparison plots for HTML report."""
    plots_html = ""

    # Create plots in batches of 4
    for i in range(0, len(combo_names), 4):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for j in range(4):
            if i + j >= len(combo_names):
                axes[j].set_visible(False)
                continue

            name = combo_names[i + j]
            sp_meas = measured_combos[name]
            levels = parse_combo_name(name, channels)

            # Original prediction
            orig_pred = predict_spectrum_from_levels(gamut, levels,
                                                     original_params.get('gammas'),
                                                     original_params.get('n'))

            # Calibrated prediction
            cal_pred = predict_spectrum_from_levels(gamut, levels,
                                                    calibrated_params.get('gammas'),
                                                    calibrated_params.get('n'))

            # Interpolate measured to match wavelengths
            if not np.array_equal(sp_meas.wavelengths, wavelengths):
                sp_meas = sp_meas.interpolate_values(wavelengths)

            # Plot
            axes[j].plot(wavelengths, sp_meas.data, 'b-', label='Measured', linewidth=2)
            axes[j].plot(wavelengths, orig_pred, 'r--', label='Original', linewidth=2)
            axes[j].plot(wavelengths, cal_pred, 'g--', label='Calibrated', linewidth=2)
            axes[j].set_title(f'{name}')
            axes[j].set_xlabel('Wavelength (nm)')
            axes[j].set_ylabel('Reflectance')
            axes[j].legend()
            axes[j].grid(True, alpha=0.3)
            axes[j].set_ylim(0, 1)

        plt.tight_layout()
        plots_html += f'<div class="plot"><img src="data:image/png;base64,{plot_to_base64(fig)}" /></div>'
        plt.close()

    return plots_html


def generate_gamut_visualization(measured_combos: dict, gamut: InkGamut, channels: List[str],
                                 wavelengths: np.ndarray, original_params: dict, calibrated_params: dict,
                                 combo_names: List[str]) -> str:
    """Generate gamut visualization plots for HTML report."""

 # Set up observer and illuminant
    d65 = Illuminant.get("d65")
    tetrachromat = Observer.tetrachromat(wavelengths=np.arange(400, 710, 10))

    # Convert spectra to Lab
    measured_lab = []
    original_lab = []
    calibrated_lab = []

    for name in combo_names:
        sp_meas = measured_combos[name]
        levels = parse_combo_name(name, channels)

        # Interpolate measured to standard wavelengths
        if not np.array_equal(sp_meas.wavelengths, wavelengths):
            sp_meas = sp_meas.interpolate_values(wavelengths)

        # Original prediction
        orig_pred = predict_spectrum_from_levels(gamut, levels,
                                                 original_params.get('gammas'),
                                                 original_params.get('n'))

        # Calibrated prediction
        cal_pred = predict_spectrum_from_levels(gamut, levels,
                                                calibrated_params.get('gammas'),
                                                calibrated_params.get('n'))

        # Convert to Lab
        sp_meas_obj = Spectra(wavelengths=wavelengths, data=sp_meas.data)
        orig_obj = Spectra(wavelengths=wavelengths, data=orig_pred)
        cal_obj = Spectra(wavelengths=wavelengths, data=cal_pred)

        measured_lab.append(sp_meas_obj.to_lab())
        original_lab.append(orig_obj.to_lab())
        calibrated_lab.append(cal_obj.to_lab())

    measured_lab = np.array(measured_lab)
    original_lab = np.array(original_lab)
    calibrated_lab = np.array(calibrated_lab)

    # Create Lab projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # L*a* projection
    axes[0].scatter(measured_lab[:, 1], measured_lab[:, 0], c='blue', alpha=0.7, label='Measured', s=50)
    axes[0].scatter(original_lab[:, 1], original_lab[:, 0], c='red', alpha=0.7, label='Original', s=30, marker='x')
    axes[0].scatter(calibrated_lab[:, 1], calibrated_lab[:, 0], c='green',
                    alpha=0.7, label='Calibrated', s=30, marker='+')
    axes[0].set_xlabel('a*')
    axes[0].set_ylabel('L*')
    axes[0].set_title('L*a* Projection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # a*b* projection
    axes[1].scatter(measured_lab[:, 1], measured_lab[:, 2], c='blue', alpha=0.7, label='Measured', s=50)
    axes[1].scatter(original_lab[:, 1], original_lab[:, 2], c='red', alpha=0.7, label='Original', s=30, marker='x')
    axes[1].scatter(calibrated_lab[:, 1], calibrated_lab[:, 2], c='green',
                    alpha=0.7, label='Calibrated', s=30, marker='+')
    axes[1].set_xlabel('a*')
    axes[1].set_ylabel('b*')
    axes[1].set_title('a*b* Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # L*b* projection
    axes[2].scatter(measured_lab[:, 2], measured_lab[:, 0], c='blue', alpha=0.7, label='Measured', s=50)
    axes[2].scatter(original_lab[:, 2], original_lab[:, 0], c='red', alpha=0.7, label='Original', s=30, marker='x')
    axes[2].scatter(calibrated_lab[:, 2], calibrated_lab[:, 0], c='green',
                    alpha=0.7, label='Calibrated', s=30, marker='+')
    axes[2].set_xlabel('b*')
    axes[2].set_ylabel('L*')
    axes[2].set_title('L*b* Projection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plots_html = f'<div class="plot"><img src="data:image/png;base64,{plot_to_base64(fig)}" /></div>'
    plt.close()

    return plots_html


def generate_error_analysis_plots(original_errors: List[float], calibrated_errors: List[float],
                                  combo_names: List[str]) -> str:
    """Generate error analysis plots for HTML report."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Error improvement histogram
    improvements = np.array(original_errors) - np.array(calibrated_errors)
    axes[0, 0].hist(improvements, bins=20, alpha=0.7, color='green')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('RMSE Improvement')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of RMSE Improvements')
    axes[0, 0].grid(True, alpha=0.3)

    # Original vs Calibrated scatter
    axes[0, 1].scatter(original_errors, calibrated_errors, alpha=0.7)
    axes[0, 1].plot([0, max(original_errors)], [0, max(original_errors)], 'r--', alpha=0.7)
    axes[0, 1].set_xlabel('Original RMSE')
    axes[0, 1].set_ylabel('Calibrated RMSE')
    axes[0, 1].set_title('Original vs Calibrated RMSE')
    axes[0, 1].grid(True, alpha=0.3)

    # Error comparison box plot
    axes[1, 0].boxplot([original_errors, calibrated_errors], labels=['Original', 'Calibrated'])
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('RMSE Distribution Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative improvement
    sorted_improvements = np.sort(improvements)[::-1]
    cumulative_improvement = np.cumsum(sorted_improvements)
    axes[1, 1].plot(range(len(cumulative_improvement)), cumulative_improvement)
    axes[1, 1].set_xlabel('Combination Rank')
    axes[1, 1].set_ylabel('Cumulative RMSE Improvement')
    axes[1, 1].set_title('Cumulative Improvement by Rank')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plots_html = f'<div class="plot"><img src="data:image/png;base64,{plot_to_base64(fig)}" /></div>'
    plt.close()

    return plots_html


# -------- Subcommands --------

def cmd_register_primaries(args):
    input_path = os.path.abspath(args.input_file)
    out_name = args.name

    # Try Nix CSV first, else assume already in standard format
    try:
        name_to_spectra, paper = read_nix_csv(input_path)

        # Discover channels from all measurements (not just 255-level)
        channels = discover_channels_from_names(list(name_to_spectra.keys()))
        if not channels:
            raise ValueError("Could not infer channels from measurement names")

        # Build complete Neugebauer primaries dictionary
        primaries_dict, wavelengths = build_complete_neugebauer_primaries(
            name_to_spectra, paper, channels, prefer_overprints=True
        )

        # Save both formats: standard CSV and primaries dictionary
        output_dir = f"data/inksets/{out_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save standard CSV format (for backward compatibility)
        csv_path = os.path.join(output_dir, f"{out_name}-inks.csv")
        save_primaries_as_csv(primaries_dict, wavelengths, csv_path)

        # 2. Save primaries dictionary as pickle (for easy loading)
        primaries_path = os.path.join(output_dir, f"{out_name}-primaries.pkl")
        save_primaries_dict(primaries_dict, primaries_path)

        # Register in library registry with both formats
        registry.register_library(
            out_name,
            f"{out_name}/{out_name}-inks.csv",
            {
                "created": "registered_primaries",
                "source_path": input_path,
                "primaries_path": f"{out_name}/{out_name}-primaries.pkl",
                "channels": channels,
                "num_primaries": len(primaries_dict),
                "primaries_keys": list(primaries_dict.keys()),
                "has_complete_neugebauer": True,
            }
        )

        print(f"Registered complete Neugebauer primaries for library '{out_name}'")
        print(f"  Channels: {channels}")
        print(f"  Number of primaries: {len(primaries_dict)}")
        print(f"  Primary keys: {sorted(primaries_dict.keys())}")
        print(f"  CSV saved to: {csv_path}")
        print(f"  Primaries dict saved to: {primaries_path}")
        return

    except Exception as e:
        print(f"Nix CSV parsing failed: {e}")
        # Fallback: assume it's already a standard inkset CSV → filter and register
        pass

    # Validate and register a standard-format inkset CSV
    try:
        inks, paper, wavelengths = load_inkset_csv(input_path, filter_clogged=True)

        # Discover channels from all measurements
        channels = discover_channels_from_names(list(inks.keys()))
        if not channels:
            raise ValueError("Could not infer channels from measurement names")

        # Build complete Neugebauer primaries dictionary
        primaries_dict, wavelengths = build_complete_neugebauer_primaries(
            inks, paper, channels, prefer_overprints=True
        )

        # Save both formats
        output_dir = f"data/inksets/{out_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save standard CSV format
        csv_path = os.path.join(output_dir, f"{out_name}-inks.csv")
        save_primaries_as_csv(primaries_dict, wavelengths, csv_path)

        # 2. Save primaries dictionary as pickle
        primaries_path = os.path.join(output_dir, f"{out_name}-primaries.pkl")
        save_primaries_dict(primaries_dict, primaries_path)

        # Register in library registry
        registry.register_library(
            out_name,
            f"{out_name}/{out_name}-inks.csv",
            {
                "created": "registered_primaries",
                "source_path": input_path,
                "primaries_path": f"{out_name}/{out_name}-primaries.pkl",
                "channels": channels,
                "num_primaries": len(primaries_dict),
                "primaries_keys": list(primaries_dict.keys()),
                "has_complete_neugebauer": True,
            }
        )

        print(f"Registered complete Neugebauer primaries for library '{out_name}'")
        print(f"  Channels: {channels}")
        print(f"  Number of primaries: {len(primaries_dict)}")
        print(f"  Primary keys: {sorted(primaries_dict.keys())}")
        print(f"  CSV saved to: {csv_path}")
        print(f"  Primaries dict saved to: {primaries_path}")

    except Exception as e:
        raise ValueError(f"Failed to load CSV '{input_path}': {e}")


def predict_spectrum_from_levels(gamut: InkGamut, levels_vec_0_255: np.ndarray,
                                 gammas: Optional[np.ndarray] = None,
                                 n_param: Optional[float] = None) -> np.ndarray:
    """Return predicted spectral data (numpy array) using Neugebauer with optional TRC gamma and n."""
    k = gamut.neugebauer.num_inks
    if levels_vec_0_255.shape[0] != k:
        raise ValueError("Combination vector length does not match number of inks")
    if gammas is None:
        gammas = np.ones(k)
    a = apply_trc_gamma(levels_vec_0_255, gammas)  # area coverages

    neug = gamut.neugebauer
    old_n = neug.n
    if n_param is not None:
        neug.n = n_param
    try:
        spectra = neug.mix(a)  # shape (wavelengths,)
        return spectra.reshape(-1)
    finally:
        neug.n = old_n


def cmd_evaluate(args):
    if not registry.library_exists(args.library_name):
        raise ValueError(f"Library '{args.library_name}' not found in registry")
    library_path = registry.resolve_library_path(args.library_name)

    gamut, channels = build_gamut_from_library(library_path)
    combos, _ = read_nix_csv(os.path.abspath(args.combos_file))
    wavelengths = combos[list(combos.keys())[0]].wavelengths

    # Create plots directory
    plots_dir = f"results/{args.library_name}_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Evaluate spectral RMSE
    rows = []
    for name, sp_meas in combos.items():
        try:
            levels = parse_combo_name(name, channels)
        except Exception as e:
            if args.strict:
                raise
            else:
                print(f"Skipping '{name}': {e}")
                continue
        spec_pred = Spectra(wavelengths=wavelengths, data=predict_spectrum_from_levels(gamut, levels))
        # Interpolate measured to gamut wavelengths if needed
        meas = sp_meas
        if not np.array_equal(meas.wavelengths, wavelengths):
            meas = meas.interpolate_values(wavelengths)
        err = Spectra.spectral_rmse(spec_pred, meas)
        de = Spectra.delta_e(spec_pred, meas)

        rows.append({"name": name, "rmse": err, "delta_e": de})

        # Create plot for this combination
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, meas.data, 'b-', label='Measured', linewidth=2)
        plt.plot(wavelengths, spec_pred.data, 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f'{name}: RMSE={err:.4f}, ΔE={de:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(wavelengths[0], wavelengths[-1])
        plt.ylim(0, 1)

        # Save plot
        safe_name = re.sub(r'[^\w\-_]', '_', name)  # Make filename safe
        plot_path = os.path.join(plots_dir, f"{safe_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No valid combinations evaluated.")
        return
    print(
        f"Count={len(df)}  "
        f"meanRMSE={df.rmse.mean():.5f}  medianRMSE={df.rmse.median():.5f}  p95RMSE={df.rmse.quantile(0.95):.5f}  "
        f"meanΔE={df.delta_e.mean():.5f}  medianΔE={df.delta_e.median():.5f}  p95ΔE={df.delta_e.quantile(0.95):.5f}"
    )
    print(f"Plots saved to: {plots_dir}")

    if args.out:
        outp = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Saved evaluation to {outp}")


def cmd_calibrate(args):
    if not registry.library_exists(args.library_name):
        raise ValueError(f"Library '{args.library_name}' not found in registry")
    library_path = registry.resolve_library_path(args.library_name)
    gamut, channels = build_gamut_from_library(library_path)
    combos, _ = read_nix_csv(os.path.abspath(args.combos_file))
    wavelengths = combos[list(combos.keys())[0]].wavelengths

    X_levels: List[np.ndarray] = []
    Y_specs: List[np.ndarray] = []
    combo_names: List[str] = []
    for name, sp in combos.items():
        try:
            levels = parse_combo_name(name, channels)
        except Exception:
            continue
        X_levels.append(levels)
        if not np.array_equal(sp.wavelengths, wavelengths):
            sp = sp.interpolate_values(wavelengths)
        Y_specs.append(sp.data)
        combo_names.append(name)
    if not X_levels:
        raise ValueError("No valid combos for calibration.")
    X = np.stack(X_levels, axis=0)
    Y = np.stack(Y_specs, axis=0)

    # Parameters: per-ink gammas and optional global n
    k = gamut.neugebauer.num_inks
    init_gammas = np.ones(k)
    init_n = 50.0

    do_fit_n = (args.fit_n is True)
    do_fit_trc = (args.fit_trc is True)

    from scipy.optimize import minimize

    def pack_params(gammas, n):
        if do_fit_n and do_fit_trc:
            return np.concatenate([gammas, [n]])
        elif do_fit_trc:
            return gammas.copy()
        elif do_fit_n:
            return np.array([n])
        else:
            return np.array([])

    def unpack_params(p):
        gam = init_gammas.copy()
        nval = None
        if do_fit_n and do_fit_trc:
            gam = p[:k]
            nval = float(p[-1])
        elif do_fit_trc:
            gam = p[:k]
        elif do_fit_n:
            nval = float(p[0])
        return gam, nval

    p0 = pack_params(init_gammas, init_n)

    def objective(p):
        gam, nval = unpack_params(p)
        errs = []
        for i in range(X.shape[0]):
            pred = predict_spectrum_from_levels(gamut, X[i], gam, nval)
            sp_pred = Spectra(wavelengths=wavelengths, data=pred)
            sp_meas = Spectra(wavelengths=wavelengths, data=Y[i])
            errs.append(Spectra.spectral_rmse(sp_pred, sp_meas))
        return float(np.mean(errs))

    bounds = None
    if do_fit_n and do_fit_trc:
        bounds = [(0.1, 5.0)] * k + [(1.0, 100.0)]
    elif do_fit_trc:
        bounds = [(0.1, 5.0)] * k
    elif do_fit_n:
        bounds = [(1.0, 100.0)]

    res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
    opt_gam, opt_n = unpack_params(res.x)

    print(f"Calibrated: meanRMSE={res.fun:.5f}")
    if do_fit_trc:
        print(f"  gammas= {np.round(opt_gam, 4).tolist()}")
    if do_fit_n:
        print(f"  n= {opt_n:.3f}")

    if args.model_out:
        model = {
            "channels": channels,
            "gammas": opt_gam.tolist() if do_fit_trc else None,
            "n": float(opt_n) if do_fit_n else None,
            "objective_mean_rmse": float(res.fun),
        }
        outp = os.path.abspath(args.model_out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as f:
            json.dump(model, f, indent=2)
        print(f"Saved calibration model to {outp}")

    # Generate HTML report if requested
    if args.visualize:
        # Calculate original errors
        original_errors = []
        calibrated_errors = []

        for i in range(X.shape[0]):
            # Original prediction
            orig_pred = predict_spectrum_from_levels(gamut, X[i], init_gammas, init_n)
            sp_orig = Spectra(wavelengths=wavelengths, data=orig_pred)
            sp_meas = Spectra(wavelengths=wavelengths, data=Y[i])
            original_errors.append(Spectra.spectral_rmse(sp_orig, sp_meas))

            # Calibrated prediction
            cal_pred = predict_spectrum_from_levels(gamut, X[i], opt_gam, opt_n)
            sp_cal = Spectra(wavelengths=wavelengths, data=cal_pred)
            calibrated_errors.append(Spectra.spectral_rmse(sp_cal, sp_meas))

        # Prepare parameters for report
        original_params = {
            'n': init_n if do_fit_n else None,
            'gammas': init_gammas.tolist() if do_fit_trc else None
        }
        calibrated_params = {
            'n': opt_n if do_fit_n else None,
            'gammas': opt_gam.tolist() if do_fit_trc else None
        }

        # Generate HTML report
        report_path = generate_calibration_report(
            library_name=args.library_name,
            original_params=original_params,
            calibrated_params=calibrated_params,
            measured_combos=combos,
            gamut=gamut,
            channels=channels,
            wavelengths=wavelengths,
            X=X,
            Y=Y,
            original_errors=original_errors,
            calibrated_errors=calibrated_errors,
            combo_names=combo_names
        )


def generate_holdout_visualization(neug: Neugebauer, model: Dict[str, any], combos: Dict[str, Spectra],
                                   paper: Spectra, output_dir: str = None) -> str:
    """Generate visualization plots for holdout validation performance."""

    if output_dir is None:
        output_dir = "results/holdout_validation"
    os.makedirs(output_dir, exist_ok=True)

    channels = model['channels']
    gammas_by_ch = model['gammas_by_ch']
    residual_scale = model.get('residual_scale')
    wavelengths = np.array(model['wavelengths'])
    holdout_names = model.get('names_holdout', [])

    if not holdout_names:
        print("No holdout samples to visualize")
        return output_dir

    # Calculate errors for all samples (both train and holdout)
    train_names = model.get('names_train', [])
    all_train_errors = []
    all_train_delta_e = []
    all_holdout_errors = []
    all_holdout_delta_e = []

    # Training samples
    for name in train_names:
        sp_meas = combos[name]
        levels = parse_combo_name(name, channels)
        pred = predict_with_model(neug, levels, channels, gammas_by_ch, residual_scale)

        # Interpolate measured to match wavelengths if needed
        if not np.array_equal(sp_meas.wavelengths, wavelengths):
            sp_meas = sp_meas.interpolate_values(wavelengths)

        # Calculate errors
        rmse = float(np.sqrt(np.mean((pred - sp_meas.data) ** 2)))
        all_train_errors.append(rmse)

        # Calculate delta E
        sp_pred = Spectra(wavelengths=wavelengths, data=pred)
        de = Spectra.delta_e(sp_pred, sp_meas)
        all_train_delta_e.append(de)

    # Holdout samples
    for name in holdout_names:
        sp_meas = combos[name]
        levels = parse_combo_name(name, channels)
        pred = predict_with_model(neug, levels, channels, gammas_by_ch, residual_scale)

        # Interpolate measured to match wavelengths if needed
        if not np.array_equal(sp_meas.wavelengths, wavelengths):
            sp_meas = sp_meas.interpolate_values(wavelengths)

        # Calculate errors
        rmse = float(np.sqrt(np.mean((pred - sp_meas.data) ** 2)))
        all_holdout_errors.append(rmse)

        # Calculate delta E
        sp_pred = Spectra(wavelengths=wavelengths, data=pred)
        de = Spectra.delta_e(sp_pred, sp_meas)
        all_holdout_delta_e.append(de)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Spectral comparison plots (top-left) - show both train and holdout
    ax1 = axes[0, 0]

    # Plot training samples (first 3)
    for i, name in enumerate(train_names[:3]):
        sp_meas = combos[name]
        levels = parse_combo_name(name, channels)
        pred = predict_with_model(neug, levels, channels, gammas_by_ch, residual_scale)

        # Interpolate measured to match wavelengths if needed
        if not np.array_equal(sp_meas.wavelengths, wavelengths):
            sp_meas = sp_meas.interpolate_values(wavelengths)

        ax1.plot(wavelengths, sp_meas.data, color=sp_meas.to_rgb(),
                 alpha=0.7, linewidth=1.5, linestyle='-', label=f'{name} (train meas)')
        ax1.plot(wavelengths, pred, color=Spectra(wavelengths=wavelengths, data=pred).to_rgb(),
                 alpha=0.7, linewidth=1.5, linestyle='--', label=f'{name} (train pred)')

    # Plot holdout samples (first 3)
    for i, name in enumerate(holdout_names[:3]):
        sp_meas = combos[name]
        levels = parse_combo_name(name, channels)
        pred = predict_with_model(neug, levels, channels, gammas_by_ch, residual_scale)

        # Interpolate measured to match wavelengths if needed
        if not np.array_equal(sp_meas.wavelengths, wavelengths):
            sp_meas = sp_meas.interpolate_values(wavelengths)

        ax1.plot(wavelengths, sp_meas.data, color=sp_meas.to_rgb(),
                 alpha=0.5, linewidth=2.0, linestyle='-', label=f'{name} (holdout meas)')
        ax1.plot(wavelengths, pred, color=Spectra(wavelengths=wavelengths, data=pred).to_rgb(),
                 alpha=0.5, linewidth=2.0, linestyle=':', label=f'{name} (holdout pred)')

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Training vs Holdout: Measured vs Predicted Spectra')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 2. Error distribution (top-right) - show both train and holdout
    ax2 = axes[0, 1]
    if (all_train_errors and all_train_delta_e) or (all_holdout_errors and all_holdout_delta_e):
        # Create dual y-axis plot
        ax2_twin = ax2.twinx()

        # Plot RMSE histograms for both train and holdout
        if all_train_errors:
            ax2.hist(all_train_errors, bins=10, alpha=0.6, color='blue', edgecolor='black',
                     label=f'Train RMSE (n={len(all_train_errors)})')
            ax2.axvline(np.mean(all_train_errors), color='darkblue', linestyle='--', linewidth=2,
                        label=f'Train RMSE Mean: {np.mean(all_train_errors):.4f}')

        if all_holdout_errors:
            ax2.hist(all_holdout_errors, bins=10, alpha=0.8, color='red', edgecolor='black',
                     label=f'Holdout RMSE (n={len(all_holdout_errors)})')
            ax2.axvline(np.mean(all_holdout_errors), color='darkred', linestyle='--', linewidth=2,
                        label=f'Holdout RMSE Mean: {np.mean(all_holdout_errors):.4f}')

        # Plot Delta E histograms on secondary axis
        if all_train_delta_e:
            ax2_twin.hist(all_train_delta_e, bins=10, alpha=0.4, color='lightblue', edgecolor='black',
                          label=f'Train ΔE (n={len(all_train_delta_e)})')
            ax2_twin.axvline(np.mean(all_train_delta_e), color='blue', linestyle=':', linewidth=2,
                             label=f'Train ΔE Mean: {np.mean(all_train_delta_e):.2f}')

        if all_holdout_delta_e:
            ax2_twin.hist(all_holdout_delta_e, bins=10, alpha=0.6, color='lightgreen', edgecolor='black',
                          label=f'Holdout ΔE (n={len(all_holdout_delta_e)})')
            ax2_twin.axvline(np.mean(all_holdout_delta_e), color='green', linestyle=':', linewidth=2,
                             label=f'Holdout ΔE Mean: {np.mean(all_holdout_delta_e):.2f}')

        ax2.set_xlabel('Error Value')
        ax2.set_ylabel('Count (RMSE)', color='black')
        ax2_twin.set_ylabel('Count (ΔE)', color='gray')
        ax2.set_title('Train vs Holdout Error Distributions')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

        ax2.grid(True, alpha=0.3)

    # 3. Error vs combination complexity (bottom-left) - show both train and holdout
    ax3 = axes[1, 0]

    # Calculate complexity for all samples
    train_complexity = []
    holdout_complexity = []

    for name in train_names:
        levels = parse_combo_name(name, channels)
        active_inks = np.count_nonzero(levels > 0)
        normalized_sum = np.sum(levels) / 255.0
        complexity = active_inks + normalized_sum
        train_complexity.append(complexity)

    for name in holdout_names:
        levels = parse_combo_name(name, channels)
        active_inks = np.count_nonzero(levels > 0)
        normalized_sum = np.sum(levels) / 255.0
        complexity = active_inks + normalized_sum
        holdout_complexity.append(complexity)

    if all_train_errors and train_complexity:
        ax3.scatter(train_complexity, all_train_errors, alpha=0.6, color='blue', s=30, label='Train')
    if all_holdout_errors and holdout_complexity:
        ax3.scatter(holdout_complexity, all_holdout_errors, alpha=0.8, color='red', s=50, label='Holdout')

    ax3.set_xlabel('Combination Complexity (active inks + normalized sum)')
    ax3.set_ylabel('Spectral RMSE')
    ax3.set_title('Error vs Combination Complexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary metrics table (bottom-right)
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    train_rmse = model.get('train_mean_rmse', 'N/A')
    holdout_rmse = model.get('holdout_mean_rmse', 'N/A')

    summary_data = [
        ['Metric', 'Value'],
        ['Fitted n', f"{model['n']:.4f}"],
        ['Train RMSE', f"{train_rmse:.5f}" if isinstance(train_rmse, float) else str(train_rmse)],
        ['Holdout RMSE', f"{holdout_rmse:.5f}" if isinstance(holdout_rmse, float) else str(holdout_rmse)],
        ['Train samples', str(model.get('num_train', 'N/A'))],
        ['Holdout samples', str(model.get('num_holdout', 'N/A'))],
        ['Residual scale', 'Enabled' if residual_scale is not None else 'Disabled']
    ]

    if all_train_errors:
        summary_data.extend([
            ['Train RMSE mean', f"{np.mean(all_train_errors):.5f}"],
            ['Train RMSE median', f"{np.median(all_train_errors):.5f}"],
            ['Train RMSE std', f"{np.std(all_train_errors):.5f}"]
        ])

    if all_holdout_errors:
        summary_data.extend([
            ['Holdout RMSE mean', f"{np.mean(all_holdout_errors):.5f}"],
            ['Holdout RMSE median', f"{np.median(all_holdout_errors):.5f}"],
            ['Holdout RMSE std', f"{np.std(all_holdout_errors):.5f}"]
        ])

    if all_train_delta_e:
        summary_data.extend([
            ['Train ΔE mean', f"{np.mean(all_train_delta_e):.2f}"],
            ['Train ΔE median', f"{np.median(all_train_delta_e):.2f}"],
            ['Train ΔE std', f"{np.std(all_train_delta_e):.2f}"]
        ])

    if all_holdout_delta_e:
        summary_data.extend([
            ['Holdout ΔE mean', f"{np.mean(all_holdout_delta_e):.2f}"],
            ['Holdout ΔE median', f"{np.median(all_holdout_delta_e):.2f}"],
            ['Holdout ΔE std', f"{np.std(all_holdout_delta_e):.2f}"],
            ['Holdout ΔE 95th %ile', f"{np.percentile(all_holdout_delta_e, 95):.2f}"]
        ])

    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Calibration Summary', pad=20)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "holdout_validation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Holdout validation plots saved to: {plot_path}")
    return output_dir


def cmd_calibrate_proc(args):
    # New procedure based on provided measurements file (dict-like names->Spectra)

    combos, paper = read_nix_csv(os.path.abspath(args.combos_file))

    neug, model = calibrate_from_measurements(
        measured=combos,
        paper=paper,
        holdout_frac=args.holdout,
        seed=args.seed,
        n_min=args.n_min,
        n_max=args.n_max,
        n_reg=args.n_reg,
        enable_residual=(not args.no_residual)
    )
    # Save neug as a pickle file if requested

    # Report
    print(f"Fitted n = {model['n']:.4f}")
    print(f"Per-ink gammas: {model['gammas_by_ch']}")
    if model['residual_scale'] is not None:
        print("Residual scale enabled (per-wavelength)")
    if model['train_mean_rmse'] is not None:
        print(f"Train mean RMSE: {model['train_mean_rmse']:.5f}")
    if model['holdout_mean_rmse'] is not None:
        print(f"Holdout mean RMSE: {model['holdout_mean_rmse']:.5f}")

    # Calculate and report Delta E statistics for holdout set
    if model.get('num_holdout', 0) > 0:
        channels = model['channels']
        gammas_by_ch = model['gammas_by_ch']
        residual_scale = model.get('residual_scale')
        wavelengths = np.array(model['wavelengths'])
        holdout_names = model.get('names_holdout', [])

        holdout_delta_e = []
        for name in holdout_names:
            sp_meas = combos[name]
            levels = parse_combo_name(name, channels)
            pred = predict_with_model(neug, levels, channels, gammas_by_ch, residual_scale)

            with open("neug.pkl", "wb") as f:
                import pickle
                pickle.dump((neug, levels, channels, gammas_by_ch, residual_scale), f)
            print(f"Saved Neugebauer model (neug) to neug.pkl")

            # Interpolate measured to match wavelengths if needed
            if not np.array_equal(sp_meas.wavelengths, wavelengths):
                sp_meas = sp_meas.interpolate_values(wavelengths)

            # Calculate delta E
            sp_pred = Spectra(wavelengths=wavelengths, data=pred)
            de = Spectra.delta_e(sp_pred, sp_meas)
            holdout_delta_e.append(de)

        if holdout_delta_e:
            print(
                f"Holdout ΔE - Mean: {np.mean(holdout_delta_e):.2f}, Median: {np.median(holdout_delta_e):.2f}, 95th %ile: {np.percentile(holdout_delta_e, 95):.2f}")

    # Generate holdout visualization if we have holdout samples
    if model.get('num_holdout', 0) > 0:
        viz_dir = generate_holdout_visualization(neug, model, combos, paper)
        print(f"Holdout validation plots saved to: {viz_dir}")

    # Optional export
    if args.model_out:
        outp = os.path.abspath(args.model_out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as f:
            json.dump(model, f, indent=2)
        print(f"Saved model to {outp}")

    # Return Neugebauer in-process by storing in a module-global for interactive use
    globals()["_last_calibrated_neugebauer"] = neug
    print("Neugebauer object available as _last_calibrated_neugebauer for programmatic use.")


def main():
    parser = argparse.ArgumentParser(description='Inkset checking and calibration CLI')
    sub = parser.add_subparsers(dest='command', help='Commands')

    # register-primaries
    p_reg = sub.add_parser('register-primaries', help='Register measured primaries CSV as an ink library')
    p_reg.add_argument('name', help='Library name to register (e.g., nix_printer_primaries_XYZ)')
    p_reg.add_argument('input_file', help='Path to measured primaries CSV (Nix or standard format)')

    # evaluate
    p_eval = sub.add_parser('evaluate', help='Evaluate Neugebauer vs measured combos (spectral RMSE)')
    p_eval.add_argument('library_name', help='Registered library name of primaries')
    p_eval.add_argument('combos_file', help='CSV of measured combinations (e.g., C255M128)')
    p_eval.add_argument('--out', help='Path to save CSV of per-combo errors')
    p_eval.add_argument('--strict', action='store_true', help='Error on bad names instead of skipping')

    # calibrate
    p_cal = sub.add_parser('calibrate', help='Fit Yule–Nielsen n and/or per-ink TRC gammas')
    p_cal.add_argument('library_name', help='Registered library name of primaries')
    p_cal.add_argument('combos_file', help='CSV of measured combinations for calibration')
    p_cal.add_argument('--fit-n', dest='fit_n', action='store_true', help='Fit global Yule–Nielsen n')
    p_cal.add_argument('--no-fit-n', dest='fit_n', action='store_false')
    p_cal.set_defaults(fit_n=True)
    p_cal.add_argument('--fit-trc', dest='fit_trc', action='store_true', help='Fit per-ink TRC gammas')
    p_cal.add_argument('--no-fit-trc', dest='fit_trc', action='store_false')
    p_cal.set_defaults(fit_trc=True)
    p_cal.add_argument('--model-out', help='Path to save fitted model JSON')
    p_cal.add_argument('--visualize', action='store_true', help='Generate HTML report with visualizations')

    # New: procedure-3 calibrate that builds Neugebauer from measurements and returns it
    p_cproc = sub.add_parser(
        'calibrate-proc', help='New procedure: tone->area, fixed primaries, fit n, optional residual')
    p_cproc.add_argument('combos_file', help='Measurements file (names->Spectra), e.g. Nix CSV')
    p_cproc.add_argument('--holdout', type=float, default=0.2, help='Holdout fraction for validation')
    p_cproc.add_argument('--seed', type=int, default=42, help='Random seed for split')
    p_cproc.add_argument('--n-min', type=float, default=1.0, help='Lower bound for n')
    p_cproc.add_argument('--n-max', type=float, default=3.0, help='Upper bound for n')
    p_cproc.add_argument('--n-reg', type=float, default=0.0, help='L2 regularization strength on n about 2.0')
    p_cproc.add_argument('--no-residual', action='store_true', help='Disable residual per-wavelength scaling')
    p_cproc.add_argument('--model-out', help='Path to save fitted model JSON')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == 'register-primaries':
        cmd_register_primaries(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'calibrate':
        cmd_calibrate(args)
    elif args.command == 'calibrate-proc':
        cmd_calibrate_proc(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
