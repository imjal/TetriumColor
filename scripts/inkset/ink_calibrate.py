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

from TetriumColor.Observer.Inks import InkGamut, Neugebauer
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
    from TetriumColor.Observer import Observer, Illuminant

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
        # Filter to keep only 255-level single-channel primaries
        filtered_spectra = {}
        for nm, sp in name_to_spectra.items():
            m = PRIMARY_RE.match(nm.strip())
            if m and m.group(2) == "255":  # Only 255-level primaries
                filtered_spectra[nm] = sp
            elif nm.strip().lower() == 'paper':  # Keep paper
                if paper is None:
                    paper = sp
        name_to_spectra = filtered_spectra

        # Compose standard inkset DataFrame
        # Use paper if present; ensure wavelengths alignment
        names = list(name_to_spectra.keys())
        wavelengths = None
        rows = []
        for nm, sp in name_to_spectra.items():
            if wavelengths is None:
                wavelengths = sp.wavelengths
            elif not np.array_equal(wavelengths, sp.wavelengths):
                sp = sp.interpolate_values(wavelengths)
            rows.append((nm, sp.data))
        if paper is None:
            raise ValueError("No paper spectra found in input Nix CSV")

        output_dir = f"data/inksets/{out_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{out_name}-inks.csv")

        # Build DataFrame (ink rows + paper)
        df = pd.DataFrame([r[1] for r in rows] + [paper.data],
                          index=[r[0] for r in rows] + ["paper"],
                          columns=wavelengths)
        df.index.name = "Name"
        df = df.reset_index()
        df.insert(0, 'ID', range(1, len(df) + 1))
        df['clogged'] = 0
        df.to_csv(output_path, index=False)

        registry.register_library(
            out_name,
            f"{out_name}/{out_name}-inks.csv",
            {
                "created": "registered_primaries",
                "source_path": input_path,
                "num_rows": len(rows),
                "filtered_to_255_only": True,
            }
        )
        print(f"Registered {len(rows)} 255-level primaries as library '{out_name}' at {output_path}")
        return
    except Exception:
        # Fallback: assume it's already a standard inkset CSV → filter and register
        pass

    # Validate and register a standard-format inkset CSV
    try:
        inks, paper, wavelengths = load_inkset_csv(input_path, filter_clogged=True)
    except Exception as e:
        raise ValueError(f"Failed to load CSV '{input_path}': {e}")

    # Filter to keep only 255-level single-channel primaries
    filtered_inks = {}
    for name, sp in inks.items():
        m = PRIMARY_RE.match(name.strip())
        if m and m.group(2) == "255":  # Only 255-level primaries
            filtered_inks[name] = sp

    if not filtered_inks:
        raise ValueError("No 255-level primaries found in input CSV")

    output_dir = f"data/inksets/{out_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{out_name}-inks.csv")
    # Copy/normalize into standard format
    df = pd.DataFrame([sp.data for sp in filtered_inks.values()] + [paper.data],
                      index=list(filtered_inks.keys()) + ["paper"],
                      columns=wavelengths)
    df.index.name = "Name"
    df = df.reset_index()
    df.insert(0, 'ID', range(1, len(df) + 1))
    df['clogged'] = 0
    df.to_csv(output_path, index=False)

    registry.register_library(
        out_name,
        f"{out_name}/{out_name}-inks.csv",
        {
            "created": "registered_primaries",
            "source_path": input_path,
            "num_rows": len(filtered_inks),
            "filtered_to_255_only": True,
        }
    )
    print(f"Registered {len(filtered_inks)} 255-level primaries as library '{out_name}' at {output_path}")


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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
