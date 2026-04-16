"""
Analyze anomaloscope match data to estimate L and M cone parameters.

Fits the Neitz nomogram model to Rayleigh match data by searching over:
  - L cone peak wavelength (550-570 nm)
  - M cone peak wavelength (520-545 nm)
  - Photopigment optical density (0.2-0.6)

Loads measured LED spectra from the measurements/ directory (via
load_primaries_from_csv format).  The 3-primary anomaloscope uses
R, G, O channels (indices 0, 1, 3 from the RGBO primary list).

A Rayleigh match occurs when the observer's L and M cone responses to the
match field (R+G mixture) equal those to the reference field (O + fixed R/G).

Since S cones contribute negligibly at these wavelengths, the match is
determined entirely by L and M cone spectral sensitivities.

Usage:
  python analyze_matches.py
  python analyze_matches.py --primaries-dir ../../measurements/2026-02-11/primaries
  python analyze_matches.py --data-dir data/2026-03-02_17-43-25
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Self-contained Neitz nomogram + pre-receptoral filtering
# (avoids importing TetriumColor which requires open3d)
# ---------------------------------------------------------------------------

WL = np.arange(380, 781, 1).astype(np.float32)


def _neitz_nomogram(wls, lambda_max):
    """Neitz nomogram (Carroll, McMahon, Neitz & Neitz 2000).
    Returns quantal sensitivity as a 1-D array."""
    wls = wls.astype(np.float32)
    A, B, C = 0.417050601, 0.002072146, 0.000163888
    D, E = -1.922880605, -16.05774461
    F, G = 0.001575426, 5.11376e-05
    H, I = 0.00157981, 6.58428e-05
    J, K = 6.68402e-05, 0.002310442
    L, M = 7.31313e-05, 1.86269e-05
    N, O = 0.002008124, 5.40717e-05
    P, Q = 5.14736e-06, 0.001455413
    R, S = 4.217640000e-05, 4.800000000e-06
    T, U = 0.001809022, 3.86677000e-05
    V, W = 2.99000000e-05, 0.001757315
    X, Y = 1.47344000e-05, 1.51000000e-05

    A2 = np.log10(1.0 / lambda_max) - np.log10(1.0 / 558.5)
    vector = np.log10(np.reciprocal(wls))
    const = 1 / np.sqrt(2 * np.pi)

    ex1 = np.log10(-E + E * np.tanh(-((10 ** (vector - A2)) - F) / G)) + D
    ex2 = A * np.tanh(-(((10 ** (vector - A2))) - B) / C)
    ex3 = -(J / I * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - H) / I) ** 2)))
    ex4 = -(M / L * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - K) / L) ** 2)))
    ex5 = -(P / O * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - N) / O) ** 2)))
    ex6 = (S / R * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - Q) / R) ** 2)))
    ex7 = (V / U * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - T) / U) ** 2))) / 10
    ex8 = (Y / X * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - W) / X) ** 2))) / 100
    ex = ex1 + ex2 + ex3 + ex4 + ex5 + ex6 + ex7 + ex8

    return np.clip(10.0 ** ex, 0, 1)


def _load_lens_absorption():
    """Load lens absorption data from TetriumColor assets."""
    import pandas as pd
    lens_path = os.path.join(_REPO_ROOT, "TetriumColor", "Assets", "Cones", "lensss_1.csv")
    df = pd.read_csv(lens_path, header=None)
    lens_wl = df.iloc[:, 0].values
    lens_data = df.iloc[:, 1].values
    return np.interp(WL, lens_wl, lens_data, left=0, right=0)


# Cache lens absorption (loaded once)
_LENS_ABSORPTION = None


def _get_lens_absorption():
    global _LENS_ABSORPTION
    if _LENS_ABSORPTION is None:
        _LENS_ABSORPTION = _load_lens_absorption()
    return _LENS_ABSORPTION


def make_cone_sensitivity(peak, od):
    """Build a cone sensitivity curve using Neitz nomogram + OD + lens filtering.

    No macular pigment (user specified none).
    Lens density fixed at 1.0.
    """
    # Raw quantal sensitivity
    raw_quantal = _neitz_nomogram(WL, peak)

    # Apply optical density (self-screening)
    od_data = (1 - np.exp(np.log(10) * -od * raw_quantal)) / (1 - 10 ** (-od))

    # Apply lens absorption (macular = 0, so only lens contributes)
    lens = _get_lens_absorption()
    denom = 10 ** lens  # lens * 1.0 + macular * 0.0 = lens
    filtered_quantal = od_data / denom

    # Normalize
    filtered_quantal = filtered_quantal / np.max(filtered_quantal)

    # Convert quantal to energy: energy = quantal * wavelength (then normalize)
    log_data = np.log(WL.astype(np.float64)) + np.log(np.clip(filtered_quantal, 1e-30, None))
    energy_data = np.exp(log_data - np.max(log_data))

    return energy_data


# ---------------------------------------------------------------------------
# Display primary spectra — loaded from measured CSV files
# ---------------------------------------------------------------------------

def _find_latest_primaries_dir():
    """Auto-detect the most recent measurements/YYYY-MM-DD/primaries/ directory."""
    meas_root = os.path.join(_REPO_ROOT, "measurements")
    if not os.path.isdir(meas_root):
        return None
    dates = sorted([d for d in os.listdir(meas_root)
                    if os.path.isdir(os.path.join(meas_root, d, "primaries"))],
                   reverse=True)
    if dates:
        return os.path.join(meas_root, dates[0], "primaries")
    return None


def _load_primary_csv(filepath):
    """Load a single primary CSV (wavelength, power) and interpolate to 1nm."""
    data = np.loadtxt(filepath, delimiter=',')
    raw_wl = data[:, 0]
    raw_power = data[:, 1]
    # Interpolate to 1nm grid matching WL
    f = interp1d(raw_wl, raw_power, kind='cubic', bounds_error=False, fill_value=0.0)
    return f(WL.astype(np.float64))


def load_display_primaries(primaries_dir):
    """Load R, G, O LED spectra from a primaries directory.

    Looks for files matching the RGBO naming convention:
      255_0_0_0.csv  (Red)
      0_255_0_0.csv  (Green)
      0_0_0_255.csv  (Orange — the 4th primary)

    Falls back to timestamped filenames (r255g0b0o0_*.csv) if the simple
    names aren't found.

    Returns: (3, N_wl) array with rows [R, G, O].
    """
    # Map: (simple_name, pattern_prefix) for R, G, O
    targets = [
        ("255_0_0_0.csv", "r255g0b0o0_"),    # Red
        ("0_255_0_0.csv", "r0g255b0o0_"),     # Green
        ("0_0_0_255.csv", "r0g0b0o255_"),     # Orange (4th channel)
    ]
    spectra = []
    for simple, prefix in targets:
        simple_path = os.path.join(primaries_dir, simple)
        if os.path.exists(simple_path):
            spectra.append(_load_primary_csv(simple_path))
            continue
        # Fall back to timestamped files — pick the latest
        candidates = sorted(glob.glob(os.path.join(primaries_dir, prefix + "*.csv")))
        if candidates:
            spectra.append(_load_primary_csv(candidates[-1]))
        else:
            raise FileNotFoundError(
                f"No primary file found for pattern '{simple}' or '{prefix}*' "
                f"in {primaries_dir}")

    return np.array(spectra)  # (3, N_wl)


# Module-level primary spectra — set by main() after parsing args
PRIMARY_SPECTRA = None  # (3, N_wl) array: [R, G, O]


# ---------------------------------------------------------------------------
# Observer model
# ---------------------------------------------------------------------------

def cone_responses(l_peak, m_peak, l_od, m_od):
    """Return (2, 3) matrix: rows=[M, L], cols=[R, G, O] cone excitations."""
    l_sens = make_cone_sensitivity(l_peak, l_od)
    m_sens = make_cone_sensitivity(m_peak, m_od)
    sensor_mat = np.array([m_sens, l_sens])  # (2, N_wl)
    responses = sensor_mat @ PRIMARY_SPECTRA.T  # (2, 3)
    return responses


# ---------------------------------------------------------------------------
# Rayleigh match prediction
# ---------------------------------------------------------------------------

def predict_match_ratio(l_peak, m_peak, l_od, m_od, ref_rgb):
    """Predict the R/(R+G) intensity ratio an observer would set to match the reference.

    Solves: resp[:, :2] @ [r, g]^T = resp @ ref
    where resp is the (2,3) cone response matrix.
    Returns the ratio r/(r+g) in primary-intensity space.
    """
    resp = cone_responses(l_peak, m_peak, l_od, m_od)
    ref = np.array(ref_rgb, dtype=float) / 255.0
    ref_excitation = resp @ ref

    A = resp[:, :2]  # R and G columns only
    try:
        match_rg = np.linalg.solve(A, ref_excitation)
    except np.linalg.LinAlgError:
        return None

    r, g = match_rg
    if r < -0.01 or g < -0.01:  # allow tiny negative for numerical noise
        return None
    r, g = max(r, 0), max(g, 0)
    total = r + g
    if total < 1e-12:
        return None
    return r / total


def observed_match_ratio(match_R, match_G):
    """Compute observed R/(R+G) ratio from the 8-bit match values.

    The app applies luminance compensation, so the recorded R and G values
    already reflect the iso-luminance constraint. The ratio R/(R+G) in
    primary-intensity space is what we compare against the model prediction.
    """
    total = match_R + match_G
    if total < 1:
        return 0.5
    return match_R / total


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def match_cost(params, matches):
    """Sum of squared errors between predicted and observed R/(R+G) ratios."""
    l_peak, m_peak, l_od, m_od = params

    # Bounds check
    if not (545 <= l_peak <= 580 and 515 <= m_peak <= 555):
        return 1e6
    if not (0.1 <= l_od <= 0.8 and 0.1 <= m_od <= 0.8):
        return 1e6
    if m_peak >= l_peak:
        return 1e6

    total_err = 0.0
    n = 0
    for m in matches:
        ref_rgb = (m['ref_R'], m['ref_G'], m['ref_O'])
        pred_ratio = predict_match_ratio(l_peak, m_peak, l_od, m_od, ref_rgb)
        if pred_ratio is None:
            total_err += 1.0  # max penalty (ratio is 0-1)
            n += 1
            continue

        obs_ratio = observed_match_ratio(m['match_R'], m['match_G'])
        total_err += (pred_ratio - obs_ratio) ** 2
        n += 1

    return total_err / max(n, 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matches(data_dirs):
    """Load all matches.csv files from the given directories."""
    matches = []
    for d in data_dirs:
        csv_path = os.path.join(d, "matches.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                matches.append({
                    'timestamp': row['timestamp'],
                    'mode': row['mode'],
                    'bit_depth': int(row['bit_depth']),
                    'angle_deg': int(row['angle_deg']),
                    'rg_ratio': float(row['rg_ratio']),
                    'match_R': int(row['match_R']),
                    'match_G': int(row['match_G']),
                    'ref_R': int(row['ref_R']),
                    'ref_G': int(row['ref_G']),
                    'ref_O': int(row['ref_O']),
                    'ref_lum': float(row['ref_lum']),
                    'orange_chrom': float(row['orange_chrom']),
                })
    return matches


def find_data_dirs(base_dir):
    """Find all timestamped data directories."""
    dirs = sorted(glob.glob(os.path.join(base_dir, "20*")))
    return [d for d in dirs if os.path.isdir(d)]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_observer(matches):
    """Fit L peak, M peak, L OD, M OD to the match data (4 parameters)."""
    # Count independent conditions
    conditions = set()
    for m in matches:
        conditions.add((m['angle_deg'], m['ref_R'], m['ref_G'], m['ref_O']))
    n_conditions = len(conditions)
    print(f"  {len(matches)} matches across {n_conditions} unique condition(s)")
    if n_conditions < 4:
        print(f"  WARNING: {n_conditions} conditions for 4 parameters -- "
              f"underdetermined. Collect matches at {4 - n_conditions} more "
              f"distinct reference settings.")

    # Grid search: L_peak x M_peak x L_OD x M_OD
    l_peaks = np.arange(545, 575, 2.0)
    m_peaks = np.arange(520, 550, 2.0)
    ods = np.arange(0.2, 0.71, 0.1)

    best_cost = np.inf
    best_params = (559, 530, 0.5, 0.5)

    print("Grid search over L_peak x M_peak x L_OD x M_OD ...")
    for lp in l_peaks:
        for mp in m_peaks:
            if mp >= lp:
                continue
            for l_od in ods:
                for m_od in ods:
                    c = match_cost((lp, mp, l_od, m_od), matches)
                    if c < best_cost:
                        best_cost = c
                        best_params = (lp, mp, l_od, m_od)

    print(f"  Grid best: L={best_params[0]:.0f}nm  M={best_params[1]:.0f}nm  "
          f"L_OD={best_params[2]:.2f}  M_OD={best_params[3]:.2f}  cost={best_cost:.4f}")

    # Refine with Nelder-Mead
    result = minimize(
        match_cost, best_params, args=(matches,),
        method='Nelder-Mead',
        options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 5000},
    )
    l_peak, m_peak, l_od, m_od = result.x
    final_cost = result.fun
    print(f"  Refined:   L={l_peak:.1f}nm  M={m_peak:.1f}nm  "
          f"L_OD={l_od:.3f}  M_OD={m_od:.3f}  cost={final_cost:.6f}")

    return l_peak, m_peak, l_od, m_od, final_cost


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyze(matches, l_peak, m_peak, l_od, m_od):
    """Print detailed analysis."""
    print("\n" + "=" * 70)
    print("  FITTED OBSERVER MODEL")
    print("=" * 70)
    print(f"  L cone peak:          {l_peak:.1f} nm")
    print(f"  M cone peak:          {m_peak:.1f} nm")
    print(f"  L photopigment OD:    {l_od:.3f}")
    print(f"  M photopigment OD:    {m_od:.3f}")
    print(f"  Lens density:         1.0 (fixed)")
    print(f"  Macular pigment:      0.0 (none)")
    print(f"  Template:             Neitz (Carroll et al. 2000)")
    print("=" * 70)

    # Group by (angle, ref) condition
    by_condition = {}
    for m in matches:
        key = (m['angle_deg'], m['ref_R'], m['ref_G'], m['ref_O'])
        by_condition.setdefault(key, []).append(m)

    print(f"\n{'Angle':>6}  {'N':>3}  {'Ref':>15}  "
          f"{'Obs ratio':>12}  {'Pred ratio':>12}  {'Error':>8}")
    print("-" * 70)

    all_errors = []

    for key in sorted(by_condition.keys()):
        angle, rr, rg, ro = key
        group = by_condition[key]

        ref_rgb = (rr, rg, ro)
        pred_ratio = predict_match_ratio(l_peak, m_peak, l_od, m_od, ref_rgb)

        obs_ratios = [observed_match_ratio(m['match_R'], m['match_G']) for m in group]
        mean_ratio = np.mean(obs_ratios)
        std_ratio = np.std(obs_ratios) if len(obs_ratios) > 1 else 0

        err = pred_ratio - mean_ratio if pred_ratio is not None else float('nan')
        all_errors.append(err)

        obs_str = f"{mean_ratio:.4f}" + (f"+/-{std_ratio:.4f}" if std_ratio > 0 else "")
        ref_str = f"({rr},{rg},{ro})"
        pred_str = f"{pred_ratio:.4f}" if pred_ratio is not None else "N/A"

        print(f"{angle:>4}deg  {len(group):>3}  {ref_str:>15}  "
              f"{obs_str:>12}  {pred_str:>12}  {err:>+8.4f}")

    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))
    print("-" * 70)
    print(f"{'RMSE':>50}  {'':>12}  {rmse:>8.4f}")

    # Match variability
    print("\n" + "=" * 70)
    print("  MATCH VARIABILITY (within-condition std dev)")
    print("=" * 70)
    for key in sorted(by_condition.keys()):
        angle, rr, rg, ro = key
        group = by_condition[key]
        rs = [m['match_R'] for m in group]
        gs = [m['match_G'] for m in group]
        ratios = [m['rg_ratio'] for m in group]
        if len(group) < 2:
            print(f"  {angle}deg ref=({rr},{rg},{ro}): n=1 (no std dev)")
        else:
            print(f"  {angle}deg ref=({rr},{rg},{ro}): n={len(group)}  "
                  f"R={np.mean(rs):.1f}+/-{np.std(rs):.1f}  "
                  f"G={np.mean(gs):.1f}+/-{np.std(gs):.1f}  "
                  f"ratio={np.mean(ratios):.4f}+/-{np.std(ratios):.4f}")

    # Compare to standard observers
    print("\n" + "=" * 70)
    print("  COMPARISON TO STANDARD OBSERVERS")
    print("=" * 70)
    # (label, l_peak, m_peak, l_od, m_od)
    standards = [
        ("Standard (559/530)",       559, 530, 0.50, 0.50),
        ("Mild protan (553/530)",    553, 530, 0.50, 0.50),
        ("Mild deutan (559/536)",    559, 536, 0.50, 0.50),
        ("High OD (559/530/0.6)",    559, 530, 0.60, 0.60),
        ("Low OD (559/530/0.3)",     559, 530, 0.30, 0.30),
        ("Fitted model",             l_peak, m_peak, l_od, m_od),
    ]
    first_m = matches[0]
    ref0 = (first_m['ref_R'], first_m['ref_G'], first_m['ref_O'])
    obs_ratio0 = observed_match_ratio(first_m['match_R'], first_m['match_G'])
    print(f"  {'Model':30s}  {'Cost':>10}  {'Pred ratio':>12}")
    print("  " + "-" * 57)
    for label, lp, mp, lo, mo in standards:
        cost = match_cost((lp, mp, lo, mo), matches)
        pred = predict_match_ratio(lp, mp, lo, mo, ref0)
        pred_str = f"{pred:.4f}" if pred is not None else "N/A"
        print(f"  {label:30s}  {cost:>10.6f}  {pred_str:>12}")

    print(f"\n  Observed first match ratio: {obs_ratio0:.4f}  "
          f"(R={first_m['match_R']} G={first_m['match_G']}, ref {ref0})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PRIMARY_SPECTRA

    parser = argparse.ArgumentParser(
        description="Analyze anomaloscope matches to estimate L/M cone parameters"
    )
    parser.add_argument(
        "--data-dir", nargs="*",
        help="Specific data directories. Default: all in scripts/anomaloscope/data/.",
    )
    parser.add_argument(
        "--primaries-dir",
        help="Path to measured primaries directory (contains 255_0_0_0.csv etc). "
             "Default: auto-detect from measurements/.",
    )
    args = parser.parse_args()

    # ── Load display primaries ──
    primaries_dir = args.primaries_dir or _find_latest_primaries_dir()
    if primaries_dir is None:
        print("ERROR: No primaries directory found. Pass --primaries-dir or "
              "ensure measurements/YYYY-MM-DD/primaries/ exists.")
        return
    print(f"Loading LED spectra from: {primaries_dir}")
    PRIMARY_SPECTRA = load_display_primaries(primaries_dir)
    print(f"  R peak: {WL[np.argmax(PRIMARY_SPECTRA[0])]}nm")
    print(f"  G peak: {WL[np.argmax(PRIMARY_SPECTRA[1])]}nm")
    print(f"  O peak: {WL[np.argmax(PRIMARY_SPECTRA[2])]}nm")

    # ── Load match data ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, "data")

    if args.data_dir:
        data_dirs = args.data_dir
    else:
        data_dirs = find_data_dirs(base_data_dir)

    if not data_dirs:
        print("No data directories found.")
        return

    print(f"\nLoading matches from {len(data_dirs)} session(s)...")
    matches = load_matches(data_dirs)

    # Filter to bipartite mode only (the actual Rayleigh matches)
    matches = [m for m in matches if m['mode'] == 'bipartite']
    if not matches:
        print("No bipartite match data found.")
        return

    print(f"Found {len(matches)} bipartite match(es)")

    # Show raw data
    print("\nRaw match data:")
    print(f"  {'Angle':>5}  {'R/G ratio':>9}  {'Match R':>7}  {'Match G':>7}  "
          f"{'Ref R':>5}  {'Ref G':>5}  {'Ref O':>5}")
    for m in matches:
        print(f"  {m['angle_deg']:>4}d  {m['rg_ratio']:>9.4f}  {m['match_R']:>7d}  "
              f"{m['match_G']:>7d}  {m['ref_R']:>5d}  {m['ref_G']:>5d}  {m['ref_O']:>5d}")

    # Fit the model
    print()
    l_peak, m_peak, l_od, m_od, cost = fit_observer(matches)

    # Detailed analysis
    analyze(matches, l_peak, m_peak, l_od, m_od)


if __name__ == "__main__":
    main()
