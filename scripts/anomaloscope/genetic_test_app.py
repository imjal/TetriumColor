"""
Genetic test plate viewer for RGO 3-primary display.

For each observer genotype in the population (from ObserverGenotypes), an
Ishihara-style plate is produced whose figure and background are ML-metamers
*for that specific genotype* — i.e. the colours stimulate M and L identically
for an observer with those exact cone peaks.  Observers with different M/L
peaks will perceive the two colours as distinct and can therefore read the
hidden figure (Landolt C).

Genotypes are enumerated using dimension=4 (tetrachromat variants) for finer
M/Q/L peak granularity.  The S cone is discarded; only M and L rows are used
to find the null direction (the Q row provides the diagnostic dQ readout).

With 3 primaries (R, G, O) and 2 ML constraints, there is exactly 1 degree
of freedom: the Q-cone separation.  The "Q value" slider controls how far
along the ML null direction the inside/outside colours are displaced.

The output image uses the HDMI channel convention:
  R -> Red LED   |   G -> Green LED   |   O(B) -> Orange LED

6-bit emulation:
  Values are rounded to the nearest of 64 levels per channel (0-255 -> 4-step grid).

Usage
-----
  python genetic_test_app.py [--primaries-dir PATH] [--coverage FLOAT]
                              [--sex {male,female,both}]
                              [--viewing-distance CM]

Controls
--------
  <- / ->         : prev / next genotype plate
  up / down       : adjust Q value (metamer separation)
  Control [C]     : show uniform plate (Q=0, figure invisible)
  Save [S]        : save current plate as PNG
  B               : cycle 8/6/4-bit mode
  L / Shift+L     : increase / decrease luminance noise
  A               : anomaloscope prediction mode (row of 5 plates)
  H               : toggle controls visibility
  Escape          : exit full-screen
  Sliders         : Q value, dot size, lum noise, visual angle
"""

from TetriumColor.Measurement.TetriumMeasurementRoutines import load_primaries_from_csv
from TetriumColor.PsychoPhys.IshiharaPlate import generate_ishihara_plate
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import Observer, Spectra, Cone
import argparse
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.linalg import null_space
from PIL import Image, ImageDraw, ImageFont, ImageTk

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Measurements auto-detection
# ---------------------------------------------------------------------------

def find_latest_primaries_dir() -> str | None:
    """
    Search for the most recent  measurements/<date>/primaries  directory
    relative to the repository root and return its path, or None if not found.
    """
    meas_root = os.path.join(_REPO_ROOT, "measurements")
    if not os.path.isdir(meas_root):
        return None

    candidates = []
    for entry in os.scandir(meas_root):
        if not entry.is_dir():
            continue
        prim_path = os.path.join(entry.path, "primaries")
        if os.path.isdir(prim_path):
            candidates.append((entry.name, prim_path))

    if not candidates:
        return None

    # Sort by date string (ISO-like names sort lexicographically)
    candidates.sort(key=lambda x: x[0], reverse=True)
    date, path = candidates[0]
    print(f"Auto-detected primaries: measurements/{date}/primaries/")
    return path


# ---------------------------------------------------------------------------
# Bit-depth helpers
# ---------------------------------------------------------------------------

BIT_MODES = [8, 6, 4]   # cycle order


def quantize_to_n_bit_arr(arr_uint8: np.ndarray, bits: int) -> np.ndarray:
    """Quantize an 8-bit image array to n-bit levels."""
    if bits >= 8:
        return arr_uint8
    levels = (1 << bits) - 1
    arr_f = arr_uint8.astype(np.float32)
    arr_q = np.round(arr_f * levels / 255.0) * 255.0 / levels
    return np.clip(arr_q, 0, 255).astype(np.uint8)


def quantize_image(img: Image.Image, bits: int) -> Image.Image:
    """Quantize a PIL image to n-bit levels."""
    if bits >= 8:
        return img
    return Image.fromarray(quantize_to_n_bit_arr(np.array(img), bits))


# ---------------------------------------------------------------------------
# Landolt C directions (replacing hidden numbers)
# ---------------------------------------------------------------------------

LANDOLT_DIRECTIONS = [
    'landolt_up', 'landolt_down', 'landolt_left', 'landolt_right',
    'landolt_up-right', 'landolt_down-right', 'landolt_up-left', 'landolt_down-left',
]

LANDOLT_LABELS = {
    'landolt_up': 'Up',
    'landolt_down': 'Down',
    'landolt_left': 'Left',
    'landolt_right': 'Right',
    'landolt_up-right': 'Up-Right',
    'landolt_down-right': 'Down-Right',
    'landolt_up-left': 'Up-Left',
    'landolt_down-left': 'Down-Left',
}


def make_synthetic_rgo_primaries(wavelengths: np.ndarray) -> list:
    """Synthetic Gaussian LED primaries: R~660 nm, G~540 nm, O~598 nm."""
    peaks, sigma = [660, 540, 598], 15.0
    result = []
    for pk in peaks:
        data = np.exp(-((wavelengths - pk) ** 2) / (2 * sigma ** 2))
        result.append(Spectra(wavelengths=wavelengths, data=data / data.max()))
    return result   # [R_spectra, G_spectra, O_spectra]


# ---------------------------------------------------------------------------
# Visual angle helpers
# ---------------------------------------------------------------------------

def deg_to_px(degrees: float, viewing_dist_cm: float, dpi: float) -> int:
    """Convert visual angle in degrees to pixels given viewing distance and screen DPI."""
    cm = 2 * viewing_dist_cm * np.tan(np.radians(degrees / 2))
    return int(cm * dpi / 2.54)


# ---------------------------------------------------------------------------
# Per-genotype ML null-direction (dimension=4 aware)
# ---------------------------------------------------------------------------

def ml_null_direction_for_genotype(
    genotype_peaks: tuple,
    wavelengths: np.ndarray,
    primaries_rgo: list,
) -> dict:
    """
    Compute the ML null-space direction and gamut midpoint for a genotype.

    With 3 primaries (R, G, O) and 2 ML constraints, the set of ML-metamers
    in display space is a 1-D line: {center + t * null_dir : t in [-t_max, +t_max]}.

    The center is the midpoint of the segment of this line that lies within [0,1]^3.
    The Q slider then controls t, with t_max chosen so both endpoints stay in gamut.

    Returns a dict with keys:
        null_dir : ndarray (3,) unit vector in (R,G,O) display space
        center   : ndarray (3,) midpoint of the gamut-clipped null line
        t_max    : float, max symmetric extent from center
        delta_q_per_t : float or None, Q-cone response change per unit t
        q_peak   : float or None
        ml_mat   : ndarray (2,3) the ML response matrix (for verification)
    """
    peaks_sorted = sorted(genotype_peaks)

    q_peak_val = None
    q_cone_obj = None

    if len(peaks_sorted) >= 4:
        # dimension=4 genotype: (S, M, Q, L) -- discard S (shortest)
        m_peak = peaks_sorted[-3]
        q_peak_val = peaks_sorted[-2]
        l_peak = peaks_sorted[-1]
        q_cone_obj = Cone.cone(q_peak_val, wavelengths=wavelengths, template="neitz", od=0.5)
    elif len(peaks_sorted) >= 2:
        m_peak = peaks_sorted[-2]
        l_peak = peaks_sorted[-1]
    else:
        m_peak = peaks_sorted[0]
        l_peak = peaks_sorted[0]

    # Build ML observer and compute response matrix
    m_cone = Cone.cone(m_peak, wavelengths=wavelengths, template="neitz", od=0.5)
    l_cone = Cone.cone(l_peak, wavelengths=wavelengths, template="neitz", od=0.5)
    ml_obs = Observer([m_cone, l_cone])

    # ML response matrix: each column is a primary's [M, L] response
    # observe_spectras returns (n_primaries, n_cones) = (3, 2)
    resp = ml_obs.observe_spectras(primaries_rgo)   # (3, 2)
    ML_mat = resp.T                                  # (2, 3)

    # Null space of ML_mat: directions in (R,G,O) that produce zero ML change
    ns = null_space(ML_mat)                          # (3, 1)
    if ns.shape[1] == 0:
        raise ValueError(f"No null space for genotype {genotype_peaks} -- degenerate system")
    null_dir = ns[:, 0]
    null_dir /= np.linalg.norm(null_dir)

    # Find the gamut segment of the null line.
    # The null line in display space is {p : ML_mat @ p = ML_mat @ p0} for any
    # starting point p0. We parameterize as p(t) = p0 + t * null_dir.
    # We need to find the segment within [0, 1]^3.

    # Pick a particular solution p0 on the null line that is inside [0,1]^3.
    # Use least-norm solution targeting center of gamut ML response.
    gamut_center = np.array([0.5, 0.5, 0.5])
    ml_target = ML_mat @ gamut_center   # target ML response = that of gamut center
    # Particular solution: ML_mat^+ @ ml_target (least-norm point with that ML response)
    p0 = np.linalg.lstsq(ML_mat, ml_target, rcond=None)[0]

    # Find the range of t such that p0 + t * null_dir stays in [0, 1]^3
    t_lo = -np.inf
    t_hi = np.inf
    for i in range(3):
        if abs(null_dir[i]) > 1e-12:
            # p0[i] + t * null_dir[i] >= 0  =>  t >= -p0[i] / null_dir[i]  (if dir > 0)
            t_for_0 = -p0[i] / null_dir[i]
            t_for_1 = (1.0 - p0[i]) / null_dir[i]
            t_lo = max(t_lo, min(t_for_0, t_for_1))
            t_hi = min(t_hi, max(t_for_0, t_for_1))

    if t_lo >= t_hi:
        # Degenerate: just use p0 as center with no extent
        center = np.clip(p0, 0, 1)
        t_max = 0.0
    else:
        # Center = midpoint of the valid segment
        t_mid = (t_lo + t_hi) / 2.0
        center = p0 + t_mid * null_dir
        center = np.clip(center, 0, 1)
        t_max = (t_hi - t_lo) / 2.0

    # Compute Q-cone response per unit t (if Q cone exists)
    delta_q_per_t = None
    if q_cone_obj is not None:
        q_obs = Observer([q_cone_obj])
        q_resp = q_obs.observe_spectras(primaries_rgo)  # (3, 1)
        q_row = q_resp.T[0]  # (3,)
        delta_q_per_t = float(q_row @ null_dir)

    # Verify: ML_mat @ null_dir should be ~0
    ml_check = ML_mat @ null_dir
    if np.linalg.norm(ml_check) > 1e-6:
        print(f"  WARNING: ML_mat @ null_dir = {ml_check} (should be ~0)")

    # Verify: center should be in [0,1]^3
    if np.any(center < -0.01) or np.any(center > 1.01):
        print(f"  WARNING: center {center} outside gamut")

    return {
        "null_dir": null_dir,
        "center": center,
        "t_max": t_max,
        "delta_q_per_t": delta_q_per_t,
        "q_peak": q_peak_val,
        "ml_mat": ML_mat,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_plate_colors(
    null_dir: np.ndarray,
    center: np.ndarray,
    t_max: float,
    q_value: float,
) -> tuple:
    """
    Compute inside / outside DISP colours for an Ishihara plate.

    q_value in [0, 1] controls how far along the null direction we go:
      inside  = center + q_value * t_max * null_dir
      outside = center - q_value * t_max * null_dir

    Both endpoints are guaranteed to stay in [0,1]^3 because center is the
    midpoint of the gamut-clipped null line and t_max is the half-extent.

    Returns
    -------
    inside_disp, outside_disp : ndarray (3,)  in [0, 1]
    """
    t = q_value * t_max
    color_a = np.clip(center + t * null_dir, 0, 1)
    color_b = np.clip(center - t * null_dir, 0, 1)
    # Assign the color with higher O (index 2) as the foreground (inside),
    # and the lower O as the background (outside).
    if color_a[2] >= color_b[2]:
        return color_a, color_b   # inside=figure (higher O), outside=bg (lower O)
    else:
        return color_b, color_a   # inside=figure (higher O), outside=bg (lower O)


# ---------------------------------------------------------------------------
# Genotype catalogue
# ---------------------------------------------------------------------------

def build_genotype_catalogue(
    wavelengths: np.ndarray,
    primaries_rgo: list,
    coverage: float = 0.90,
    sex: str = "male",
) -> list:
    """
    Build catalogue of genotypes with their null directions and gamut midpoints.

    Uses ObserverGenotypes with dimension=4 for tetrachromat genotype variants,
    providing finer M/Q/L peak granularity.
    """
    print(f"Building genotype catalogue (coverage={coverage*100:.0f}%, sex={sex})...")
    og = ObserverGenotypes(wavelengths=wavelengths, dimensions=[3])
    genotypes = og.get_genotypes_covering_probability(coverage, sex=sex)
    probs = og.get_pdf(sex)

    catalogue = []
    for gt in genotypes:
        prob = probs.get(gt, 0.0)
        try:
            info = ml_null_direction_for_genotype(gt, wavelengths, primaries_rgo)
        except ValueError as e:
            print(f"  Skipping genotype {gt}: {e}")
            continue

        peaks_sorted = sorted(gt)
        if len(peaks_sorted) >= 4:
            m_pk = peaks_sorted[-3]
            q_pk = peaks_sorted[-2]
            l_pk = peaks_sorted[-1]
            label = f"M={m_pk:.0f} Q={q_pk:.0f} L={l_pk:.0f}  ({prob*100:.1f}%)"
        elif len(peaks_sorted) >= 2:
            m_pk = peaks_sorted[-2]
            l_pk = peaks_sorted[-1]
            label = f"M={m_pk:.0f} L={l_pk:.0f}  ({prob*100:.1f}%)"
        else:
            label = f"peak={peaks_sorted[0]:.0f}  ({prob*100:.1f}%)"

        catalogue.append({
            "label":         label,
            "peaks":         gt,
            "null_dir":      info["null_dir"],
            "center":        info["center"],
            "t_max":         info["t_max"],
            "delta_q_per_t": info["delta_q_per_t"],
            "q_peak":        info["q_peak"],
            "ml_mat":        info["ml_mat"],
            "prob":          prob,
        })

        # Print verification info
        nd = info["null_dir"]
        c = info["center"]
        dq = info["delta_q_per_t"]
        print(f"  {label}  center={c.round(3)}  t_max={info['t_max']:.3f}  "
              f"dQ/t={dq:.4f}" if dq is not None else f"  {label}  center={c.round(3)}  t_max={info['t_max']:.3f}")

    # Sort by probability (most common genotype first)
    catalogue.sort(key=lambda x: x["prob"], reverse=True)
    print(f"  {len(catalogue)} genotypes loaded.")
    return catalogue


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GeneticTestApp:
    """
    Pseudo-isochromatic plate viewer cycling through population genotypes.

    For each genotype the plate colours are ML-metamers for *that* genotype.
    The only free parameter is the Q-cone separation (the "Q value" slider).
    """

    PLATE_PX = 1024

    # DLP display native resolution
    DLP_W = 912
    DLP_H = 1140

    # Physical display dimensions in cm (measured)
    PHYS_W_CM = 59.0
    PHYS_H_CM = 37.0

    BG_COLOR = "#1e1e1e"
    FG_COLOR = "#dddddd"
    VAL_COLOR = "#88ccff"
    HINT_COLOR = "#666666"
    GEN_COLOR = "#ffcc66"

    PROP_STEP = 0.02

    def __init__(self, root: tk.Tk,
                 primaries_dir: str | None = None,
                 coverage: float = 0.90,
                 sex: str = "both",
                 viewing_distance_cm: float = 149.0):
        self.root = root
        self.root.title("Genetic Test -- RGO 3-Primary Display")
        self.root.configure(bg=self.BG_COLOR)

        # ── state ──────────────────────────────────────────────────────────
        self.bit_depth = tk.IntVar(value=8)       # 8, 6, or 4
        self.q_value = tk.DoubleVar(value=0.5)
        self.dot_size_scale = tk.DoubleVar(value=1.0)
        self.lum_noise = tk.DoubleVar(value=0.015)
        self.visual_angle = tk.DoubleVar(value=4.0)
        self._viewing_dist_cm = viewing_distance_cm
        self._gt_idx = 0
        self._plate_cache: Image.Image | None = None
        self._photo = None
        self._generating = False
        self._pending_request = False
        self._controls_visible = True
        self._plate_info = {}   # metadata drawn onto the frame

        # ── spectral setup ─────────────────────────────────────────────────
        self.wavelengths = np.arange(380, 781, 1)

        # ── load RGO primaries: real (preferred) or synthetic fallback ─────
        pdir = primaries_dir or find_latest_primaries_dir()
        if pdir is not None:
            try:
                primaries_4 = load_primaries_from_csv(pdir, primary_order="RGBO")
                self.primaries = [primaries_4[0], primaries_4[1], primaries_4[3]]
                print(f"Loaded RGO primaries from: {pdir}")
            except Exception as exc:
                print(f"Could not load primaries from {pdir} ({exc}). Using synthetic.")
                self.primaries = make_synthetic_rgo_primaries(self.wavelengths)
        else:
            print("No measurements folder found. Using synthetic Gaussian primaries "
                  "(R=660, G=540, O=598 nm).")
            self.primaries = make_synthetic_rgo_primaries(self.wavelengths)

        # Rendering ColorSpace: standard trichromat + 3 RGO primaries
        self._render_observer = Observer.trichromat(wavelengths=self.wavelengths)
        self._render_cs = ColorSpace(self._render_observer,
                                     display_primaries=self.primaries)

        # Genotype catalogue (ML null directions for each genotype)
        self._catalogue = build_genotype_catalogue(
            self.wavelengths, self.primaries, coverage=coverage, sex=sex
        )
        if not self._catalogue:
            raise RuntimeError("No genotypes found -- check coverage/sex parameters.")

        # ── UI ────────────────────────────────────────────────────────────
        self._build_ui()
        self._bind_keys()
        self._request_plate()

        # Force keyboard focus so arrow keys work immediately
        self.root.focus_force()
        self.root.after(100, lambda: self.root.focus_force())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # canvas (centred, fills available space)
        self.canvas = tk.Canvas(
            self.root, width=self.DLP_W, height=self.DLP_H,
            bg="#333333", bd=0, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # status bar (for transient messages like "Generating..." / errors)
        self.status = tk.Label(
            self.root, text="Starting up...", font=("Courier", 8),
            bg=self.BG_COLOR, fg=self.VAL_COLOR, anchor="w", padx=10,
        )
        self.status.pack(fill=tk.X)

        # controls panel (toggleable with H)
        self.ctrl_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=10)
        self.ctrl_frame.pack(fill=tk.X, pady=4)

        # buttons
        btn_row = tk.Frame(self.ctrl_frame, bg=self.BG_COLOR)
        btn_row.pack(fill=tk.X, pady=4)
        for text, cmd in [
            ("< Prev", self._prev_genotype),
            ("> Next", self._next_genotype),
            ("Control [C]", self._control_plate),
            ("Save [S]",    self._save_plate),
        ]:
            tk.Button(btn_row, text=text, command=cmd, width=12,
                      bg="#aaaaaa", fg="black",
                      activebackground="#cccccc").pack(side=tk.LEFT, padx=3)

        self.bit_btn = tk.Button(
            btn_row, text="8-bit [B]", command=self._toggle_bit,
            width=10, bg="#aaaaaa", fg="black", activebackground="#cccccc",
            relief="flat",
        )
        self.bit_btn.pack(side=tk.LEFT, padx=3)

        # sliders -- only Q value is the free variable for plate colors
        self._make_slider(
            self.ctrl_frame, "Q value:", self.q_value, 0, 1,
            "0 = ML-metamer (invisible)   1 = max Q separation",
        )
        self._make_slider(
            self.ctrl_frame, "Dot size:", self.dot_size_scale, 0.5, 2.0,
            "Scale factor for dot radii  (smaller = more dots)",
        )
        self._make_slider(
            self.ctrl_frame, "Lum noise:", self.lum_noise, 0.0, 0.1,
            "Luminance noise added to plate circles",
        )
        self._make_slider(
            self.ctrl_frame, "Visual angle:", self.visual_angle, 1.0, 20.0,
            "Plate diameter in degrees of visual angle",
            fmt="{:.1f}deg",
        )

        # catalogue info
        n = len(self._catalogue)
        tk.Label(
            self.root,
            text=f"{n} genotypes loaded  |  HDMI: R->Red  G->Green  O->Orange",
            fg="#888888", font=("Helvetica", 9, "italic"), bg=self.BG_COLOR,
        ).pack(pady=(2, 8))

    def _make_slider(self, parent, label, var, from_, to, hint, fmt="{:.2f}"):
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, bg=self.BG_COLOR, fg=self.FG_COLOR,
                 width=15, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=from_, to=to, variable=var,
                  command=lambda _: self._request_plate()
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        val_lbl = tk.Label(row, text=fmt.format(var.get()), width=8,
                           bg=self.BG_COLOR, fg=self.VAL_COLOR,
                           font=("Courier", 10))
        val_lbl.pack(side=tk.LEFT)
        tk.Label(row, text=hint, bg=self.BG_COLOR, fg=self.HINT_COLOR,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=6)
        var.trace_add("write", lambda *_: val_lbl.config(text=fmt.format(var.get())))

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self):
        # Use bind_all so keys work regardless of which widget has focus
        self.root.bind_all("<Right>", lambda _: self._next_genotype())
        self.root.bind_all("<Left>", lambda _: self._prev_genotype())
        self.root.bind_all("<Up>", lambda _: self._step_q(+self.PROP_STEP))
        self.root.bind_all("<Down>", lambda _: self._step_q(-self.PROP_STEP))
        self.root.bind_all("<KeyPress-n>", lambda _: self._next_genotype())
        self.root.bind_all("<KeyPress-N>", lambda _: self._next_genotype())
        self.root.bind_all("<KeyPress-p>", lambda _: self._prev_genotype())
        self.root.bind_all("<KeyPress-P>", lambda _: self._prev_genotype())
        self.root.bind_all("<KeyPress-c>", lambda _: self._control_plate())
        self.root.bind_all("<KeyPress-C>", lambda _: self._control_plate())
        self.root.bind_all("<KeyPress-b>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-B>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-s>", lambda _: self._save_plate())
        self.root.bind_all("<KeyPress-S>", lambda _: self._save_plate())
        self.root.bind_all("<KeyPress-h>", lambda _: self._toggle_controls())
        self.root.bind_all("<KeyPress-H>", lambda _: self._toggle_controls())
        self.root.bind_all("<KeyPress-l>", lambda _: self._step_lum_noise(+0.005))
        self.root.bind_all("<KeyPress-L>", lambda _: self._step_lum_noise(-0.005))
        self.root.bind_all("<KeyPress-a>", lambda _: self._show_anomaloscope_predictions())
        self.root.bind_all("<KeyPress-A>", lambda _: self._show_anomaloscope_predictions())
        self.root.bind_all("<Escape>", lambda _: self.root.attributes("-fullscreen", False))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _next_genotype(self):
        self._gt_idx = (self._gt_idx + 1) % len(self._catalogue)
        self._request_plate()

    def _prev_genotype(self):
        self._gt_idx = (self._gt_idx - 1) % len(self._catalogue)
        self._request_plate()

    def _step_q(self, delta):
        self.q_value.set(np.clip(self.q_value.get() + delta, 0.0, 1.0))
        self._request_plate()

    def _step_lum_noise(self, delta):
        self.lum_noise.set(np.clip(self.lum_noise.get() + delta, 0.0, 0.1))
        self._request_plate()

    def _toggle_bit(self):
        cur = self.bit_depth.get()
        idx = (BIT_MODES.index(cur) + 1) % len(BIT_MODES)
        nxt = BIT_MODES[idx]
        self.bit_depth.set(nxt)
        colors = {8: ("#aaaaaa", "black"), 6: ("#cc8800", "black"), 4: ("#cc4400", "white")}
        bg, fg = colors[nxt]
        self.bit_btn.config(text=f"{nxt}-bit [B]", bg=bg, fg=fg)
        if self._plate_cache is not None:
            self._display_plate(self._plate_cache)

    def _toggle_controls(self):
        if self._controls_visible:
            self.ctrl_frame.pack_forget()
        else:
            self.ctrl_frame.pack(fill=tk.X, pady=4, after=self.status)
        self._controls_visible = not self._controls_visible

    def _control_plate(self):
        """Show a control plate at Q=0 (figure invisible)."""
        saved = self.q_value.get()
        self.q_value.set(0.0)
        self._generate_plate_sync()
        self.q_value.set(saved)

    def _save_plate(self):
        if self._plate_cache is None:
            return
        gt = self._catalogue[self._gt_idx]
        path = f"genetic_plate_gt{self._gt_idx}_{gt['peaks']}.png"
        img = quantize_image(self._plate_cache, self.bit_depth.get())
        img.save(path)
        print(f"Saved -> {path}")

    # ------------------------------------------------------------------
    # Anomaloscope prediction mode (A key)
    # ------------------------------------------------------------------

    # Anomaloscope primary range caps (must match anomaloscope_app.py)
    ANOM_R_MAX = 128   # red primary max (0-255)
    ANOM_G_MAX = 64    # green primary max (0-255)
    ANOM_O_MAX = 255   # orange primary max (0-255)

    def _solve_anomaloscope_match(self, ml_mat: np.ndarray) -> dict | None:
        """
        Solve for the anomaloscope match point for a given genotype.

        The anomaloscope has two free parameters:
          ratio ∈ [0,1]:      R = ratio * R_MAX,  G = (1-ratio) * G_MAX
          orange_lum ∈ [0,1]: O = orange_lum * O_MAX

        Match condition (ML responses equal on both fields):
          ML_mat @ [R/255, G/255, 0] = ML_mat @ [0, 0, O/255]

        Substituting the parameterization and rearranging:
          ratio * (ML_col_R * R_MAX - ML_col_G * G_MAX) + ML_col_G * G_MAX
              = orange_lum * ML_col_O * O_MAX

        This is a 2×2 linear system in (ratio, orange_lum).

        Returns dict with ratio, orange_lum, R, G, O (8-bit values) or None.
        """
        # ML response columns scaled by primary ranges
        a = ml_mat[:, 0] * self.ANOM_R_MAX / 255.0   # per-unit-ratio from R
        b = ml_mat[:, 1] * self.ANOM_G_MAX / 255.0   # per-unit-(1-ratio) from G
        c = ml_mat[:, 2] * self.ANOM_O_MAX / 255.0   # per-unit-orange_lum from O

        # System: ratio * (a - b) - orange_lum * c = -b
        # [[a-b | -c]] @ [ratio, orange_lum]^T = -b
        d = a - b
        mat = np.column_stack([d, -c])   # (2, 2)
        rhs = -b                          # (2,)

        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            return None

        ratio, orange_lum = sol

        # Check valid range
        if ratio < -0.01 or ratio > 1.01 or orange_lum < -0.01 or orange_lum > 1.01:
            return None

        ratio = float(np.clip(ratio, 0, 1))
        orange_lum = float(np.clip(orange_lum, 0, 1))

        R = int(round(ratio * self.ANOM_R_MAX))
        G = int(round((1 - ratio) * self.ANOM_G_MAX))
        O = int(round(orange_lum * self.ANOM_O_MAX))

        return {
            "ratio": ratio,
            "orange_lum": orange_lum,
            "R": R, "G": G, "O": O,
            # Display-space [0,1] colors for plate generation
            "rg_disp": np.array([R / 255.0, G / 255.0, 0.0]),
            "o_disp": np.array([0.0, 0.0, O / 255.0]),
        }

    def _show_anomaloscope_predictions(self):
        """
        Show a row of 5 pseudo-isochromatic plates (one per genotype).
        Inside = orange at predicted luminance.
        Outside = R+G at predicted ratio.
        Both fields are ML-metameric for that genotype.
        """
        self.status.config(text="Generating anomaloscope predictions...")
        threading.Thread(
            target=self._generate_anomaloscope_predictions, daemon=True
        ).start()

    def _generate_anomaloscope_predictions(self):
        n_plates = min(5, len(self._catalogue))

        plates = []
        labels = []
        for i in range(n_plates):
            gt = self._catalogue[i]
            ml_mat = gt["ml_mat"]
            match = self._solve_anomaloscope_match(ml_mat)

            if match is None:
                print(f"  Genotype {gt['label']}: no valid anomaloscope match")
                continue

            # Inside (figure) = orange at solved luminance
            inside_disp = match["o_disp"]
            # Outside (background) = R+G at solved ratio
            outside_disp = match["rg_disp"]

            # Verify ML metamerism
            ml_diff = np.linalg.norm(ml_mat @ inside_disp - ml_mat @ outside_disp)
            if ml_diff > 1e-4:
                print(f"  WARNING: ML_diff={ml_diff:.6f} for {gt['label']}")

            inside_cone = self._render_cs.convert(
                inside_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)
            outside_cone = self._render_cs.convert(
                outside_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)

            secret = LANDOLT_DIRECTIONS[i % len(LANDOLT_DIRECTIONS)]
            dot_scale = self.dot_size_scale.get()
            base_sizes = [16, 22, 28]
            dot_sizes = [max(4, int(s * dot_scale)) for s in base_sizes]

            try:
                result = generate_ishihara_plate(
                    inside_cone, outside_cone, self._render_cs,
                    secret=secret,
                    image_size=self.PLATE_PX,
                    dot_sizes=dot_sizes,
                    output_space=ColorSpaceType.DISP,
                    background_color=np.array([0, 0, 0], dtype=int),
                    lum_noise=self.lum_noise.get(),
                    seed=42 + i,
                )
                plates.append(result[0])
            except Exception as exc:
                print(f"  Genotype {gt['label']}: plate error: {exc}")
                continue

            R, G, O = match["R"], match["G"], match["O"]
            ratio_pct = match["ratio"] * 100
            labels.append(
                f"{gt['label']}\n"
                f"ratio={ratio_pct:.1f}% O={O}\n"
                f"R={R} G={G}"
            )
            print(f"  {gt['label']}: ratio={match['ratio']:.3f}  "
                  f"orange_lum={match['orange_lum']:.3f}  "
                  f"R={R} G={G} O={O}  ML_diff={ml_diff:.2e}")

        if not plates:
            self.root.after(0, lambda: self.status.config(
                text="No valid anomaloscope predictions"))
            return

        # Compose plates in a row on the DLP frame
        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)

        # Each plate gets an equal slice of the width
        plate_w = render_w // len(plates)
        plate_sz = min(plate_w - 10, render_h // 2)

        frame = Image.new("RGB", (render_w, render_h), (0, 0, 0))
        draw = ImageDraw.Draw(frame)
        font = self._get_label_font(14)

        for idx, (plate, label) in enumerate(zip(plates, labels)):
            resized = plate.resize((plate_sz, plate_sz), Image.LANCZOS)
            x = idx * plate_w + (plate_w - plate_sz) // 2
            y = (render_h - plate_sz) // 2 - 30
            frame.paste(resized, (x, max(0, y)))

            for li, line in enumerate(label.split('\n')):
                draw.text((x, y + plate_sz + 5 + li * 18),
                          line, fill=(180, 180, 180), font=font)

        # Draw title
        title_font = self._get_label_font(20)
        draw.text((10, 10),
                  f"Anomaloscope Predictions  "
                  f"(R_MAX={self.ANOM_R_MAX} G_MAX={self.ANOM_G_MAX} O_MAX={self.ANOM_O_MAX})",
                  fill=(255, 200, 100), font=title_font)

        frame = frame.resize((self.DLP_W, self.DLP_H), Image.NEAREST)

        def _show():
            self._photo = ImageTk.PhotoImage(frame)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            self.status.config(text="Anomaloscope prediction mode [press any nav key to return]")

        self.root.after(0, _show)

    # ------------------------------------------------------------------
    # Plate generation (threaded)
    # ------------------------------------------------------------------

    def _request_plate(self):
        if self._generating:
            self._pending_request = True
            return
        self._generating = True
        self._pending_request = False
        self.status.config(text="Generating plate...")
        threading.Thread(target=self._generate_plate_thread, daemon=True).start()

    def _generate_plate_thread(self):
        try:
            self._generate_plate_sync()
        finally:
            self._generating = False
            if self._pending_request:
                self.root.after(0, self._request_plate)

    def _generate_plate_sync(self):
        gt = self._catalogue[self._gt_idx]
        null_dir = gt["null_dir"]
        center = gt["center"]
        t_max = gt["t_max"]
        delta_q_per_t = gt["delta_q_per_t"]
        q_peak = gt["q_peak"]
        ml_mat = gt["ml_mat"]

        q_val = self.q_value.get()
        dot_scale = self.dot_size_scale.get()
        lum_noise = self.lum_noise.get()
        secret = LANDOLT_DIRECTIONS[self._gt_idx % len(LANDOLT_DIRECTIONS)]

        # Scale dot sizes
        base_sizes = [16, 22, 28]
        dot_sizes = [max(4, int(s * dot_scale)) for s in base_sizes]

        # Compute plate colors from the genotype's precomputed center and null_dir
        inside_disp, outside_disp = compute_plate_colors(
            null_dir, center, t_max, q_val,
        )

        # Verify ML metamerism for this genotype's cones
        ml_inside = ml_mat @ inside_disp
        ml_outside = ml_mat @ outside_disp
        ml_diff = np.linalg.norm(ml_inside - ml_outside)
        if ml_diff > 1e-4:
            print(f"  WARNING: ML responses differ by {ml_diff:.6f} "
                  f"(inside={ml_inside.round(6)}, outside={ml_outside.round(6)})")

        # Convert DISP -> CONE using the standard trichromat rendering ColorSpace
        inside_cone = self._render_cs.convert(
            inside_disp,  ColorSpaceType.DISP, ColorSpaceType.CONE)
        outside_cone = self._render_cs.convert(
            outside_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)

        bg_255 = np.array([0, 0, 0], dtype=int)  # black background

        try:
            plates = generate_ishihara_plate(
                inside_cone, outside_cone, self._render_cs,
                secret=secret,
                image_size=self.PLATE_PX,
                dot_sizes=dot_sizes,
                output_space=ColorSpaceType.DISP,
                background_color=bg_255,
                lum_noise=lum_noise,
                seed=42 + self._gt_idx,
            )
            plate = plates[0]

            # Apply luminance noise to the O channel (index 2 in RGB encoding).
            # The library's lum_noise operates in cone space and maps primarily
            # to R/G in display space, so we add O-channel noise directly.
            if lum_noise > 0:
                rng = np.random.default_rng(seed=42 + self._gt_idx + 1000)
                arr = np.array(plate)
                mask = np.any(arr > 0, axis=2)
                noise_scale = lum_noise * 255
                o_noise = rng.normal(0, noise_scale, size=arr.shape[:2])
                arr[:, :, 2] = np.where(
                    mask,
                    np.clip(arr[:, :, 2].astype(np.float32) + o_noise, 0, 255),
                    arr[:, :, 2],
                ).astype(np.uint8)
                plate = Image.fromarray(arr)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status.config(text=f"Error: {exc}"))
            return

        self._plate_cache = plate

        # inside_disp = foreground (Landolt C figure), outside_disp = background
        fi, gi_f, oi_f = (inside_disp * 255).astype(int)
        rb, gb, ob = (outside_disp * 255).astype(int)
        direction_label = LANDOLT_LABELS.get(secret, secret)

        # Build delta-Q readout
        dq_txt = ""
        if delta_q_per_t is not None:
            t = q_val * t_max
            dq_val = abs(delta_q_per_t * t)
            dq_txt = f"dQ={dq_val:.4f} Q@{q_peak:.0f}nm dQ/t={delta_q_per_t:.4f}"

        # Store metadata for drawing on the frame
        self._plate_info = {
            "gt_lbl": f"Genotype {self._gt_idx+1}/{len(self._catalogue)}: {gt['label']}",
            "direction": direction_label,
            "fig_rgb": (fi, gi_f, oi_f),
            "bg_rgb": (rb, gb, ob),
            "dq_txt": dq_txt,
            "center": center.round(3),
            "t_max": t_max,
            "ml_diff": ml_diff,
            "q_val": q_val,
        }

        self.root.after(0, lambda: self._display_plate(plate))

    # ------------------------------------------------------------------
    # Font helper
    # ------------------------------------------------------------------

    def _get_label_font(self, size=20):
        """Return a PIL font for drawing on the image, with robust fallback."""
        cache_key = f'_label_font_{size}'
        if not hasattr(self, cache_key):
            font = None
            for path in [
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.dfont",
                "/System/Library/Fonts/Courier.dfont",
                "/System/Library/Fonts/SFNSMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "C:/Windows/Fonts/consola.ttf",
            ]:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
            if font is None:
                try:
                    font = ImageFont.load_default(size=size)
                except TypeError:
                    font = ImageFont.load_default()
            setattr(self, cache_key, font)
        return getattr(self, cache_key)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_plate(self, img: Image.Image):
        bits = self.bit_depth.get()
        disp = quantize_image(img, bits)

        # Compute plate size in pixels at the physical aspect ratio.
        # The plate circle is 95% of image_size (see _generate_geometry),
        # so scale the image so the inner circle matches the visual angle.
        render_h = self.DLP_H
        px_per_cm = render_h / self.PHYS_H_CM
        angle_deg = self.visual_angle.get()
        circle_diameter_cm = 2 * self._viewing_dist_cm * np.tan(np.radians(angle_deg / 2))
        circle_diameter_px = int(circle_diameter_cm * px_per_cm)
        # image_size = circle_diameter / 0.95 (circle is 95% of image)
        target_px = max(100, int(circle_diameter_px / 0.95))

        # Resize the plate to the target size in the square-pixel space
        if disp.size != (target_px, target_px):
            disp = disp.resize((target_px, target_px), Image.LANCZOS)

        # Compose onto a full-frame black image at physical aspect ratio
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)
        frame = Image.new("RGB", (render_w, render_h), (0, 0, 0))
        paste_x = (render_w - target_px) // 2
        paste_y = (render_h - target_px) // 2
        frame.paste(disp, (paste_x, paste_y))

        # Draw info text onto the frame (before squashing)
        draw = ImageDraw.Draw(frame)
        font = self._get_label_font(20)
        font_sm = self._get_label_font(16)
        lc = (180, 180, 180)    # label color
        hc = (255, 200, 100)    # highlight color
        y = 10

        info = self._plate_info
        if info:
            # Upper left: genotype, direction, dQ, q_value
            draw.text((10, y), info.get("gt_lbl", ""), fill=hc, font=font)
            y += 26
            draw.text((10, y), f"[{info.get('direction', '')}]  Q={info.get('q_val', 0):.2f}", fill=lc, font=font)
            y += 26
            if info.get("dq_txt"):
                draw.text((10, y), info["dq_txt"], fill=(255, 136, 68), font=font_sm)
                y += 22
            fi, gi_f, oi_f = info.get("fig_rgb", (0, 0, 0))
            rb, gb, ob = info.get("bg_rgb", (0, 0, 0))
            draw.text((10, y), f"fig(R={fi} G={gi_f} O={oi_f})", fill=lc, font=font_sm)
            y += 20
            draw.text((10, y), f"bg (R={rb} G={gb} O={ob})", fill=lc, font=font_sm)
            y += 20
            draw.text((10, y),
                       f"center={info.get('center', '')}  t_max={info.get('t_max', 0):.3f}",
                       fill=(100, 100, 100), font=font_sm)

            # Upper right: bit-depth, visual angle, ML_diff
            bit_color = {8: (200, 200, 200), 6: (204, 136, 0), 4: (204, 68, 0)}[bits]
            draw.text((render_w - 120, 10), f"{bits}-bit", fill=bit_color, font=font)
            draw.text((render_w - 120, 36), f"{angle_deg:.1f}deg", fill=lc, font=font)
            draw.text((render_w - 200, 62),
                       f"ML_diff={info.get('ml_diff', 0):.2e}", fill=(100, 100, 100), font=font_sm)

        # Squash to DLP native resolution
        frame = frame.resize((self.DLP_W, self.DLP_H), Image.NEAREST)

        self._photo = ImageTk.PhotoImage(frame)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _position_on_monitor(root: tk.Tk, monitor_idx: int):
    """Position the window on the specified monitor (0=primary, 1=HDMI/secondary)."""
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if monitor_idx < len(monitors):
            m = monitors[monitor_idx]
            root.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
            print(f"Positioned on monitor {monitor_idx}: {m.width}x{m.height} at ({m.x},{m.y})")
            return
        print(f"Monitor {monitor_idx} not found ({len(monitors)} available). Using default.")
    except ImportError:
        if monitor_idx > 0:
            print("screeninfo not installed -- cannot target specific monitor. "
                  "Install with: pip install screeninfo")


def main():
    parser = argparse.ArgumentParser(
        description="Genetic test plate viewer for RGO 3-primary display"
    )
    parser.add_argument(
        "--primaries-dir",
        help="Directory with measured primary CSV files (R, G, B, O in RGBO order). "
             "If omitted, synthetic Gaussian primaries are used.",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.90,
        help="Fraction of population to cover with genotypes (default 0.90 = 90%%).",
    )
    parser.add_argument(
        "--sex", choices=["male", "female", "both"], default="both",
        help="Sex for genotype frequency statistics (default: both).",
    )
    parser.add_argument(
        "--viewing-distance", type=float, default=149.0,
        help="Viewing distance in cm (default: 149 cm).",
    )
    parser.add_argument(
        "--monitor", type=int, default=1,
        help="Monitor index (0=primary, 1=HDMI/secondary, default: 1).",
    )
    args = parser.parse_args()

    print("""
============================================
  Genetic Test -- RGO 3-Primary Display
============================================
  HDMI mapping: R->Red LED  G->Green LED  O->Orange LED

  Keyboard controls:
    Left / Right    Prev / next genotype plate
    Up / Down       Adjust Q value (metamer separation)
    C               Control plate (Q=0, figure invisible)
    S               Save current plate as PNG
    B               Cycle 8/6/4-bit mode
    L / Shift+L     Increase / decrease luminance noise
    A               Anomaloscope prediction mode (5 plates)
    H               Toggle controls visibility
    Escape          Exit full-screen
============================================
""")

    root = tk.Tk()
    _position_on_monitor(root, args.monitor)
    root.attributes("-fullscreen", True)
    GeneticTestApp(
        root,
        primaries_dir=args.primaries_dir,
        coverage=args.coverage,
        sex=args.sex,
        viewing_distance_cm=args.viewing_distance,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
