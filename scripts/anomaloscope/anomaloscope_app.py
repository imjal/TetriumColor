"""
Anomaloscope split-field app for RGO 3-primary display.

Displays a Rayleigh anomaloscope task:
  Left field:  R + G mixture  (the "test" light — ratio adjustable)
  Right field: Orange          (the "reference" light — luminance adjustable)

The output image is sent over HDMI where the channel mapping is:
  R channel -> Red LED
  G channel -> Green LED
  B channel -> Orange LED  (not blue on the physical display)

Controls:
  Arrow keys:    up/down -> orange lum   |   left/right -> R/G ratio
  B key:         toggle 6-bit / 8-bit mode
  S key:         save current display frame as PNG
  Escape:        exit full-screen
  Gamepad:       left stick X -> ratio, left stick Y -> orange lum

Usage:
  python anomaloscope_app.py
  python anomaloscope_app.py --monitor 0
"""

import argparse
import csv
import json
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

try:
    from TetriumColor.PsychoPhys.IshiharaPlate import generate_ishihara_plate
    from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
    from TetriumColor.Observer import Observer
    _HAS_PLATE = True
except ImportError:
    _HAS_PLATE = False

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# ---------------------------------------------------------------------------
# Bit-depth helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Primary range caps (8-bit values).  The R/G ratio slider spans
# [0, R_MAX] for the red channel and [0, G_MAX] for the green channel.
# ---------------------------------------------------------------------------
R_MAX = 128  # red primary maximum (0-255) at SCALE=1.0
G_MAX = 64   # green primary maximum (0-255) at SCALE=1.0
O_MAX = 255   # orange primary maximum (0-255)
# SCALE is now computed dynamically from the reference field so the match
# field always has enough R+G headroom to reach the reference luminance.

# LED luminances (cd/m² at full 255). Used to keep match field at
# constant luminance as R/G ratio changes.
R_LUM = 2.0   # red LED luminance (cd/m²) at 255
G_LUM = 4.0   # green LED luminance (cd/m²) at 255
O_LUM = 2.0   # orange LED luminance (cd/m²) at 255

# Reference field chromaticity: fixed R/G ratio for the "orange" side.
# k=1.0 → pure red, k=0.0 → pure green, k~0.6 → orange-ish.
# The reference field is:  R_ref = k * REF_R_MAX * lum,
#                          G_ref = (1-k) * REF_G_MAX * lum,
#                          O_ref = O_chrom * O_MAX
# Three knobs: rg_ratio (match side), ref_lum (reference brightness),
#              orange_chrom (orange LED added to reference)
REF_K = 0.6       # fixed R/G chromaticity ratio for reference field
REF_R_MAX = 255   # max R for reference field
REF_G_MAX = 255   # max G for reference field

BLINK_ON_MS = 2000   # stimulus visible duration (ms)
BLINK_OFF_MS = 500  # black screen duration (ms)
PLATE_BLUR_RADIUS = 3.0  # Gaussian blur applied to pseudo-isochromatic plate (0 = off)
LUM_NOISE_LEVELS = [0.0, 0.005, 0.015, 0.03, 0.06]  # lum noise std-dev (cone-space), cycled with N

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_SCRIPT_DIR, "anomaloscope_settings.json")
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

BIT_MODES = [8, 6, 4]   # cycle order

LANDOLT_DIRECTIONS = [
    ('landolt_up', 'Up'), ('landolt_down', 'Down'),
    ('landolt_left', 'Left'), ('landolt_right', 'Right'),
    ('landolt_up-right', 'Up-Right'), ('landolt_down-right', 'Down-Right'),
    ('landolt_up-left', 'Up-Left'), ('landolt_down-left', 'Down-Left'),
]

BLOB_DIRECTIONS = [
    ('up', 'Up'), ('down', 'Down'),
    ('left', 'Left'), ('right', 'Right'),
]

TEST_MODES = ['bipartite', 'plate', 'blob']

MEASURE_PRIMARIES = [
    ((255, 0, 0), "Red 255"),
    ((0, 255, 0), "Green 255"),
    ((0, 0, 255), "Orange(B) 255"),
]


def quantize_to_n_bit(v_255: int, bits: int) -> int:
    """Round an 8-bit value (0-255) to the nearest n-bit representation."""
    levels = (1 << bits) - 1   # 255 for 8, 63 for 6, 15 for 4
    vq = round(v_255 * levels / 255)
    return round(vq * 255 / levels)


def to_int8(value_01: float, bits: int) -> int:
    """Convert [0, 1] float -> 8-bit int, optionally quantised to n-bit."""
    v = int(round(np.clip(value_01, 0.0, 1.0) * 255))
    return quantize_to_n_bit(v, bits) if bits < 8 else v


def _print_controls():
    """Print keyboard/gamepad controls to the terminal."""
    print("""
========================================
  Anomaloscope -- RGO 3-Primary Display
========================================
  HDMI mapping: R->Red LED  G->Green LED  O->Orange LED

  Keyboard controls:
    Up / Down       Orange chromaticity (orange LED in reference)
    Shift+Up/Down   Reference R+G luminance (bottom field)
    Left / Right    R/G ratio (top match field)
    B               Toggle 8/6/4-bit mode
    T               Cycle mode: bipartite / plate / blob
    X               Toggle blink on/off
    M               Toggle measurement patch (Left/Right cycles R/G/O)
    N               Cycle luminance noise (0/.005/.015/.03/.06)
    P               Start/stop trial session (plate/blob 4AFC)
    L               Log current match to data/ folder
    +/-             Stimulus size (2/4/6/8/10 deg)
    S               Save current frame as PNG
    Escape          Exit full-screen

  Gamepad (if detected):
    Left stick X    R/G ratio
    Left stick Y    Orange luminance
    Button A / B    Orange lum down / up
========================================
""")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AnomaloscopeApp:
    """
    Rayleigh anomaloscope for a 3-primary (R, G, O) display.

    The left field contains the R+G mixture (test light).
    The right field contains the orange primary (reference light).
    Both halves are rendered into a single RGB image; the B channel carries
    the Orange LED signal on the physical HDMI display.

    The bipartite field is rendered as a circle with a vertical divider.
    """

    # DLP display native resolution
    DLP_W = 912
    DLP_H = 1140
    STEP_FOR_BITS = {8: 1/255, 6: 4/255, 4: 17/255}   # one step per bit depth

    BG_COLOR = "#1e1e1e"
    FG_COLOR = "#dddddd"
    VAL_COLOR = "#88ccff"
    HINT_COLOR = "#666666"

    def __init__(self, root: tk.Tk, debug: bool = False):
        self.root = root
        self._debug = debug
        self.root.title("Anomaloscope -- RGO 3-Primary Display")
        self.root.resizable(True, True)
        self.root.configure(bg=self.BG_COLOR)

        # In debug mode, match the laptop screen resolution instead of DLP
        if self._debug:
            self.root.update_idletasks()
            self.DLP_W = self.root.winfo_screenwidth()
            self.DLP_H = self.root.winfo_screenheight()

        # ── state (defaults, overridden by saved settings below) ─────────
        self.bit_depth = tk.IntVar(value=8)          # 8, 6, or 4
        self.rg_ratio = tk.DoubleVar(value=0.5)     # 0=all-green  1=all-red (match side)
        self.ref_lum = tk.DoubleVar(value=0.0)      # 0-1: reference field R+G brightness
        self.orange_chrom = tk.DoubleVar(value=0.0)  # 0-1: orange LED added to reference
        self._visual_angle_steps = [2, 4, 6, 8, 10]  # degrees
        self._visual_angle_idx = 0                     # start at 2°

        self._photo = None   # keep reference so GC doesn't collect it
        self._blink_on = True  # True = stimulus visible, False = black
        self._test_mode = 'bipartite'  # 'bipartite' | 'plate' | 'blob'
        self._plate_cache = None  # cached plate image
        self._plate_dirty = True  # True = need to regenerate plate
        self._landolt_dir = 'landolt_right'  # current Landolt C direction
        self._landolt_label = 'Right'         # human-readable label
        self._blob_direction = 'right'  # current blob 4AFC position
        self._blob_label = 'Right'
        self._lum_noise_idx = 0  # index into LUM_NOISE_LEVELS
        self._measure_mode = False
        self._measure_idx = 0  # index into MEASURE_PRIMARIES

        # ── trial session state ───────────────────────────────────────────
        self._session_active = False
        self._session_trials = []   # list of (correct_answer, user_answer, correct_bool)
        self._session_awaiting = False  # True while stimulus is on and waiting for response

        # Load previous session settings
        self._load_settings()

        # Set up plate rendering (for T toggle)
        self._render_cs = None
        if _HAS_PLATE:
            try:
                wl = np.arange(380, 781, 1)
                from TetriumColor.Observer import Spectra
                peaks, sigma = [660, 540, 598], 15.0
                primaries = []
                for pk in peaks:
                    data = np.exp(-((wl - pk) ** 2) / (2 * sigma ** 2))
                    primaries.append(Spectra(wavelengths=wl, data=data / data.max()))
                obs = Observer.trichromat(wavelengths=wl)
                self._render_cs = ColorSpace(obs, display_primaries=primaries)
            except Exception as exc:
                print(f"Plate mode unavailable: {exc}")

        # ── logging session ────────────────────────────────────────────────
        self._log_dir = None   # created on first log entry
        self._log_count = 0

        self._build_ui()
        self._bind_keys()
        self._init_gamepad()

        # Save settings on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Wait for window to be mapped so we can read actual screen size
        self.root.update_idletasks()
        self._screen_w = self.root.winfo_screenwidth()
        self._screen_h = self.root.winfo_screenheight()

        self._update_display()
        self._blink_cycle()

        # Force keyboard focus to root so arrow keys work immediately
        self.root.focus_force()
        self.root.after(100, lambda: self.root.focus_force())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── canvas ──
        self.canvas = tk.Canvas(
            self.root, width=self.DLP_W, height=self.DLP_H,
            bg="black", bd=0, highlightthickness=0,
        )
        self.canvas.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

        # ── debug controls panel at bottom ──
        self.ctrl_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=10)
        self.ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=4)

        # value readout
        self.val_label = tk.Label(
            self.ctrl_frame, text="", font=("Courier", 10),
            bg=self.BG_COLOR, fg=self.VAL_COLOR, anchor="w", padx=12,
        )
        self.val_label.pack(fill=tk.X)

        # bit-mode row
        bit_row = tk.Frame(self.ctrl_frame, bg=self.BG_COLOR)
        bit_row.pack(fill=tk.X, pady=2)
        tk.Label(bit_row, text="Bit mode:", bg=self.BG_COLOR, fg=self.FG_COLOR,
                 width=14, anchor="w").pack(side=tk.LEFT)
        self.bit_btn = tk.Button(
            bit_row, text="8-bit", width=8, command=self._toggle_bit,
            bg="#3c3c3c", fg="#dddddd", activebackground="#555555",
            relief="flat",
        )
        self.bit_btn.pack(side=tk.LEFT, padx=4)
        tk.Label(bit_row, text="[B] cycle 8/6/4-bit",
                 bg=self.BG_COLOR, fg=self.HINT_COLOR, font=("Helvetica", 9),
                 ).pack(side=tk.LEFT, padx=6)

        # sliders
        self._make_slider(self.ctrl_frame, "R / G ratio:", self.rg_ratio,
                          "Left/Right -- match field  (0=all green, 1=all red)")
        self._make_slider(self.ctrl_frame, "Orange chrom:", self.orange_chrom,
                          "Up/Down -- orange LED added to reference")
        self._make_slider(self.ctrl_frame, "Ref lum:", self.ref_lum,
                          "Shift+Up/Down -- reference R+G brightness (fixed k)")

        # HDMI note
        tk.Label(self.ctrl_frame,
                 text="HDMI output:  R -> Red LED     G -> Green LED     O(B) -> Orange LED",
                 fg="#888888", font=("Helvetica", 9, "italic"), bg=self.BG_COLOR,
                 ).pack(pady=(2, 2))

    def _make_slider(self, parent, label: str, var: tk.DoubleVar, hint: str):
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, pady=2)

        tk.Label(row, text=label, bg=self.BG_COLOR, fg=self.FG_COLOR,
                 width=17, anchor="w").pack(side=tk.LEFT)

        style = ttk.Style()
        style.configure("Horizontal.TScale", background=self.BG_COLOR)
        slider = ttk.Scale(row, from_=0, to=1, variable=var,
                           command=lambda _: self._update_display())
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        val_lbl = tk.Label(row, text="0.500", width=6,
                           bg=self.BG_COLOR, fg=self.VAL_COLOR,
                           font=("Courier", 10))
        val_lbl.pack(side=tk.LEFT)

        int_lbl = tk.Label(row, text="(128)", width=6,
                           bg=self.BG_COLOR, fg=self.HINT_COLOR,
                           font=("Courier", 9))
        int_lbl.pack(side=tk.LEFT)

        tk.Label(row, text=hint, bg=self.BG_COLOR, fg=self.HINT_COLOR,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=6)

        def _refresh(*_):
            fv = var.get()
            iv = to_int8(fv, self.bit_depth.get())
            val_lbl.config(text=f"{fv:.3f}")
            int_lbl.config(text=f"({iv:3d})")

        var.trace_add("write", _refresh)
        self.bit_depth.trace_add("write", _refresh)

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self):
        def step(var, delta):
            var.set(np.clip(var.get() + delta, 0.0, 1.0))
            self._update_display()

        def ss(): return self.STEP_FOR_BITS.get(self.bit_depth.get(), 1/255)

        def on_arrow(direction):
            # During an active trial session in plate/blob mode,
            # arrow keys are 4AFC response inputs instead of adjustments.
            if self._session_active and self._session_awaiting:
                self._submit_response(direction)
                return
            # Otherwise, normal adjustment behaviour
            if direction == 'Up':
                step(self.orange_chrom, +ss())
            elif direction == 'Down':
                step(self.orange_chrom, -ss())
            elif direction == 'Left':
                if self._measure_mode:
                    self._cycle_measure_primary(-1)
                else:
                    step(self.rg_ratio, -ss())
            elif direction == 'Right':
                if self._measure_mode:
                    self._cycle_measure_primary(+1)
                else:
                    step(self.rg_ratio, +ss())

        # Use bind_all so keys work regardless of which widget has focus
        self.root.bind_all("<Up>", lambda _: on_arrow('Up'))
        self.root.bind_all("<Down>", lambda _: on_arrow('Down'))
        self.root.bind_all("<Right>", lambda _: on_arrow('Right'))
        self.root.bind_all("<Left>", lambda _: on_arrow('Left'))
        self.root.bind_all("<Shift-Up>", lambda _: step(self.ref_lum, +ss()))
        self.root.bind_all("<Shift-Down>", lambda _: step(self.ref_lum, -ss()))
        self.root.bind_all("<KeyPress-b>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-B>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-s>", lambda _: self._save_frame())
        self.root.bind_all("<KeyPress-S>", lambda _: self._save_frame())
        self.root.bind_all("<KeyPress-plus>", lambda _: self._step_angle(+1))
        self.root.bind_all("<KeyPress-equal>", lambda _: self._step_angle(+1))
        self.root.bind_all("<KeyPress-minus>", lambda _: self._step_angle(-1))
        self.root.bind_all("<KeyPress-t>", lambda _: self._toggle_plate_mode())
        self.root.bind_all("<KeyPress-T>", lambda _: self._toggle_plate_mode())
        self.root.bind_all("<KeyPress-x>", lambda _: self._toggle_blink())
        self.root.bind_all("<KeyPress-X>", lambda _: self._toggle_blink())
        self.root.bind_all("<KeyPress-m>", lambda _: self._toggle_measure())
        self.root.bind_all("<KeyPress-M>", lambda _: self._toggle_measure())
        self.root.bind_all("<KeyPress-n>", lambda _: self._cycle_noise())
        self.root.bind_all("<KeyPress-N>", lambda _: self._cycle_noise())
        self.root.bind_all("<KeyPress-l>", lambda _: self._log_match())
        self.root.bind_all("<KeyPress-L>", lambda _: self._log_match())
        self.root.bind_all("<KeyPress-p>", lambda _: self._toggle_session())
        self.root.bind_all("<KeyPress-P>", lambda _: self._toggle_session())
        self.root.bind_all("<Escape>", lambda _: self.root.attributes("-fullscreen", False))

    # ------------------------------------------------------------------
    # Gamepad input
    # ------------------------------------------------------------------

    def _init_gamepad(self):
        self._joystick = None
        if not _HAS_PYGAME:
            print("pygame not installed -- gamepad input disabled.")
            return
        # Only init joystick subsystem -- pygame.init() steals keyboard
        # focus from Tkinter on macOS
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            print(f"Gamepad detected: {self._joystick.get_name()}")
            self._poll_gamepad()
        else:
            print("No gamepad detected -- keyboard/slider input only.")

    def _poll_gamepad(self):
        if self._joystick is None:
            return
        pygame.event.pump()

        DEADZONE = 0.15
        SPEED = 0.008   # per poll step

        if self._session_active and self._session_awaiting:
            # In trial mode, map stick flicks to 4AFC responses
            # Use a threshold + cooldown to avoid repeats
            if not hasattr(self, '_gp_cooldown'):
                self._gp_cooldown = 0
            if self._gp_cooldown > 0:
                self._gp_cooldown -= 1
            else:
                x = self._joystick.get_axis(0)
                y = self._joystick.get_axis(1)
                THRESH = 0.6
                if abs(x) > abs(y) and abs(x) > THRESH:
                    self._submit_response('Right' if x > 0 else 'Left')
                    self._gp_cooldown = 15  # ~250ms at 60Hz
                elif abs(y) > THRESH:
                    self._submit_response('Up' if y < 0 else 'Down')
                    self._gp_cooldown = 15
        else:
            # Normal adjustment mode
            x = self._joystick.get_axis(0)
            if abs(x) > DEADZONE:
                self.rg_ratio.set(np.clip(self.rg_ratio.get() + x * SPEED, 0.0, 1.0))

            y = self._joystick.get_axis(1)
            if abs(y) > DEADZONE:
                self.ref_lum.set(np.clip(self.ref_lum.get() - y * SPEED, 0.0, 1.0))

            if self._joystick.get_numbuttons() > 1:
                if self._joystick.get_button(0):
                    self.ref_lum.set(np.clip(self.ref_lum.get() - SPEED, 0.0, 1.0))
                if self._joystick.get_button(1):
                    self.ref_lum.set(np.clip(self.ref_lum.get() + SPEED, 0.0, 1.0))

        self.root.after(16, self._poll_gamepad)   # ~60 Hz

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle_bit(self):
        cur = self.bit_depth.get()
        idx = (BIT_MODES.index(cur) + 1) % len(BIT_MODES)
        nxt = BIT_MODES[idx]
        self.bit_depth.set(nxt)
        colors = {8: ("#3c3c3c", "#dddddd"), 6: ("#cc8800", "white"), 4: ("#cc4400", "white")}
        bg, fg = colors[nxt]
        self.bit_btn.config(text=f"{nxt}-bit", bg=bg, fg=fg)
        self._update_display()

    def _step_angle(self, direction):
        self._visual_angle_idx = np.clip(
            self._visual_angle_idx + direction,
            0, len(self._visual_angle_steps) - 1,
        )
        self._update_display()

    def _save_frame(self):
        path = "anomaloscope_frame.png"
        img = self._render_image()
        img.save(path)
        print(f"Saved frame -> {path}")

    def _toggle_blink(self):
        self._blink_enabled = not getattr(self, '_blink_enabled', True)
        if self._blink_enabled:
            self._blink_on = True
            print("Blink ON")
        else:
            self._blink_on = True
            self._refresh_canvas()
            print("Blink OFF")

    def _toggle_measure(self):
        if self._measure_mode:
            self._measure_mode = False
            print("Measurement mode OFF")
        else:
            self._measure_mode = True
            color, label = MEASURE_PRIMARIES[self._measure_idx]
            print(f"Measurement mode ON: {label}  {color}")
        self._update_display()

    def _cycle_measure_primary(self, direction):
        self._measure_idx = (self._measure_idx + direction) % len(MEASURE_PRIMARIES)
        color, label = MEASURE_PRIMARIES[self._measure_idx]
        print(f"Measure primary: {label}  {color}")
        self._update_display()

    def _cycle_noise(self):
        self._lum_noise_idx = (self._lum_noise_idx + 1) % len(LUM_NOISE_LEVELS)
        lvl = LUM_NOISE_LEVELS[self._lum_noise_idx]
        self._plate_dirty = True
        print(f"Luminance noise: {lvl}")
        self._update_display()

    # ------------------------------------------------------------------
    # Trial session (4AFC for plate / blob modes)
    # ------------------------------------------------------------------

    def _toggle_session(self):
        """Start or stop a trial session.  P key toggles."""
        if self._test_mode == 'bipartite':
            print("Trial sessions only available in plate or blob mode.")
            return
        if self._session_active:
            self._end_session()
        else:
            self._start_session()

    def _start_session(self):
        self._session_active = True
        self._session_trials = []
        self._session_awaiting = True  # first stimulus already on screen
        print(f"\n{'='*40}")
        print(f"  SESSION STARTED  ({self._test_mode} mode)")
        print(f"  Respond with arrow keys or gamepad.")
        print(f"  Press P to end session.")
        print(f"{'='*40}\n")
        self._update_display()

    def _end_session(self):
        self._session_active = False
        self._session_awaiting = False
        total = len(self._session_trials)
        correct = sum(1 for _, _, c in self._session_trials if c)
        pct = (correct / total * 100) if total > 0 else 0

        summary = (f"Session complete: {correct}/{total} correct ({pct:.0f}%)")
        print(f"\n{'='*40}")
        print(f"  {summary}")
        for i, (ans, resp, ok) in enumerate(self._session_trials, 1):
            mark = "+" if ok else "X"
            print(f"    [{mark}] Trial {i}: answer={ans}, response={resp}")
        print(f"{'='*40}\n")

        # Log session to CSV
        self._log_session()

        # Show result on screen via the val_label
        self.val_label.config(text=summary)
        self._update_display()

    def _submit_response(self, direction: str):
        """Record the subject's response for the current trial."""
        if not self._session_awaiting:
            return
        self._session_awaiting = False

        if self._test_mode == 'plate':
            correct_answer = self._landolt_label
        elif self._test_mode == 'blob':
            correct_answer = self._blob_label
        else:
            return

        is_correct = (direction == correct_answer)
        self._session_trials.append((correct_answer, direction, is_correct))
        n = len(self._session_trials)
        mark = "+" if is_correct else "X"
        print(f"  [{mark}] Trial {n}: answer={correct_answer}, response={direction}")

        # Brief feedback then advance to next trial via the blink cycle.
        # The next blink-on will set _session_awaiting = True again.

    def _log_session(self):
        """Write completed session trials to the data folder."""
        if not self._session_trials:
            return
        if self._log_dir is None:
            session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._log_dir = os.path.join(_DATA_DIR, session_ts)
            os.makedirs(self._log_dir, exist_ok=True)

        match_rgb, ref_rgb = self.get_display_values()
        bits = self.bit_depth.get()
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        noise = LUM_NOISE_LEVELS[self._lum_noise_idx]
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(self._log_dir, f"session_{self._test_mode}_{ts}.csv")

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "correct_answer", "response", "correct",
                "mode", "bit_depth", "angle_deg", "lum_noise",
                "match_R", "match_G", "ref_R", "ref_G", "ref_O",
                "rg_ratio", "ref_lum", "orange_chrom",
            ])
            for i, (ans, resp, ok) in enumerate(self._session_trials, 1):
                writer.writerow([
                    i, ans, resp, int(ok),
                    self._test_mode, bits, angle_deg, noise,
                    match_rgb[0], match_rgb[1],
                    ref_rgb[0], ref_rgb[1], ref_rgb[2],
                    f"{self.rg_ratio.get():.6f}",
                    f"{self.ref_lum.get():.6f}",
                    f"{self.orange_chrom.get():.6f}",
                ])
        total = len(self._session_trials)
        correct = sum(1 for _, _, c in self._session_trials if c)
        print(f"Session logged -> {log_path}  ({correct}/{total})")

    def _toggle_plate_mode(self):
        idx = TEST_MODES.index(self._test_mode)
        nxt = TEST_MODES[(idx + 1) % len(TEST_MODES)]
        if nxt == 'plate' and (not _HAS_PLATE or self._render_cs is None):
            nxt = TEST_MODES[(idx + 2) % len(TEST_MODES)]
        self._test_mode = nxt
        self._plate_dirty = True
        print(f"Display mode: {self._test_mode}")
        self._update_display()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self):
        """Load parameters from the previous session, if available."""
        if not os.path.exists(_SETTINGS_PATH):
            return
        try:
            with open(_SETTINGS_PATH, "r") as f:
                s = json.load(f)
            self.bit_depth.set(s.get("bit_depth", 8))
            self.rg_ratio.set(s.get("rg_ratio", 0.5))
            self.ref_lum.set(s.get("ref_lum", 0.0))
            self.orange_chrom.set(s.get("orange_chrom", 0.0))
            self._visual_angle_idx = s.get("visual_angle_idx", 0)
            saved_mode = s.get("test_mode", s.get("plate_mode", "bipartite"))
            if saved_mode is True:
                self._test_mode = 'plate'
            elif saved_mode is False:
                self._test_mode = 'bipartite'
            elif saved_mode in TEST_MODES:
                self._test_mode = saved_mode
            self._lum_noise_idx = s.get("lum_noise_idx", 0) % len(LUM_NOISE_LEVELS)
            print(f"Loaded settings from {_SETTINGS_PATH}")
        except Exception as exc:
            print(f"Could not load settings: {exc}")

    def _save_settings(self):
        """Persist current parameters to disk."""
        s = {
            "bit_depth": self.bit_depth.get(),
            "rg_ratio": self.rg_ratio.get(),
            "ref_lum": self.ref_lum.get(),
            "orange_chrom": self.orange_chrom.get(),
            "visual_angle_idx": int(self._visual_angle_idx),
            "test_mode": self._test_mode,
            "lum_noise_idx": self._lum_noise_idx,
        }
        try:
            with open(_SETTINGS_PATH, "w") as f:
                json.dump(s, f, indent=2)
            print(f"Saved settings to {_SETTINGS_PATH}")
        except Exception as exc:
            print(f"Could not save settings: {exc}")

    def _on_close(self):
        """Save settings and exit."""
        self._save_settings()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Match logging
    # ------------------------------------------------------------------

    def _log_match(self):
        """Log current match data to a timestamped CSV in the data folder."""
        # Create session directory on first log
        if self._log_dir is None:
            session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._log_dir = os.path.join(_DATA_DIR, session_ts)
            os.makedirs(self._log_dir, exist_ok=True)

        self._log_count += 1
        match_rgb, ref_rgb = self.get_display_values()
        mr, mg, _ = match_rgb
        rr, rg_v, ro = ref_rgb
        bits = self.bit_depth.get()
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        mode = self._test_mode
        timestamp = datetime.now().isoformat()

        log_path = os.path.join(self._log_dir, "matches.csv")
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "trial", "mode", "bit_depth", "angle_deg",
                    "rg_ratio", "match_R", "match_G",
                    "ref_R", "ref_G", "ref_O",
                    "ref_lum", "orange_chrom",
                ])
            writer.writerow([
                timestamp, self._log_count, mode, bits, angle_deg,
                f"{self.rg_ratio.get():.6f}", mr, mg,
                rr, rg_v, ro,
                f"{self.ref_lum.get():.6f}", f"{self.orange_chrom.get():.6f}",
            ])
        print(f"[LOG #{self._log_count}]  Match R={mr} G={mg}  |  "
              f"Ref R={rr} G={rg_v} O={ro}  |  {angle_deg}deg {bits}-bit  ->  {log_path}")

    def _generate_plate(self) -> Image.Image | None:
        """Generate a pseudo-isochromatic plate with the current match and reference colors."""
        if self._render_cs is None:
            return None

        match_rgb, ref_rgb = self.get_display_values()

        # Convert 8-bit display values to [0,1] for DISP space
        match_disp = np.array([c / 255.0 for c in match_rgb])
        ref_disp = np.array([c / 255.0 for c in ref_rgb])

        match_cone = self._render_cs.convert(match_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)
        ref_cone = self._render_cs.convert(ref_disp, ColorSpaceType.DISP, ColorSpaceType.CONE)

        try:
            plates = generate_ishihara_plate(
                match_cone, ref_cone, self._render_cs,
                secret=self._landolt_dir,
                image_size=1024,
                dot_sizes=[16, 22, 28],
                output_space=ColorSpaceType.DISP,
                background_color=np.array([0, 0, 0], dtype=int),
                lum_noise=LUM_NOISE_LEVELS[self._lum_noise_idx],
                seed=42,
            )
            return plates[0]
        except Exception as exc:
            print(f"Plate generation error: {exc}")
            return None

    def _render_plate_image(self) -> Image.Image:
        """Render the pseudo-isochromatic plate onto the DLP frame."""
        if self._plate_dirty or self._plate_cache is None:
            self._plate_cache = self._generate_plate()
            self._plate_dirty = False

        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)

        frame = Image.new("RGB", (render_w, render_h), (0, 0, 0))

        if self._plate_cache is not None:
            # Size the plate to the current visual angle
            viewing_dist_cm = 149.0
            angle_deg = self._visual_angle_steps[self._visual_angle_idx]
            diameter_cm = 2 * viewing_dist_cm * np.tan(np.radians(angle_deg / 2))
            px_per_cm = render_h / self.PHYS_H_CM
            # Plate circle is 95% of image_size
            circle_px = int(diameter_cm * px_per_cm)
            target_px = max(100, int(circle_px / 0.95))

            plate = self._plate_cache.resize((target_px, target_px), Image.LANCZOS)
            if PLATE_BLUR_RADIUS > 0:
                from PIL import ImageFilter
                plate = plate.filter(ImageFilter.GaussianBlur(radius=PLATE_BLUR_RADIUS))
            paste_x = (render_w - target_px) // 2
            paste_y = (render_h - target_px) // 2
            frame.paste(plate, (paste_x, paste_y))

        # Draw labels
        draw = ImageDraw.Draw(frame)
        rg_rgb, orange_rgb = self.get_display_values()
        r, g, _ = rg_rgb
        rr, rg_v, o = orange_rgb
        font = self._get_label_font()
        label_color = (180, 180, 180)
        draw.text((10, 10), f"Match R={r:3d} G={g:3d}", fill=label_color, font=font)
        draw.text((render_w - 350, 10), f"Ref R={rr:3d} G={rg_v:3d} O={o:3d}",
                  fill=label_color, font=font)

        bits = self.bit_depth.get()
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        noise_std = LUM_NOISE_LEVELS[self._lum_noise_idx]
        bit_color = {8: (200, 200, 200), 6: (204, 136, 0), 4: (204, 68, 0)}[bits]
        cx = render_w // 2
        cy = render_h // 2
        draw.text((10, cy - 24), f"{bits}-bit", fill=bit_color, font=font)
        draw.text((10, cy + 4), f"{angle_deg}deg", fill=(200, 200, 200), font=font)
        mode_label = "PLATE"
        if self._session_active:
            mode_label = "PLATE  SESSION"
        draw.text((10, cy + 30), f"{mode_label}  noise={noise_std}", fill=(255, 200, 100), font=font)

        # Answer label — hidden during active session
        if not self._session_active:
            draw.text((10, cy + 56), f"Ans: {self._landolt_label}",
                      fill=(120, 120, 120), font=font)

        frame = frame.resize((self.DLP_W, self.DLP_H), Image.NEAREST)
        return frame

    # ------------------------------------------------------------------
    # Gaussian blob 4AFC
    # ------------------------------------------------------------------

    def _render_blob_image(self) -> Image.Image:
        """Render a 4AFC gaussian blob detection stimulus.

        A circular patch (visual-angle sized) is filled with the match R/G
        color plus per-pixel Gaussian luminance noise.  A small Gaussian blob
        (diameter ≈ Landolt-C gap) of the reference O+R/G color is placed at
        one of 4 cardinal offsets within the circle.
        """
        match_rgb, ref_rgb = self.get_display_values()

        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)

        cx, cy = render_w // 2, render_h // 2

        viewing_dist_cm = 149.0
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        diameter_cm = 2 * viewing_dist_cm * np.tan(np.radians(angle_deg / 2))
        px_per_cm = render_h / self.PHYS_H_CM
        radius = int(diameter_cm * px_per_cm / 2)

        # Landolt-C gap = 1/5 of diameter
        gap_px = radius * 2 * 0.2
        # Blob σ so visible extent (4σ diameter) ≈ gap
        blob_sigma = max(gap_px / 4.0, 1.0)

        # Blob offset: place at half-radius from centre
        offset = radius * 0.5
        positions = {
            'up':    (cx, cy - int(offset)),
            'down':  (cx, cy + int(offset)),
            'left':  (cx - int(offset), cy),
            'right': (cx + int(offset), cy),
        }
        bx, by = positions[self._blob_direction]

        # --- build frame as float64 for blending ---
        bg = np.array(match_rgb, dtype=np.float64)
        fg = np.array(ref_rgb, dtype=np.float64)

        frame = np.zeros((render_h, render_w, 3), dtype=np.float64)

        # Circular mask at visual-angle size
        Y, X = np.ogrid[:render_h, :render_w]
        dist_sq = (X - cx).astype(np.float64)**2 + (Y - cy).astype(np.float64)**2
        circle_mask = dist_sq <= radius**2

        # Fill circle with background color + per-pixel luminance noise
        # LUM_NOISE_LEVELS are in cone-space; scale to 8-bit for pixel noise
        noise_std_8bit = LUM_NOISE_LEVELS[self._lum_noise_idx] * 255.0
        if noise_std_8bit > 0:
            lum_noise = np.random.normal(0.0, noise_std_8bit, (render_h, render_w))
        else:
            lum_noise = 0.0
        for c in range(3):
            plane = np.where(circle_mask, bg[c] + lum_noise, 0.0)
            frame[:, :, c] = plane

        # --- composite Gaussian blob (tight bbox for speed) ---
        hw = int(np.ceil(4 * blob_sigma))
        x0, x1 = max(bx - hw, 0), min(bx + hw + 1, render_w)
        y0, y1 = max(by - hw, 0), min(by + hw + 1, render_h)

        xs = np.arange(x0, x1, dtype=np.float64) - bx
        ys = np.arange(y0, y1, dtype=np.float64) - by
        Gx = np.exp(-xs**2 / (2 * blob_sigma**2))
        Gy = np.exp(-ys**2 / (2 * blob_sigma**2))
        alpha = Gy[:, None] * Gx[None, :]

        # Only blend inside the circle
        patch_mask = circle_mask[y0:y1, x0:x1]
        alpha = alpha * patch_mask

        for c in range(3):
            patch = frame[y0:y1, x0:x1, c]
            frame[y0:y1, x0:x1, c] = patch * (1.0 - alpha) + fg[c] * alpha

        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), 'RGB')

        # Labels
        draw = ImageDraw.Draw(img)
        mr, mg, _ = match_rgb
        rr, rg_v, ro = ref_rgb
        font = self._get_label_font()
        label_color = (180, 180, 180)
        draw.text((10, 10), f"Match R={mr:3d} G={mg:3d}", fill=label_color, font=font)
        draw.text((render_w - 350, 10), f"Ref R={rr:3d} G={rg_v:3d} O={ro:3d}",
                  fill=label_color, font=font)

        bits = self.bit_depth.get()
        bit_color = {8: (200, 200, 200), 6: (204, 136, 0), 4: (204, 68, 0)}[bits]
        draw.text((10, cy - 24), f"{bits}-bit", fill=bit_color, font=font)
        draw.text((10, cy + 4), f"{angle_deg}deg", fill=(200, 200, 200), font=font)
        noise_lvl = LUM_NOISE_LEVELS[self._lum_noise_idx]
        mode_label = "BLOB"
        if self._session_active:
            mode_label = "BLOB  SESSION"
        draw.text((10, cy + 30), f"{mode_label}  noise={noise_lvl}", fill=(100, 200, 255), font=font)
        if not self._session_active:
            draw.text((10, cy + 56), f"Ans: {self._blob_label}",
                      fill=(120, 120, 120), font=font)

        img = img.resize((self.DLP_W, self.DLP_H), Image.NEAREST)
        return img

    # ------------------------------------------------------------------
    # Measurement patch
    # ------------------------------------------------------------------

    def _render_measure_image(self) -> Image.Image:
        """Render a single-primary circle for spectroradiometer measurement."""
        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)

        img = Image.new("RGB", (render_w, render_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = render_w // 2, render_h // 2

        viewing_dist_cm = 149.0
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        diameter_cm = 2 * viewing_dist_cm * np.tan(np.radians(angle_deg / 2))
        px_per_cm = render_h / self.PHYS_H_CM
        radius = int(diameter_cm * px_per_cm / 2)

        color, label = MEASURE_PRIMARIES[self._measure_idx]
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=color)

        font = self._get_label_font()
        draw.text((10, 10), f"MEASURE: {label}", fill=(180, 180, 180), font=font)
        draw.text((10, cy + 4), f"{angle_deg}deg", fill=(200, 200, 200), font=font)
        draw.text((10, cy + 30), "Left/Right to cycle primary",
                  fill=(120, 120, 120), font=font)

        img = img.resize((self.DLP_W, self.DLP_H), Image.NEAREST)
        return img

    # ------------------------------------------------------------------
    # Font helper
    # ------------------------------------------------------------------

    def _get_label_font(self):
        """Return a PIL font for drawing on the image, with robust fallback."""
        if not hasattr(self, '_label_font'):
            self._label_font = None
            # Try common macOS/Linux/Windows monospace fonts
            for path in [
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.dfont",
                "/System/Library/Fonts/Courier.dfont",
                "/System/Library/Fonts/SFNSMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "C:/Windows/Fonts/consola.ttf",
            ]:
                try:
                    self._label_font = ImageFont.truetype(path, 22)
                    break
                except Exception:
                    continue
            if self._label_font is None:
                try:
                    # PIL >= 10.1 default font supports size
                    self._label_font = ImageFont.load_default(size=22)
                except TypeError:
                    self._label_font = ImageFont.load_default()
        return self._label_font

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def get_display_values(self):
        """
        Return (match_rgb, ref_rgb) as (R, G, B) tuples with values in [0, 255].

        Match field (top):
            R+G mixture controlled by rg_ratio slider.
            SCALE is computed dynamically so the match field can reach
            the reference field's total luminance.

        Reference field (bottom):
            Orange LED (adjustable via orange_chrom) plus a fixed-chromaticity
            R+G component (adjustable brightness via ref_lum).
            R_ref = REF_K * REF_R_MAX * lum
            G_ref = (1-REF_K) * REF_G_MAX * lum
            O_ref = orange_chrom * O_MAX
        """
        bits = self.bit_depth.get()

        # ── Reference field (bottom): O + fixed R/G ratio ──
        # Compute reference first so we can derive SCALE for the match field.
        lum = np.clip(self.ref_lum.get(), 0.0, 1.0)
        rr_raw = int(round(REF_K * REF_R_MAX * lum))
        gr_raw = int(round((1.0 - REF_K) * REF_G_MAX * lum))
        or_raw = int(round(np.clip(self.orange_chrom.get(), 0.0, 1.0) * O_MAX))
        rr = quantize_to_n_bit(min(rr_raw, 255), bits) if bits < 8 else min(rr_raw, 255)
        gr = quantize_to_n_bit(min(gr_raw, 255), bits) if bits < 8 else min(gr_raw, 255)
        orv = quantize_to_n_bit(min(or_raw, 255), bits) if bits < 8 else min(or_raw, 255)
        ref_rgb = (rr, gr, orv)

        # ── Compute SCALE from reference field luminance ──
        # Only boost SCALE when R/G is actually added to the reference side.
        # When ref_lum is 0 (orange-only reference), use the static default.
        DEFAULT_SCALE = 1.5
        if lum > 1e-8:
            ref_lum_total = (rr * R_LUM + gr * G_LUM + orv * O_LUM) / 255.0
            match_base_lum = (0.5 * R_MAX * R_LUM + 0.5 * G_MAX * G_LUM) / 255.0
            if match_base_lum > 1e-8:
                scale = max(DEFAULT_SCALE, 1.2 * ref_lum_total / match_base_lum)
            else:
                scale = DEFAULT_SCALE
        else:
            scale = DEFAULT_SCALE

        # ── Match field (top): adjustable R/G ratio, constant luminance ──
        r_max = min(int(R_MAX * scale), 255)
        g_max = min(int(G_MAX * scale), 255)
        ratio = np.clip(self.rg_ratio.get(), 0.0, 1.0)

        # Unnormalized values
        r_un = ratio * r_max
        g_un = (1.0 - ratio) * g_max

        # Luminance at this ratio (cd/m²)
        lum_current = r_un * R_LUM / 255.0 + g_un * G_LUM / 255.0

        # Target luminance: use luminance at ratio=0.5 as reference
        r_mid = 0.5 * r_max
        g_mid = 0.5 * g_max
        lum_target = r_mid * R_LUM / 255.0 + g_mid * G_LUM / 255.0

        # Scale to keep luminance constant as ratio changes
        if lum_current > 1e-8:
            lum_scale = lum_target / lum_current
        else:
            lum_scale = 1.0

        r_raw = int(round(min(r_un * lum_scale, 255)))
        g_raw = int(round(min(g_un * lum_scale, 255)))
        r = quantize_to_n_bit(r_raw, bits) if bits < 8 else r_raw
        g = quantize_to_n_bit(g_raw, bits) if bits < 8 else g_raw
        match_rgb = (r, g, 0)

        return match_rgb, ref_rgb

    # Physical display dimensions in cm (measured)
    PHYS_W_CM = 59.0
    PHYS_H_CM = 37.0

    def _render_image(self) -> Image.Image:
        """Render at the physical aspect ratio (16:10) with square pixels,
        then resize to DLP native resolution (912x1140) for output."""
        match_rgb, ref_rgb = self.get_display_values()

        # Rasterize at physical aspect ratio so circles are true circles.
        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)

        img = Image.new("RGB", (render_w, render_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx = render_w // 2
        cy = render_h // 2

        # Visual angle at 149 cm viewing distance
        viewing_dist_cm = 149.0
        angle_deg = self._visual_angle_steps[self._visual_angle_idx]
        diameter_cm = 2 * viewing_dist_cm * np.tan(np.radians(angle_deg / 2))
        px_per_cm = render_h / self.PHYS_H_CM
        radius = int(diameter_cm * px_per_cm / 2)

        # Draw top semicircle (match field: adjustable R+G)
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=180, end=360, fill=match_rgb,
        )
        # Draw bottom semicircle (reference field: fixed-chrom R+G + orange)
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=0, end=180, fill=ref_rgb,
        )
        # Thin horizontal divider at centre
        draw.line([cx - radius, cy, cx + radius, cy], fill=(60, 60, 60), width=3)

        # Draw values in the top corners
        mr, mg, _ = match_rgb
        rr, rg_v, ro = ref_rgb
        font = self._get_label_font()
        label_color = (180, 180, 180)
        draw.text((10, 10), f"Match R={mr:3d} G={mg:3d}", fill=label_color, font=font)
        draw.text((render_w - 350, 10), f"Ref R={rr:3d} G={rg_v:3d} O={ro:3d}",
                  fill=label_color, font=font)

        # Bit-depth and angle indicator on the left, vertically centred
        bits = self.bit_depth.get()
        bit_color = {8: (200, 200, 200), 6: (204, 136, 0), 4: (204, 68, 0)}[bits]
        draw.text((10, cy - 24), f"{bits}-bit", fill=bit_color, font=font)
        draw.text((10, cy + 4), f"{angle_deg}deg", fill=(200, 200, 200), font=font)

        # Squash to DLP native resolution — the non-square DLP pixels
        # will stretch it back to the physical aspect ratio on screen.
        img = img.resize((self.DLP_W, self.DLP_H), Image.NEAREST)
        return img

    def _blink_cycle(self):
        """Toggle stimulus on/off to prevent adaptation."""
        if getattr(self, '_blink_enabled', True):
            self._blink_on = not self._blink_on
            if self._blink_on:
                import random
                if self._test_mode == 'plate':
                    # During a session, restrict to 4 cardinal directions
                    # so arrow keys can be used as responses
                    if self._session_active:
                        pool = [d for d in LANDOLT_DIRECTIONS
                                if d[1] in ('Up', 'Down', 'Left', 'Right')]
                    else:
                        pool = LANDOLT_DIRECTIONS
                    self._landolt_dir, self._landolt_label = random.choice(pool)
                    self._plate_dirty = True
                elif self._test_mode == 'blob':
                    self._blob_direction, self._blob_label = random.choice(BLOB_DIRECTIONS)
                # In a trial session, mark that we're waiting for a response
                if self._session_active:
                    self._session_awaiting = True
            self._refresh_canvas()
        delay = BLINK_ON_MS if self._blink_on else BLINK_OFF_MS
        self.root.after(delay, self._blink_cycle)

    def _refresh_canvas(self):
        """Redraw the canvas based on current blink state."""
        if self._measure_mode:
            img = self._render_measure_image()
        elif self._blink_on:
            if self._test_mode == 'plate':
                img = self._render_plate_image()
            elif self._test_mode == 'blob':
                img = self._render_blob_image()
            else:
                img = self._render_image()
        else:
            img = Image.new("RGB", (self.DLP_W, self.DLP_H), (0, 0, 0))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _update_display(self, *_):
        self._plate_dirty = True  # values changed, regenerate plate on next render
        self._refresh_canvas()

        # value readout
        match_rgb, ref_rgb = self.get_display_values()
        mr, mg, _ = match_rgb
        rr, rg_v, ro = ref_rgb
        mode = f"{self.bit_depth.get()}-bit"
        self.val_label.config(
            text=(f"[{mode}]   "
                  f"Match:  R={mr:3d}  G={mg:3d}   |   "
                  f"Ref:  R={rr:3d}  G={rg_v:3d}  O={ro:3d}")
        )


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
        description="Anomaloscope for RGO 3-primary display"
    )
    parser.add_argument(
        "--primaries-dir",
        help="(Informational only) Directory with measured primary CSV files. "
             "This app shows raw RGB values; pass --primaries-dir to "
             "genetic_test_app.py for colour-science plate generation.",
    )
    parser.add_argument(
        "--monitor", type=int, default=1,
        help="Monitor index (0=primary, 1=HDMI/secondary, default: 1).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: half-resolution window on laptop (no DLP needed).",
    )
    args = parser.parse_args()
    if args.primaries_dir:
        print("Note: --primaries-dir has no effect on the anomaloscope app. "
              "Pass it to genetic_test_app.py instead.")

    _print_controls()

    root = tk.Tk()
    if args.debug:
        print("=== DEBUG MODE: fullscreen on laptop ===")
        root.attributes("-fullscreen", True)
    else:
        _position_on_monitor(root, args.monitor)
        root.attributes("-fullscreen", True)
    AnomaloscopeApp(root, debug=args.debug)
    root.mainloop()


if __name__ == "__main__":
    main()
