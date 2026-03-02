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
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
R_MAX = 128   # red primary maximum (0-255)
G_MAX = 64    # green primary maximum (0-255)
O_MAX = 255   # orange primary maximum (0-255)

BLINK_ON_MS = 1000   # stimulus visible duration (ms)
BLINK_OFF_MS = 250  # black screen duration (ms)

BIT_MODES = [8, 6, 4]   # cycle order


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
    Up / Down       Orange luminance (right field)
    Left / Right    R/G ratio (left field)
    B               Toggle 8/6/4-bit mode
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

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Anomaloscope -- RGO 3-Primary Display")
        self.root.resizable(True, True)
        self.root.configure(bg=self.BG_COLOR)

        # ── state ──────────────────────────────────────────────────────────
        self.bit_depth = tk.IntVar(value=8)          # 8, 6, or 4
        self.orange_lum = tk.DoubleVar(value=0.5)    # 0-1: right field intensity
        self.rg_ratio = tk.DoubleVar(value=0.5)     # 0=all-green  1=all-red
        self._visual_angle_steps = [2, 4, 6, 8, 10]  # degrees
        self._visual_angle_idx = 0                     # start at 2°

        self._photo = None   # keep reference so GC doesn't collect it
        self._blink_on = True  # True = stimulus visible, False = black

        self._build_ui()
        self._bind_keys()
        self._init_gamepad()

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
        self._make_slider(self.ctrl_frame, "Orange lum (O):", self.orange_lum,
                          "Up/Down -- right field (O channel = Orange LED)")
        self._make_slider(self.ctrl_frame, "R / G ratio:", self.rg_ratio,
                          "Left/Right -- left field  (0=all green, 1=all red)")

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

        # Use bind_all so keys work regardless of which widget has focus
        self.root.bind_all("<Up>", lambda _: step(self.orange_lum, +ss()))
        self.root.bind_all("<Down>", lambda _: step(self.orange_lum, -ss()))
        self.root.bind_all("<Right>", lambda _: step(self.rg_ratio,   +ss()))
        self.root.bind_all("<Left>", lambda _: step(self.rg_ratio,   -ss()))
        self.root.bind_all("<KeyPress-b>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-B>", lambda _: self._toggle_bit())
        self.root.bind_all("<KeyPress-s>", lambda _: self._save_frame())
        self.root.bind_all("<KeyPress-S>", lambda _: self._save_frame())
        self.root.bind_all("<KeyPress-plus>", lambda _: self._step_angle(+1))
        self.root.bind_all("<KeyPress-equal>", lambda _: self._step_angle(+1))
        self.root.bind_all("<KeyPress-minus>", lambda _: self._step_angle(-1))
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

        # Left stick X-axis -> rg_ratio
        x = self._joystick.get_axis(0)
        if abs(x) > DEADZONE:
            self.rg_ratio.set(np.clip(self.rg_ratio.get() + x * SPEED, 0.0, 1.0))

        # Left stick Y-axis -> orange_lum (inverted: up = negative)
        y = self._joystick.get_axis(1)
        if abs(y) > DEADZONE:
            self.orange_lum.set(np.clip(self.orange_lum.get() - y * SPEED, 0.0, 1.0))

        # Button A (0) / B (1) -> orange_lum adjust
        if self._joystick.get_numbuttons() > 1:
            if self._joystick.get_button(0):
                self.orange_lum.set(np.clip(self.orange_lum.get() - SPEED, 0.0, 1.0))
            if self._joystick.get_button(1):
                self.orange_lum.set(np.clip(self.orange_lum.get() + SPEED, 0.0, 1.0))

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
        Return (rg_rgb, orange_rgb) as (R, G, B) tuples with values in [0, 255].
        rg_rgb    = (R, G, 0)   -- top field:  R+G mixture
        orange_rgb = (0, 0, O)  -- bottom field: Orange (sent as B channel)

        R scales over [0, R_MAX], G over [0, G_MAX], O over [0, O_MAX].
        """
        bits = self.bit_depth.get()

        # orange field: only B channel (= Orange LED on HDMI)
        o_raw = int(round(np.clip(self.orange_lum.get(), 0.0, 1.0) * O_MAX))
        o = quantize_to_n_bit(o_raw, bits) if bits < 8 else o_raw
        orange_rgb = (0, 0, o)

        # R+G field: ratio controls mix within their respective ranges
        ratio = self.rg_ratio.get()
        r_raw = int(round(np.clip(ratio, 0.0, 1.0) * R_MAX))
        g_raw = int(round(np.clip(1.0 - ratio, 0.0, 1.0) * G_MAX))
        r = quantize_to_n_bit(r_raw, bits) if bits < 8 else r_raw
        g = quantize_to_n_bit(g_raw, bits) if bits < 8 else g_raw
        rg_rgb = (r, g, 0)

        return rg_rgb, orange_rgb

    # Physical display dimensions in cm (measured)
    PHYS_W_CM = 59.0
    PHYS_H_CM = 37.0

    def _render_image(self) -> Image.Image:
        """Render at the physical aspect ratio (16:10) with square pixels,
        then resize to DLP native resolution (912x1140) for output."""
        rg_rgb, orange_rgb = self.get_display_values()

        # Rasterize at physical aspect ratio so circles are true circles.
        # Use DLP_H as the vertical resolution, scale width to match
        # the physical 59:37 aspect ratio.
        render_h = self.DLP_H
        render_w = int(render_h * self.PHYS_W_CM / self.PHYS_H_CM)  # 1140 * 59/37 ≈ 1818

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

        # Draw top semicircle (R+G test field)
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=180, end=360, fill=rg_rgb,
        )
        # Draw bottom semicircle (orange reference field)
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=0, end=180, fill=orange_rgb,
        )
        # Thin horizontal divider at centre
        draw.line([cx - radius, cy, cx + radius, cy], fill=(60, 60, 60), width=3)

        # Draw R,G and O values in the top corners
        r, g, _ = rg_rgb
        _, _, o = orange_rgb
        font = self._get_label_font()
        label_color = (180, 180, 180)
        draw.text((10, 10), f"R={r:3d}  G={g:3d}", fill=label_color, font=font)
        draw.text((render_w - 160, 10), f"O={o:3d}", fill=label_color, font=font)

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
        self._blink_on = not self._blink_on
        self._refresh_canvas()
        delay = BLINK_ON_MS if self._blink_on else BLINK_OFF_MS
        self.root.after(delay, self._blink_cycle)

    def _refresh_canvas(self):
        """Redraw the canvas based on current blink state."""
        if self._blink_on:
            img = self._render_image()
        else:
            img = Image.new("RGB", (self.DLP_W, self.DLP_H), (0, 0, 0))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _update_display(self, *_):
        # Update the canvas immediately (respects current blink state)
        self._refresh_canvas()

        # value readout
        rg_rgb, orange_rgb = self.get_display_values()
        r, g, _ = rg_rgb
        _, _, o = orange_rgb
        mode = f"{self.bit_depth.get()}-bit"
        self.val_label.config(
            text=(f"[{mode}]   "
                  f"R+G field:  R={r:3d}  G={g:3d}  O=  0   |   "
                  f"Orange field:  R=  0  G=  0  O={o:3d}")
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
    args = parser.parse_args()
    if args.primaries_dir:
        print("Note: --primaries-dir has no effect on the anomaloscope app. "
              "Pass it to genetic_test_app.py instead.")

    _print_controls()

    root = tk.Tk()
    _position_on_monitor(root, args.monitor)
    root.attributes("-fullscreen", True)
    AnomaloscopeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
