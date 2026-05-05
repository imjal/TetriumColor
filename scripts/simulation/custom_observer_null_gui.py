#!/usr/bin/env python3
"""
Interactive matplotlib GUI for custom-observer display null directions.

Builds Observer.custom_observer(..., illuminant="raw"), computes
ColorSpace.get_cone_contrast_null_direction_in_disp() from display midgray,
finds the two display-gamut endpoints along that direction, and plots:
  - display spectra s1 and s2
  - s1 - s2
  - hyperobserver raw excitation |delta|
  - custom cone fundamentals with Stockman-Sharpe fundamentals overlaid in black
"""

from __future__ import annotations
from TetriumColor.Observer.Spectra import Spectra
from TetriumColor.Observer import Cone, Observer
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximumIn1DimDirection
from TetriumColor.ColorSpace import ColorSpace

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, TextBox

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


TEMPLATE_OPTIONS = ("stockman", "neitz", "govardovskii", "baylor", "lamb")
CONE_NAMES = ("S", "M", "Q", "L")
CONE_COLORS = {
    "S": "royalblue",
    "M": "seagreen",
    "Q": "goldenrod",
    "L": "firebrick",
}
TEXT_DEFAULTS = {
    "macular": 1.0,
    "lens": 1.0,
    "od": 0.5,
    "proportion": 1.0,
    "s_peak": 419.0,
    "m_peak": 530.0,
    "q_peak": 547.0,
    "l_peak": 559.0,
}
HYPER_PEAKS = [420, 530, 533, 536, 547, 551, 552, 553, 555, 556, 556.5, 559]
HYPER_LABELS = [
    "S\n420", "M\n530", "M\n533", "M\n536",
    "L\n547", "L\n551", "L\n552", "L\n553",
    "L\n555", "L\n556", "L\n556.5", "L\n559",
]


@dataclass
class NullResult:
    observer: Observer
    direction: np.ndarray
    disp_points_float: np.ndarray
    disp_points: np.ndarray
    disp_codes: np.ndarray
    spectra: np.ndarray
    background_spectrum: np.ndarray
    custom_excitation_float: np.ndarray
    custom_excitation_quantized: np.ndarray
    custom_diff_float: np.ndarray
    custom_diff_quantized: np.ndarray
    hyper_diffs: dict[str, np.ndarray]


class CustomObserverNullGUI:
    def __init__(self, primaries_dir: str, primary_order: str = "BGOR"):
        self.wavelengths = np.arange(380, 781, 1)
        self.primary_order = primary_order
        self.display_scaling_factor = 1.0
        self.primaries = load_primaries_from_csv(
            primaries_dir,
            extract_zero=False,
            primary_order=primary_order,
        )
        if len(self.primaries) != 4:
            raise ValueError(f"Expected 4 display primaries, got {len(self.primaries)}")

        self.primary_matrix = np.stack([
            p.interpolate_values(self.wavelengths).data for p in self.primaries
        ])

        self.template = "stockman"
        self.fig = plt.figure(figsize=(14, 8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Custom Observer Null Direction")

        self.ax_spectra = self.fig.add_axes([0.06, 0.58, 0.46, 0.34])
        self.ax_hyper = [
            self.fig.add_axes([0.58, 0.58 + (4 - i) * 0.068, 0.38, 0.055])
            for i in range(len(TEMPLATE_OPTIONS))
        ]
        self.ax_fund = self.fig.add_axes([0.06, 0.10, 0.46, 0.34])
        self.ax_info = self.fig.add_axes([0.56, 0.10, 0.25, 0.34])
        self.ax_controls = self.fig.add_axes([0.83, 0.10, 0.13, 0.34])
        self.ax_info.axis("off")
        self.ax_controls.axis("off")

        self.text_boxes: dict[str, TextBox] = {}
        self._build_controls()
        self._compute_and_draw()

    def _build_controls(self) -> None:
        self.ax_controls.text(0.0, 1.0, "Parameters", va="top", fontsize=11, fontweight="bold")
        names = list(TEXT_DEFAULTS)
        for i, name in enumerate(names):
            y = 0.405 - i * 0.043
            ax = self.fig.add_axes([0.85, y, 0.10, 0.03])
            box = TextBox(ax, name.replace("_", " "), initial=f"{TEXT_DEFAULTS[name]:.3g}")
            self.text_boxes[name] = box

        radio_ax = self.fig.add_axes([0.79, 0.49, 0.17, 0.08])
        self.template_radio = RadioButtons(radio_ax, TEMPLATE_OPTIONS, active=0)
        for label in self.template_radio.labels:
            label.set_fontsize(7)
        self.template_radio.on_clicked(self._set_template)

        compute_ax = self.fig.add_axes([0.82, 0.045, 0.06, 0.04])
        self.compute_button = Button(compute_ax, "Compute")
        self.compute_button.on_clicked(lambda _: self._compute_and_draw())

        reset_ax = self.fig.add_axes([0.90, 0.045, 0.05, 0.04])
        self.reset_button = Button(reset_ax, "Reset")
        self.reset_button.on_clicked(self._reset)

    def _set_template(self, label: str) -> None:
        self.template = label

    def _reset(self, _: object) -> None:
        for name, value in TEXT_DEFAULTS.items():
            self.text_boxes[name].set_val(f"{value:.3g}")
        self.template = "stockman"
        self.template_radio.set_active(0)
        self._compute_and_draw()

    def _values(self) -> dict[str, float]:
        values = {}
        for name, box in self.text_boxes.items():
            try:
                values[name] = float(box.text)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number, got {box.text!r}") from exc
        return values

    def _current_observer(self) -> Observer:
        values = self._values()
        return Observer.custom_observer(
            self.wavelengths,
            dimension=4,
            s_cone_peak=values["s_peak"],
            m_cone_peak=values["m_peak"],
            q_cone_peak=values["q_peak"],
            l_cone_peak=values["l_peak"],
            od=values["od"],
            macular=values["macular"],
            lens=values["lens"],
            template=self.template,
            degree=None,
            illuminant="raw",
        )

    def _spectrum_from_disp(self, disp: np.ndarray) -> np.ndarray:
        return self.display_scaling_factor * (disp @ self.primary_matrix)

    def _spectrum_rgb(self, data: np.ndarray) -> np.ndarray:
        spectrum = Spectra(wavelengths=self.wavelengths, data=data, normalized=False)
        return np.clip(spectrum.to_rgb(), 0.0, 1.0)

    def _compute(self) -> NullResult:
        observer = self._current_observer()
        cs = ColorSpace(observer, display_primaries=self.primaries)
        self.display_scaling_factor = cs._disp_metadata.get("scaling_factor", 1.0)
        background = np.full(cs.dim, 0.5)

        direction = cs.get_cone_contrast_null_direction_in_disp(background=background)
        direction = direction / np.linalg.norm(direction)
        disp_points = np.clip(
            np.array(FindMaximumIn1DimDirection(background, direction, np.eye(cs.dim))),
            0.0,
            1.0,
        )

        distances = np.linalg.norm(disp_points - background, axis=1)
        disp_points = disp_points[np.argsort(distances)[::-1]]
        proportion = self._values()["proportion"]
        disp_points = background + proportion * (disp_points - background)
        disp_points = np.clip(disp_points, 0.0, 1.0)
        disp_points_float = disp_points.copy()
        disp_codes = np.clip(np.round(disp_points * 255.0), 0, 255).astype(int)
        disp_points = disp_codes.astype(float) / 255.0
        spectra = np.stack([self._spectrum_from_disp(point) for point in disp_points])
        spectra_float = np.stack([self._spectrum_from_disp(point) for point in disp_points_float])
        background_spectrum = self._spectrum_from_disp(background)

        custom_excitation_float = np.stack([
            observer.sensor_matrix @ spectra_float[0],
            observer.sensor_matrix @ spectra_float[1],
        ])
        custom_excitation_quantized = np.stack([
            observer.sensor_matrix @ spectra[0],
            observer.sensor_matrix @ spectra[1],
        ])
        custom_diff_float = custom_excitation_float[0] - custom_excitation_float[1]
        custom_diff_quantized = custom_excitation_quantized[0] - custom_excitation_quantized[1]

        hyper_diffs = {}
        for template in TEMPLATE_OPTIONS:
            hyperobserver = Observer.hyperobserver(
                wavelengths=self.wavelengths,
                template=template,
                od=self._values()["od"],
                illuminant="raw",
                degree=None,
            )
            raw_1 = hyperobserver.sensor_matrix @ spectra[0]
            raw_2 = hyperobserver.sensor_matrix @ spectra[1]
            hyper_diffs[template] = np.abs(raw_1 - raw_2)

        return NullResult(
            observer,
            direction,
            disp_points_float,
            disp_points,
            disp_codes,
            spectra,
            background_spectrum,
            custom_excitation_float,
            custom_excitation_quantized,
            custom_diff_float,
            custom_diff_quantized,
            hyper_diffs,
        )

    def _compute_and_draw(self) -> None:
        try:
            result = self._compute()
        except Exception as exc:
            self.ax_info.clear()
            self.ax_info.axis("off")
            self.ax_info.text(0.0, 0.95, f"Compute failed:\n{exc}", va="top", color="crimson")
            self.fig.canvas.draw_idle()
            return

        self._draw_spectra(result)
        self._draw_hyper_diff(result)
        self._draw_fundamentals(result)
        self._draw_info(result)
        self.fig.canvas.draw_idle()

    def _draw_spectra(self, result: NullResult) -> None:
        self.ax_spectra.clear()
        rgb1 = self._spectrum_rgb(result.spectra[0])
        rgb2 = self._spectrum_rgb(result.spectra[1])
        bg_rgb = self._spectrum_rgb(result.background_spectrum)
        self.ax_spectra.plot(self.wavelengths, result.spectra[0], color=rgb1, label="s1")
        self.ax_spectra.plot(self.wavelengths, result.spectra[1], color=rgb2, label="s2")
        self.ax_spectra.plot(self.wavelengths, result.background_spectrum, color=bg_rgb, alpha=0.45, label="midpoint")
        self.ax_spectra.plot(
            self.wavelengths,
            result.spectra[0] - result.spectra[1],
            color="0.2",
            linestyle="--",
            label="s1 - s2",
        )
        self.ax_spectra.set_title("Display Spectra")
        self.ax_spectra.set_xlabel("Wavelength (nm)")
        self.ax_spectra.set_ylabel("Power")
        self.ax_spectra.legend(loc="upper right")
        self.ax_spectra.grid(alpha=0.25)
        for i, rgb in enumerate((rgb1, rgb2, bg_rgb)):
            self.ax_spectra.add_patch(
                plt.Rectangle(
                    (0.02 + i * 0.055, 0.88),
                    0.04,
                    0.06,
                    transform=self.ax_spectra.transAxes,
                    facecolor=rgb,
                    edgecolor="black",
                    linewidth=0.7,
                )
            )

    def _draw_hyper_diff(self, result: NullResult) -> None:
        x = np.arange(len(HYPER_LABELS))
        max_y = max(float(np.max(diff)) for diff in result.hyper_diffs.values())
        y_top = max(max_y * 1.08, 1e-12)

        for i, (template, ax) in enumerate(zip(TEMPLATE_OPTIONS, self.ax_hyper)):
            diff = result.hyper_diffs[template]
            ax.clear()
            ax.bar(
                x,
                diff,
                color="steelblue" if template != self.template else "darkorange",
                alpha=0.85,
                edgecolor="black",
                linewidth=0.35,
            )
            ax.axhline(0, color="0.2", linewidth=0.8)
            ax.axvspan(-0.5, 0.5, alpha=0.06, color="blue")
            ax.axvspan(0.5, 3.5, alpha=0.06, color="green")
            ax.axvspan(3.5, 11.5, alpha=0.06, color="red")
            ax.set_ylim(0, y_top)
            ax.set_xlim(-0.6, len(HYPER_LABELS) - 0.4)
            ax.text(
                0.01,
                0.82,
                template,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                fontweight="bold" if template == self.template else "normal",
            )
            ax.grid(axis="y", alpha=0.22, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title("Hyperobserver Raw Excitation |Q @ (s1 - s2)| by Cone Template")
            if i == len(TEMPLATE_OPTIONS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(HYPER_LABELS, fontsize=7)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if i == 2:
                ax.set_ylabel("Abs. excitation diff", fontsize=8)

    def _draw_fundamentals(self, result: NullResult) -> None:
        self.ax_fund.clear()
        for name, sensor in zip(CONE_NAMES, result.observer.sensors):
            data = sensor.interpolate_values(self.wavelengths).data
            self.ax_fund.plot(
                self.wavelengths,
                data / np.max(data),
                color=CONE_COLORS[name],
                label=f"custom {name}",
            )

        reference_cones = [
            ("S", Cone.s_cone(self.wavelengths, template=None)),
            ("M", Cone.m_cone(self.wavelengths, template=None)),
            ("L", Cone.l_cone(self.wavelengths, template=None)),
        ]
        for i, (name, cone) in enumerate(reference_cones):
            data = cone.interpolate_values(self.wavelengths).data
            self.ax_fund.plot(
                self.wavelengths,
                data / np.max(data),
                color="black",
                linestyle=["-", "--", ":"][i],
                linewidth=1.5,
                alpha=0.85,
                label=f"S-S {name}",
            )

        self.ax_fund.set_title("Custom Fundamentals with Stockman-Sharpe Reference")
        self.ax_fund.set_xlabel("Wavelength (nm)")
        self.ax_fund.set_ylabel("Normalized sensitivity")
        self.ax_fund.legend(loc="upper right", fontsize=8, ncol=2)
        self.ax_fund.grid(alpha=0.25)

    def _draw_info(self, result: NullResult) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")
        disp1, disp2 = result.disp_points
        code1, code2 = result.disp_codes
        extra_codes = []
        if self.primary_order == "BGOR":
            rgbo1 = code1[[3, 1, 0, 2]]
            rgbo2 = code2[[3, 1, 0, 2]]
            extra_codes = [
                "s1 RGBO 8-bit:",
                np.array2string(rgbo1),
                "s2 RGBO 8-bit:",
                np.array2string(rgbo2),
            ]
        text = "\n".join([
            f"Template: {self.template}",
            f"Primary order: {self.primary_order}",
            f"Display scale: {self.display_scaling_factor:.6g}",
            f"Observer: {result.observer}",
            "",
            "Null direction in DISP:",
            np.array2string(result.direction, precision=5),
            "",
            "s1/s2 DISP float:",
            np.array2string(result.disp_points_float, precision=5),
            "",
            "s1 DISP:",
            np.array2string(disp1, precision=5),
            "s2 DISP:",
            np.array2string(disp2, precision=5),
            "s1 8-bit:",
            np.array2string(code1),
            "s2 8-bit:",
            np.array2string(code2),
            *extra_codes,
            "",
            "custom Q@s float:",
            np.array2string(result.custom_excitation_float, precision=5),
            "custom diff float:",
            np.array2string(result.custom_diff_float, precision=5),
            "custom Q@s 8-bit:",
            np.array2string(result.custom_excitation_quantized, precision=5),
            "custom diff 8-bit:",
            np.array2string(result.custom_diff_quantized, precision=5),
            "",
            f"||s1-s2||_2: {np.linalg.norm(result.spectra[0] - result.spectra[1]):.6g}",
        ])
        self.ax_info.text(0.0, 1.0, text, va="top", family="monospace", fontsize=7)

    def show(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primaries_dir",
        default="measurements/2026-05-04/primaries/",
        help="Directory containing measured display primary CSV files.",
    )
    parser.add_argument(
        "--primary_order",
        default="BGOR",
        choices=("RGBO", "BGOR"),
        help="Primary order returned by load_primaries_from_csv.",
    )
    args = parser.parse_args()

    app = CustomObserverNullGUI(args.primaries_dir, primary_order=args.primary_order)
    app.show()


if __name__ == "__main__":
    main()
