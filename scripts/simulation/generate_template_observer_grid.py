#!/usr/bin/env python3
"""
Generate an RGO/BGO metamer grid across observer genotypes and cone templates.

Rows cycle over the five cone nomogram templates. Columns are the top N
ObserverGenotypes observers with a 547 nm Q cone added. Each cell contains a
side-by-side pair of 8-bit quantized display endpoints found from the cone
contrast null direction at display midpoint.

Outputs:
  - metamer_grid_RGO.png / metamer_grid_BGO.png
  - per-cell RGO/BGO images
  - per-cell hyperobserver bar graphs, with all five hyperobserver templates
  - metadata CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from TetriumColor.ColorSpace import ColorSpace
from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximumIn1DimDirection
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer import Observer
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.Spectra import Spectra


TEMPLATES = ("stockman", "neitz", "govardovskii", "baylor", "lamb")
HYPER_LABELS = [
    "S\n420", "M\n530", "M\n533", "M\n536",
    "L\n547", "L\n551", "L\n552", "L\n553",
    "L\n555", "L\n556", "L\n556.5", "L\n559",
]


def bgor_to_rgo_bgo(bgor_code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert BGOR 8-bit code [B,G,O,R] to RGO and BGO RGB pixels."""
    b, g, o, r = [int(v) for v in bgor_code]
    return np.array([r, g, o], dtype=np.uint8), np.array([b, g, o], dtype=np.uint8)


def make_pair_cell(
    bgor_1: np.ndarray,
    bgor_2: np.ndarray,
    cell_size: int,
    sigma_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGO/BGO cell images with two Gaussian blobs on DISP 0.5 background."""
    h = cell_size
    w = cell_size
    background = np.array([128, 128, 128, 128], dtype=float)

    yy, xx = np.mgrid[0:h, 0:w]
    sigma = max(float(sigma_frac) * cell_size, 1.0)
    centers = [(0.32 * (w - 1), 0.50 * (h - 1)), (0.68 * (w - 1), 0.50 * (h - 1))]
    alpha_1 = np.exp(-((xx - centers[0][0]) ** 2 + (yy - centers[0][1]) ** 2) / (2.0 * sigma ** 2))
    alpha_2 = np.exp(-((xx - centers[1][0]) ** 2 + (yy - centers[1][1]) ** 2) / (2.0 * sigma ** 2))

    bgor = (
        background[None, None, :]
        + alpha_1[:, :, None] * (bgor_1.astype(float) - background)[None, None, :]
        + alpha_2[:, :, None] * (bgor_2.astype(float) - background)[None, None, :]
    )
    bgor = np.clip(np.round(bgor), 0, 255).astype(np.uint8)

    b = bgor[:, :, 0]
    g = bgor[:, :, 1]
    o = bgor[:, :, 2]
    r = bgor[:, :, 3]
    rgo = np.stack([r, g, o], axis=2)
    bgo = np.stack([b, g, o], axis=2)
    return rgo, bgo


def make_single_blob_cell_srgb(
    fg_srgb: np.ndarray,
    bg_srgb: np.ndarray,
    cell_size: int,
    sigma_frac: float,
    position: str,
    exposure: float,
) -> np.ndarray:
    """Return an sRGB cell with one Gaussian blob on a metamer background."""
    h = cell_size
    w = cell_size
    fg = np.clip(np.asarray(fg_srgb, dtype=float) * exposure, 0.0, 1.0)
    bg = np.clip(np.asarray(bg_srgb, dtype=float) * exposure, 0.0, 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    sigma = max(float(sigma_frac) * cell_size, 1.0)
    centers = {
        "center": (0.50 * (w - 1), 0.50 * (h - 1)),
        "left": (0.32 * (w - 1), 0.50 * (h - 1)),
        "right": (0.68 * (w - 1), 0.50 * (h - 1)),
        "up": (0.50 * (w - 1), 0.32 * (h - 1)),
        "down": (0.50 * (w - 1), 0.68 * (h - 1)),
    }
    cx, cy = centers[position]
    alpha = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))

    srgb = bg[None, None, :] + alpha[:, :, None] * (fg - bg)[None, None, :]
    return np.clip(np.round(srgb[:, :, :3] * 255.0), 0, 255).astype(np.uint8)


def place_cell(canvas: np.ndarray, cell: np.ndarray, row: int, col: int, cell_size: int, gap: int) -> None:
    y = row * (cell_size + gap)
    x = col * (cell_size + gap)
    canvas[y:y + cell_size, x:x + cell_size] = cell


def observer_from_genotype(
    wavelengths: np.ndarray,
    genotype: tuple[float, ...],
    template: str,
    od: float,
    macular: float,
    lens: float,
) -> Observer:
    """Create S/M/Q/L custom observer from a trichromat genotype plus Q=547."""
    peaks = sorted(genotype)
    if len(peaks) != 2:
        raise ValueError(f"Expected trichromat genotype with two M/L peaks, got {genotype}")
    m_peak, l_peak = peaks
    return Observer.custom_observer(
        wavelengths,
        dimension=4,
        s_cone_peak=420,
        m_cone_peak=m_peak,
        q_cone_peak=547,
        l_cone_peak=l_peak,
        od=od,
        macular=macular,
        lens=lens,
        template=template,
        degree=None,
        illuminant="raw",
    )


def solve_bgor_pair(
    observer: Observer,
    primaries,
    proportion: float,
    metameric_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Return quantized BGOR codes, normalized BGOR points, spectra, and display scale."""
    cs = ColorSpace(observer, display_primaries=primaries, metameric_axis=metameric_axis)
    scale = cs._disp_metadata.get("scaling_factor", 1.0)
    background = np.full(cs.dim, 0.5)

    direction = cs.get_cone_contrast_null_direction_in_disp(
        metameric_axis_num=metameric_axis,
        background=background,
    )
    direction = direction / np.linalg.norm(direction)
    endpoints = np.clip(
        np.array(FindMaximumIn1DimDirection(background, direction, np.eye(cs.dim))),
        0.0,
        1.0,
    )
    distances = np.linalg.norm(endpoints - background, axis=1)
    endpoints = endpoints[np.argsort(distances)[::-1]]
    endpoints = background + proportion * (endpoints - background)
    endpoints = np.clip(endpoints, 0.0, 1.0)
    codes = np.clip(np.round(endpoints * 255.0), 0, 255).astype(np.uint8)
    quantized = codes.astype(float) / 255.0

    primary_matrix = np.stack([
        p.interpolate_values(observer.wavelengths).data for p in primaries
    ])
    spectra = scale * (quantized @ primary_matrix)
    midpoint_spectrum = scale * (background @ primary_matrix)
    return codes, quantized, spectra, scale, midpoint_spectrum


def hyperobserver_diffs(
    spectra: np.ndarray,
    hyperobservers: dict[str, Observer],
) -> dict[str, np.ndarray]:
    diffs = {}
    for template, hyper in hyperobservers.items():
        raw_1 = hyper.sensor_matrix @ spectra[0]
        raw_2 = hyper.sensor_matrix @ spectra[1]
        diffs[template] = np.abs(raw_1 - raw_2)
    return diffs


def save_hyperobserver_graph(
    path: Path,
    diffs: dict[str, np.ndarray],
    title: str,
    selected_template: str,
) -> None:
    x = np.arange(len(HYPER_LABELS))
    max_y = max(float(np.max(v)) for v in diffs.values())
    y_top = max(max_y * 1.08, 1e-12)

    fig, axes = plt.subplots(
        len(TEMPLATES),
        1,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for i, (template, ax) in enumerate(zip(TEMPLATES, axes)):
        color = "darkorange" if template == selected_template else "steelblue"
        ax.bar(x, diffs[template], color=color, alpha=0.85, edgecolor="black", linewidth=0.35)
        ax.axvspan(-0.5, 0.5, alpha=0.06, color="blue")
        ax.axvspan(0.5, 3.5, alpha=0.06, color="green")
        ax.axvspan(3.5, 11.5, alpha=0.06, color="red")
        ax.set_ylim(0, y_top)
        ax.set_xlim(-0.6, len(HYPER_LABELS) - 0.4)
        ax.text(0.01, 0.82, template, transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 2:
            ax.set_ylabel("Abs. excitation diff")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(HYPER_LABELS, fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hyperobserver_contact_sheet(
    path: Path,
    all_diffs: dict[tuple[int, int], dict[str, np.ndarray]],
    genotypes: list[tuple[float, ...]],
) -> None:
    fig, axes = plt.subplots(
        len(TEMPLATES),
        len(genotypes),
        figsize=(3.0 * len(genotypes), 2.0 * len(TEMPLATES)),
        sharex=True,
        constrained_layout=True,
    )
    global_max = max(float(np.max(v)) for diffs in all_diffs.values() for v in diffs.values())
    y_top = max(global_max * 1.08, 1e-12)
    x = np.arange(len(HYPER_LABELS))

    for row, template in enumerate(TEMPLATES):
        for col, genotype in enumerate(genotypes):
            ax = axes[row, col]
            diff = all_diffs[(row, col)][template]
            ax.bar(x, diff, color="steelblue", alpha=0.85, linewidth=0)
            ax.set_ylim(0, y_top)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{col + 1}\n{genotype}", fontsize=8)
            if col == 0:
                ax.set_ylabel(template, fontsize=9)
    fig.suptitle("Per-cell matched-template hyperobserver raw excitation differences", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_fixed_hyperobserver_contact_sheet(
    path: Path,
    all_diffs: dict[tuple[int, int], dict[str, np.ndarray]],
    genotypes: list[tuple[float, ...]],
    hyper_template: str,
) -> None:
    """Save one grid evaluated everywhere with a single hyperobserver template."""
    fig, axes = plt.subplots(
        len(TEMPLATES),
        len(genotypes),
        figsize=(3.0 * len(genotypes), 2.0 * len(TEMPLATES)),
        sharex=True,
        constrained_layout=True,
    )
    global_max = max(float(np.max(diffs[hyper_template])) for diffs in all_diffs.values())
    y_top = max(global_max * 1.08, 1e-12)
    x = np.arange(len(HYPER_LABELS))

    for row, observer_template in enumerate(TEMPLATES):
        for col, genotype in enumerate(genotypes):
            ax = axes[row, col]
            diff = all_diffs[(row, col)][hyper_template]
            ax.bar(x, diff, color="steelblue", alpha=0.85, linewidth=0)
            ax.set_ylim(0, y_top)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{col + 1}\n{genotype}", fontsize=8)
            if col == 0:
                ax.set_ylabel(observer_template, fontsize=9)
    fig.suptitle(
        f"Hyperobserver raw excitation differences: {hyper_template} hyperobserver",
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primaries_dir", default="measurements/2026-05-04/primaries/")
    parser.add_argument("--output_dir", default="outputs/template_observer_grid")
    parser.add_argument("--num_observers", type=int, default=10)
    parser.add_argument("--sex", choices=("male", "female", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--od", type=float, default=0.5)
    parser.add_argument("--macular", type=float, default=1.0)
    parser.add_argument("--lens", type=float, default=1.0)
    parser.add_argument("--cell_size", type=int, default=96)
    parser.add_argument("--cell_gap", type=int, default=6)
    parser.add_argument(
        "--single_blob_position",
        choices=("left", "right", "up", "down"),
        default="right",
        help="Cardinal position for the additional one-blob sRGB grid.",
    )
    parser.add_argument(
        "--srgb_exposure",
        type=float,
        default=1.0,
        help="Display exposure multiplier for spectra-to-sRGB single-blob cells.",
    )
    parser.add_argument(
        "--blob_sigma_frac",
        type=float,
        default=0.16,
        help="Gaussian blob sigma as a fraction of cell_size.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cells_dir = output_dir / "cells"
    graphs_dir = output_dir / "hyperobserver_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    primaries = load_primaries_from_csv(args.primaries_dir, extract_zero=False, primary_order="BGOR")
    wavelengths = primaries[0].wavelengths
    observer_genotypes = ObserverGenotypes(wavelengths=wavelengths, dimensions=[3], seed=args.seed)
    genotypes = list(observer_genotypes.get_pdf(args.sex).keys())[:args.num_observers]

    hyperobservers = {
        template: Observer.hyperobserver(
            wavelengths=wavelengths,
            template=template,
            od=args.od,
            illuminant="raw",
            degree=None,
        )
        for template in TEMPLATES
    }

    rows = len(TEMPLATES)
    cols = len(genotypes)
    h = rows * args.cell_size + (rows - 1) * args.cell_gap
    w = cols * args.cell_size + (cols - 1) * args.cell_gap
    rgo_grid = np.zeros((h, w, 3), dtype=np.uint8)
    bgo_grid = np.zeros((h, w, 3), dtype=np.uint8)
    srgb_blob_grid = np.zeros((h, w, 3), dtype=np.uint8)

    metadata_rows = []
    all_diffs: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for row, template in enumerate(TEMPLATES):
        for col, genotype in enumerate(genotypes):
            observer = observer_from_genotype(
                wavelengths=wavelengths,
                genotype=genotype,
                template=template,
                od=args.od,
                macular=args.macular,
                lens=args.lens,
            )
            codes, bgor, spectra, scale, midpoint_spectrum = solve_bgor_pair(observer, primaries, args.proportion)
            diffs = hyperobserver_diffs(spectra, hyperobservers)
            all_diffs[(row, col)] = diffs
            endpoint_srgb = np.array([
                Spectra(wavelengths=wavelengths, data=spectra[0], normalized=False).to_rgb(),
                Spectra(wavelengths=wavelengths, data=midpoint_spectrum, normalized=False).to_rgb(),
            ])

            rgo_cell, bgo_cell = make_pair_cell(codes[0], codes[1], args.cell_size, args.blob_sigma_frac)
            srgb_blob_cell = make_single_blob_cell_srgb(
                endpoint_srgb[0],
                endpoint_srgb[1],
                args.cell_size,
                args.blob_sigma_frac,
                args.single_blob_position,
                args.srgb_exposure,
            )
            place_cell(rgo_grid, rgo_cell, row, col, args.cell_size, args.cell_gap)
            place_cell(bgo_grid, bgo_cell, row, col, args.cell_size, args.cell_gap)
            place_cell(srgb_blob_grid, srgb_blob_cell, row, col, args.cell_size, args.cell_gap)

            prefix = f"row{row:02d}_{template}_obs{col:02d}"
            Image.fromarray(rgo_cell).save(cells_dir / f"{prefix}_RGO.png")
            Image.fromarray(bgo_cell).save(cells_dir / f"{prefix}_BGO.png")
            Image.fromarray(srgb_blob_cell).save(cells_dir / f"{prefix}_SRGB_single_blob.png")

            title = (
                f"{template} observer row {row + 1}, observer {col + 1}: genotype {genotype}\n"
                f"BGOR s1={codes[0].tolist()} s2={codes[1].tolist()}"
            )
            save_hyperobserver_graph(
                graphs_dir / f"{prefix}_hyperobserver.png",
                diffs,
                title,
                selected_template=template,
            )

            rgbo_1 = codes[0][[3, 1, 0, 2]]
            rgbo_2 = codes[1][[3, 1, 0, 2]]
            metadata_rows.append({
                "row": row,
                "col": col,
                "template": template,
                "genotype": repr(genotype),
                "m_peak": sorted(genotype)[0],
                "q_peak": 547,
                "l_peak": sorted(genotype)[1],
                "display_scale": scale,
                "bgor_1": codes[0].tolist(),
                "bgor_2": codes[1].tolist(),
                "rgbo_1": rgbo_1.tolist(),
                "rgbo_2": rgbo_2.tolist(),
                "matched_template_hyper_max": float(np.max(diffs[template])),
                "matched_template_hyper_l559": float(diffs[template][-1]),
            })
            print(f"{prefix}: BGOR {codes[0].tolist()} / {codes[1].tolist()}")

    Image.fromarray(rgo_grid).save(output_dir / "metamer_grid_RGO.png")
    Image.fromarray(bgo_grid).save(output_dir / "metamer_grid_BGO.png")
    Image.fromarray(srgb_blob_grid).save(output_dir / "metamer_grid_SRGB_single_blob.png")
    save_hyperobserver_contact_sheet(output_dir / "hyperobserver_grid.png", all_diffs, genotypes)
    for hyper_template in TEMPLATES:
        save_fixed_hyperobserver_contact_sheet(
            output_dir / f"hyperobserver_grid_{hyper_template}.png",
            all_diffs,
            genotypes,
            hyper_template,
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(metadata_rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Wrote {output_dir / 'metamer_grid_RGO.png'}")
    print(f"Wrote {output_dir / 'metamer_grid_BGO.png'}")
    print(f"Wrote {output_dir / 'metamer_grid_SRGB_single_blob.png'}")
    print(f"Wrote {output_dir / 'hyperobserver_grid.png'}")
    for hyper_template in TEMPLATES:
        print(f"Wrote {output_dir / f'hyperobserver_grid_{hyper_template}.png'}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {len(metadata_rows)} per-cell graph/image sets under {output_dir}")


if __name__ == "__main__":
    main()
