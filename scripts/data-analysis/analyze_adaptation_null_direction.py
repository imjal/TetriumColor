#!/usr/bin/env python3
"""
Analyze how adaptation assumptions move display metamer/null directions.

This script is intentionally explicit about the model:

  raw cone response to display stimulus x:      C @ P @ x
  row-gain adapted response under spectrum a:  diag(1 / (C @ a)) @ C @ P @ x

where C is the observer sensor matrix and P contains measured RGBO primary spectra.
That is the von-Kries / row-gain interpretation: adaptation scales cone fundamentals.

The script can also run the project-style "illuminant_multiply" model:

  diag(1 / (C @ a)) @ C @ diag(a) @ P @ x

That treats display spectra as reflectances under an illuminant. It can rotate null
directions because it changes the effective primary spectra. For an emissive display,
that is usually the wrong physical model, but it is useful for diagnosing why an
adaptation correction may shift the inferred match point.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import null_space


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
TETRIUM_COLOR_ROOT = REPO_ROOT / "extern" / "TetriumColor"
sys.path.insert(0, str(TETRIUM_COLOR_ROOT))

from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer import Cone, Observer
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.Spectra import Illuminant


RGBO_LABELS = ("R", "G", "B", "O")
DEFAULT_ADAPTANTS = ("d65", "disp05", "disp025")


@dataclass
class Genotype:
    rank: int
    peaks: tuple[float, ...]
    probability: float

    @property
    def label(self) -> str:
        return "(" + ",".join(format_peak(p) for p in self.peaks) + ")"


@dataclass
class AdaptationResult:
    genotype: Genotype
    model: str
    adaptant: str
    gains: np.ndarray
    response_matrix: np.ndarray
    null_direction: np.ndarray | None
    anomaloscope: dict[str, float] | None
    stim_delta_norm: float | None = None
    stim_delta_relative: float | None = None


def format_peak(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def parse_rgbo(text: str) -> np.ndarray:
    values = np.array([float(x.strip()) for x in text.split(",")], dtype=float)
    if values.shape != (4,):
        raise argparse.ArgumentTypeError("Expected four comma-separated RGBO values.")
    if np.any(values > 1.0):
        values = values / 255.0
    return values


def latest_primaries_dir() -> Path:
    measurement_root = TETRIUM_COLOR_ROOT / "measurements"
    candidates = sorted(
        p / "primaries"
        for p in measurement_root.iterdir()
        if (p / "primaries").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No primaries directories found under {measurement_root}")
    return candidates[-1]


def interpolate_vector(spectrum, wavelengths: np.ndarray) -> np.ndarray:
    return spectrum.interpolate_values(wavelengths).data.astype(float)


def make_observer(peaks: Iterable[float], wavelengths: np.ndarray, degree: float) -> Observer:
    # ObserverGenotypes stores variable M/L peaks. Add the standard S cone first.
    with contextlib.redirect_stdout(io.StringIO()):
        sensors = [Cone.cone(420, wavelengths=wavelengths, template="neitz", degree=degree)]
        sensors.extend(
            Cone.cone(float(peak), wavelengths=wavelengths, template="neitz", degree=degree)
            for peak in peaks
        )
    return Observer(sensors, illuminant="raw", degree=degree)


def observer_genotypes(count: int, sex: str, wavelengths: np.ndarray) -> list[Genotype]:
    genotypes = ObserverGenotypes(wavelengths=wavelengths, dimensions=[3]).get_pdf(sex)
    rows: list[Genotype] = []
    for idx, (peaks, prob) in enumerate(genotypes.items(), start=1):
        rows.append(Genotype(rank=idx, peaks=tuple(float(p) for p in peaks), probability=float(prob)))
        if len(rows) >= count:
            break
    return rows


def display_spd(primaries: list, rgbo: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    out = np.zeros_like(wavelengths, dtype=float)
    for primary, weight in zip(primaries, rgbo):
        out += float(weight) * interpolate_vector(primary, wavelengths)
    return out


def adaptant_spd(name: str, primaries: list, wavelengths: np.ndarray) -> np.ndarray:
    key = name.lower()
    if key in {"d65", "illuminant:d65"}:
        return Illuminant.get("D65").interpolate_values(wavelengths).data.astype(float)
    if key == "disp05":
        return display_spd(primaries, np.full(4, 0.5), wavelengths)
    if key == "disp025":
        return display_spd(primaries, np.full(4, 0.25), wavelengths)
    if key.startswith("disp:"):
        return display_spd(primaries, parse_rgbo(key.split(":", 1)[1]), wavelengths)
    raise ValueError(f"Unknown adaptant '{name}'. Use d65, disp05, disp025, or disp:r,g,b,o.")


def response_matrix(
    observer: Observer,
    primaries: list,
    adapt_spd: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    sensor = observer.get_sensor_matrix(observer.wavelengths)
    white = sensor @ adapt_spd
    if np.any(np.abs(white) < 1e-12):
        raise ValueError("Adaptation white point produced near-zero cone response.")
    gains = 1.0 / white

    primary_matrix = np.column_stack(
        [interpolate_vector(primary, observer.wavelengths) for primary in primaries]
    )
    if model == "row_gain":
        raw = sensor @ primary_matrix
    elif model == "illuminant_multiply":
        raw = sensor @ (adapt_spd[:, None] * primary_matrix)
    else:
        raise ValueError(f"Unknown model '{model}'")
    return gains[:, None] * raw, gains


def unit_null_direction(matrix: np.ndarray, cone_subset: tuple[int, ...]) -> np.ndarray | None:
    sub = matrix[list(cone_subset), :]
    ns = null_space(sub)
    if ns.size == 0:
        return None
    direction = ns[:, 0]
    direction = direction / np.linalg.norm(direction)
    # Fix sign for stable reporting.
    first = np.flatnonzero(np.abs(direction) > 1e-9)
    if first.size and direction[first[0]] < 0:
        direction = -direction
    return direction


def angle_degrees(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def solve_anomaloscope_match(matrix: np.ndarray, r_max: float, g_max: float, o_max: float) -> dict[str, float] | None:
    # Use M/L rows. Observer order is S, then genotype peaks sorted low-to-high.
    ml = matrix[1:3, [0, 1, 3]]
    a = ml[:, 0] * r_max / 255.0
    b = ml[:, 1] * g_max / 255.0
    c = ml[:, 2] * o_max / 255.0
    system = np.column_stack([a - b, -c])
    rhs = -b
    try:
        ratio, orange = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None
    if ratio < -0.01 or ratio > 1.01 or orange < -0.01 or orange > 1.01:
        return None
    ratio = float(np.clip(ratio, 0, 1))
    orange = float(np.clip(orange, 0, 1))
    return {
        "ratio": ratio,
        "orange_lum": orange,
        "R": round(ratio * r_max),
        "G": round((1.0 - ratio) * g_max),
        "O": round(orange * o_max),
    }


def analyze(
    genotypes: list[Genotype],
    primaries: list,
    models: list[str],
    adaptants: list[str],
    cone_subset: tuple[int, ...],
    degree: float,
    stim_a: np.ndarray | None,
    stim_b: np.ndarray | None,
    r_max: float,
    g_max: float,
    o_max: float,
) -> list[AdaptationResult]:
    wavelengths = primaries[0].wavelengths
    adapt_spds = {name: adaptant_spd(name, primaries, wavelengths) for name in adaptants}
    results: list[AdaptationResult] = []

    for genotype in genotypes:
        observer = make_observer(genotype.peaks, wavelengths, degree)
        for model in models:
            for name, spd in adapt_spds.items():
                mat, gains = response_matrix(observer, primaries, spd, model)
                direction = unit_null_direction(mat, cone_subset)
                anom = solve_anomaloscope_match(mat, r_max, g_max, o_max)

                delta_norm = None
                delta_relative = None
                if stim_a is not None and stim_b is not None:
                    delta = mat @ (stim_a - stim_b)
                    mean_response = mat @ ((stim_a + stim_b) * 0.5)
                    delta_norm = float(np.linalg.norm(delta))
                    denom = float(np.linalg.norm(mean_response))
                    delta_relative = delta_norm / denom if denom > 1e-12 else float("nan")

                results.append(
                    AdaptationResult(
                        genotype=genotype,
                        model=model,
                        adaptant=name,
                        gains=gains,
                        response_matrix=mat,
                        null_direction=direction,
                        anomaloscope=anom,
                        stim_delta_norm=delta_norm,
                        stim_delta_relative=delta_relative,
                    )
                )
    return results


def write_csv(path: Path, results: list[AdaptationResult], baseline: str) -> None:
    by_key = {(r.genotype.rank, r.model, r.adaptant): r for r in results}
    fields = [
        "rank", "genotype", "probability", "model", "adaptant",
        "gain_S", "gain_M", "gain_L",
        "gain_M_over_L", "gain_S_over_L",
        "null_R", "null_G", "null_B", "null_O",
        f"null_angle_vs_{baseline}_deg",
        "anom_ratio", "anom_orange_lum", "anom_R", "anom_G", "anom_O",
        "stim_delta_norm", "stim_delta_relative",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            base = by_key.get((result.genotype.rank, result.model, baseline))
            angle = angle_degrees(
                result.null_direction,
                base.null_direction if base is not None else None,
            )
            row = {
                "rank": result.genotype.rank,
                "genotype": result.genotype.label,
                "probability": result.genotype.probability,
                "model": result.model,
                "adaptant": result.adaptant,
                "gain_S": result.gains[0],
                "gain_M": result.gains[1],
                "gain_L": result.gains[2],
                "gain_M_over_L": result.gains[1] / result.gains[2],
                "gain_S_over_L": result.gains[0] / result.gains[2],
                "stim_delta_norm": result.stim_delta_norm,
                "stim_delta_relative": result.stim_delta_relative,
                f"null_angle_vs_{baseline}_deg": angle,
            }
            if result.null_direction is not None:
                row.update({f"null_{label}": value for label, value in zip(RGBO_LABELS, result.null_direction)})
            if result.anomaloscope is not None:
                row.update({
                    "anom_ratio": result.anomaloscope["ratio"],
                    "anom_orange_lum": result.anomaloscope["orange_lum"],
                    "anom_R": result.anomaloscope["R"],
                    "anom_G": result.anomaloscope["G"],
                    "anom_O": result.anomaloscope["O"],
                })
            writer.writerow(row)


def print_summary(results: list[AdaptationResult], models: list[str], adaptants: list[str], baseline: str) -> None:
    by_rank = {}
    for result in results:
        by_rank.setdefault((result.model, result.genotype.rank), {})[result.adaptant] = result

    print("\nNull direction/adaptation summary")
    print("model                 rank  genotype     adaptant  angle_vs_base  M/L_gain  S/L_gain  null[RGBO]                 anomaloscope")
    for model in models:
        for rank in sorted(rank for m, rank in by_rank if m == model):
            rows = by_rank[(model, rank)]
            base = rows.get(baseline)
            for adaptant in adaptants:
                r = rows[adaptant]
                angle = angle_degrees(r.null_direction, base.null_direction if base else None)
                null_text = "none"
                if r.null_direction is not None:
                    null_text = "[" + " ".join(f"{x:+.3f}" for x in r.null_direction) + "]"
                anom = "invalid"
                if r.anomaloscope:
                    anom = (
                        f"ratio={r.anomaloscope['ratio']:.3f} "
                        f"O={r.anomaloscope['orange_lum']:.3f} "
                        f"RGB=({r.anomaloscope['R']:.0f},{r.anomaloscope['G']:.0f},{r.anomaloscope['O']:.0f})"
                    )
                angle_text = "base" if r.adaptant == baseline else f"{angle:.3f}"
                print(
                    f"{model:<21} {rank:>4}  {r.genotype.label:<11} {adaptant:<8} {angle_text:>12} "
                    f"{r.gains[1] / r.gains[2]:>9.4g} {r.gains[0] / r.gains[2]:>9.4g} "
                    f"{null_text:<27} {anom}"
                )

    if any(r.stim_delta_relative is not None for r in results):
        print("\nBest genotype for supplied stimulus pair, by adaptant")
        for model in models:
            print(f"  model={model}:")
            for adaptant in adaptants:
                rows = [
                    r for r in results
                    if r.model == model and r.adaptant == adaptant and r.stim_delta_relative is not None
                ]
                rows.sort(key=lambda r: r.stim_delta_relative)
                print(f"    {adaptant}:")
                for r in rows[:5]:
                    print(
                        f"      rank={r.genotype.rank:>2} genotype={r.genotype.label:<11} "
                        f"relative_delta={r.stim_delta_relative:.6g} raw_delta={r.stim_delta_norm:.6g}"
                    )

    print("\nInterpretation note:")
    print("  Under a pure row-gain adaptation model, DISP 0.25 has the same chromatic")
    print("  adaptation ratios and null directions as DISP 0.5; it only changes absolute")
    print("  adapted response scale. If the illuminant_multiply model shifts the null")
    print("  while row_gain does not, that is evidence that the adaptation correction is")
    print("  changing the stimulus spectra rather than only adapting cone gains.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primaries-dir", type=Path, default=None)
    parser.add_argument("--observer-count", type=int, default=10)
    parser.add_argument("--sex", choices=("male", "female", "both"), default="both")
    parser.add_argument("--degree", type=float, default=2.0)
    parser.add_argument("--adaptants", nargs="+", default=list(DEFAULT_ADAPTANTS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("row_gain", "illuminant_multiply"),
        default=["row_gain", "illuminant_multiply"],
    )
    parser.add_argument(
        "--cone-subset",
        default="SML",
        help="Cone rows used for the null direction. Use SML for full trichromat or ML for Rayleigh plane.",
    )
    parser.add_argument("--stim-a", type=parse_rgbo, default=None, help="First presented RGBO, e.g. 0.5,0.5,0,0")
    parser.add_argument("--stim-b", type=parse_rgbo, default=None, help="Second presented RGBO, e.g. 0,0,0,0.5")
    parser.add_argument("--r-max", type=float, default=128)
    parser.add_argument("--g-max", type=float, default=64)
    parser.add_argument("--o-max", type=float, default=255)
    parser.add_argument("--output", type=Path, default=Path("adaptation_null_direction_summary.csv"))
    args = parser.parse_args()

    primaries_dir = args.primaries_dir or latest_primaries_dir()
    primaries = load_primaries_from_csv(str(primaries_dir), primary_order="RGBO")
    wavelengths = primaries[0].wavelengths

    cone_name_to_index = {"S": 0, "M": 1, "L": 2}
    cone_subset = tuple(cone_name_to_index[c] for c in args.cone_subset.upper())
    if len(cone_subset) < 3:
        print("Warning: fewer than 3 cone constraints gives a null plane; reporting one basis vector only.")

    if (args.stim_a is None) != (args.stim_b is None):
        raise SystemExit("Provide both --stim-a and --stim-b, or neither.")

    genotypes = observer_genotypes(args.observer_count, args.sex, wavelengths)
    results = analyze(
        genotypes=genotypes,
        primaries=primaries,
        models=args.models,
        adaptants=args.adaptants,
        cone_subset=cone_subset,
        degree=args.degree,
        stim_a=args.stim_a,
        stim_b=args.stim_b,
        r_max=args.r_max,
        g_max=args.g_max,
        o_max=args.o_max,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = args.adaptants[0]
    write_csv(args.output, results, baseline)

    print(f"Primaries: {primaries_dir}")
    print(f"Wrote: {args.output}")
    print_summary(results, args.models, args.adaptants, baseline)


if __name__ == "__main__":
    main()
