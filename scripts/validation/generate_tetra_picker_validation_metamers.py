#!/usr/bin/env python3
"""
Generate display-validation measurements from TetraColorPicker.QuestColorGenerator.

This is intentionally different from generate_display_validation_metamers.py:
the DISP values are not reconstructed analytically here. They are read from the
trial metadata recorded by QuestColorGenerator, so the JSON captures the exact
stimuli that the picker would hand to plate generation when color_picking_space
is cone_contrast.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from TetriumColor.ColorSpace import ColorSpaceType
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.TetraColorPicker import QuestColorGenerator

from generate_display_validation_metamers import (
    _nominal_observer_params,
    _reconstruct_spectrum,
    monte_carlo_metamer_robustness,
)


def _as_color_space_type(value: str) -> ColorSpaceType:
    return ColorSpaceType(value.lower())


def _probability_for_picker_genotype(pdf: dict, genotype: tuple, peak_to_test: float) -> float:
    """Return the population probability for a QuestColorGenerator metadata genotype."""
    if genotype in pdf:
        return float(pdf[genotype])

    candidates = []
    genotype_list = list(genotype)
    if peak_to_test in genotype_list:
        reduced = genotype_list.copy()
        reduced.remove(peak_to_test)
        candidates.append(tuple(reduced))

    candidates.extend(tuple(v for i, v in enumerate(genotype) if i != j) for j in range(len(genotype)))

    for candidate in candidates:
        if candidate in pdf:
            return float(pdf[candidate])
    return 0.0


def _format_space(space: ColorSpaceType) -> str:
    return str(space).split(".")[-1]


def generate_picker_metamers(
    output: str,
    primaries_path: str,
    num_observers: int = 8,
    sex: str = "both",
    percentage_screened: float = 0.999,
    dimensions: list[int] | None = None,
    metameric_axis: int = 2,
    peak_to_test: float = 547,
    seed: int = 42,
    degree: float = 4.0,
    trials_per_direction: int = 1,
    mcs_k: int = 1,
    bipolar: bool = False,
    background_luminance: float = 0.5,
    adapting_background: list[float] | None = None,
    adapting_background_space: ColorSpaceType = ColorSpaceType.DISP,
    mc_samples: int = 1000,
) -> dict[str, Any]:
    if dimensions is None:
        dimensions = [3]
    if mcs_k <= 0:
        raise ValueError(
            "This exporter requires --mcs-k > 0 so QuestColorGenerator can emit a "
            "deterministic list of picker trials without simulated observer responses."
        )

    display_primaries = load_primaries_from_csv(primaries_path)
    wavelengths = display_primaries[0].wavelengths
    dim = dimensions[0] + 1
    background = np.asarray(
        adapting_background if adapting_background is not None else np.ones(dim) * background_luminance,
        dtype=float,
    )
    observer_indices = list(range(num_observers))

    generator = QuestColorGenerator(
        sex=sex,
        percentage_screened=percentage_screened,
        peak_to_test=peak_to_test,
        luminance=background_luminance,
        dimensions=dimensions,
        seed=seed,
        trials_per_direction=trials_per_direction,
        metameric_axes=[metameric_axis],
        bipolar=bipolar,
        degree=degree,
        mcs_k=mcs_k,
        observer_indices=observer_indices,
        color_picking_space="cone_contrast",
        adapting_background=background,
        adapting_background_space=adapting_background_space,
        display_primaries=display_primaries,
        wavelengths=wavelengths,
        illuminant="raw",
        template="baylor",
    )

    expected_trials = generator.get_num_samples()
    observers_by_genotype: dict[tuple, dict[str, Any]] = {}
    pdf = generator.observer_genotypes.get_pdf(sex)
    nom = _nominal_observer_params(degree=degree)

    result = generator.NewColor()
    trial_count = 0
    while result is not None:
        inside_cone, outside_cone, color_space, proportion = result
        trial_metadata = generator.GetCurrentTrialMetadata()
        direction_metadata = generator.direction_metadata[trial_metadata["quest_direction_idx"]]

        genotype = tuple(direction_metadata["genotype"])
        observer_metameric_axis = int(direction_metadata["metameric_axis"])
        metameric_peak = sorted(set([420] + list(genotype)))[observer_metameric_axis]

        inside_disp = np.asarray(trial_metadata["quest_inside_disp"], dtype=float)
        outside_disp = np.asarray(trial_metadata["quest_outside_disp"], dtype=float)
        bgyr_pair = color_space.convert(
            np.stack([inside_disp, outside_disp]),
            ColorSpaceType.DISP,
            ColorSpaceType.BGYR,
        )

        picker_cone_1 = np.asarray(inside_cone, dtype=float)
        picker_cone_2 = np.asarray(outside_cone, dtype=float)
        raw_display_to_cone = color_space.get_raw_display_to_cone_matrix()
        raw_cone_1 = raw_display_to_cone @ inside_disp
        raw_cone_2 = raw_display_to_cone @ outside_disp
        raw_cone_delta = raw_cone_1 - raw_cone_2
        cone_contrast_delta = raw_cone_delta

        avg = np.maximum((raw_cone_1 + raw_cone_2) / 2.0, 1e-10)
        delta = raw_cone_delta / avg
        lms_idx = [i for i in range(len(raw_cone_1)) if i != observer_metameric_axis]
        nominal_d_lms = float(np.sqrt(np.sum(delta[lms_idx] ** 2)))

        spec_1 = _reconstruct_spectrum(inside_disp, display_primaries, wavelengths)
        spec_2 = _reconstruct_spectrum(outside_disp, display_primaries, wavelengths)
        mc_result = monte_carlo_metamer_robustness(
            peaks=genotype,
            wavelengths=wavelengths,
            spectrum_1_data=spec_1,
            spectrum_2_data=spec_2,
            n_samples=mc_samples,
            metameric_axis=observer_metameric_axis,
            illuminant_data="raw",
            template="baylor",
            seed=seed + trial_count,
            mpod_mean=nom["mpod"],
            lens_mean=nom["lens"],
            od_lm_mean=nom["od_lm"],
            od_s_mean=nom["od_s"],
        )

        observer_record = observers_by_genotype.setdefault(
            genotype,
            {
                "observer_index": len(observers_by_genotype),
                "genotype": list(genotype),
                "probability": _probability_for_picker_genotype(pdf, genotype, peak_to_test),
                "requested_metameric_axis": int(metameric_axis),
                "metameric_axis": observer_metameric_axis,
                "metameric_peak_nm": float(metameric_peak),
                "metamers": [],
            },
        )

        pair_index = len(observer_record["metamers"])
        observer_record["metamers"].append(
            {
                "pair_index": pair_index,
                "grid_position": [pair_index, 0],
                "cone_1": raw_cone_1.tolist(),
                "cone_2": raw_cone_2.tolist(),
                "raw_cone_1": raw_cone_1.tolist(),
                "raw_cone_2": raw_cone_2.tolist(),
                "raw_cone_delta": raw_cone_delta.tolist(),
                "picker_cone_1": picker_cone_1.tolist(),
                "picker_cone_2": picker_cone_2.tolist(),
                "metamer_difference": float(abs(raw_cone_delta[observer_metameric_axis])),
                "requested_metameric_axis": int(metameric_axis),
                "metameric_axis": observer_metameric_axis,
                "metameric_peak_nm": float(metameric_peak),
                "rgbo_1": inside_disp.tolist(),
                "rgbo_2": outside_disp.tolist(),
                "bgyr_1": bgyr_pair[0].tolist(),
                "bgyr_2": bgyr_pair[1].tolist(),
                "nominal_d_lms": nominal_d_lms,
                "observer_variability": mc_result,
                "quest_direction_idx": int(trial_metadata["quest_direction_idx"]),
                "quest_proportion": float(proportion),
                "quest_bipolar": bool(trial_metadata["quest_bipolar"]),
                "quest_color_picking_space": trial_metadata["quest_color_picking_space"],
                "quest_cone_contrast_delta": cone_contrast_delta.tolist(),
                "quest_raw_cone_delta": raw_cone_delta.tolist(),
            }
        )

        trial_count += 1
        if trial_count >= expected_trials:
            break
        result = generator.GetColor(None)

    observers_data = sorted(
        observers_by_genotype.values(),
        key=lambda obs: (-float(obs.get("probability", 0.0)), tuple(obs.get("genotype", []))),
    )
    for observer_index, observer_record in enumerate(observers_data):
        observer_record["observer_index"] = observer_index
    primaries_serialized = [
        {
            "wavelengths": primary.wavelengths.tolist(),
            "data": primary.data.tolist(),
        }
        for primary in display_primaries
    ]

    output_data = {
        "metadata": {
            "num_observers": num_observers,
            "num_pairs_per_observer": (
                len(observers_data[0]["metamers"]) if observers_data else 0
            ),
            "sex": sex,
            "grid_size": 1,
            "use_display_midpoint": False,
            "proportion": None,
            "luminance": background_luminance,
            "saturation": None,
            "cube_face": None,
            "requested_metameric_axis": metameric_axis,
            "metameric_axis": metameric_axis,
            "metameric_axis_note": (
                "Legacy fallback only. Each observer/metamer stores the resolved "
                "QuestColorGenerator axis as metameric_axis."
            ),
            "metameric_peak_nm": peak_to_test,
            "observer_degree": degree,
            "observer_template": "baylor",
            "observer_illuminant": "raw",
            "seed": seed,
            "sampling_space": "RGBO",
            "storage_space": "BGYR",
            "color_picking_space": "cone_contrast",
            "cone_contrast_background": background.tolist(),
            "cone_contrast_background_space": _format_space(adapting_background_space),
            "cone_contrast_basis": "QuestColorGenerator via ColorSpace.get_cone_contrast_null_direction_in_disp",
            "used_display_primaries": True,
            "primaries_path": str(primaries_path),
            "primaries_order": "RGBO",
            "wavelength_range": [int(wavelengths[0]), int(wavelengths[-1])],
            "total_metamer_pairs": sum(len(obs["metamers"]) for obs in observers_data),
            "description": (
                "Validation stimuli captured from TetraColorPicker.QuestColorGenerator "
                "with color_picking_space=cone_contrast. rgbo_1/rgbo_2 are the exact "
                "DISP values recorded in Quest trial metadata."
            ),
            "method": (
                "QuestColorGenerator.NewColor()/GetColor() with mcs_k="
                f"{mcs_k}, bipolar={bipolar}; DISP values from GetCurrentTrialMetadata()."
            ),
            "percentage_screened": percentage_screened,
            "dimensions": dimensions,
            "trials_per_direction": trials_per_direction,
            "mcs_k": mcs_k,
            "bipolar": bipolar,
            "mc_samples": mc_samples,
            "mc_params": {
                "note": f"means are degree-adjusted to match nominal observer at {degree:g} deg",
                "mpod": {"mean": nom["mpod"], "std": 0.25, "min": 0.0, "max": 2.0},
                "lens": {"mean": nom["lens"], "half_range_frac": 0.25},
                "od_lm": {"mean": nom["od_lm"], "half_range": 0.1},
                "od_s": {"mean": nom["od_s"], "half_range": 0.1},
            },
        },
        "display_primaries": primaries_serialized,
        "observers": observers_data,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation JSON from TetraColorPicker Quest cone_contrast DISP values"
    )
    parser.add_argument("--output", default="config/display_validation_metamers_tetra_picker.json")
    parser.add_argument("--primaries-path", required=True)
    parser.add_argument("--num-observers", type=int, default=10)
    parser.add_argument("--sex", choices=["male", "female", "both"], default="both")
    parser.add_argument("--percentage-screened", type=float, default=0.999)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3])
    parser.add_argument("--metameric-axis", type=int, default=2)
    parser.add_argument("--peak-to-test", type=float, default=547)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degree", type=float, default=2.0)
    parser.add_argument("--trials-per-direction", type=int, default=1)
    parser.add_argument("--mcs-k", type=int, default=1)
    parser.add_argument("--bipolar", action="store_true")
    parser.add_argument("--background-luminance", type=float, default=0.5)
    parser.add_argument("--adapting-background", type=float, nargs="+", default=None)
    parser.add_argument("--adapting-background-space", type=_as_color_space_type, default=ColorSpaceType.DISP)
    parser.add_argument("--mc-samples", type=int, default=1000)
    args = parser.parse_args()

    output = generate_picker_metamers(
        output=args.output,
        primaries_path=args.primaries_path,
        num_observers=args.num_observers,
        sex=args.sex,
        percentage_screened=args.percentage_screened,
        dimensions=args.dimensions,
        metameric_axis=args.metameric_axis,
        peak_to_test=args.peak_to_test,
        seed=args.seed,
        degree=args.degree,
        trials_per_direction=args.trials_per_direction,
        mcs_k=args.mcs_k,
        bipolar=args.bipolar,
        background_luminance=args.background_luminance,
        adapting_background=args.adapting_background,
        adapting_background_space=args.adapting_background_space,
        mc_samples=args.mc_samples,
    )
    print(f"Saved picker validation configuration to: {args.output}")
    print(f"Total observers: {len(output['observers'])}")
    print(f"Total metamer pairs: {output['metadata']['total_metamer_pairs']}")


if __name__ == "__main__":
    main()
