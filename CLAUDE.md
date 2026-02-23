# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda create -n tetriumcolor python=3.11  # open3d requires 3.11
conda activate tetriumcolor
pip install -e /path/to/TetriumColor
```

Additional runtime dependencies not in `setup.py`: `joblib`, `scikit-learn`, `tetrapolyscope`, `pyglm` (see `requirements.txt`).

## Running Scripts

Scripts are standalone and run directly; no build step needed.

```bash
# Run a validation script
python scripts/validation/validate_display_measurements.py --help

# Run a tetrium calibration script
python scripts/tetrium-routines/generateCubeMap.py
```

There is no test suite. `test_validation/` contains integration-style scripts that exercise the full pipeline.

## Architecture

### Core Data Types

- **`Spectra`** (`Observer/Spectra.py`): Spectral power distribution, wrapping `(wavelengths, data)` arrays. Base class for `Cone`.
- **`Cone`** (`Observer/Observer.py`): Single photoreceptor sensitivity curve, with nomogram templates (Neitz, Govardovskii, Stockman-Sharpe, etc.) and pre-receptoral filtering.
- **`Observer`** (`Observer/Observer.py`): A collection of `Cone` sensors forming a complete visual system. Static constructors: `Observer.trichromat()`, `Observer.tetrachromat()`, `Observer.bird(name)`, etc. Sensor order is always `[S, M, (Q,) L]`.
- **`Illuminant`** (`Observer/Spectra.py`): Named illuminants (D65, E, etc.).

### Color Space System

`ColorSpace` (`ColorSpace.py`) is the central class. It takes an `Observer` plus optional display primaries.

**All conversions route through `CONE` space.** The `convert(points, from_space, to_space)` method handles the routing. `ColorSpaceType` is an enum listing all supported spaces:

| Space | Description |
|---|---|
| `CONE` | Raw cone excitations (SMQL order) |
| `MAXBASIS` | Optimal basis spanning gamut |
| `HERING` | Opponent color axes from MaxBasis |
| `VSH` | Value-Saturation-Hue (spherical from Hering) |
| `BGYR` | Spectral step-function basis (cutpoints 493/563/608 nm) |
| `HERING_BGYR` | Hering opponent of BGYR |
| `VSH_BGYR` | Spherical coords of HERING_BGYR |
| `DISP` | Display primary weights (RGBO) |
| `DISP_6P` | 6-channel even/odd display representation |
| `SRGB`, `XYZ`, `OKLAB`, `CIELAB` | Standard tristimulus spaces (3D only) |

Transformation matrices are lazy-computed and cached as `_cone_to_*` attributes.

### MaxBasis

`MaxBasis` (`Observer/MaxBasis.py`) computes the optimal basis for an observer's gamut using the zonotope / optimal reflectance approach. `MaxBasisFactory` caches results in `TetriumColor/Assets/Cache/max-basis-cache.pkl`.

### ColorSampler

`ColorSampler` (`ColorSampler.py`) provides gamut-aware color sampling from a `ColorSpace`. It builds and caches a lookup table (LUT) of gamut boundaries (`gamut_lut_*.pkl` in `Assets/Cache/`). Key method: `get_gamut_lut()`.

### Measurement

`Measurement/` handles hardware interaction:
- `PR650.py`: Photoresearch PR-650 spectroradiometer (serial port, Mac only)
- `MeasurementGUI.py`: Interactive display measurement tool
- `MeasurementRoutines.py`: `MeasurePrimaries()`, `SaveRGBOtoSixChannel()`
- `TetriumMeasurementRoutines.py`: Tetrium-specific routines

`load_primaries_from_csv()` is the main entry point for loading measured display primaries.

### Visualization

`Visualization/` uses Polyscope (`tetrapolyscope`) for 3D/4D color solid rendering. `PolyscopeDisplayType` (in `ColorSpace.py`) enumerates valid display spaces for Polyscope (a subset of `ColorSpaceType`).

### PsychoPhys

`PsychoPhys/` implements psychophysics:
- `Quest.py`: QUEST adaptive threshold procedure
- `IshiharaPlate.py`: Pseudo-isochromatic plate generation
- `HueSphere.py`, `HyperspectralImage.py`

### Scripts

`scripts/` are not part of the installed package:
- `tetrium-routines/`: Generate calibration data and cube maps for the Tetrium display
- `validation/`: Validate display measurements against expected LMSQ responses
- `paper-viz/`, `printing/`, `simulating-population/`: Analysis and visualization

### Caching Architecture

Expensive computations are persisted as pickle files in `TetriumColor/Assets/Cache/`:
- `observer-cache.pkl`: Cached `Observer` objects keyed by `stable_hash(observer)`
- `max-basis-cache.pkl`: Cached `MaxBasis` objects
- `gamut_lut_<hash>.pkl`: Cached gamut LUTs for `ColorSampler`
- `convex_hull_search_cache_<hash>.pkl`: Convex hull search results

The hash for cache keys is computed by `Utils/Hash.py:stable_hash()`.

## Key Design Patterns

- **Observer cone order**: S=0, M=1, Q=2 (tetrachromats only), L=2 or 3. The `metameric_axis` parameter in `ColorSpace` identifies the Q cone index (default=2).
- **DISP_6P / led_mapping**: The Tetrium display uses 6 LEDs in even/odd frames. `led_mapping` (default `[0,1,3,2,1,3]`) maps 6 frame slots to 4 RGBO primaries.
- **Tetrachromat convention**: `Observer.tetrachromat()` uses Stockman-Sharpe tabulated data for S/M/L and a Neitz-template Q cone at 545 nm.
