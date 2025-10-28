#!/usr/bin/env python3
"""
Ink library processing CLI tool.
Convert, combine, and create ink libraries from raw data.
"""

from TetriumColor.Observer.Inks import combine_inksets, save_combined_inkset_to_csv, load_inkset
from TetriumColor.Measurement.Nix import read_nix_csv
from TetriumColor.Observer.Spectra import Spectra
import numpy as np
import pandas as pd
from library_registry import registry
import argparse
import os
import sys
import json
import colorsys
from pathlib import Path
from typing import List, Dict, Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))


def convert_nix_to_csv(input_path: str, library_name: str, output_dir: Optional[str] = None,
                       filter_clogged: bool = True, wavelength_range: str = "400-700") -> str:
    """
    Convert nix files to processed CSV format.

    Args:
        input_path: Path to nix CSV file or directory of nix files
        library_name: Name of target ink library
        output_dir: Output directory (default: data/inksets/{library_name}/)
        filter_clogged: Filter out clogged inks
        wavelength_range: Wavelength range (e.g., "400-700")

    Returns:
        Path to the created CSV file
    """

    # Parse wavelength range
    try:
        start_wl, end_wl = map(int, wavelength_range.split('-'))
        wavelengths = np.arange(start_wl, end_wl + 1, 10)
    except ValueError:
        raise ValueError(f"Invalid wavelength range format: {wavelength_range}. Use format like '400-700'")

    # Set up output directory
    if output_dir is None:
        output_dir = f"data/inksets/{library_name}"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{library_name}-inks.csv")

    # Collect all spectra from input files
    all_spectra = {}
    paper_spectra = None

    input_path = Path(input_path)
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = list(input_path.glob("*.csv"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not input_files:
        raise ValueError(f"No CSV files found in {input_path}")

    print(f"Processing {len(input_files)} input files...")

    for file_path in input_files:
        print(f"Processing: {file_path}")
        try:
            # Read nix CSV
            name_to_spectra, paper = read_nix_csv(str(file_path))

            # Add to collection
            for name, spectra in name_to_spectra.items():
                # Interpolate to standard wavelengths if needed
                if not np.array_equal(spectra.wavelengths, wavelengths):
                    spectra = spectra.interpolate_values(wavelengths)

                all_spectra[name] = spectra

            # Use first paper found
            if paper_spectra is None and paper is not None:
                if not np.array_equal(paper.wavelengths, wavelengths):
                    paper = paper.interpolate_values(wavelengths)
                paper_spectra = paper

        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue

    if not all_spectra:
        raise ValueError("No valid spectra found in input files")

    # Try to find paper if not already set
    if paper_spectra is None:
        # Look for a spectra named "paper" (case insensitive)
        for name, spectra in all_spectra.items():
            if name.lower() == "paper":
                paper_spectra = spectra
                break

    if paper_spectra is None:
        print("Warning: No paper spectra found. Creating library without paper.")

    # Create DataFrame in the standard format
    ink_names = list(all_spectra.keys())
    reflectance_data = np.array([all_spectra[name].data for name in ink_names])

    # Create DataFrame
    df = pd.DataFrame(reflectance_data, index=ink_names, columns=wavelengths)
    df.index.name = "Name"

    # Reset index to make Name a column
    df = df.reset_index()

    # Add Index column
    df.insert(0, 'ID', range(1, len(df) + 1))

    # Add clogged column (all False/0 for converted inksets)
    df['clogged'] = 0

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Register the library
    registry.register_library(
        library_name,
        f"{library_name}/{library_name}-inks.csv",
        {
            "created": "converted_from_nix",
            "input_files": [str(f) for f in input_files],
            "wavelength_range": wavelength_range,
            "filter_clogged": filter_clogged,
            "num_inks": len(all_spectra)
        }
    )

    print(f"Successfully converted {len(all_spectra)} inks to {output_path}")
    return output_path


def _parse_names_file(names_file: str) -> List[str]:
    """
    Parse a text file containing ink names. Supports two common formats:
    1) Simple one-name-per-line
    2) Ranked lists like " 1. Ink Name" possibly with headers.

    Returns a deduplicated list preserving first-seen order.
    """
    import re

    names: List[str] = []
    seen = set()

    with open(names_file, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Skip obvious headers/underlines
            lower = line.lower()
            if lower.startswith('top ') or set(line) == {'='}:
                continue

            # Match numbered format: " 12. Name here"
            m = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if m:
                candidate = m.group(1).strip()
            else:
                candidate = line

            if candidate and candidate not in seen:
                seen.add(candidate)
                names.append(candidate)

    return names


def filter_library_by_names(source_library: str, output_name: str, names_file: str,
                            filter_clogged: bool = True) -> str:
    """
    Create a filtered ink library that contains only the inks listed in names_file.

    Args:
        source_library: Name of an existing library in the registry to filter.
        output_name: Name for the new filtered library (e.g., fp_top40).
        names_file: Path to a text file listing ink names to keep.
        filter_clogged: Whether to filter clogged inks when loading source.

    Returns:
        Path to the created CSV file for the filtered library.
    """
    # Resolve and load the source library
    if not registry.library_exists(source_library):
        raise ValueError(f"Library '{source_library}' not found")

    source_path = registry.resolve_library_path(source_library)
    library, paper, wavelengths = load_inkset(source_path, filter_clogged=filter_clogged)

    # Parse names to keep
    names_to_keep = _parse_names_file(names_file)
    if not names_to_keep:
        raise ValueError(f"No ink names found in {names_file}")

    # Build filtered library, preserving order from names_to_keep
    filtered: Dict[str, Spectra] = {}
    missing: List[str] = []
    for name in names_to_keep:
        if name in library:
            filtered[name] = library[name]
        else:
            missing.append(name)

    if not filtered:
        raise ValueError("None of the requested names were found in the source library")

    # Set up output
    output_dir = f"data/inksets/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}-inks.csv")

    # Create DataFrame in the standard format (including a paper row)
    ink_names = list(filtered.keys())
    reflectance_data = np.array([filtered[name].data for name in ink_names] + [paper.data])

    df = pd.DataFrame(reflectance_data, index=ink_names + ["paper"], columns=wavelengths)
    df.index.name = "Name"
    df = df.reset_index()
    df.insert(0, 'ID', range(1, len(df) + 1))
    df['clogged'] = 0
    df.to_csv(output_path, index=False)

    # Register the new filtered library
    registry.register_library(
        output_name,
        f"{output_name}/{output_name}-inks.csv",
        {
            "created": "filtered",
            "source_library": source_library,
            "names_file": os.path.abspath(names_file),
            "filter_clogged": filter_clogged,
            "requested": len(names_to_keep),
            "kept": len(filtered),
            "missing": missing,
        }
    )

    print(f"Filtered {len(filtered)} inks (missing {len(missing)}) from '{source_library}' into {output_path}")
    if missing:
        print("Missing names not found in source:")
        for n in missing:
            print(f"  - {n}")

    return output_path


def filter_by_color(source_library: str, output_name: str, exclude_colors: List[str],
                    filter_clogged: bool = True, hue_tolerance: float = 30.0,
                    saturation_threshold: float = 0.15, value_threshold: float = 0.15,
                    keep_names: Optional[List[str]] = None) -> str:
    """
    Filter an ink library by removing inks of specified colors based on HSV analysis.

    Args:
        source_library: Name of an existing library in the registry to filter.
        output_name: Name for the new filtered library.
        exclude_colors: List of color names to exclude. Supported: 'blue', 'yellow', 'cyan', 'red', 'green', 'purple'
        filter_clogged: Whether to filter clogged inks when loading source.
        hue_tolerance: How many degrees around the hue center to exclude (default: 30).
        saturation_threshold: Minimum saturation to consider an ink as colored (default: 0.15).
        value_threshold: Minimum value/brightness to consider an ink (default: 0.15).
        keep_names: Optional list of ink names to always keep regardless of color filtering.

    Returns:
        Path to the created CSV file for the filtered library.
    """
    # HSV hue ranges (in degrees, 0-360)
    color_hue_centers = {
        'red': 0,       # 0 degrees (wraps around)
        'yellow': 60,   # 60 degrees
        'green': 120,   # 120 degrees
        'cyan': 180,    # 180 degrees
        'blue': 240,    # 240 degrees (NOT purple)
        'purple': 300,  # 300 degrees (magenta/purple)
    }

    # Validate exclude_colors
    for color in exclude_colors:
        if color not in color_hue_centers:
            raise ValueError(f"Unknown color: {color}. Supported colors: {list(color_hue_centers.keys())}")

    # Resolve and load the source library
    if not registry.library_exists(source_library):
        raise ValueError(f"Library '{source_library}' not found")

    source_path = registry.resolve_library_path(source_library)
    library, paper, wavelengths = load_inkset(source_path, filter_clogged=filter_clogged)

    def is_ink_excluded(ink_name: str, spectra: Spectra) -> bool:
        """Check if an ink should be excluded based on its color."""
        try:
            # Convert spectra to RGB (0-1 range)
            rgb = spectra.to_rgb()

            # Convert to HSV (hue in 0-1, saturation 0-1, value 0-1)
            h, s, v = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

            # Convert hue to degrees (0-360)
            hue_deg = h * 360

            # Skip very desaturated (gray/white/black) inks
            if s < saturation_threshold or v < value_threshold:
                return False

            # Check if hue matches any excluded color
            for color in exclude_colors:
                center = color_hue_centers[color]

                # Handle wraparound for red (0/360 degrees)
                if color == 'red':
                    # Red wraps around 0/360
                    if hue_deg < hue_tolerance or hue_deg > (360 - hue_tolerance):
                        print(f"  Excluding {ink_name}: {color} (H={hue_deg:.1f}°, S={s:.2f}, V={v:.2f})")
                        return True
                else:
                    # Check if within tolerance of the color center
                    if abs(hue_deg - center) < hue_tolerance:
                        print(f"  Excluding {ink_name}: {color} (H={hue_deg:.1f}°, S={s:.2f}, V={v:.2f})")
                        return True

            return False
        except Exception as e:
            print(f"  Warning: Failed to analyze color for {ink_name}: {e}")
            return False

    # Filter the library
    filtered: Dict[str, Spectra] = {}
    excluded_count = 0
    keep_set = set(keep_names) if keep_names else set()

    print(f"Filtering inks, excluding colors: {exclude_colors}")
    if keep_names:
        print(f"Always keeping: {keep_names}")

    for name, spectra in library.items():
        # Always keep whitelisted inks
        if name in keep_set:
            print(f"  Keeping {name} (in whitelist)")
            filtered[name] = spectra
        elif is_ink_excluded(name, spectra):
            excluded_count += 1
        else:
            filtered[name] = spectra

    if not filtered:
        raise ValueError("All inks were filtered out!")

    # Set up output
    output_dir = f"data/inksets/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}-inks.csv")

    # Create DataFrame in the standard format (including paper if available)
    ink_names = list(filtered.keys())
    if paper is not None:
        reflectance_data = np.array([filtered[name].data for name in ink_names] + [paper.data])
        index_list = ink_names + ["paper"]
    else:
        reflectance_data = np.array([filtered[name].data for name in ink_names])
        index_list = ink_names

    df = pd.DataFrame(reflectance_data, index=index_list, columns=wavelengths)
    df.index.name = "Name"
    df = df.reset_index()
    df.insert(0, 'ID', range(1, len(df) + 1))
    df['clogged'] = 0
    df.to_csv(output_path, index=False)

    # Register the new filtered library
    registry.register_library(
        output_name,
        f"{output_name}/{output_name}-inks.csv",
        {
            "created": "color_filtered",
            "source_library": source_library,
            "excluded_colors": exclude_colors,
            "keep_names": keep_names,
            "filter_clogged": filter_clogged,
            "original_count": len(library),
            "kept": len(filtered),
            "excluded": excluded_count,
        }
    )

    print(f"\nFiltered library saved to {output_path}")
    print(f"Kept {len(filtered)} inks, excluded {excluded_count} inks")

    return output_path


def combine_libraries(library_names: List[str], output_name: str,
                      prefixes: Optional[List[str]] = None, paper_source: str = "first",
                      filter_clogged: bool = True) -> str:
    """
    Combine multiple ink libraries into one.

    Args:
        library_names: List of library names to combine
        output_name: Name for combined library
        prefixes: Optional prefixes for each library
        paper_source: Which library to use for paper ("first", "last", or library name)
        filter_clogged: Filter out clogged inks

    Returns:
        Path to the created CSV file
    """

    # Resolve library paths
    inkset_paths = []
    for name in library_names:
        if not registry.library_exists(name):
            raise ValueError(f"Library '{name}' not found")
        inkset_paths.append(registry.resolve_library_path(name))

    # Set up output path
    output_dir = f"data/inksets/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}-inks.csv")

    # Combine inksets
    combined_inkset = combine_inksets(
        inkset_paths=inkset_paths,
        output_path=output_path,
        name_prefixes=prefixes,
        paper_source=paper_source,
        filter_clogged=filter_clogged
    )

    # Register the combined library
    registry.register_library(
        output_name,
        f"{output_name}/{output_name}-inks.csv",
        {
            "created": "combined",
            "source_libraries": library_names,
            "prefixes": prefixes,
            "paper_source": paper_source,
            "filter_clogged": filter_clogged,
            "num_inks": len(combined_inkset.library)
        }
    )

    print(f"Successfully combined {len(library_names)} libraries into {output_path}")
    return output_path


def create_library_from_files(library_name: str, input_files: List[str],
                              paper_file: Optional[str] = None, format_type: str = "auto",
                              filter_clogged: bool = True) -> str:
    """
    Create new ink library from scratch.

    Args:
        library_name: Name for new library
        input_files: List of input files (nix, CSV, or mat files)
        paper_file: Specific file containing paper spectra
        format_type: Input format (auto-detect, nix, csv, mat)
        filter_clogged: Filter out clogged inks

    Returns:
        Path to the created CSV file
    """

    # Set up output path
    output_dir = f"data/inksets/{library_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{library_name}-inks.csv")

    all_spectra = {}
    paper_spectra = None

    print(f"Processing {len(input_files)} input files...")

    for file_path in input_files:
        print(f"Processing: {file_path}")

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            if format_type == "nix" or (format_type == "auto" and file_path.endswith('.csv')):
                # Try to read as nix CSV
                name_to_spectra, paper = read_nix_csv(file_path)

                for name, spectra in name_to_spectra.items():
                    all_spectra[name] = spectra

                if paper_spectra is None and paper is not None:
                    paper_spectra = paper

            elif format_type == "csv" or (format_type == "auto" and file_path.endswith('.csv')):
                # Try to read as standard CSV
                library, paper, wavelengths = load_inkset(file_path, filter_clogged=filter_clogged)

                for name, spectra in library.items():
                    all_spectra[name] = spectra

                if paper_spectra is None:
                    paper_spectra = paper

            elif format_type == "mat" or (format_type == "auto" and file_path.endswith('.mat')):
                # Handle .mat files (similar to existing combine_data.py)
                from scipy.io import loadmat
                data = loadmat(file_path)
                # This would need to be customized based on the .mat file structure
                print(f"Warning: .mat file processing not yet implemented for {file_path}")
                continue

        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue

    # Handle paper file if specified
    if paper_file and os.path.exists(paper_file):
        try:
            if paper_file.endswith('.csv'):
                name_to_spectra, paper = read_nix_csv(paper_file)
                if paper is not None:
                    paper_spectra = paper
                else:
                    # Look for paper in the spectra
                    for name, spectra in name_to_spectra.items():
                        if name.lower() == "paper":
                            paper_spectra = spectra
                            break
        except Exception as e:
            print(f"Warning: Failed to process paper file {paper_file}: {e}")

    if not all_spectra:
        raise ValueError("No valid spectra found in input files")

    # Try to find paper if not already set
    if paper_spectra is None:
        # Look for a spectra named "paper" (case insensitive)
        for name, spectra in all_spectra.items():
            if name.lower() == "paper":
                paper_spectra = spectra
                break

    # Create DataFrame - paper is optional
    ink_names = list(all_spectra.keys())

    # Get wavelengths from paper if available, otherwise from first ink
    if paper_spectra is not None:
        wavelengths = paper_spectra.wavelengths
        reflectance_data = np.array([all_spectra[name].data for name in ink_names] + [paper_spectra.data])
        index_list = ink_names + ["paper"]
    else:
        print("Warning: No paper spectra found. Creating library without paper.")
        wavelengths = next(iter(all_spectra.values())).wavelengths
        reflectance_data = np.array([all_spectra[name].data for name in ink_names])
        index_list = ink_names

    # Remove trailing spaces from all ink_names
    ink_names = [name.rstrip() for name in ink_names]
    index_list = [name.rstrip() for name in index_list]

    # Create DataFrame
    df = pd.DataFrame(reflectance_data, index=index_list, columns=wavelengths)
    df.index.name = "Name"

    # Reset index to make Name a column
    df = df.reset_index()

    # Add Index column
    df.insert(0, 'ID', range(1, len(df) + 1))

    # Add clogged column (all False/0 for created inksets)
    df['clogged'] = 0

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Register the library
    registry.register_library(
        library_name,
        f"{library_name}/{library_name}-inks.csv",
        {
            "created": "created_from_files",
            "input_files": input_files,
            "paper_file": paper_file,
            "format_type": format_type,
            "filter_clogged": filter_clogged,
            "num_inks": len(all_spectra)
        }
    )

    print(f"Successfully created library with {len(all_spectra)} inks: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Process ink libraries: convert, combine, and create')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert nix files to processed CSV format')
    convert_parser.add_argument('input_path', help='Path to nix CSV file or directory of nix files')
    convert_parser.add_argument('library_name', help='Name of target ink library')
    convert_parser.add_argument('--output-dir', help='Output directory (default: data/inksets/{library_name}/)')
    convert_parser.add_argument('--filter-clogged', action='store_true', default=True,
                                help='Filter out clogged inks (default: True)')
    convert_parser.add_argument('--no-filter-clogged', action='store_true', help='Do not filter out clogged inks')
    convert_parser.add_argument('--wavelength-range', default='400-700', help='Wavelength range (default: 400-700)')

    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple ink libraries into one')
    combine_parser.add_argument('library_names', help='Comma-separated list of library names to combine')
    combine_parser.add_argument('output_name', help='Name for combined library')
    combine_parser.add_argument('--prefixes', help='Comma-separated prefixes for each library (optional)')
    combine_parser.add_argument('--paper-source', default='first',
                                help='Which library to use for paper: first, last, or library name')
    combine_parser.add_argument('--filter-clogged', action='store_true', default=True,
                                help='Filter out clogged inks (default: True)')
    combine_parser.add_argument('--no-filter-clogged', action='store_true', help='Do not filter out clogged inks')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new ink library from scratch')
    create_parser.add_argument('library_name', help='Name for new library')
    create_parser.add_argument('input_files', help='Comma-separated list of input files (nix, CSV, or mat files)')
    create_parser.add_argument('--paper-file', help='Specific file containing paper spectra')
    create_parser.add_argument('--format', default='auto',
                               choices=['auto', 'nix', 'csv', 'mat'], help='Input format (default: auto-detect)')
    create_parser.add_argument('--filter-clogged', action='store_true', default=True,
                               help='Filter out clogged inks (default: True)')
    create_parser.add_argument('--no-filter-clogged', action='store_true', help='Do not filter out clogged inks')

    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter an ink library to a subset by names file')
    filter_parser.add_argument('source_library', help='Existing library name in registry to filter')
    filter_parser.add_argument('names_file', help='Path to a text file of ink names to keep')
    filter_parser.add_argument('output_name', help='Name for the new filtered library')
    filter_parser.add_argument('--filter-clogged', action='store_true', default=True,
                               help='Filter out clogged inks when loading source (default: True)')
    filter_parser.add_argument('--no-filter-clogged', action='store_true', help='Do not filter out clogged inks')

    # Filter by color command
    filter_color_parser = subparsers.add_parser(
        'filter-color', help='Filter an ink library by removing specific colors')
    filter_color_parser.add_argument('source_library', help='Existing library name in registry to filter')
    filter_color_parser.add_argument('output_name', help='Name for the new filtered library')
    filter_color_parser.add_argument('--exclude', required=True,
                                     help='Comma-separated list of colors to exclude (blue, yellow, cyan, red, green, purple)')
    filter_color_parser.add_argument('--keep',
                                     help='Comma-separated list of ink names to always keep (e.g., C255,Y255)')
    filter_color_parser.add_argument('--hue-tolerance', type=float, default=30.0,
                                     help='Degrees around hue center to exclude (default: 30.0)')
    filter_color_parser.add_argument('--saturation-threshold', type=float, default=0.15,
                                     help='Minimum saturation to consider colored (default: 0.15)')
    filter_color_parser.add_argument('--value-threshold', type=float, default=0.15,
                                     help='Minimum brightness to consider (default: 0.15)')
    filter_color_parser.add_argument('--filter-clogged', action='store_true', default=True,
                                     help='Filter out clogged inks when loading source (default: True)')
    filter_color_parser.add_argument('--no-filter-clogged', action='store_true', help='Do not filter out clogged inks')

    # List command
    list_parser = subparsers.add_parser('list', help='List all available ink libraries')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a library from the registry (does not delete files)')
    delete_parser.add_argument('library_name', help='Name of the library to remove from registry')

    # Rename command
    rename_parser = subparsers.add_parser('rename', help='Rename a library in the registry (does not move files)')
    rename_parser.add_argument('old_name', help='Existing library name in registry')
    rename_parser.add_argument('new_name', help='New library name to use in registry')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Handle filter_clogged logic
    if hasattr(args, 'no_filter_clogged') and args.no_filter_clogged:
        filter_clogged = False
    elif hasattr(args, 'filter_clogged'):
        filter_clogged = args.filter_clogged
    else:
        filter_clogged = True

    try:
        if args.command == 'convert':
            convert_nix_to_csv(
                args.input_path,
                args.library_name,
                args.output_dir,
                filter_clogged,
                args.wavelength_range
            )

        elif args.command == 'combine':
            library_names = [name.strip() for name in args.library_names.split(',')]
            prefixes = None
            if args.prefixes:
                prefixes = [p.strip() for p in args.prefixes.split(',')]
                if len(prefixes) != len(library_names):
                    print(
                        f"Error: Number of prefixes ({len(prefixes)}) must match number of libraries ({len(library_names)})")
                    sys.exit(1)

            combine_libraries(
                library_names,
                args.output_name,
                prefixes,
                args.paper_source,
                filter_clogged
            )

        elif args.command == 'create':
            input_files = [f.strip() for f in args.input_files.split(',')]
            create_library_from_files(
                args.library_name,
                input_files,
                args.paper_file,
                args.format,
                filter_clogged
            )

        elif args.command == 'filter':
            filter_library_by_names(
                args.source_library,
                args.output_name,
                args.names_file,
                filter_clogged
            )

        elif args.command == 'filter-color':
            exclude_colors = [c.strip().lower() for c in args.exclude.split(',')]
            keep_names = [n.strip() for n in args.keep.split(',')] if args.keep else None
            filter_by_color(
                args.source_library,
                args.output_name,
                exclude_colors,
                filter_clogged,
                args.hue_tolerance,
                args.saturation_threshold,
                args.value_threshold,
                keep_names
            )

        elif args.command == 'list':
            # Auto-discover libraries if registry is empty
            if not registry.list_libraries():
                print("Auto-discovering ink libraries...")
                registry.auto_discover_libraries()

            libraries = registry.list_libraries()
            if libraries:
                print("Available ink libraries:")
                for lib in libraries:
                    metadata = registry.get_library_metadata(lib)
                    num_inks = metadata.get('num_inks', 'unknown')
                    created = metadata.get('created', 'unknown')
                    print(f"  {lib}: {num_inks} inks ({created})")
            else:
                print("No ink libraries found. Use 'ink_processing create' to create a new library.")

        elif args.command == 'delete':
            if not registry.library_exists(args.library_name):
                print(f"Library '{args.library_name}' not found in registry.")
                sys.exit(1)
            if registry.delete_library(args.library_name):
                print(f"Removed '{args.library_name}' from registry. Files were not deleted.")
            else:
                print(f"Failed to remove '{args.library_name}'.")

        elif args.command == 'rename':
            if not registry.library_exists(args.old_name):
                print(f"Library '{args.old_name}' not found in registry.")
                sys.exit(1)
            if registry.library_exists(args.new_name):
                print(f"A library named '{args.new_name}' already exists in registry.")
                sys.exit(1)
            if registry.rename_library(args.old_name, args.new_name):
                print(f"Renamed library '{args.old_name}' to '{args.new_name}'. Files were not moved.")
            else:
                print(f"Failed to rename '{args.old_name}' to '{args.new_name}'.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
