from __future__ import annotations

import csv
import re
from typing import Dict, Optional, Tuple, List

import numpy as np

from TetriumColor.Observer.Spectra import Spectra


def read_nix_csv(csv_path: str) -> Tuple[Dict[str, Spectra], Optional[Spectra]]:
    """Parse a Nix-exported CSV and return a mapping of name->Spectra and an optional paper Spectra.

    The CSV is expected to include columns named like "R400 nm", "R410 nm", ..., "R700 nm" and one or more
    name columns such as "User Color Name" or "Original Color Name". Rows before the header (including a
    possible leading "sep=;" and metadata lines) are ignored.

    Args:
        csv_path: Absolute or relative path to the CSV exported from a Nix device.

    Returns:
        (name_to_spectra, paper_spectra)
        - name_to_spectra: Dict mapping the chosen row name to its Spectra
        - paper_spectra: The Spectra for the row whose name is exactly "paper" (case-insensitive), or None
    """
    name_to_spectra: Dict[str, Spectra] = {}
    paper_spectra: Optional[Spectra] = None

    # Regex to match spectral columns like "R400 nm"
    spectral_col_pattern = re.compile(r"^R\s*(\d{3})\s*nm$", re.IGNORECASE)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")

        header: Optional[List[str]] = None
        spectral_indices: List[Tuple[int, int]] = []  # (col_idx, wavelength_nm)

        # Identify header row containing spectral columns
        for row in reader:
            if not row:
                continue
            # Some Nix files begin with a sep=; line; skip it
            if len(row) == 1 and row[0].strip().lower().startswith("sep="):
                continue

            # Determine whether this row is the header by searching for R*** nm columns
            candidate_spectral_indices: List[Tuple[int, int]] = []
            for idx, col in enumerate(row):
                m = spectral_col_pattern.match(col.strip())
                if m:
                    candidate_spectral_indices.append((idx, int(m.group(1))))

            if candidate_spectral_indices:
                header = [c.strip() for c in row]
                spectral_indices = candidate_spectral_indices
                break
            # Otherwise keep scanning until we find the header

        if header is None or not spectral_indices:
            raise ValueError("Could not find a header row with spectral columns like 'R400 nm' in the CSV.")

        # Determine name column preference order
        preferred_name_columns = [
            "User Color Name",
            "Original Color Name",
            "Saved Collection Name",
            "Original Library Name",
            "Scan Type",
        ]
        name_col_idx: Optional[int] = None
        for col_name in preferred_name_columns:
            try:
                name_col_idx = header.index(col_name)
                break
            except ValueError:
                continue

        # If none of the preferred columns exist, fall back to any non-spectral column
        if name_col_idx is None:
            non_spectral_candidates = [
                i for i, col in enumerate(header)
                if not spectral_col_pattern.match(col)
            ]
            if not non_spectral_candidates:
                raise ValueError("No suitable name column found in CSV header.")
            name_col_idx = non_spectral_candidates[0]

        # Sort spectral indices by wavelength to ensure ascending order
        spectral_indices.sort(key=lambda x: x[1])
        wavelengths = np.array([w for _, w in spectral_indices], dtype=float)

        # Parse data rows
        row_counter = 0
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # Defensive: pad short rows
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))

            raw_name = row[name_col_idx].strip()
            # Skip placeholder names if entirely empty and attempt to construct a fallback
            if raw_name in ("", "-"):
                raw_name = f"sample_{row_counter}"

            # Extract reflectance values
            reflectances: List[float] = []
            valid_row = True
            for col_idx, _wv in spectral_indices:
                val = row[col_idx].strip()
                if val == "":
                    valid_row = False
                    break
                try:
                    # Some locales might use comma decimal; try both
                    try:
                        fv = float(val)
                    except ValueError:
                        fv = float(val.replace(",", "."))
                except ValueError:
                    valid_row = False
                    break
                reflectances.append(fv)

            if not valid_row:
                continue

            data = np.array(reflectances, dtype=float)
            # Clip to [0,1] as these are reflectances
            data = np.clip(data, 0.0, 1.0)

            spectra = Spectra(wavelengths=wavelengths, data=data, normalized=True)

            # Ensure unique keys if duplicates occur
            name = raw_name.strip()
            if name in name_to_spectra:
                suffix = 2
                while f"{name}_{suffix}" in name_to_spectra:
                    suffix += 1
                name = f"{name}_{suffix}"

            name_to_spectra[name] = spectra

            if paper_spectra is None and raw_name.strip().lower() == "paper":
                paper_spectra = spectra

            row_counter += 1

    return name_to_spectra, paper_spectra
