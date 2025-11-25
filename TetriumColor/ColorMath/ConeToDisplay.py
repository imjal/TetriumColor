"""
Flexible methods for mapping cone responses to display primaries.

This module provides various strategies for computing CONE->DISP transformations,
supporting different scenarios like equal dimensions, overcomplete systems, and
optimization-based primary selection.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import minimize

from TetriumColor.Observer import Observer, Spectra


def compute_cone_to_display_direct(
    observer: Observer,
    display_primaries: List[Spectra],
    white_point_normalize: bool = True,
    scaling_factor: float = 10000
) -> Tuple[npt.NDArray, Dict[str, Any]]:
    """
    Direct matrix inverse when N cones = N primaries.

    Args:
        observer: Observer model
        display_primaries: List of display primary spectra
        white_point_normalize: Whether to normalize to white point
        scaling_factor: Scaling factor for display measurements

    Returns:
        Tuple of (cone_to_disp_matrix, metadata_dict)

    Raises:
        ValueError: If number of primaries doesn't match observer dimension
    """
    n_cones = observer.dimension
    n_primaries = len(display_primaries)

    if n_cones != n_primaries:
        raise ValueError(
            f"Direct method requires equal dimensions: "
            f"{n_cones} cones but {n_primaries} primaries"
        )

    # Observe the primaries
    disp_responses = observer.observe_spectras(display_primaries)
    intensities = disp_responses.T * scaling_factor

    # Compute white point
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))

    # Compute weights to achieve white point
    white_weights = np.linalg.inv(intensities) @ white_pt

    # Rescale intensities so white point is achieved at equal weights
    if white_point_normalize:
        new_intensities = intensities * np.max(white_weights)
        cone_to_disp = np.linalg.inv(new_intensities)
        rescaled_white_weights = np.ones(n_cones)
    else:
        cone_to_disp = np.linalg.inv(intensities)
        rescaled_white_weights = white_weights

    metadata = {
        'method': 'direct',
        'white_weights': rescaled_white_weights,
        'n_primaries': n_primaries,
        'primary_indices': list(range(n_primaries))
    }

    return cone_to_disp, metadata


def compute_cone_to_display_lsq(
    observer: Observer,
    display_primaries: List[Spectra],
    white_point_normalize: bool = True,
    scaling_factor: float = 100
) -> Tuple[npt.NDArray, Dict[str, Any]]:
    """
    Least-squares/pseudoinverse for overcomplete systems (N primaries > M cones).

    Args:
        observer: Observer model
        display_primaries: List of display primary spectra
        white_point_normalize: Whether to normalize to white point
        scaling_factor: Scaling factor for display measurements

    Returns:
        Tuple of (cone_to_disp_matrix, metadata_dict)

    Raises:
        ValueError: If number of primaries is less than observer dimension
    """
    n_cones = observer.dimension
    n_primaries = len(display_primaries)

    if n_primaries < n_cones:
        raise ValueError(
            f"LSQ method requires at least as many primaries as cones: "
            f"{n_cones} cones but only {n_primaries} primaries"
        )

    # Observe the primaries
    disp_responses = observer.observe_spectras(display_primaries)
    intensities = disp_responses.T * scaling_factor

    # Compute white point
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))

    # Use pseudoinverse for least-squares solution
    cone_to_disp = np.linalg.pinv(intensities)

    # Compute white weights
    white_weights = cone_to_disp @ white_pt

    # Normalize if requested
    if white_point_normalize:
        scaling = np.max(white_weights)
        cone_to_disp = cone_to_disp / scaling
        rescaled_white_weights = white_weights / scaling
    else:
        rescaled_white_weights = white_weights

    metadata = {
        'method': 'lsq',
        'white_weights': rescaled_white_weights,
        'n_primaries': n_primaries,
        'primary_indices': list(range(n_primaries))
    }

    return cone_to_disp, metadata


def compute_cone_to_display_optimized(
    observer: Observer,
    display_primaries: List[Spectra],
    target_cone_idx: int = 0,
    maximize: bool = True,
    n_primaries_to_select: Optional[int] = None,
    white_point_normalize: bool = True,
    scaling_factor: float = 100
) -> Tuple[npt.NDArray, Dict[str, Any]]:
    """
    Optimization-based primary selection (e.g., maximize L-cone contrast).

    Selects a subset of primaries that optimize contrast in a target cone type.

    Args:
        observer: Observer model
        display_primaries: List of display primary spectra
        target_cone_idx: Index of cone to optimize (0=S, 1=M, 2=Q, 3=L for tetrachromats)
        maximize: If True, maximize contrast; if False, minimize
        n_primaries_to_select: Number of primaries to select (defaults to observer.dimension)
        white_point_normalize: Whether to normalize to white point
        scaling_factor: Scaling factor for display measurements

    Returns:
        Tuple of (cone_to_disp_matrix, metadata_dict)

    Raises:
        ValueError: If insufficient primaries available
    """
    n_cones = observer.dimension
    n_primaries = len(display_primaries)

    if n_primaries_to_select is None:
        n_primaries_to_select = n_cones

    if n_primaries < n_primaries_to_select:
        raise ValueError(
            f"Need at least {n_primaries_to_select} primaries, "
            f"but only {n_primaries} available"
        )

    if n_primaries_to_select < n_cones:
        raise ValueError(
            f"Cannot select fewer primaries ({n_primaries_to_select}) "
            f"than cones ({n_cones})"
        )

    # Observe all primaries
    all_responses = observer.observe_spectras(display_primaries)

    # If selecting exactly the number of cones, try all combinations
    if n_primaries_to_select == n_cones:
        from itertools import combinations

        best_score = -np.inf if maximize else np.inf
        best_indices = None

        for indices in combinations(range(n_primaries), n_cones):
            selected_responses = all_responses[list(indices)]

            # Check if this combination is invertible
            if np.linalg.matrix_rank(selected_responses.T) < n_cones:
                continue

            # Compute cone response range for target cone
            try:
                intensities = selected_responses.T * scaling_factor
                cone_to_disp = np.linalg.inv(intensities)

                # Score based on dynamic range in target cone
                unit_cube_corners = np.array(
                    [[i, j, k] if n_cones == 3 else [i, j, k, l]
                     for i in [0, 1] for j in [0, 1] for k in [0, 1]
                     for l in ([0, 1] if n_cones == 4 else [0])]
                )[:2**n_cones]

                cone_responses = np.linalg.inv(cone_to_disp) @ unit_cube_corners.T
                score = np.ptp(cone_responses[target_cone_idx, :])  # peak-to-peak

                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_indices = list(indices)
            except np.linalg.LinAlgError:
                continue

        if best_indices is None:
            raise ValueError("No valid combination of primaries found")

        selected_primaries = [display_primaries[i] for i in best_indices]
        cone_to_disp, metadata = compute_cone_to_display_direct(
            observer, selected_primaries, white_point_normalize, scaling_factor
        )
        metadata['method'] = 'optimized'
        metadata['target_cone_idx'] = target_cone_idx
        metadata['maximize'] = maximize
        metadata['primary_indices'] = best_indices
        metadata['optimization_score'] = float(best_score)

        return cone_to_disp, metadata

    else:
        # For overcomplete case, use LSQ on all primaries
        # Could implement more sophisticated optimization here
        cone_to_disp, metadata = compute_cone_to_display_lsq(
            observer, display_primaries, white_point_normalize, scaling_factor
        )
        metadata['method'] = 'optimized_lsq'
        metadata['target_cone_idx'] = target_cone_idx

        return cone_to_disp, metadata


def compute_cone_to_display_subset(
    observer: Observer,
    display_primaries: List[Spectra],
    primary_indices: List[int],
    white_point_normalize: bool = True,
    scaling_factor: float = 100
) -> Tuple[npt.NDArray, Dict[str, Any]]:
    """
    User-specified subset of primaries.

    Args:
        observer: Observer model
        display_primaries: List of display primary spectra
        primary_indices: Indices of primaries to use
        white_point_normalize: Whether to normalize to white point
        scaling_factor: Scaling factor for display measurements

    Returns:
        Tuple of (cone_to_disp_matrix, metadata_dict)

    Raises:
        ValueError: If invalid indices or insufficient primaries
    """
    n_cones = observer.dimension
    n_selected = len(primary_indices)

    if n_selected < n_cones:
        raise ValueError(
            f"Must select at least {n_cones} primaries for {n_cones} cones, "
            f"but only {n_selected} indices provided"
        )

    # Validate indices
    if any(idx < 0 or idx >= len(display_primaries) for idx in primary_indices):
        raise ValueError(f"Invalid primary indices: {primary_indices}")

    # Select primaries
    selected_primaries = [display_primaries[i] for i in primary_indices]

    # Use appropriate method based on number of selected primaries
    if n_selected == n_cones:
        cone_to_disp, metadata = compute_cone_to_display_direct(
            observer, selected_primaries, white_point_normalize, scaling_factor
        )
    else:
        cone_to_disp, metadata = compute_cone_to_display_lsq(
            observer, selected_primaries, white_point_normalize, scaling_factor
        )

    metadata['method'] = 'subset'
    metadata['primary_indices'] = primary_indices

    return cone_to_disp, metadata
