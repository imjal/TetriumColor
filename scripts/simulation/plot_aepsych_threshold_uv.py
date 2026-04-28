#!/usr/bin/env python3
"""Plot an exported AEPsych threshold contour as r*(u, v)."""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def _field(data, name, default=None):
    return data[name] if name in data.files else default


def load_contour(path):
    data = np.load(path, allow_pickle=False)
    if "patch_uv" not in data.files or "r_star" not in data.files:
        raise ValueError(f"{path} must contain patch_uv and r_star")
    uv = np.asarray(data["patch_uv"], dtype=float)
    r_star = np.asarray(data["r_star"], dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"patch_uv must have shape (N, 2), got {uv.shape}")
    if r_star.shape != (len(uv),):
        raise ValueError(f"r_star must have shape ({len(uv)},), got {r_star.shape}")
    return data, uv, r_star


def infer_grid(uv, r_star, grid_shape):
    if grid_shape is None:
        return None
    na, nb = [int(v) for v in grid_shape]
    if na * nb != len(r_star):
        return None
    return (
        uv[:, 0].reshape(na, nb),
        uv[:, 1].reshape(na, nb),
        r_star.reshape(na, nb),
    )


def plot_uv(data, uv, r_star, clipped, found, null_idx, args):
    grid = infer_grid(uv, r_star, _field(data, "grid_shape"))
    not_found = ~np.asarray(found, dtype=bool)

    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    tri = Triangulation(uv[:, 0], uv[:, 1])

    if grid is not None:
        U, V, R = grid
        surf = ax3d.plot_surface(
            U, V, R, cmap="viridis", linewidth=0.25,
            edgecolor="k", alpha=0.9, antialiased=True,
        )
        heat = ax2d.tricontourf(tri, r_star, levels=32, cmap="viridis")
    else:
        surf = ax3d.plot_trisurf(
            tri, r_star, cmap="viridis", linewidth=0.25,
            edgecolor="k", alpha=0.9, antialiased=True,
        )
        heat = ax2d.tricontourf(tri, r_star, levels=24, cmap="viridis")

    fig.colorbar(surf, ax=ax3d, shrink=0.72, pad=0.08, label="r*")
    fig.colorbar(heat, ax=ax2d, label="r*")

    if np.any(clipped):
        ax3d.scatter(uv[clipped, 0], uv[clipped, 1], r_star[clipped],
                     c="red", s=16, label="clipped/no threshold")
        ax2d.scatter(uv[clipped, 0], uv[clipped, 1],
                     c="red", s=12, label="clipped/no threshold")
    elif np.any(not_found):
        ax3d.scatter(uv[not_found, 0], uv[not_found, 1], r_star[not_found],
                     c="red", s=16, label="no threshold")
        ax2d.scatter(uv[not_found, 0], uv[not_found, 1],
                     c="red", s=12, label="no threshold")

    if args.mark_null:
        ax3d.scatter(uv[null_idx, 0], uv[null_idx, 1], r_star[null_idx],
                     c="lime", s=80, marker="D", label="nearest null")
        ax2d.scatter(uv[null_idx, 0], uv[null_idx, 1],
                     c="lime", s=70, marker="D", edgecolors="black",
                     label="nearest null")

    ax3d.set_title("Threshold Surface: r*(u, v)")
    ax3d.set_xlabel("u")
    ax3d.set_ylabel("v")
    ax3d.set_zlabel("r*")
    ax3d.view_init(elev=args.elev, azim=args.azim)

    ax2d.set_title("Threshold Heatmap")
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")
    ax2d.set_aspect("equal", adjustable="box")
    return fig, (ax3d, ax2d)


def gaussian_threshold_curve(angles, sigma, p_chance, threshold, cap):
    frac = (threshold - p_chance) / max(1.0 - p_chance, 1e-12)
    frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
    signal = sigma * np.sqrt(-2.0 * np.log(1.0 - frac))
    sin_a = np.sin(angles)
    raw = np.full_like(angles, np.inf, dtype=float)
    valid = sin_a > 1e-12
    raw[valid] = signal / sin_a[valid]
    return np.minimum(raw, cap)


def gaussian_threshold_values(data, r_star, null_idx, args):
    directions, d_null = load_directions(data, null_idx)
    angles = np.arccos(np.clip(directions @ d_null, -1.0, 1.0))
    sigma, p_chance, threshold, cap = get_gaussian_params(data, r_star, args)
    caps = np.full(len(r_star), cap, dtype=float)
    r_max_gamut = _field(data, "r_max_gamut")
    if r_max_gamut is not None:
        caps = np.minimum(caps, np.asarray(r_max_gamut, dtype=float))

    frac = (threshold - p_chance) / max(1.0 - p_chance, 1e-12)
    frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
    signal = sigma * np.sqrt(-2.0 * np.log(1.0 - frac))
    sin_a = np.sin(angles)
    raw = np.full(len(r_star), np.inf, dtype=float)
    valid = sin_a > 1e-12
    raw[valid] = signal / sin_a[valid]
    return np.minimum(raw, caps), raw > caps, angles


def get_gaussian_params(data, r_star, args):
    sigma_arr = _field(data, "simulation_sigma")
    sigma = float(args.sigma if args.sigma is not None
                  else sigma_arr[0] if sigma_arr is not None else 0.010)
    p_chance_arr = _field(data, "simulation_p_chance")
    p_chance = float(args.p_chance if args.p_chance is not None
                     else p_chance_arr[0] if p_chance_arr is not None else 0.25)
    threshold_arr = _field(data, "threshold_level")
    threshold = float(args.threshold if args.threshold is not None
                      else threshold_arr[0] if threshold_arr is not None else 0.75)
    actual_max = _field(data, "actual_max_radius")
    cap = float(args.max_radius if args.max_radius is not None
                else actual_max[0] if actual_max is not None else np.nanmax(r_star))
    return sigma, p_chance, threshold, cap


def load_directions(data, null_idx):
    directions = _field(data, "directions_3d")
    if directions is None:
        raise ValueError("angle plots require directions_3d in the NPZ")
    directions = np.asarray(directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    d_null = directions[null_idx]
    return directions, d_null


def angular_tangent_coordinates(data, null_idx):
    directions, d_null = load_directions(data, null_idx)
    angles = np.arccos(np.clip(directions @ d_null, -1.0, 1.0))

    pca_e1 = _field(data, "pca_e1")
    if pca_e1 is None:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(float(candidate @ d_null)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
    else:
        candidate = np.asarray(pca_e1, dtype=float)

    e1 = candidate - float(candidate @ d_null) * d_null
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(d_null, e1)
    e2 /= np.linalg.norm(e2) + 1e-12

    tangent = directions - (directions @ d_null)[:, None] * d_null
    tangent_norm = np.linalg.norm(tangent, axis=1)
    unit_tangent = np.zeros_like(tangent)
    valid = tangent_norm > 1e-12
    unit_tangent[valid] = tangent[valid] / tangent_norm[valid, None]

    x = angles * (unit_tangent @ e1)
    y = angles * (unit_tangent @ e2)
    x[~valid] = 0.0
    y[~valid] = 0.0
    return np.column_stack([x, y]), angles


def plot_angle(data, uv, r_star, clipped, found, null_idx, args):
    directions, d_null = load_directions(data, null_idx)
    angles = np.arccos(np.clip(directions @ d_null, -1.0, 1.0))
    sigma, p_chance, threshold, cap = get_gaussian_params(data, r_star, args)

    order = np.argsort(angles)
    curve_angles = np.linspace(0.0, max(float(np.max(angles)), 1e-5), 600)
    curve = gaussian_threshold_curve(curve_angles, sigma, p_chance, threshold, cap)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax_zoom = fig.add_subplot(1, 2, 2)

    colors = np.where(clipped, "red", "tab:blue")
    colors = np.where(~np.asarray(found, dtype=bool), "red", colors)
    ax.scatter(angles, r_star, c=colors, s=18, alpha=0.75, label="AEPsych estimate")
    ax.plot(curve_angles, curve, color="black", linewidth=2.0,
            label="Gaussian ground truth")
    ax.scatter(angles[null_idx], r_star[null_idx], c="lime", s=80,
               marker="D", edgecolors="black", label="nearest null")

    ax_zoom.scatter(angles, r_star, c=colors, s=18, alpha=0.75)
    ax_zoom.plot(curve_angles, curve, color="black", linewidth=2.0)
    ax_zoom.scatter(angles[null_idx], r_star[null_idx], c="lime", s=80,
                    marker="D", edgecolors="black")

    zoom_hi = np.percentile(angles, 35)
    if zoom_hi <= 0:
        zoom_hi = max(float(np.max(angles)), 1e-4)
    ax_zoom.set_xlim(0, zoom_hi)

    for target in (ax, ax_zoom):
        target.set_xlabel("angle from null direction (rad)")
        target.set_ylabel("r*")
        target.grid(True, alpha=0.25)
        target.set_ylim(bottom=0.0)
    ax.set_title("Threshold vs Angular Distance")
    ax_zoom.set_title("Near-Null Zoom")
    ax.legend(fontsize=8)

    print("Gaussian overlay:")
    print(f"  sigma={sigma:.5f}")
    print(f"  p_chance={p_chance:.3f}")
    print(f"  threshold={threshold:.3f}")
    print(f"  cap={cap:.5f}")
    print(f"  nearest null angle={angles[null_idx]:.5e}, r*={r_star[null_idx]:.5f}")
    return fig, (ax, ax_zoom)


def plot_angle2d(data, uv, r_star, clipped, found, null_idx, args):
    angle_xy, angles = angular_tangent_coordinates(data, null_idx)
    sigma, p_chance, threshold, cap = get_gaussian_params(data, r_star, args)
    truth_r_star, truth_clipped, _ = gaussian_threshold_values(
        data, r_star, null_idx, args)

    tri = Triangulation(angle_xy[:, 0], angle_xy[:, 1])
    radius = np.linalg.norm(angle_xy, axis=1)
    grid_lim = max(float(np.max(np.abs(angle_xy))), 1e-4)
    n = 151
    gx = np.linspace(-grid_lim, grid_lim, n)
    gy = np.linspace(-grid_lim, grid_lim, n)
    GX, GY = np.meshgrid(gx, gy)
    GR = np.sqrt(GX ** 2 + GY ** 2)
    truth = gaussian_threshold_curve(GR, sigma, p_chance, threshold, cap)
    truth[GR > max(float(np.max(radius)), 1e-8)] = np.nan

    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    surf = ax3d.plot_trisurf(
        tri, r_star, cmap="viridis", linewidth=0.25,
        edgecolor="k", alpha=0.9, antialiased=True,
    )
    if args.values == "compare":
        ax3d.plot_trisurf(
            tri, truth_r_star, color="tab:orange", linewidth=0.0,
            alpha=0.32, antialiased=True, label="ground truth",
        )
        ax3d.scatter([], [], [], c="tab:blue", label="AEPsych estimate")
        ax3d.scatter([], [], [], c="tab:orange", alpha=0.45,
                     label="ground truth")
    contour = ax2d.tricontourf(tri, r_star, levels=32, cmap="viridis")
    ax2d.contour(GX, GY, truth, levels=10, colors="black",
                 linewidths=0.8, alpha=0.65)

    fig.colorbar(surf, ax=ax3d, shrink=0.72, pad=0.08, label="r*")
    fig.colorbar(contour, ax=ax2d, label="r*")

    bad = clipped | ~np.asarray(found, dtype=bool)
    if args.values == "compare":
        bad = bad | truth_clipped
    if np.any(bad):
        ax3d.scatter(angle_xy[bad, 0], angle_xy[bad, 1], r_star[bad],
                     c="red", s=16, label="clipped/no threshold")
        ax2d.scatter(angle_xy[bad, 0], angle_xy[bad, 1],
                     c="red", s=12, label="clipped/no threshold")

    if args.mark_null:
        ax3d.scatter(angle_xy[null_idx, 0], angle_xy[null_idx, 1], r_star[null_idx],
                     c="lime", s=80, marker="D", edgecolors="black",
                     label="nearest null")
        ax2d.scatter(angle_xy[null_idx, 0], angle_xy[null_idx, 1],
                     c="lime", s=70, marker="D", edgecolors="black",
                     label="nearest null")

    ax3d.set_title("Threshold Surface: r*(angular x, angular y)")
    ax3d.set_xlabel("angular x (rad)")
    ax3d.set_ylabel("angular y (rad)")
    ax3d.set_zlabel("r*")
    ax3d.view_init(elev=args.elev, azim=args.azim)

    ax2d.set_title("Angular Threshold Heatmap\nblack contours = analytic Gaussian")
    ax2d.set_xlabel("angular x (rad)")
    ax2d.set_ylabel("angular y (rad)")
    ax2d.set_aspect("equal", adjustable="box")

    print("Gaussian overlay:")
    print(f"  sigma={sigma:.5f}")
    print(f"  p_chance={p_chance:.3f}")
    print(f"  threshold={threshold:.3f}")
    print(f"  cap={cap:.5f}")
    print(f"  angle radius range=[{np.min(angles):.5e}, {np.max(angles):.5e}]")
    print(f"  nearest null angle={angles[null_idx]:.5e}, r*={r_star[null_idx]:.5f}")
    if args.values == "compare":
        print(f"  ground truth r* range=[{truth_r_star.min():.5f}, {truth_r_star.max():.5f}]")
        print(f"  ground truth nearest null r*={truth_r_star[null_idx]:.5f}")
    return fig, (ax3d, ax2d)


def main():
    parser = argparse.ArgumentParser(
        description="Plot threshold amplitude over AEPsych patch coordinates.")
    parser.add_argument("contour", help="NPZ exported by aepsych_contour_simulation.py")
    parser.add_argument("--coords", choices=["uv", "angle", "angle2d"], default="uv",
                        help="Plot r* over patch coordinates or angle from null")
    parser.add_argument("--values", choices=["estimate", "ground-truth", "error", "compare"],
                        default="estimate",
                        help="Plot estimate, ground truth, estimate-ground truth, or both")
    parser.add_argument("--output", default=None,
                        help="Optional figure path. If omitted, show interactively.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--mark-null", action="store_true", default=True,
                        help="Mark nearest point to u=v=0")
    parser.add_argument("--no-mark-null", dest="mark_null", action="store_false")
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian sigma for --coords angle overlay")
    parser.add_argument("--p-chance", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Cap for --coords angle overlay")
    args = parser.parse_args()

    data, uv, r_star = load_contour(args.contour)
    grid = infer_grid(uv, r_star, _field(data, "grid_shape"))
    clipped = _field(data, "gamut_clipped", np.zeros(len(r_star), dtype=bool))
    found = _field(data, "threshold_found", np.ones(len(r_star), dtype=bool))
    clipped = np.asarray(clipped, dtype=bool)

    null_idx = int(np.argmin(np.linalg.norm(uv, axis=1)))
    estimate_r_star = r_star.copy()
    if args.values != "estimate" and args.values != "compare":
        truth_r_star, truth_clipped, _ = gaussian_threshold_values(
            data, estimate_r_star, null_idx, args)
        if args.values == "ground-truth":
            r_star = truth_r_star
            clipped = truth_clipped
            found = ~truth_clipped
        else:
            r_star = estimate_r_star - truth_r_star
            clipped = clipped | truth_clipped
    if args.coords == "uv":
        fig, axes = plot_uv(data, uv, r_star, clipped, found, null_idx, args)
    elif args.coords == "angle":
        fig, axes = plot_angle(data, uv, r_star, clipped, found, null_idx, args)
    else:
        fig, axes = plot_angle2d(data, uv, r_star, clipped, found, null_idx, args)

    title = args.title or args.contour
    title += f"\nvalues={args.values}"
    completed = _field(data, "completed_trials")
    sigma = _field(data, "simulation_sigma")
    if completed is not None:
        title += f"\ntrials={int(completed[0])}"
    if sigma is not None:
        title += f", sigma={float(sigma[0]):.4f}"

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    print(f"Loaded {args.contour}")
    print(f"  values={args.values}")
    print(f"  count={len(r_star)}")
    print(f"  r* range=[{r_star.min():.5f}, {r_star.max():.5f}]")
    print(f"  nearest null uv={uv[null_idx]}, r*={r_star[null_idx]:.5f}")
    if "r_max_gamut" in data.files:
        print(f"  nearest null r_max_gamut={data['r_max_gamut'][null_idx]:.5f}")
    print(f"  clipped_count={int(np.sum(clipped))}")
    if found is not None:
        print(f"  threshold_found_count={int(np.sum(found))}")

    if args.output:
        fig.savefig(args.output, dpi=160)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
