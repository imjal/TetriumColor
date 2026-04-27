#!/usr/bin/env python3
"""
Interactive null-direction theory visualizer (polyscope).

Shows, for a set of trichromat+Q CMFs:
  1. Display gamut slice at the luminance plane of (0.5,0.5,0.5,0.5)
  2. Null directions for each CMF as points on the gamut surface
  3. Gamut surface patch covering the theta/phi region near the null directions
  4. A peaky 3D Gaussian threshold contour with adjustable sigma

Usage:
    python null_direction_viewer.py --primaries_dir ../../measurements/2026-03-03/primaries/
"""

from TetriumColor.Visualization.PolyscopeUtils import (
    Render3DLine,
    Render3DMesh,
    RenderBGYRGamut,
    RenderGamutSlices,
    RenderPointCloud,
)
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer import GetHeringMatrix
from TetriumColor import ColorSpace, ColorSpaceType, PolyscopeDisplayType
import argparse
import os
import sys

import numpy as np
import tetrapolyscope as ps
import tetrapolyscope.imgui as psim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Chromatic-subspace helpers (adapted from plot_null_direction_theory.py)
# ---------------------------------------------------------------------------

def build_chromatic_basis(cst: ColorSpace, q_axis: int):
    """Return (w0_disp (4,), chrom_basis (4,3)).

    w0 is fixed to the physical display midgray [0.5, 0.5, 0.5, 0.5] for all
    observers so that all gamut cross-sections share a common centre point.

    chrom_basis spans the 3D subspace orthogonal to the *trichromat* luminance
    direction in DISP space.  The Q cone's contribution is zeroed out of the
    Hering luminance row before computing the SVD complement, so the null
    direction e_q (only Q changes) lies exactly in this subspace.
    """
    cone_to_disp = cst._get_cone_to_disp()
    disp_to_cone = np.linalg.inv(cone_to_disp)

    # Common physical background — independent of observer-specific MaxBasis
    n_disp = cone_to_disp.shape[1]   # == 4 for a 4-primary display
    w0 = np.full(n_disp, 0.5)

    # Trichromat luminance: zero out the Q contribution
    lum_cone = cst._get_cone_to_hering()[0, :].copy()
    lum_cone[q_axis] = 0.0
    lum_cone /= np.linalg.norm(lum_cone)

    lum_disp = disp_to_cone.T @ lum_cone
    _, _, Vt = np.linalg.svd(lum_disp.reshape(1, -1), full_matrices=True)
    chrom_basis = Vt[1:, :].T   # (4, 3)
    return w0, chrom_basis


def r_max_in_direction(w0, chrom_basis, d3):
    """Largest r such that  w0 + r*(chrom_basis @ d3)  stays in [0,1]^4."""
    dir4 = chrom_basis @ d3
    r = np.inf
    for i in range(4):
        if dir4[i] > 1e-12:
            r = min(r, (1.0 - w0[i]) / dir4[i])
        elif dir4[i] < -1e-12:
            r = min(r, -w0[i] / dir4[i])
    return float(r) if r < 1e9 else 1.0


def r_max_batch(w0, chrom_basis, d3_batch):
    """Vectorised r_max: d3_batch (N,3) → r (N,).

    For each row d3, returns the largest r keeping
    w0 + r*(chrom_basis @ d3) inside [0,1]^4.
    """
    dir4   = d3_batch @ chrom_basis.T          # (N, 4)
    n_disp = dir4.shape[1]                     # 4  (number of DISP primaries)
    r      = np.full(len(d3_batch), np.inf)
    for i in range(n_disp):
        pos = dir4[:, i] >  1e-12
        neg = dir4[:, i] < -1e-12
        r[pos] = np.minimum(r[pos], (1.0 - w0[i]) / dir4[pos, i])
        r[neg] = np.minimum(r[neg],      -w0[i]   / dir4[neg, i])
    r[r > 1e9] = 1.0
    return r


def sphere_grid_batch(thetas, phis):
    """Return all (theta, phi) grid points as a (N,3) array of unit vectors."""
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')   # (N_T, N_P)
    TH, PH = TH.ravel(), PH.ravel()
    return np.stack([np.sin(TH) * np.cos(PH),
                     np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=1)                # (N, 3)


def sph_to_cart(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class NullDirectionViewer:
    # GUI-adjustable params
    sigma_major: float = 0.30   # ellipsoid semi-axis along null direction (viz units)
    sigma_minor: float = 0.06   # ellipsoid semi-axis perpendicular to null direction
    patch_angle: float = 0.35   # angular radius (rad) for gamut patch
    point_size: float = 0.015

    # Sphere sampling resolution
    N_THETA = 120
    N_PHI   = 240

    OBSERVER_COLORS = [
        [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
        [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        [1.0, 0.6, 0.0], [0.6, 0.3, 1.0], [0.3, 0.8, 0.8],
        [0.8, 0.8, 0.3],
    ]

    def __init__(self, primaries_dir: str, num_observers: int = 10,
                 display_basis=PolyscopeDisplayType.HERING_BGYR):
        self.num_observers = num_observers
        self.display_basis = display_basis
        self.window_open = True

        print(f"Loading primaries from {primaries_dir}…")
        self.primaries = load_primaries_from_csv(primaries_dir, extract_zero=False)
        print(f"Loaded {len(self.primaries)} primaries")

        og = ObserverGenotypes(wavelengths=np.arange(380, 781, 5),
                               dimensions=[3], seed=42)
        genotypes = og.get_genotypes_covering_probability(
            target_probability=0.999, sex='both')[:num_observers]
        print(f"Selected {len(genotypes)} trichromatic genotypes")

        self.color_spaces: list[ColorSpace] = []
        self.q_axes:        list[int] = []
        self.w0s:           list[np.ndarray] = []   # background in DISP
        self.chrom_bases:   list[np.ndarray] = []   # (4,3) each
        self.null_dirs_3d:  list[np.ndarray] = []   # unit vec in chrom coords

        for g in genotypes:
            gq = tuple(sorted(g + (547,)))
            obs = og.get_observer_for_peaks(gq, degree=4.0)
            q_axis = next((i for i, c in enumerate(obs.sensors)
                           if abs(c.peak - 547.0) < 1.0), 2)

            cst = ColorSpace(obs, self.primaries)
            w0, cb = build_chromatic_basis(cst, q_axis)

            meta4 = cst.get_metameric_axis_in(ColorSpaceType.DISP,
                                              metameric_axis_num=q_axis)
            meta4 /= np.linalg.norm(meta4)
            d3 = cb.T @ meta4
            norm = np.linalg.norm(d3)
            d3 = d3 / norm if norm > 1e-8 else d3

            self.color_spaces.append(cst)
            self.q_axes.append(q_axis)
            self.w0s.append(w0)
            self.chrom_bases.append(cb)
            self.null_dirs_3d.append(d3)

            theta = float(np.arccos(np.clip(d3[2], -1, 1)))
            phi = float(np.arctan2(d3[1], d3[0]))
            print(f"  {gq}  q_axis={q_axis}  d_null=(θ={theta:.3f}, φ={phi:.3f})")

        self._verify_null_dirs()

        # Polyscope init
        ps.init()
        ps.set_transparency_render_passes(24)
        ps.set_transparency_peel_epsilon(1e-7)
        ps.set_ground_plane_mode("none")
        ps.set_always_redraw(True)
        ps.set_user_callback(self._callback)

        self._render()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _disp_to_viz(self, cst, disp_pts):
        """DISP (N,4) → polyscope 3-D (N,3)."""
        cone = cst.convert(disp_pts, ColorSpaceType.DISP, ColorSpaceType.CONE)
        return cst.convert_to_polyscope(cone, ColorSpaceType.CONE, self.display_basis)

    def _null_gamut_disp(self, idx, sign=1):
        """DISP point at the gamut boundary along the null direction.

        With the trichromat-luminance chrom_basis, meta4 lies exactly in the
        column space of cb, so  cb @ (cb.T @ meta4) == meta4  and walking along
        cb @ d3 changes only Q — verified by _verify_null_dirs.
        """
        cst = self.color_spaces[idx]
        w0, cb = self.w0s[idx], self.chrom_bases[idx]

        meta4 = cst.get_metameric_axis_in(ColorSpaceType.DISP,
                                          metameric_axis_num=self.q_axes[idx])
        d3 = cb.T @ meta4
        norm = np.linalg.norm(d3)
        if norm < 1e-8:
            return w0.copy()
        d3 = sign * d3 / norm

        r = r_max_in_direction(w0, cb, d3)
        return w0 + r * (cb @ d3)

    def _null_gamut_pt(self, idx, sign=1):
        """Viz coords of the null-direction gamut boundary for observer idx.

        Always uses observer 0's cst for DISP→viz so all null-direction points
        live in the same coordinate frame as the gamut solid.
        """
        disp_pt = self._null_gamut_disp(idx, sign)
        return self._disp_to_viz(self.color_spaces[0], disp_pt.reshape(1, -1))[0]

    def _verify_null_dirs(self):
        """Print cone-space differences at the null-direction gamut endpoints.

        For observer idx the two DISP endpoints (+/-) should be metamers:
        identical L, M, S responses but different Q response.
        Uses each observer's own cone sensitivities for the check.
        """
        print("\n=== Null-direction metamer verification ===")
        header = f"  {'observer':<28}  {'ΔS':>8}  {'ΔM':>8}  {'ΔQ':>8}  {'ΔL':>8}  {'|ΔLMS|':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx, (cst, q_axis) in enumerate(zip(self.color_spaces, self.q_axes)):
            peaks = tuple(int(round(c.peak)) for c in cst.observer.sensors)

            dp = self._null_gamut_disp(idx, +1)
            dn = self._null_gamut_disp(idx, -1)

            # Convert both DISP endpoints to CONE using observer idx's sensitivities
            cp = cst.convert(dp.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            cn = cst.convert(dn.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

            diff = cp - cn                          # (dim,) — should be zero except at q_axis
            non_q = [i for i in range(cst.dim) if i != q_axis]
            lms_residual = np.linalg.norm(diff[non_q])

            # Print one row per observer (cone order is S, M, [Q,] L)
            vals = "  ".join(f"{diff[i]:+8.5f}" for i in range(cst.dim))
            print(f"  {str(peaks):<28}  {vals}  |ΔLMS|={lms_residual:.2e}")

        print("=== end verification ===\n")

    def _background_viz(self, idx=0):
        cst, w0 = self.color_spaces[idx], self.w0s[idx]
        return self._disp_to_viz(cst, w0.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    # Surface computation
    # ------------------------------------------------------------------

    def _sphere_grid(self):
        """Return (N_THETA, N_PHI) grids of (theta, phi) values."""
        thetas = np.linspace(0.01, np.pi - 0.01, self.N_THETA)
        phis = np.linspace(-np.pi, np.pi, self.N_PHI, endpoint=False)
        return thetas, phis

    def _grid_faces(self, nt, np_):
        """Triangle faces for an (nt × np_) spherical grid (phi wraps). Vectorised."""
        i  = np.arange(nt - 1)
        j  = np.arange(np_)
        II, JJ = np.meshgrid(i, j, indexing='ij')   # (nt-1, np_)
        II, JJ = II.ravel(), JJ.ravel()
        JN = (JJ + 1) % np_
        k00 = II * np_ + JJ
        k01 = II * np_ + JN
        k10 = (II + 1) * np_ + JJ
        k11 = (II + 1) * np_ + JN
        tri_a = np.stack([k00, k01, k11], axis=1)
        tri_b = np.stack([k00, k11, k10], axis=1)
        return np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)

    def _null_dir_viz(self, idx):
        """Unit null direction in polyscope 3-D viz coords for observer `idx`.

        Uses observer 0's cst so the direction is consistent with the gamut solid.
        """
        cst0 = self.color_spaces[0]
        w0, cb, d3 = self.w0s[idx], self.chrom_bases[idx], self.null_dirs_3d[idx]
        eps = 1e-3
        v1 = self._disp_to_viz(cst0, (w0 + eps * (cb @ d3)).reshape(1, -1))[0]
        v0 = self._disp_to_viz(cst0, w0.reshape(1, -1))[0]
        d = v1 - v0
        n = np.linalg.norm(d)
        return d / n if n > 1e-10 else d

    def _threshold_surface(self, idx=0):
        """Ellipsoid contour in polyscope viz coords with null direction as major axis.

        Polar form of the ellipsoid centered at bg_viz:

            r(α) = 1 / sqrt( cos²(α)/σ_major² + sin²(α)/σ_minor² )

        where α = angle between sample direction and d_null (treated ±symmetrically).
        This gives r = σ_major at the null axis and r = σ_minor perpendicular,
        with a perfectly smooth surface everywhere.

        Returns
        -------
        viz_pts   : (N,3) polyscope coords
        r_stars   : (N,)  radius at each sample
        gamut_mask: (N,)  all False (kept for API compatibility)
        faces     : (F,3) triangle indices
        """
        bg_viz = self._background_viz(idx)
        null_viz = self._null_dir_viz(idx)

        sm2 = max(self.sigma_major, 1e-4) ** 2
        sn2 = max(self.sigma_minor, 1e-4) ** 2

        thetas, phis = self._sphere_grid()
        d3_all = sphere_grid_batch(thetas, phis)                     # (N, 3)

        cos_a  = np.clip(d3_all @ null_viz, -1., 1.)                 # (N,)
        sin_a  = np.sqrt(np.maximum(1. - cos_a ** 2, 0.))
        r_stars = 1.0 / np.sqrt(cos_a ** 2 / sm2 + sin_a ** 2 / sn2)  # (N,)

        viz_pts = bg_viz + r_stars[:, None] * d3_all                 # (N, 3)
        faces   = self._grid_faces(self.N_THETA, self.N_PHI)
        return viz_pts, r_stars, np.zeros(len(r_stars), dtype=bool), faces

    def _compute_vl_hyperplane_solid(self, idx=0):
        """Boundary of the 4D gamut intersected with the V(λ)-constant hyperplane.

        The hyperplane is defined by:
            V(λ)_disp · (x - w0) = 0,   x ∈ [0,1]^4

        where V(λ)_disp is the luminance direction in DISP space, derived from
        the first row of the Hering matrix pushed through the cone→DISP transform.
        The complement 3D subspace is exactly chrom_basis (built in __init__).

        Boundary is sampled by sweeping all sphere directions d3 and finding the
        gamut extent r_max along  w0 + r * (chrom_basis @ d3).

        Returns
        -------
        viz_pts : (N,3) polyscope coordinates
        faces   : (F,3) triangle indices (spherical grid topology, phi-wrapping)
        colors  : (N,3) white-balanced linear-sRGB for hue visualisation
        """
        cst, w0, cb = (self.color_spaces[idx], self.w0s[idx],
                       self.chrom_bases[idx])
        thetas, phis = self._sphere_grid()
        d3_all   = sphere_grid_batch(thetas, phis)               # (N, 3)
        rm       = r_max_batch(w0, cb, d3_all)                   # (N,)
        disp_pts = w0 + rm[:, None] * (d3_all @ cb.T)           # (N, 4)

        viz_pts = self._disp_to_viz(cst, disp_pts)              # (N, 3)
        faces   = self._grid_faces(self.N_THETA, self.N_PHI)    # (F, 3)

        # White-balance to background so we see hue rather than absolute radiometry
        cone_pts = cst.convert(disp_pts, ColorSpaceType.DISP, ColorSpaceType.CONE)
        srgb_raw = cst.convert(cone_pts, ColorSpaceType.CONE,
                               ColorSpaceType.LINEAR_SRGB)
        bg_cone = cst.convert(w0.reshape(1, -1),
                              ColorSpaceType.DISP, ColorSpaceType.CONE)
        bg_srgb = np.maximum(
            cst.convert(bg_cone, ColorSpaceType.CONE, ColorSpaceType.LINEAR_SRGB),
            1e-8)
        colors = np.clip(srgb_raw / bg_srgb, 0, 1)

        return viz_pts, faces, colors

    def _gamut_patch(self, idx=0):
        """Gamut boundary points within `patch_angle` of the null direction.

        Returns viz_pts (M,3) and faces (F,3).  Rows outside the patch are
        replaced by the background point so the mesh stays well-formed.
        """
        cst, w0, cb, d_null = (self.color_spaces[idx], self.w0s[idx],
                               self.chrom_bases[idx], self.null_dirs_3d[idx])
        thetas, phis = self._sphere_grid()
        bg_viz  = self._background_viz(idx)
        d3_all  = sphere_grid_batch(thetas, phis)                  # (N, 3)

        cos_a    = np.clip(d3_all @ d_null, -1., 1.)
        in_patch = np.arccos(np.abs(cos_a)) <= self.patch_angle    # (N,) bool

        # Gamut boundary only where in patch; background elsewhere (masked out)
        rm       = r_max_batch(w0, cb, d3_all)                     # (N,)
        disp_pts = np.where(in_patch[:, None],
                            w0 + rm[:, None] * (d3_all @ cb.T),
                            w0)                                     # (N, 4)

        viz_pts = self._disp_to_viz(cst, disp_pts)
        viz_pts[~in_patch] = bg_viz

        all_faces   = self._grid_faces(self.N_THETA, self.N_PHI)
        patch_faces = all_faces[np.all(in_patch[all_faces], axis=1)]

        return viz_pts, patch_faces

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        print("\n=== Rendering ===")
        self._clear()

        cst_ref = self.color_spaces[0]

        # --- 1. V(λ)-constant hyperplane solid (gamut intersection) ----------
        # The hyperplane normal is V(λ) in DISP space, passing through w0.
        # Its 3D chromatic complement is chrom_basis (built in __init__).
        print("  Computing V(λ) hyperplane solid…")
        solid_pts, solid_faces, solid_colors = self._compute_vl_hyperplane_solid(0)
        try:
            ps_solid = ps.register_surface_mesh(
                "vl_hyperplane_solid", solid_pts, solid_faces,
                back_face_policy='identical', material='wax', smooth_shade=True)
            ps_solid.add_color_quantity("vl_hyperplane_solid_colors",
                                        solid_colors, defined_on='vertices',
                                        enabled=True)
            ps_solid.set_transparency(0.25)
        except Exception as e:
            print(f"  Solid mesh failed ({e}), using point cloud")
            RenderPointCloud("vl_hyperplane_solid", solid_pts, solid_colors,
                             radius=0.007)
        print(f"  V(λ) hyperplane: {len(solid_pts)} verts, {len(solid_faces)} faces")

        # --- 2. Null direction points (± both sides, all observers) ----------
        null_pts, null_colors = [], []
        for i in range(len(self.color_spaces)):
            col = np.array(self.OBSERVER_COLORS[i % len(self.OBSERVER_COLORS)])
            for sign in (+1, -1):
                null_pts.append(self._null_gamut_pt(i, sign))
                null_colors.append(col * (1.0 if sign > 0 else 0.45))

        null_pts = np.array(null_pts)
        null_colors = np.clip(np.array(null_colors), 0, 1)
        RenderPointCloud("null_pts", null_pts, null_colors, radius=self.point_size)

        # Reference null direction arrow (first observer)
        bg_viz = self._background_viz(0)
        null_end = self._null_gamut_pt(0, +1)
        Render3DLine("null_arrow",
                     np.array([bg_viz, null_end]),
                     np.array([0.2, 1.0, 0.2]),
                     radius=0.004)

        # --- 3. Gamut surface patch near null directions (first observer) ----
        patch_viz, patch_faces = self._gamut_patch(0)
        if len(patch_faces) > 0:
            patch_colors = np.tile([0.15, 0.75, 1.0], (len(patch_viz), 1))
            try:
                ps_patch = ps.register_surface_mesh(
                    "null_gamut_patch", patch_viz, patch_faces,
                    back_face_policy='identical', material='wax', smooth_shade=True)
                ps_patch.add_color_quantity("null_gamut_patch_colors",
                                            patch_colors, defined_on='vertices',
                                            enabled=True)
                ps_patch.set_transparency(0.55)
            except Exception as e:
                print(f"  Patch mesh failed ({e}), using point cloud")
                RenderPointCloud("null_gamut_patch", patch_viz[
                    np.unique(patch_faces.ravel())],
                    np.tile([0.15, 0.75, 1.0], (len(np.unique(patch_faces.ravel())), 1)),
                    radius=0.01)

        # --- 4. Gaussian threshold contour surface ---------------------------
        viz_pts, r_stars, gamut_mask, faces = self._threshold_surface(0)

        # Color by r* value: minor-axis equator is dark, major-axis tip is bright
        r_norm = np.clip(
            (r_stars - self.sigma_minor) / max(self.sigma_major - self.sigma_minor, 1e-8),
            0, 1)
        surf_colors = np.zeros((len(viz_pts), 3))
        surf_colors[:, 0] = np.clip(r_norm * 2,       0, 1)
        surf_colors[:, 1] = np.clip(r_norm * 2 - 0.5, 0, 1)
        surf_colors[:, 2] = np.clip(r_norm * 2 - 1.5, 0, 1)

        try:
            ps_surf = ps.register_surface_mesh(
                "gaussian_contour", viz_pts, faces,
                back_face_policy='identical', material='wax', smooth_shade=True)
            ps_surf.add_color_quantity("gaussian_contour_colors",
                                       surf_colors, defined_on='vertices',
                                       enabled=True)
            ps_surf.set_transparency(0.65)
        except Exception as e:
            print(f"  Contour mesh failed ({e}), using point cloud")
            RenderPointCloud("gaussian_contour", viz_pts, surf_colors, radius=0.007)

        # Background reference sphere
        RenderPointCloud("bg_sphere", bg_viz.reshape(1, -1),
                         np.array([[1., 1., 1.]]), radius=self.point_size * 1.5)

        print(f"  r* range: [{r_stars.min():.4f}, {r_stars.max():.4f}]")
        print("=== Done ===\n")

    def _clear(self):
        for name in ["gaussian_contour", "null_gamut_patch", "vl_hyperplane_solid"]:
            try:
                ps.remove_surface_mesh(name)
            except:
                pass
        for name in ["null_pts", "bg_sphere", "gaussian_contour",
                     "null_gamut_patch", "vl_hyperplane_solid"]:
            try:
                ps.remove_point_cloud(name)
            except:
                pass
        try:
            ps.remove_curve_network("null_arrow")
        except:
            pass

    # ------------------------------------------------------------------
    # GUI callback
    # ------------------------------------------------------------------

    def _callback(self):
        opened, self.window_open = psim.Begin("Null Direction Viewer", self.window_open)
        if not opened:
            psim.End()
            return

        psim.Text(f"{len(self.color_spaces)} trichromat observers  (+547 nm Q-cone)")
        psim.Separator()

        psim.Text("Ellipsoid contour  (major axis = null direction)")
        _, new_sm = psim.SliderFloat("sigma_major (null axis)", self.sigma_major, 0.01, 1.5)
        _, new_sn = psim.SliderFloat("sigma_minor (perp)",      self.sigma_minor, 0.01, 1.5)

        psim.Separator()
        psim.Text("Gamut patch")
        _, new_patch = psim.SliderFloat("Patch angle (rad)", self.patch_angle, 0.05, 1.2)

        psim.Separator()
        _, new_pt_size = psim.SliderFloat("Point size", self.point_size, 0.002, 0.05)

        changed = (
            abs(new_sm - self.sigma_major) > 1e-4 or
            abs(new_sn - self.sigma_minor) > 1e-4 or
            abs(new_patch - self.patch_angle) > 1e-4 or
            abs(new_pt_size - self.point_size) > 1e-4
        )

        self.sigma_major = new_sm
        self.sigma_minor = new_sn
        self.patch_angle = new_patch
        self.point_size = new_pt_size

        if changed:
            self._render()

        psim.Text(f"\nLegend:")
        psim.Text("  Colored pts    – null direction ± gamut endpoints per CMF")
        psim.Text("  Green arrow    – null direction (observer 1)")
        psim.Text("  Cyan surface   – gamut patch near null direction")
        psim.Text("  Hot surface    – ellipsoid: r = 1/sqrt(cos²α/σ_maj² + sin²α/σ_min²)")

        psim.End()

    def show(self):
        ps.show()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive null-direction theory visualizer (polyscope)")
    parser.add_argument("--primaries_dir", type=str,
                        default="../../measurements/2026-03-03/primaries/")
    parser.add_argument("--num_observers", type=int, default=10)
    args = parser.parse_args()

    viewer = NullDirectionViewer(
        primaries_dir=args.primaries_dir,
        num_observers=args.num_observers,
    )
    viewer.show()


if __name__ == "__main__":
    main()
