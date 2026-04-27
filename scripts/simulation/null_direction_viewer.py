#!/usr/bin/env python3
"""
Interactive null-direction theory visualizer (polyscope).

Shows, for a set of trichromat+Q CMFs:
  1. Display gamut slice at the luminance plane of (0.5,0.5,0.5,0.5)
  2. Null directions for each CMF as points on the gamut surface
  3. Gamut surface patch covering the theta/phi region near the null directions,
     parameterized along the PCA axes of population null-direction variation
  4. A peaky 3D Gaussian threshold contour with adjustable sigma

Usage:
    python null_direction_viewer.py --primaries_dir ../../measurements/2026-03-03/primaries/
"""

from TetriumColor.Visualization.PolyscopeUtils import (
    Render3DLine,
    RenderPointCloud,
)
from TetriumColor.Measurement import load_primaries_from_csv
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes
from TetriumColor.Observer.CMFSampler import CMFSampler
from TetriumColor.Observer import Observer
from TetriumColor.Observer.Observer import Cone
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

    chrom_basis spans the 3D subspace orthogonal to an observer-specific
    non-Q cone luminance functional in DISP space.  This keeps the true
    Q-isolating null direction inside the slice without constructing MaxBasis.
    """
    cone_to_disp = cst._get_cone_to_disp()
    disp_to_cone = np.linalg.inv(cone_to_disp)

    # Common physical background — independent of observer-specific MaxBasis
    n_disp = cone_to_disp.shape[1]   # == 4 for a 4-primary display
    w0 = np.full(n_disp, 0.5)

    # Luminance for this viewer is a weighted sum of the non-Q cones.  Because
    # true Q-isolating directions have zero non-Q cone change, they are
    # automatically isoluminant under this functional.
    lum_cone = np.ones(cst.dim)
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
    dir4 = d3_batch @ chrom_basis.T          # (N, 4)
    n_disp = dir4.shape[1]                     # 4  (number of DISP primaries)
    r = np.full(len(d3_batch), np.inf)
    for i in range(n_disp):
        pos = dir4[:, i] > 1e-12
        neg = dir4[:, i] < -1e-12
        r[pos] = np.minimum(r[pos], (1.0 - w0[i]) / dir4[pos, i])
        r[neg] = np.minimum(r[neg],      -w0[i] / dir4[neg, i])
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
    patch_angle_major: float = 0.25  # PCA major-axis angular half-width (rad)
    patch_angle_minor: float = 0.10  # PCA minor-axis angular half-width (rad)
    point_size: float = 0.015
    font_size: float = 6.0           # billboard label font size
    labels_enabled: bool = False

    # Sphere sampling resolution
    N_THETA = 180
    N_PHI = 360

    # PCA samples drawn from CMFSampler
    N_PCA_SAMPLES = 1000

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

        self._og_wavelengths = np.arange(380, 781, 5)
        self._og = ObserverGenotypes(wavelengths=self._og_wavelengths,
                                     dimensions=[3], seed=42)
        genotypes = self._og.get_genotypes_covering_probability(
            target_probability=0.999, sex='both')[:num_observers]
        print(f"Selected {len(genotypes)} trichromatic genotypes")

        most_common = self._og.get_most_common_genotype('both')
        self.ref_obs = self._og.get_observer_for_peaks(
            tuple(sorted(most_common + (547,))), degree=4.0
        )
        self.ref_cst = ColorSpace(self.ref_obs, self.primaries)

        self.color_spaces: list[ColorSpace] = []
        self.q_axes:        list[int] = []
        self.w0s:           list[np.ndarray] = []   # background in DISP
        self.chrom_bases:   list[np.ndarray] = []   # (4,3) each
        self.null_dirs_3d:  list[np.ndarray] = []   # unit vec in chrom coords
        self.genotype_labels: list[str] = []        # human-readable genotype

        for g in genotypes:
            gq = tuple(sorted(g + (547,)))
            obs = self._create_observer_for_peaks(gq)
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

            # Label: use the stored genotype peaks from `g`, not recalculated sensor peaks
            ml_peaks = [p for p in sorted(g) if p not in (420, 547)]
            if not ml_peaks:
                ml_peaks = list(sorted(g))
            label = "/".join(str(p) for p in ml_peaks)
            self.genotype_labels.append(label)

            theta = float(np.arccos(np.clip(d3[2], -1, 1)))
            phi = float(np.arctan2(d3[1], d3[0]))
            print(f"  {gq}  q_axis={q_axis}  d_null=(θ={theta:.3f}, φ={phi:.3f})  label='{label}'")

        self._verify_null_dirs()

        # Compute PCA of null direction variation across the CMF population
        self._compute_pca_null_dirs()

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

    def _disp_to_ref_viz(self, disp_pts):
        """DISP (N,4) -> polyscope 3-D through the CMF sampler viewer reference."""
        return self._disp_to_viz(self.ref_cst, disp_pts)

    def _create_observer_for_peaks(self, peaks_with_q, params=None):
        """Create observers the same way cmf_sampler_viewer creates nominal samples."""
        if params is None:
            params = {
                'od_lm': 0.5,
                'od_s': 0.4,
                'macular': 1.0,
                'lens': 1.0,
            }

        peaks = tuple(sorted(peaks_with_q))
        if 420 not in peaks:
            peaks = tuple(sorted((420,) + peaks))

        cones = []
        for peak in peaks:
            od = params['od_s'] if peak == 420 else params['od_lm']
            cone = Cone.templates['neitz'](
                self._og_wavelengths, peak
            ).with_preceptoral(od=od, macular=params['macular'], lens=params['lens'])
            cone.peak = int(peak)
            cones.append(cone)

        return Observer(cones, illuminant=None)

    def _maximal_null_disp_pts(self, cst, q_axis):
        """Return both display gamut endpoints using the sampler viewer path."""
        background = np.ones(cst.dim) * 0.5
        result = cst.get_maximal_pair_in_disp_from_pt(
            pt=background,
            metameric_axis=q_axis,
            input_space=ColorSpaceType.DISP,
            output_space=ColorSpaceType.DISP,
            proportion=1.0,
        )
        if result is None:
            return background.reshape(1, -1).repeat(2, axis=0)
        disp1, disp2, _ = result
        return np.array([disp1, disp2])

    def _null_gamut_disp(self, idx, sign=1):
        """DISP point at the gamut boundary along the null direction.

        With the trichromat-luminance chrom_basis, meta4 lies exactly in the
        column space of cb, so  cb @ (cb.T @ meta4) == meta4  and walking along
        cb @ d3 changes only Q — verified by _verify_null_dirs.
        """
        cst = self.color_spaces[idx]
        endpoints = self._maximal_null_disp_pts(cst, self.q_axes[idx])
        return endpoints[0 if sign > 0 else 1].copy()

    def _null_gamut_pt(self, idx, sign=1):
        """Viz coords of the null-direction gamut boundary for observer idx.

        Uses the same fixed reference observer/conversion as cmf_sampler_viewer.
        """
        disp_pt = self._null_gamut_disp(idx, sign)
        return self._disp_to_ref_viz(disp_pt.reshape(1, -1))[0]

    def _verify_null_dirs(self):
        """Print cone-space differences at the null-direction gamut endpoints."""
        print("\n=== Null-direction metamer verification ===")
        header = f"  {'observer':<28}  {'ΔS':>8}  {'ΔM':>8}  {'ΔQ':>8}  {'ΔL':>8}  {'|ΔLMS|':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx, (cst, q_axis) in enumerate(zip(self.color_spaces, self.q_axes)):
            peaks = tuple(int(round(c.peak)) for c in cst.observer.sensors)

            dp = self._null_gamut_disp(idx, +1)
            dn = self._null_gamut_disp(idx, -1)

            cp = cst.convert(dp.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]
            cn = cst.convert(dn.reshape(1, -1), ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

            diff = cp - cn
            non_q = [i for i in range(cst.dim) if i != q_axis]
            lms_residual = np.linalg.norm(diff[non_q])

            vals = "  ".join(f"{diff[i]:+8.5f}" for i in range(cst.dim))
            print(f"  {str(peaks):<28}  {vals}  |ΔLMS|={lms_residual:.2e}")

        print("=== end verification ===\n")

    def _background_viz(self, idx=0):
        w0 = self.w0s[idx]
        return self._disp_to_ref_viz(w0.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    # PCA of null-direction variation
    # ------------------------------------------------------------------

    def _compute_pca_null_dirs(self):
        """Sample observers via CMFSampler and PCA their null directions.

        Uses top_n_genotypes=10 and varies od/macular/lens continuously.
        Results are stored as:
          pca_mean_null : (3,) mean null direction in obs-0 chrom space
          pca_e1, pca_e2: (3,) PCA eigenvectors in tangent plane (e1 = largest var)

        Also sets patch_angle_major / patch_angle_minor to 2.5-sigma coverage.
        """
        print(f"Computing PCA of null directions via CMFSampler "
              f"({self.N_PCA_SAMPLES} samples, top-10 genotypes)…")

        # Match cmf_sampler_viewer defaults and top-genotype sampling so this
        # cloud is comparable to that viewer's match point distribution.
        sampler = CMFSampler(
            self._og,
            wavelengths=self._og_wavelengths,
            seed=42,
            top_n_genotypes=10,
        )
        samples = sampler.sample(self.N_PCA_SAMPLES, sex='both')

        cb0 = self.chrom_bases[0]   # obs-0 chrom basis: (4, 3)

        null_dirs_sampled = []
        null_viz_raw = []   # gamut-boundary viz point per sample
        od_lm_vals, lens_vals, macular_vals = [], [], []

        for obs_tri, params in samples:
            genotype = params['genotype']
            peaks_with_q = tuple(sorted(genotype + (547,)))
            if 420 not in peaks_with_q:
                peaks_with_q = (420,) + peaks_with_q
            peaks_with_q = tuple(sorted(peaks_with_q))

            cones = []
            for peak in peaks_with_q:
                od = params['od_s'] if peak == 420 else params['od_lm']
                cone = Cone.templates['neitz'](self._og_wavelengths, peak).with_preceptoral(
                    od=od, macular=params['macular'], lens=params['lens']
                )
                cone.peak = int(peak)
                cones.append(cone)

            try:
                obs_full = Observer(cones, illuminant=None)
                cst_i = ColorSpace(obs_full, self.primaries)
                q_axis_i = next(
                    (i for i, c in enumerate(obs_full.sensors) if abs(c.peak - 547.0) < 1.0),
                    2)

                meta4 = cst_i.get_metameric_axis_in(
                    ColorSpaceType.DISP,
                    metameric_axis_num=q_axis_i,
                )
                norm_m = np.linalg.norm(meta4)
                if norm_m < 1e-8:
                    continue
                meta4 /= norm_m

                d3 = cb0.T @ meta4
                norm = np.linalg.norm(d3)
                if norm < 1e-8:
                    continue
                d3 /= norm

                # Direction sign can be standardized here if the PCA should only
                # use one pole of the null axis.
                # if np.dot(d3, self.null_dirs_3d[0]) < 0:
                #     d3    = -d3
                #     meta4 = -meta4

                null_dirs_sampled.append(d3)

                # Match cmf_sampler_viewer's maximal-pair path, but render both
                # endpoints of the null axis.
                disp_pts = self._maximal_null_disp_pts(cst_i, q_axis_i)
                viz_pts = self._disp_to_ref_viz(disp_pts)
                null_viz_raw.extend(viz_pts)

                for _ in range(len(viz_pts)):
                    od_lm_vals.append(params['od_lm'])
                    lens_vals.append(params['lens'])
                    macular_vals.append(params['macular'])
            except Exception:
                pass

        null_dirs_sampled = np.array(null_dirs_sampled)  # (N, 3)
        print(f"  {len(null_dirs_sampled)} null directions collected")

        # Build colour array: R=od_lm, G=lens, B=macular (each normalized 0-1)
        od_arr = np.array(od_lm_vals)
        ln_arr = np.array(lens_vals)
        mac_arr = np.array(macular_vals)

        def _norm(a):
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo + 1e-8)
        pca_colors = np.stack([_norm(od_arr), _norm(ln_arr), _norm(mac_arr)], axis=1)

        self.pca_null_viz_pts = np.array(null_viz_raw)  # (N, 3) – for rendering
        self.pca_null_colors = pca_colors               # (N, 3)

        # Mean direction on sphere
        mean_dir = null_dirs_sampled.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)

        # Project to tangent plane of mean_dir
        cos_t = null_dirs_sampled @ mean_dir                    # (N,)
        tangent = null_dirs_sampled - cos_t[:, None] * mean_dir  # (N, 3)

        # PCA via SVD on tangent matrix
        _, s, Vt = np.linalg.svd(tangent, full_matrices=False)
        e1 = Vt[0]   # direction of largest variance in tangent plane
        e2 = Vt[1]   # second largest

        # Angular spread: 5-sigma so the patch comfortably covers all sampled variation
        t_e1 = tangent @ e1
        t_e2 = tangent @ e2
        spread_e1 = float(np.std(t_e1) * 5.0)
        spread_e2 = float(np.std(t_e2) * 5.0)
        print(f"  PCA spread → major={spread_e1:.4f} rad  minor={spread_e2:.4f} rad")

        # Diagnostic: angle between cloud mean and observer-0 nominal null direction
        obs0_null = self.null_dirs_3d[0]
        cos_align = float(np.clip(np.dot(mean_dir, obs0_null), -1., 1.))
        align_deg = np.degrees(np.arccos(cos_align))
        print(f"  Cloud mean vs obs-0 null: angle = {align_deg:.2f}°  "
              f"(0° = perfectly aligned; large value = parameter mismatch remains)")

        self.pca_mean_null = mean_dir
        self.pca_e1 = e1
        self.pca_e2 = e2
        self.patch_angle_major = max(spread_e1, 0.15)
        self.patch_angle_minor = max(spread_e2, 0.06)

    # ------------------------------------------------------------------
    # Surface computation
    # ------------------------------------------------------------------

    def _sphere_grid(self):
        thetas = np.linspace(0.01, np.pi - 0.01, self.N_THETA)
        phis = np.linspace(-np.pi, np.pi, self.N_PHI, endpoint=False)
        return thetas, phis

    def _grid_faces(self, nt, np_):
        """Triangle faces for an (nt × np_) spherical grid (phi wraps). Vectorised."""
        i = np.arange(nt - 1)
        j = np.arange(np_)
        II, JJ = np.meshgrid(i, j, indexing='ij')
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

        Polar form:  r(α) = 1 / sqrt( cos²(α)/σ_major² + sin²(α)/σ_minor² )
        """
        bg_viz = self._background_viz(idx)
        null_viz = self._null_dir_viz(idx)

        sm2 = max(self.sigma_major, 1e-4) ** 2
        sn2 = max(self.sigma_minor, 1e-4) ** 2

        thetas, phis = self._sphere_grid()
        d3_all = sphere_grid_batch(thetas, phis)

        cos_a = np.clip(d3_all @ null_viz, -1., 1.)
        sin_a = np.sqrt(np.maximum(1. - cos_a ** 2, 0.))
        r_stars = 1.0 / np.sqrt(cos_a ** 2 / sm2 + sin_a ** 2 / sn2)

        viz_pts = bg_viz + r_stars[:, None] * d3_all
        faces = self._grid_faces(self.N_THETA, self.N_PHI)
        return viz_pts, r_stars, np.zeros(len(r_stars), dtype=bool), faces

    def _compute_vl_hyperplane_solid(self, idx=0):
        """Boundary of the 4D gamut intersected with the V(λ)-constant hyperplane."""
        cst, w0, cb = (self.color_spaces[idx], self.w0s[idx],
                       self.chrom_bases[idx])
        thetas, phis = self._sphere_grid()
        d3_all = sphere_grid_batch(thetas, phis)
        rm = r_max_batch(w0, cb, d3_all)
        disp_pts = w0 + rm[:, None] * (d3_all @ cb.T)

        viz_pts = self._disp_to_viz(cst, disp_pts)
        faces = self._grid_faces(self.N_THETA, self.N_PHI)

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

    def _gamut_patch_pca(self, idx=0):
        """Gamut boundary patch: elliptical cap parameterized along PCA axes.

        Uses pca_mean_null as the patch centre and pca_e1/e2 as the major/minor
        axes.  The in-patch criterion is:

            (t_e1 / sin(patch_angle_major))² + (t_e2 / sin(patch_angle_minor))² ≤ 1

        where t_e1, t_e2 are the tangent-plane projections of each direction
        onto e1 and e2 (equal to sin(angle)*cos(φ), sin(angle)*sin(φ)).
        This naturally covers both ±mean_null poles.
        """
        cst, w0, cb = (self.color_spaces[idx], self.w0s[idx],
                       self.chrom_bases[idx])
        thetas, phis = self._sphere_grid()
        bg_viz = self._background_viz(idx)
        d3_all = sphere_grid_batch(thetas, phis)              # (N, 3)

        mean_null = self.pca_mean_null                         # (3,) unit
        e1, e2 = self.pca_e1, self.pca_e2

        cos_total = d3_all @ mean_null                         # (N,)
        tangent = d3_all - cos_total[:, None] * mean_null   # (N, 3)
        t_e1 = tangent @ e1                                    # (N,)
        t_e2 = tangent @ e2                                    # (N,)

        sin_maj = np.sin(max(self.patch_angle_major, 1e-4))
        sin_min = np.sin(max(self.patch_angle_minor, 1e-4))
        in_patch = (t_e1 / sin_maj) ** 2 + (t_e2 / sin_min) ** 2 <= 1  # (N,)

        rm = r_max_batch(w0, cb, d3_all)
        disp_pts = np.where(in_patch[:, None],
                            w0 + rm[:, None] * (d3_all @ cb.T),
                            w0)

        viz_pts = self._disp_to_viz(cst, disp_pts)
        viz_pts[~in_patch] = bg_viz

        all_faces = self._grid_faces(self.N_THETA, self.N_PHI)
        patch_faces = all_faces[np.all(in_patch[all_faces], axis=1)]

        return viz_pts, patch_faces

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        print("\n=== Rendering ===")
        self._clear()

        # --- 1. V(λ)-constant hyperplane solid ----------------------------
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

        # --- 2. Null direction points (± both sides, all observers) -------
        null_pts, null_colors = [], []
        for i in range(len(self.color_spaces)):
            col = np.array(self.OBSERVER_COLORS[i % len(self.OBSERVER_COLORS)])
            for sign in (+1, -1):
                null_pts.append(self._null_gamut_pt(i, sign))
                null_colors.append(col * (1.0 if sign > 0 else 0.45))

        null_pts = np.array(null_pts)
        null_colors = np.clip(np.array(null_colors), 0, 1)
        RenderPointCloud("null_pts", null_pts, null_colors, radius=self.point_size)

        if self.labels_enabled:
            # Billboard text labels at each +side null-direction point, offset outward
            for i in range(len(self.color_spaces)):
                null_pt = self._null_gamut_pt(i, sign=+1)
                null_dir = self._null_dir_viz(i)             # unit direction in viz
                label_pos = null_pt + null_dir * 0.08         # push label away from surface
                col = self.OBSERVER_COLORS[i % len(self.OBSERVER_COLORS)]
                try:
                    ps.register_billboard_text(
                        f"label_obs_{i:02d}",
                        self.genotype_labels[i],
                        label_pos,
                        enabled=True,
                        font_size=self.font_size,
                        text_color=col,
                    )
                except Exception:
                    pass

        # CMF sampled null-direction points (small, R=od_lm, G=lens, B=macular)
        if hasattr(self, 'pca_null_viz_pts') and len(self.pca_null_viz_pts) > 0:
            RenderPointCloud("cmf_null_pts", self.pca_null_viz_pts,
                             self.pca_null_colors, radius=self.point_size * 0.25)

        # Reference null direction arrow (first observer)
        bg_viz = self._background_viz(0)
        null_end = self._null_gamut_pt(0, +1)
        Render3DLine("null_arrow",
                     np.array([bg_viz, null_end]),
                     np.array([0.2, 1.0, 0.2]),
                     radius=0.004)

        # --- 3. Gamut surface patch (PCA-parameterized elliptical cap) ----
        print("  Computing PCA gamut patch…")
        patch_viz, patch_faces = self._gamut_patch_pca(0)
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
                idx_used = np.unique(patch_faces.ravel())
                RenderPointCloud("null_gamut_patch",
                                 patch_viz[idx_used],
                                 np.tile([0.15, 0.75, 1.0], (len(idx_used), 1)),
                                 radius=0.01)

        # --- 4. Ellipsoid threshold contour surface -----------------------
        viz_pts, r_stars, gamut_mask, faces = self._threshold_surface(0)

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
        for name in ["null_pts", "cmf_null_pts", "bg_sphere", "gaussian_contour",
                     "null_gamut_patch", "vl_hyperplane_solid"]:
            try:
                ps.remove_point_cloud(name)
            except:
                pass
        try:
            ps.remove_curve_network("null_arrow")
        except:
            pass
        for i in range(self.num_observers):
            try:
                ps.remove_billboard_text(f"label_obs_{i:02d}")
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
        psim.Text("Gamut patch  (PCA elliptical cap)")
        _, new_pa_maj = psim.SliderFloat("Patch angle major (rad)", self.patch_angle_major, 0.02, 1.2)
        _, new_pa_min = psim.SliderFloat("Patch angle minor (rad)", self.patch_angle_minor, 0.01, 1.2)

        psim.Separator()
        _, new_pt_size = psim.SliderFloat("Point size",  self.point_size,  0.002, 0.05)
        new_font = self.font_size

        changed = (
            abs(new_sm - self.sigma_major) > 1e-4 or
            abs(new_sn - self.sigma_minor) > 1e-4 or
            abs(new_pa_maj - self.patch_angle_major) > 1e-4 or
            abs(new_pa_min - self.patch_angle_minor) > 1e-4 or
            abs(new_pt_size - self.point_size) > 1e-4 or
            abs(new_font - self.font_size) > 1e-4
        )

        self.sigma_major = new_sm
        self.sigma_minor = new_sn
        self.patch_angle_major = new_pa_maj
        self.patch_angle_minor = new_pa_min
        self.point_size = new_pt_size
        self.font_size = new_font

        if changed:
            self._render()

        psim.Separator()
        psim.Text("Observer genotype legend  (M-peak / L-peak nm):")
        for i, lbl in enumerate(self.genotype_labels):
            col = self.OBSERVER_COLORS[i % len(self.OBSERVER_COLORS)]
            psim.Text(f"  [{i+1:2d}] {lbl}  (RGB ~ {col[0]:.1f},{col[1]:.1f},{col[2]:.1f})")

        psim.Separator()
        psim.Text("Legend:")
        psim.Text("  Colored pts    – null direction ± gamut endpoints per CMF")
        psim.Text("  Green arrow    – null direction (observer 1)")
        psim.Text("  Cyan surface   – PCA-parameterized gamut patch")
        psim.Text("  Hot surface    – ellipsoid: r=1/sqrt(cos²α/σ_maj²+sin²α/σ_min²)")

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
