"""
Monte Carlo sampling of cone spectral sensitivities (CMFs) with physiological variation.

Samples observer genotypes (discrete, population-weighted) and combines with continuous
sampling of photopigment optical density, macular pigment density, and lens density
to generate a population of observer CMFs.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import matplotlib.pyplot as plt

from TetriumColor.Observer import Observer, Cone

if TYPE_CHECKING:
    from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes


class CMFSampler:
    """
    Monte Carlo sampler for observer CMF variation.

    Combines discrete genotype sampling (weighted by population frequency) with
    continuous sampling of physiological parameters (od_lm, od_s, macular, lens).
    """

    def __init__(
        self,
        observer_genotypes: 'ObserverGenotypes',
        od_lm_mean: float = 0.5,
        od_lm_std: float = 0.05,
        od_s_mean: float = 0.4,
        od_s_std: float = 0.04,
        macular_mean: float = 1.0,
        macular_std: float = 0.25,
        lens_mean: float = 1.0,
        lens_std: float = 0.1,
        template: str = 'baylor',
        wavelengths: Optional[np.ndarray] = None,
        seed: int = 42,
        params: Optional[Dict[str, float]] = None,
        top_n_genotypes: Optional[int] = None,
    ):
        """
        Initialize CMFSampler.

        Args:
            observer_genotypes: ObserverGenotypes instance (provides genotype PDFs)
            od_lm_mean, od_lm_std: L/M cone photopigment OD distribution
            od_s_mean, od_s_std: S cone photopigment OD distribution
            macular_mean, macular_std: Macular pigment density distribution
            lens_mean, lens_std: Lens density distribution
            template: Cone nomogram template ('baylor', 'neitz', etc.)
            wavelengths: Wavelength array; defaults to observer_genotypes.wavelengths
            seed: Random seed
            params: Optional dict of parameter overrides. Allows specifying all or some
                   parameters as a dict, e.g., params={'macular_std': 0.0, 'od_lm_std': 0.1}
                   Dict values override individual kwargs.
            top_n_genotypes: If set, only sample from the top N most probable genotypes
                           instead of the entire PDF. Default None (use all genotypes).

        Examples:
            # Vary only macular pigment (fix od and lens to defaults)
            sampler = CMFSampler(og, macular_std=0.1, od_lm_std=0.0, od_s_std=0.0, lens_std=0.0)

            # Same thing using params dict
            sampler = CMFSampler(og, params={'macular_std': 0.1, 'od_lm_std': 0.0,
                                              'od_s_std': 0.0, 'lens_std': 0.0})

            # Mix both (dict overrides kwargs)
            sampler = CMFSampler(og, macular_std=0.5, params={'od_lm_std': 0.0})
        """
        # Apply params dict overrides if provided
        if params is not None:
            if 'od_lm_mean' in params:
                od_lm_mean = params['od_lm_mean']
            if 'od_lm_std' in params:
                od_lm_std = params['od_lm_std']
            if 'od_s_mean' in params:
                od_s_mean = params['od_s_mean']
            if 'od_s_std' in params:
                od_s_std = params['od_s_std']
            if 'macular_mean' in params:
                macular_mean = params['macular_mean']
            if 'macular_std' in params:
                macular_std = params['macular_std']
            if 'lens_mean' in params:
                lens_mean = params['lens_mean']
            if 'lens_std' in params:
                lens_std = params['lens_std']
        self.observer_genotypes = observer_genotypes
        self.od_lm_mean = od_lm_mean
        self.od_lm_std = od_lm_std
        self.od_s_mean = od_s_mean
        self.od_s_std = od_s_std
        self.macular_mean = macular_mean
        self.macular_std = macular_std
        self.lens_mean = lens_mean
        self.lens_std = lens_std
        self.template = template
        self.wavelengths = wavelengths if wavelengths is not None else observer_genotypes.wavelengths
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.top_n_genotypes = top_n_genotypes

    def sample(
        self,
        n_samples: int,
        sex: str = 'both',
    ) -> List[Tuple[Observer, Dict[str, float]]]:
        """
        Generate n_samples of (Observer, params_dict) pairs.

        Args:
            n_samples: Number of samples to generate
            sex: 'male', 'female', or 'both'

        Returns:
            List of (Observer, {genotype, od_lm, od_s, macular, lens}) tuples
        """
        # Get genotype distribution
        pdf = self.observer_genotypes.get_pdf(sex)
        genotypes = list(pdf.keys())
        probabilities = np.array(list(pdf.values()))

        # Filter to top N genotypes if specified
        if self.top_n_genotypes is not None:
            # Sort by probability (descending) and take top N
            sorted_indices = np.argsort(probabilities)[::-1][:self.top_n_genotypes]
            genotypes = [genotypes[i] for i in sorted_indices]
            probabilities = probabilities[sorted_indices]
            # Renormalize probabilities
            probabilities = probabilities / probabilities.sum()

        # Sample genotype indices (np.choice preserves tuple structure better via indexing)
        indices = self.rng.choice(len(genotypes), p=probabilities, size=n_samples)
        sampled_genotypes = [genotypes[i] for i in indices]

        # Sample physiological parameters (continuous, clipped to reasonable ranges)
        od_lm = self.rng.normal(self.od_lm_mean, self.od_lm_std, n_samples).clip(0.1, 0.9)
        od_s = self.rng.normal(self.od_s_mean, self.od_s_std, n_samples).clip(0.1, 0.9)
        macular = self.rng.normal(self.macular_mean, self.macular_std, n_samples).clip(0.01, 3.0)
        lens = self.rng.normal(self.lens_mean, self.lens_std, n_samples).clip(0.01, 3.0)

        # Build observers
        observers_and_params = []
        for i in range(n_samples):
            genotype = sampled_genotypes[i]
            observer = self._create_observer(
                genotype=genotype,
                od_lm=od_lm[i],
                od_s=od_s[i],
                macular=macular[i],
                lens=lens[i],
            )

            params = {
                'genotype': genotype,
                'od_lm': float(od_lm[i]),
                'od_s': float(od_s[i]),
                'macular': float(macular[i]),
                'lens': float(lens[i]),
            }

            observers_and_params.append((observer, params))

        return observers_and_params

    def _create_observer(
        self,
        genotype: Tuple[float, ...],
        od_lm: float,
        od_s: float,
        macular: float,
        lens: float,
    ) -> Observer:
        """
        Create an Observer for a specific genotype and physiological parameters.

        Args:
            genotype: Tuple of cone peak wavelengths
            od_lm: L/M photopigment optical density
            od_s: S photopigment optical density
            macular: Macular pigment density multiplier
            lens: Lens density multiplier

        Returns:
            Observer object
        """
        # Ensure S cone is present (peak 420)
        peaks = genotype if 420 in genotype else (420,) + genotype
        peaks = tuple(sorted(peaks))

        cones = []
        for peak in peaks:
            # Choose OD based on cone type
            od = od_s if peak == 420 else od_lm

            # Create cone via nomogram template + pre-receptoral filtering
            # (bypasses Cone.cone() OD restriction by calling with_preceptoral directly)
            template = 'neitz' if peak == 420 else self.template
            cone = Cone.templates[template](self.wavelengths, peak).with_preceptoral(
                od=od, macular=macular, lens=lens
            )
            # Preserve original peak (with_preceptoral recalculates peak from modified data)
            cone.peak = int(peak)
            cones.append(cone)

        return Observer(cones, illuminant='raw')

    def plot_sampled_cmfs(
        self,
        samples: List[Tuple[Observer, Dict[str, float]]],
        output_path: str = "sampled_cmfs.png",
        figsize: Tuple[int, int] = (12, 8),
        single_plot: bool = True,
    ) -> None:
        """
        Plot all sampled CMFs (sensor matrices) and save to image.

        Args:
            samples: List of (Observer, params_dict) tuples from sample()
            output_path: Path to save the plot image (default "sampled_cmfs.png")
            figsize: Figure size in inches (default (12, 8))
            single_plot: If True, plot all cones on one axis with each observer as a color.
                        If False, use subplots for each cone position (default True)
        """
        if single_plot:
            self._plot_cmfs_single_axis(samples, output_path, figsize)
        else:
            self._plot_cmfs_subplots(samples, output_path, figsize)

    def _plot_cmfs_single_axis(
        self,
        samples: List[Tuple[Observer, Dict[str, float]]],
        output_path: str,
        figsize: Tuple[int, int],
    ) -> None:
        """Plot all CMFs on a single axis, each observer as a different color."""
        fig, ax = plt.subplots(figsize=figsize)

        # Use a colormap to color each observer
        cmap = plt.cm.get_cmap('tab20' if len(samples) <= 20 else 'hsv')

        # Plot each observer's cones
        for sample_idx, (observer, params) in enumerate(samples):
            color = cmap(sample_idx / len(samples))

            for cone in observer.sensors:
                ax.plot(
                    cone.wavelengths,
                    cone.data,
                    color=color,
                    alpha=0.6,
                    linewidth=1.0,
                    label=f"Observer {sample_idx}" if cone == observer.sensors[0] else "",
                )

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_xlim(380, 780)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sampled CMFs - All Cones ({len(samples)} observers)', fontsize=14)

        # Add legend (show every 5th observer to avoid crowding)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles[::max(1, len(handles)//5)], labels[::max(1, len(labels)//5)],
                     loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved CMF plot to {output_path}")
        plt.close()

    def _plot_cmfs_subplots(
        self,
        samples: List[Tuple[Observer, Dict[str, float]]],
        output_path: str,
        figsize: Tuple[int, int],
    ) -> None:
        """Plot CMFs in subplots, one per cone position."""
        n_cones = len(samples[0][0].sensors)
        cone_colors = {
            420: 'purple',     # S cone
            530: 'green',      # M cone
            533: 'green',
            536: 'green',
            545: 'orange',     # Q cone (if present)
            547: 'orange',
            551: 'orange',
            552: 'red',        # L cone
            553: 'red',
            555: 'red',
            556: 'red',
            559: 'red',
        }

        fig, axes = plt.subplots(1, n_cones, figsize=figsize)
        if n_cones == 1:
            axes = [axes]

        # Plot each cone type across all samples
        for sample_idx, (observer, params) in enumerate(samples):
            for cone_idx, cone in enumerate(observer.sensors):
                peak = cone.peak
                color = cone_colors.get(peak, 'gray')
                alpha = 0.1 + (0.3 * (sample_idx / len(samples)))

                ax = axes[cone_idx]
                ax.plot(
                    cone.wavelengths,
                    cone.data,
                    color=color,
                    alpha=alpha,
                    linewidth=0.5,
                )

        # Format axes
        for cone_idx, ax in enumerate(axes):
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Sensitivity')
            ax.set_xlim(380, 780)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

            # Get cone types in this position across samples
            peaks = [s[0].sensors[cone_idx].peak for s in samples]
            peak_str = ', '.join(map(str, sorted(set(peaks))))
            ax.set_title(f'Cone #{cone_idx} (peaks: {peak_str})')

        plt.suptitle(
            f'Sampled CMFs ({len(samples)} observers)\n'
            f'Color indicates cone type, alpha indicates sample index',
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved CMF plot to {output_path}")
        plt.close()
