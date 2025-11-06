"""
Chromaticity analysis utilities for Quest threshold measurements.

This module provides tools for:
1. Fitting ellipsoids to threshold data in DISP space
2. Visualizing thresholds in DISP space diagrams
3. Classifying genotypes based on threshold patterns

Note: All analysis is performed in DISP (display) space, not chromaticity space.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType
from TetriumColor.Observer.ObserverGenotypes import ObserverGenotypes


class EllipsoidFitter:
    """Fit ellipsoids to threshold data in 2D or 3D chromaticity space."""

    def __init__(self, dimension: int = 2):
        """Initialize ellipsoid fitter.

        Args:
            dimension: Dimensionality of chromaticity space (2 for trichromats, 3 for tetrachromats)
        """
        self.dimension = dimension
        self.center = None
        self.radii = None
        self.rotation = None
        self.covariance = None

    @staticmethod
    def fit_ellipse_2d(points: npt.NDArray) -> Dict:
        """Fit 2D ellipse to points using algebraic fit.

        Args:
            points: Nx2 array of points in 2D chromaticity space

        Returns:
            Dictionary containing: center, radii (semi-major/minor axes), angle, covariance
        """
        x = points[:, 0]
        y = points[:, 1]

        # Center the data
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_c = x - x_mean
        y_c = y - y_mean

        # Compute covariance matrix
        cov = np.cov(x_c, y_c)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Semi-axes lengths (scale by chi-squared for 95% confidence)
        # For 2D, chi-squared with 2 DOF at 95% is ~5.991
        scale = np.sqrt(5.991)
        radii = scale * np.sqrt(eigenvalues)

        # Rotation angle
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        return {
            'center': np.array([x_mean, y_mean]),
            'radii': radii,
            'angle': angle,
            'rotation': eigenvectors,
            'covariance': cov,
            'eigenvalues': eigenvalues
        }

    @staticmethod
    def fit_ellipsoid_3d(points: npt.NDArray) -> Dict:
        """Fit 3D ellipsoid to points.

        Args:
            points: Nx3 array of points in 3D chromaticity space

        Returns:
            Dictionary containing: center, radii, rotation matrix, covariance
        """
        # Center the data
        center = np.mean(points, axis=0)
        points_centered = points - center

        # Compute covariance matrix
        cov = np.cov(points_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalues (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Semi-axes lengths (scale by chi-squared for 95% confidence)
        # For 3D, chi-squared with 3 DOF at 95% is ~7.815
        scale = np.sqrt(7.815)
        radii = scale * np.sqrt(eigenvalues)

        return {
            'center': center,
            'radii': radii,
            'rotation': eigenvectors,
            'covariance': cov,
            'eigenvalues': eigenvalues
        }

    def fit(self, directions: npt.NDArray, thresholds: npt.NDArray) -> Dict:
        """Fit ellipsoid to threshold data.

        Args:
            directions: Nxdim array of normalized direction vectors
            thresholds: N array of threshold values (distances)

        Returns:
            Dictionary with ellipsoid parameters
        """
        # Convert to Cartesian coordinates
        points = directions * thresholds[:, np.newaxis]

        if self.dimension == 2:
            result = self.fit_ellipse_2d(points)
        elif self.dimension == 3:
            result = self.fit_ellipsoid_3d(points)
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")

        # Store parameters
        self.center = result['center']
        self.radii = result['radii']
        self.rotation = result['rotation']
        self.covariance = result['covariance']

        return result

    def get_ellipse_points(self, n_points: int = 100) -> npt.NDArray:
        """Generate points on the fitted ellipse (2D only).

        Args:
            n_points: Number of points to generate

        Returns:
            Array of points on ellipse
        """
        if self.dimension != 2:
            raise ValueError("get_ellipse_points only works for 2D")

        if self.center is None:
            raise ValueError("Must fit ellipse first")

        # Generate points on unit circle
        theta = np.linspace(0, 2*np.pi, n_points)
        circle = np.array([np.cos(theta), np.sin(theta)]).T

        # Scale by radii
        ellipse = circle * self.radii

        # Rotate
        ellipse = ellipse @ self.rotation.T

        # Translate to center
        ellipse = ellipse + self.center

        return ellipse

    def get_ellipsoid_surface(self, n_points: int = 50) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generate surface mesh for fitted ellipsoid (3D only).

        Args:
            n_points: Number of points per dimension

        Returns:
            Tuple of (X, Y, Z) meshgrid arrays
        """
        if self.dimension != 3:
            raise ValueError("get_ellipsoid_surface only works for 3D")

        if self.center is None:
            raise ValueError("Must fit ellipsoid first")

        # Generate sphere
        u = np.linspace(0, 2*np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Stack into points
        sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Scale by radii
        ellipsoid_points = sphere_points * self.radii

        # Rotate
        ellipsoid_points = ellipsoid_points @ self.rotation.T

        # Translate
        ellipsoid_points = ellipsoid_points + self.center

        # Reshape back to mesh
        X = ellipsoid_points[:, 0].reshape(x.shape)
        Y = ellipsoid_points[:, 1].reshape(y.shape)
        Z = ellipsoid_points[:, 2].reshape(z.shape)

        return X, Y, Z


class ChromaticityVisualizer:
    """Visualize thresholds and gamuts in DISP space."""

    def __init__(self, color_space: ColorSpace):
        """Initialize visualizer with a color space.

        Args:
            color_space: ColorSpace object defining the observer and display
        """
        self.color_space = color_space
        self.dimension = color_space.dim  # DISP space dimension

    def plot_thresholds_2d(self, thresholds_dict: Dict,
                           background_luminance: float = 0.5,
                           show_ellipse: bool = True,
                           show_gamut: bool = True,
                           ax: Optional[plt.Axes] = None,
                           dims: Tuple[int, int] = (0, 1)) -> plt.Figure:
        """Plot 2D projection of DISP space with thresholds.

        Args:
            thresholds_dict: Dictionary from QuestColorGenerator.get_thresholds()
            background_luminance: Luminance level for background (unused in DISP space)
            show_ellipse: Whether to show fitted ellipse
            show_gamut: Whether to show display gamut boundary
            ax: Optional matplotlib axes to plot on
            dims: Which two dimensions to plot (default (0,1) for R-G)

        Returns:
            Figure object
        """

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Extract directions and thresholds
        directions = np.array([data['direction'] for data in thresholds_dict.values()])
        thresholds = np.array([data['threshold'] for data in thresholds_dict.values()])

        # Project to 2D using specified dimensions
        directions_2d = directions[:, [dims[0], dims[1]]]

        # Convert to Cartesian coordinates
        threshold_points = directions_2d * thresholds[:, np.newaxis]

        # Background is at 0.5 in DISP space
        background_2d = np.array([0.5, 0.5])

        # Plot threshold points relative to background
        ax.scatter(threshold_points[:, 0], threshold_points[:, 1],
                   c='blue', s=50, alpha=0.6, label='Measured Thresholds')

        # Plot direction vectors from origin
        for direction_2d, threshold in zip(directions_2d, thresholds):
            point = direction_2d * threshold
            ax.plot([0, point[0]], [0, point[1]], 'b-', alpha=0.3, linewidth=0.5)

        # Fit and plot ellipse in 2D projection
        if show_ellipse:
            fitter = EllipsoidFitter(dimension=2)
            ellipse_params = fitter.fit(directions_2d, thresholds)
            ellipse_points = fitter.get_ellipse_points(n_points=100)
            ax.plot(ellipse_points[:, 0], ellipse_points[:, 1],
                    'r-', linewidth=2, label='Fitted Ellipse')

            # Plot center
            ax.scatter(*ellipse_params['center'], c='red', s=100,
                       marker='x', label='Ellipse Center')

        # Plot display gamut (unit cube edges in selected dimensions)
        if show_gamut:
            # Plot unit square boundary
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0],
                    'k-', linewidth=1.5, alpha=0.7, label='Display Gamut')

        # Mark origin and background
        ax.scatter(0, 0, c='black', s=100, marker='+', linewidth=2, label='Origin')
        ax.scatter(*background_2d, c='gray', s=100, marker='o', linewidth=2, label='Background')

        dim_names = ['R', 'G', 'B', 'O']
        ax.set_xlabel(f'DISP {dim_names[dims[0]]} Dimension', fontsize=12)
        ax.set_ylabel(f'DISP {dim_names[dims[1]]} Dimension', fontsize=12)
        ax.set_title('DISP Space Threshold Diagram', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        return fig

    def plot_thresholds_3d(self, thresholds_dict: Dict,
                           background_luminance: float = 0.5,
                           show_ellipsoid: bool = True,
                           show_gamut: bool = True,
                           ax: Optional[Axes3D] = None,
                           dims: Tuple[int, int, int] = (0, 1, 2)) -> plt.Figure:
        """Plot 3D projection of DISP space with thresholds.

        Args:
            thresholds_dict: Dictionary from QuestColorGenerator.get_thresholds()
            background_luminance: Luminance level for background (unused in DISP space)
            show_ellipsoid: Whether to show fitted ellipsoid
            show_gamut: Whether to show display gamut boundary
            ax: Optional 3D axes to plot on
            dims: Which three dimensions to plot (default (0,1,2) for R-G-B)

        Returns:
            Figure object
        """

        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # Extract directions and thresholds
        directions = np.array([data['direction'] for data in thresholds_dict.values()])
        thresholds = np.array([data['threshold'] for data in thresholds_dict.values()])

        # Project to 3D using specified dimensions
        directions_3d = directions[:, [dims[0], dims[1], dims[2]]]

        # Convert to Cartesian coordinates
        threshold_points = directions_3d * thresholds[:, np.newaxis]

        # Background is at 0.5 in DISP space
        background_3d = np.array([0.5, 0.5, 0.5])

        # Plot threshold points
        ax.scatter(threshold_points[:, 0], threshold_points[:, 1], threshold_points[:, 2],
                   c='blue', s=50, alpha=0.6, label='Measured Thresholds')

        # Plot direction vectors from origin
        for direction_3d, threshold in zip(directions_3d, thresholds):
            point = direction_3d * threshold
            ax.plot([0, point[0]], [0, point[1]], [0, point[2]],
                    'b-', alpha=0.3, linewidth=0.5)

        # Fit and plot ellipsoid in 3D projection
        if show_ellipsoid:
            fitter = EllipsoidFitter(dimension=3)
            ellipsoid_params = fitter.fit(directions_3d, thresholds)
            X, Y, Z = fitter.get_ellipsoid_surface(n_points=30)
            ax.plot_surface(X, Y, Z, alpha=0.3, color='red')

            # Plot center
            center = ellipsoid_params['center']
            ax.scatter(*center, c='red', s=100, marker='x', label='Ellipsoid Center')

        # Plot display gamut (unit cube edges in selected dimensions)
        if show_gamut:
            # Plot unit cube wireframe
            from itertools import product, combinations
            cube_corners = np.array(list(product([0, 1], repeat=3)))
            for s, e in combinations(range(8), 2):
                if np.sum(np.abs(cube_corners[s] - cube_corners[e])) == 1:
                    ax.plot([cube_corners[s, 0], cube_corners[e, 0]],
                            [cube_corners[s, 1], cube_corners[e, 1]],
                            [cube_corners[s, 2], cube_corners[e, 2]],
                            'k-', linewidth=1.5, alpha=0.7)

        # Mark origin and background
        ax.scatter(0, 0, 0, c='black', s=100, marker='+', linewidth=2, label='Origin')
        ax.scatter(*background_3d, c='gray', s=100, marker='o', linewidth=2, label='Background')

        dim_names = ['R', 'G', 'B', 'O']
        ax.set_xlabel(f'DISP {dim_names[dims[0]]}', fontsize=12)
        ax.set_ylabel(f'DISP {dim_names[dims[1]]}', fontsize=12)
        ax.set_zlabel(f'DISP {dim_names[dims[2]]}', fontsize=12)
        ax.set_title('3D DISP Space Threshold Diagram', fontsize=14)
        ax.legend()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 1.1)

        return fig

    def _get_gamut_boundary_2d(self, luminance: float, n_samples: int = 100) -> Optional[npt.NDArray]:
        """Get display gamut boundary in 2D chromaticity space."""
        # Sample gamut at constant luminance
        # Generate grid in DISP space
        disp_values = np.linspace(0, 1, 10)
        gamut_points = []

        # Sample all combinations of display values
        for r in disp_values:
            for g in disp_values:
                for b in disp_values:
                    if self.color_space.dim == 3:
                        disp_point = np.array([r, g, b])
                    elif self.color_space.dim == 4:
                        for o in disp_values:
                            disp_point = np.array([r, g, b, o])
                    else:
                        continue

                    # Convert to cone
                    cone = self.color_space.convert(np.array([disp_point]),
                                                    ColorSpaceType.DISP, ColorSpaceType.CONE)[0]

                    # Convert to HERING_CHROM
                    chrom = self.color_space.convert(np.array([cone]),
                                                     ColorSpaceType.CONE, ColorSpaceType.HERING_CHROM)[0]

                    gamut_points.append(chrom)

        if len(gamut_points) > 0:
            return np.array(gamut_points)
        return None

    def _get_gamut_boundary_3d(self, luminance: float, n_samples: int = 20) -> Optional[npt.NDArray]:
        """Get display gamut boundary in 3D chromaticity space."""
        # Similar to 2D but for 4D observers
        return self._get_gamut_boundary_2d(luminance, n_samples)


class GenotypeClassifier:
    """Classify observer genotype based on threshold patterns."""

    def __init__(self, observer_genotypes: ObserverGenotypes):
        """Initialize classifier with observer genotypes.

        Args:
            observer_genotypes: ObserverGenotypes object with population statistics
        """
        self.observer_genotypes = observer_genotypes

    def classify_from_thresholds(self, thresholds_dict: Dict,
                                 color_space: ColorSpace,
                                 metameric_axis: int = 2,
                                 sex: str = 'both') -> List[Tuple[Tuple, float]]:
        """Classify most likely genotypes based on threshold pattern.

        Args:
            thresholds_dict: Dictionary from QuestColorGenerator.get_thresholds()
            color_space: ColorSpace used for testing
            metameric_axis: Index of metameric axis (cone dimension to test)
            sex: Population to consider ('male', 'female', or 'both')

        Returns:
            List of (genotype, probability) tuples, sorted by probability
        """
        # Extract threshold pattern features
        directions = np.array([data['direction'] for data in thresholds_dict.values()])
        thresholds = np.array([data['threshold'] for data in thresholds_dict.values()])

        # Fit ellipsoid to get shape parameters
        fitter = EllipsoidFitter(dimension=color_space.dim - 1)
        ellipse_params = fitter.fit(directions, thresholds)

        # Key features for classification:
        # 1. Elongation ratio (ratio of largest to smallest axis)
        radii = ellipse_params['radii']
        elongation = radii[0] / radii[-1]

        # 2. Orientation of major axis relative to metameric direction
        major_axis_direction = ellipse_params['rotation'][:, 0]

        # 3. Overall threshold magnitude (mean threshold)
        mean_threshold = np.mean(thresholds)

        # Score each genotype
        genotypes = self.observer_genotypes.get_genotypes_by_probability(sex=sex)
        probabilities = self.observer_genotypes.get_probabilities_by_genotype(sex=sex)

        scores = []
        for genotype, prior_prob in zip(genotypes, probabilities):
            # Compute expected features for this genotype
            score = self._score_genotype(
                genotype, elongation, major_axis_direction,
                mean_threshold, metameric_axis, color_space
            )

            # Combine with prior probability
            posterior = score * prior_prob
            scores.append((genotype, posterior))

        # Normalize probabilities
        total = sum(score for _, score in scores)
        if total > 0:
            scores = [(genotype, score/total) for genotype, score in scores]

        # Sort by probability
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def _score_genotype(self, genotype: Tuple, elongation: float,
                        major_axis_direction: npt.NDArray, mean_threshold: float,
                        metameric_axis: int, color_space: ColorSpace) -> float:
        """Score how well measured features match expected features for a genotype.

        Args:
            genotype: Genotype to score
            elongation: Measured elongation ratio
            major_axis_direction: Measured major axis direction
            mean_threshold: Mean threshold value
            metameric_axis: Metameric axis being tested
            color_space: ColorSpace for reference

        Returns:
            Score (higher is better match)
        """
        # Create color space for this genotype
        try:
            genotype_cs = self.observer_genotypes.get_color_space_for_peaks(
                genotype,
                display_primaries=color_space.display_primaries
            )
        except:
            return 0.0

        # Expected features:
        # If genotype is missing the tested peak, elongation should be HIGH
        # (thresholds will be very different along metameric vs. non-metameric directions)

        tested_dimension = len(genotype)
        has_metameric_sensitivity = tested_dimension > color_space.dim - 1

        # Score based on elongation
        if has_metameric_sensitivity:
            # Should have low elongation (more circular pattern)
            expected_elongation = 1.5
        else:
            # Should have high elongation (elongated pattern)
            expected_elongation = 3.0

        elongation_score = np.exp(-((elongation - expected_elongation) ** 2) / 2.0)

        # Score based on mean threshold
        # Lower thresholds = better color discrimination
        threshold_score = np.exp(-mean_threshold / 0.1)

        # Combine scores
        total_score = elongation_score * threshold_score

        return total_score

    def plot_classification_results(self, classification_results: List[Tuple[Tuple, float]],
                                    top_n: int = 10) -> plt.Figure:
        """Plot classification results as bar chart.

        Args:
            classification_results: Output from classify_from_thresholds()
            top_n: Number of top genotypes to display

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get top N genotypes
        top_results = classification_results[:top_n]
        genotype_labels = [', '.join(map(str, genotype)) for genotype, _ in top_results]
        probabilities = [prob for _, prob in top_results]

        # Create bar chart
        bars = ax.bar(range(len(genotype_labels)), probabilities)
        ax.set_xticks(range(len(genotype_labels)))
        ax.set_xticklabels(genotype_labels, rotation=45, ha='right')
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xlabel('Genotype (Peak Wavelengths)', fontsize=12)
        ax.set_title('Genotype Classification Results', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig


def create_full_analysis_report(quest_generator, output_dir: str = '.'):
    """Create complete analysis report with visualizations.

    Args:
        quest_generator: QuestColorGenerator instance with completed thresholds
        output_dir: Directory to save output files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get thresholds
    thresholds = quest_generator.get_thresholds()

    # Export CSV
    csv_path = os.path.join(output_dir, 'thresholds.csv')
    quest_generator.export_thresholds(csv_path)

    # Create visualizer
    visualizer = ChromaticityVisualizer(quest_generator.color_space)

    # Plot thresholds
    if quest_generator.color_space.dim == 3:
        # For trichromats, plot R-G projection
        fig = visualizer.plot_thresholds_2d(thresholds, dims=(0, 1))
        fig.savefig(os.path.join(output_dir, 'disp_diagram_RG.png'), dpi=150)
        plt.close(fig)

        # Also plot R-B projection
        fig = visualizer.plot_thresholds_2d(thresholds, dims=(0, 2))
        fig.savefig(os.path.join(output_dir, 'disp_diagram_RB.png'), dpi=150)
        plt.close(fig)
    elif quest_generator.color_space.dim == 4:
        # For tetrachromats, plot 3D R-G-B and multiple 2D projections
        fig = visualizer.plot_thresholds_3d(thresholds, dims=(0, 1, 2))
        fig.savefig(os.path.join(output_dir, 'disp_diagram_3d_RGB.png'), dpi=150)
        plt.close(fig)

        # 2D projections
        for dims, name in [((0, 1), 'RG'), ((0, 2), 'RB'), ((0, 3), 'RO'),
                           ((1, 2), 'GB'), ((1, 3), 'GO'), ((2, 3), 'BO')]:
            fig = visualizer.plot_thresholds_2d(thresholds, dims=dims)
            fig.savefig(os.path.join(output_dir, f'disp_diagram_{name}.png'), dpi=150)
            plt.close(fig)

    # Genotype classification
    observer_genotypes = ObserverGenotypes(dimensions=[quest_generator.color_space.dim - 1])
    classifier = GenotypeClassifier(observer_genotypes)

    classification = classifier.classify_from_thresholds(
        thresholds, quest_generator.color_space, sex='both'
    )

    # Plot classification
    fig = classifier.plot_classification_results(classification, top_n=15)
    fig.savefig(os.path.join(output_dir, 'genotype_classification.png'), dpi=150)
    plt.close(fig)

    # Save text report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Chromaticity Threshold Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total directions tested: {len(thresholds)}\n")
        f.write(f"Color space dimension: {quest_generator.color_space.dim}\n")
        f.write(f"Mode: {quest_generator.mode}\n\n")

        f.write("Top 10 Most Likely Genotypes:\n")
        f.write("-" * 50 + "\n")
        for i, (genotype, prob) in enumerate(classification[:10], 1):
            peaks_str = ', '.join(f"{p:.1f}" for p in genotype)
            f.write(f"{i}. [{peaks_str}] nm - Probability: {prob:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Results saved to: {output_dir}\n")

    print(f"Analysis complete. Results saved to {output_dir}")
    return classification
