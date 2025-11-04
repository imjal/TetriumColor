import numpy as np
import random
import pandas as pd
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from TetriumColor.Observer import Observer, Cone
from TetriumColor.ColorSpace import ColorSpace


class ObserverGenotypes:
    """
    A class to handle observer genotype simulations and color space generation.

    This class refactors the functionality from the Opsin Haplotype Prevalence notebook
    to provide a clean interface for generating observer genotypes and their corresponding
    color spaces.
    """

    # Joint statistics from Davidoff 2015 and Stockman 1998 papers
    JOINT_STATS = {
        'M_opsin': {
            'trichromat': {
                'data': {
                    (True, True): 0,
                    (False, True): 2,
                    (True, False): 60,
                    (False, False): 872
                },
                'source': 'Davidoff 2015'
            },
            'dichromat': {
                'data': {
                    (True, True): 1,
                    (True, False): 5,
                    (False, True): 1,
                    (False, False): 6
                },
                'source': 'Stockman 1998'
            }
        },
        'L_opsin': {
            'trichromat': {
                'data': {
                    (True, True, True): 1,
                    (False, True, True): 8,
                    (True, False, True): 0,
                    (True, True, False): 3,
                    (False, False, True,): 15,
                    (False, True, False,): 308,
                    (True, False, False): 13,
                    (False, False, False): 674
                },
                'source': 'Davidoff 2015'
            },
            'dichromat': {
                'data': {
                    (True, True, True): 0,
                    (False, True, True): 0,
                    (True, False, True): 0,
                    (True, True, False): 1,
                    (False, False, True): 0,
                    (False, True, False): 6,
                    (True, False, False): 1,
                    (False, False, False): 20
                },
                'source': 'Stockman 1998'
            }
        },
    }

    # Peak wavelengths for different SNP combinations
    PEAKS = {
        # M peaks
        (True, True): 536,
        (True, False): 533,
        (False, True): 533,
        (False, False): 530,

        # L peaks
        (True, True, True): 547,
        (True, True, False): 552,
        (True, False, True): 553,
        (True, False, False): 556.5,
        (False, True, True): 551,
        (False, True, False): 555,
        (False, False, True): 556,
        (False, False, False): 559
    }

    def __init__(self, wavelengths: Optional[np.ndarray] = None, dimensions: Optional[List[int]] = None, seed: int = 42):
        """
        Initialize ObserverGenotypes with a list of peak wavelengths.

        Args:
            wavelengths: Optional wavelength array, defaults to 360-830nm in 1nm steps
            dimensions: Optional list of dimensions to filter (e.g., [2] for trichromats only).
                       If None, includes all dimensions.
        """
        self.seed = seed
        random.seed(self.seed)
        if wavelengths is None:
            self.wavelengths = np.arange(360, 831, 1)
        else:
            self.wavelengths = wavelengths

        # Build PDFs for both males and females
        self.male_pdf = self._build_pdf('male')
        self.female_pdf = self._build_pdf('female')

        # Build sorted PDFs (ordered by probability, highest first)
        self.male_pdf_sorted = self._sort_pdf_by_probability(self.male_pdf)
        self.female_pdf_sorted = self._sort_pdf_by_probability(self.female_pdf)
        self.both_pdf_sorted = self._build_both_pdf()

        # Build true CDFs (cumulative probabilities)
        self.male_cdf = self._build_cdf(self.male_pdf)
        self.female_cdf = self._build_cdf(self.female_pdf)
        self.both_cdf = self._build_cdf(self.both_pdf_sorted)

        # Store dimension filter
        self.dimensions = dimensions

        # Apply dimension filtering if specified
        if dimensions is not None:
            self.male_pdf_sorted = self._filter_pdf_by_dimension(self.male_pdf_sorted, dimensions)
            self.female_pdf_sorted = self._filter_pdf_by_dimension(self.female_pdf_sorted, dimensions)
            self.both_pdf_sorted = self._filter_pdf_by_dimension(self.both_pdf_sorted, dimensions)
            # Rebuild CDFs with filtered PDFs
            self.male_cdf = self._build_cdf(self.male_pdf_sorted)
            self.female_cdf = self._build_cdf(self.female_pdf_sorted)
            self.both_cdf = self._build_cdf(self.both_pdf_sorted)

    def _build_joint_pdf(self, measurements: Dict, alpha: float = 0.5) -> Dict:
        """Build a joint probability density function from measurements."""
        pdf = {}
        total = 0
        for combo, count in measurements.items():
            pdf[combo] = count + alpha
            total += count + alpha
        for combo in pdf:
            pdf[combo] /= total
        return pdf

    def _build_pdf(self, sex: str = 'male') -> Dict[Tuple[float, ...], float]:
        """Build the probability density function for genotype combinations."""
        # Use trichromat data for building PDFs
        l_joint_pdf = self._build_joint_pdf(self.JOINT_STATS['L_opsin']['trichromat']['data'])
        m_joint_pdf = self._build_joint_pdf(self.JOINT_STATS['M_opsin']['trichromat']['data'])

        # Generate all possible combinations and their probabilities
        pdf = {}

        # Simulate a large number of observers to get empirical distribution
        n_simulations = 100000
        if sex == 'male':
            simulations = self._simulate_males(n_simulations)
        elif sex == 'female':
            simulations = self._simulate_females(n_simulations)
        else:
            raise ValueError("Sex must be 'male' or 'female'")

        # Count occurrences and convert to probabilities
        counter = Counter(simulations)
        total = sum(counter.values())

        for genotype, count in counter.items():
            pdf[genotype] = count / total

        return pdf

    def _build_both_pdf(self, male_proportion: float = 0.5) -> Dict[Tuple[float, ...], float]:
        """Build the probability density function for genotype combinations."""
        # Combine male and female PDFs with specified proportions
        female_proportion = 1.0 - male_proportion
        combined_pdf: Dict[Tuple[float, ...], float] = {}

        for genotype, prob in self.male_pdf.items():
            combined_pdf[genotype] = combined_pdf.get(genotype, 0.0) + male_proportion * prob
        for genotype, prob in self.female_pdf.items():
            combined_pdf[genotype] = combined_pdf.get(genotype, 0.0) + female_proportion * prob

        return self._sort_pdf_by_probability(combined_pdf)

    def _sort_pdf_by_probability(self, pdf: Dict) -> OrderedDict:
        """
        Sort PDF by probability (highest first).
        Returns an OrderedDict of genotype -> probability pairs.
        """
        sorted_pdf = sorted(pdf.items(), key=lambda x: x[1], reverse=True)
        return OrderedDict(sorted_pdf)

    def _filter_pdf_by_dimension(self, pdf: OrderedDict, dimensions: List[int]) -> OrderedDict:
        """
        Filter PDF by dimensions and renormalize.

        Args:
            pdf: OrderedDict of genotype -> probability
            dimensions: List of dimensions to include

        Returns:
            Filtered and renormalized OrderedDict
        """
        filtered_pdf = OrderedDict()
        for genotype, prob in pdf.items():
            if len(genotype) in dimensions:
                filtered_pdf[genotype] = prob

        # Renormalize probabilities
        total = sum(filtered_pdf.values())
        if total > 0:
            for genotype in filtered_pdf:
                filtered_pdf[genotype] /= total

        return filtered_pdf

    def _build_cdf(self, pdf: Dict) -> OrderedDict:
        """
        Build cumulative distribution function from PDF.

        Args:
            pdf: Dictionary mapping genotypes to probabilities

        Returns:
            OrderedDict mapping genotypes to cumulative probabilities
            (sorted by individual probability, highest first)
        """
        # Sort by probability (highest first)
        sorted_items = sorted(pdf.items(), key=lambda x: x[1], reverse=True)

        # Build cumulative distribution
        cdf = OrderedDict()
        cumulative = 0.0

        for genotype, prob in sorted_items:
            cumulative += prob
            cdf[genotype] = cumulative

        return cdf

    def _sample(self, joint_pdf: Dict) -> Tuple:
        """Sample from a joint probability distribution."""
        r = random.random()
        cumulative = 0.0
        for combo, prob in joint_pdf.items():
            cumulative += prob
            if r < cumulative:
                return combo
        return combo

    def _sample_peaks(self, case: str) -> List[float]:
        """Sample peak wavelengths for a given genotype case."""
        l_joint_pdf = self._build_joint_pdf(self.JOINT_STATS['L_opsin']['trichromat']['data'])
        m_joint_pdf = self._build_joint_pdf(self.JOINT_STATS['M_opsin']['trichromat']['data'])

        snps = []
        if case == "ML":
            snps = [m_joint_pdf, l_joint_pdf]
        elif case == "M":
            snps = [m_joint_pdf]
        elif case == "MM":
            snps = [m_joint_pdf, m_joint_pdf]
        elif case == "L":
            snps = [l_joint_pdf]
        elif case == "LL":
            snps = [l_joint_pdf, l_joint_pdf]

        return [self.PEAKS[self._sample(snp)] for snp in snps]

    def _get_random_genotype(self, n: int) -> List[List[float]]:
        """Generate random genotypes based on population frequencies."""
        genotypes = ["ML", "M", "MM", "L", "LL"]
        weights = [92, 0.21, 1.89, 0.86, 5.04]
        genotype_cases = random.choices(genotypes, weights, k=n)
        return [self._sample_peaks(g) for g in genotype_cases]

    def _simulate_males(self, n: int) -> List[Tuple[float, ...]]:
        """Simulate n male observers and return their functional genotypes."""
        male_simulations = self._get_random_genotype(n)
        return [self._functional_genotype(peaks) for peaks in male_simulations]

    def _simulate_females(self, n: int) -> List[Tuple[float, ...]]:
        """Simulate n female observers and return their functional genotypes."""
        # Females have two X chromosomes, so we combine two random genotypes
        male_simulations_1 = self._get_random_genotype(n)
        male_simulations_2 = self._get_random_genotype(n)
        female_simulations = [x1 + x2 for x1, x2 in zip(male_simulations_1, male_simulations_2)]
        return [self._functional_genotype(peaks) for peaks in female_simulations]

    def _functional_genotype(self, peaks: List[float]) -> Tuple[float, ...]:
        """Convert a list of peaks to a functional genotype (unique sorted peaks)."""
        return tuple(sorted(set(peaks)))

    def generate_table(self, n_simulations: int = 100000, sex: str = 'male') -> pd.DataFrame:
        """
        Generate a table showing genotype frequencies.

        Args:
            n_simulations: Number of simulations to run
            sex: 'male', 'female', or 'both'

        Returns:
            DataFrame with columns: Peaks, Count, Dimension, Percentage
        """
        if sex == 'male':
            simulations = self._simulate_males(n_simulations)
        elif sex == 'female':
            simulations = self._simulate_females(n_simulations)
        elif sex == 'both':
            # Sample one X chromosome half the time, two the other half
            n_male = n_simulations // 2
            n_female = n_simulations - n_male

            male_simulations = self._simulate_males(n_male)
            female_simulations = self._simulate_females(n_female)
            simulations = male_simulations + female_simulations
        else:
            raise ValueError("Sex must be 'male', 'female', or 'both'")

        counter = Counter(simulations)

        results = pd.DataFrame(
            [(peaks, count, len(peaks)) for peaks, count in counter.most_common()],
            columns=['Peaks', 'Count', 'Dimension']
        )

        results['Percentage'] = 100 * results['Count'] / n_simulations
        results['Peaks'] = results['Peaks'].apply(lambda x: ', '.join(map(str, x)))
        results['Percentage'] = results['Percentage'].round(4)

        return results

    def get_cdf(self, sex: str = 'male') -> OrderedDict:
        """
        Get the cumulative distribution function for a given sex.
        """
        if sex == 'male':
            return self.male_cdf
        elif sex == 'female':
            return self.female_cdf
        elif sex == 'both':
            return self.both_cdf
        else:
            raise ValueError("Sex must be 'male', 'female', or 'both'")

    def get_pdf(self, sex: str = 'male') -> OrderedDict:
        """
        Get the probability density function for a given sex.
        """
        if sex == 'male':
            return self.male_pdf_sorted
        elif sex == 'female':
            return self.female_pdf_sorted
        elif sex == 'both':
            return self.both_pdf_sorted
        else:
            raise ValueError("Sex must be 'male', 'female', or 'both'")

    def get_color_spaces_by_probability(self, sex: str = 'male') -> List[ColorSpace]:
        """
        Get ColorSpace objects for each genotype ordered by probability (highest first).

        Args:
            sex: 'male', 'female', or 'both'
            male_proportion: When sex='both', proportion of males (default 0.5)

        Returns:
            List of ColorSpace objects ordered by probability (highest first)
        """
        color_spaces = []

        # Use the appropriate sorted PDF based on sex
        pdf_to_use = self.get_pdf(sex)

        for genotype_peaks in pdf_to_use.keys():
            # Create cones for this genotype
            cones = []
            for peak in genotype_peaks:
                cone = Cone.cone(peak, wavelengths=self.wavelengths, template='neitz')
                cones.append(cone)

            observer = Observer(cones, illuminant=None)
            color_space = ColorSpace(observer)
            color_spaces.append(color_space)

        return color_spaces

    def get_genotypes_by_probability(self, sex: str = 'male') -> List[Tuple[float, ...]]:
        """
        Get the list of genotypes ordered by probability (highest first).

        Args:
            sex: 'male' or 'female'

        Returns:
            List of genotype tuples ordered by probability
        """
        return list(self.get_pdf(sex).keys())

    def get_probabilities_by_genotype(self, sex: str = 'male') -> List[float]:
        """
        Get the list of probabilities ordered by genotype probability (highest first).

        Args:
            sex: 'male' or 'female'

        Returns:
            List of probabilities corresponding to genotypes
        """
        return list(self.get_pdf(sex).values())

    def get_quantile(self, quantile: float, sex: str = 'male') -> Tuple[float, ...]:
        """
        Get the genotype at a specific quantile of the CDF.

        Args:
            quantile: Value between 0 and 1 (e.g., 0.5 for median, 0.9 for 90th percentile)
            sex: 'male' or 'female'

        Returns:
            Genotype tuple at the specified quantile
        """
        cdf = self.get_cdf(sex)

        for genotype, cumulative_prob in cdf.items():
            if cumulative_prob >= quantile:
                return genotype

        # Return last genotype if quantile is 1.0
        return list(cdf.keys())[-1]

    def get_genotypes_covering_probability(self, target_probability: float,
                                           sex: str = 'male') -> List[Tuple[float, ...]]:
        """
        Get the minimum set of genotypes that cover at least target_probability of the population.

        Args:
            target_probability: Target cumulative probability (e.g., 0.9 for 90% coverage)
            sex: 'male' or 'female'

        Returns:
            List of genotypes that cover the target probability
        """
        cdf = self.get_cdf(sex)

        genotypes = []
        for genotype, cumulative_prob in cdf.items():
            genotypes.append(genotype)
            if cumulative_prob >= target_probability:
                break

        return genotypes

    def get_color_spaces_covering_probability(self, target_probability: float,
                                              sex: str = 'male', **kwargs) -> List[ColorSpace]:
        """
        Get the list of color spaces that cover at least target_probability of the population.

        Args:
            target_probability: Target cumulative probability (e.g., 0.9 for 90% coverage)
            sex: 'male' or 'female'
            **kwargs: Additional arguments to pass to get_color_space_for_peaks
        """
        genotypes = self.get_genotypes_covering_probability(target_probability, sex)
        return [self.get_color_space_for_peaks(genotype, **kwargs) for genotype in tqdm(genotypes)]

    def plot_pdf(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8), sex: str = 'male') -> plt.Figure:
        """
        Plot the probability distribution showing probability for each genotype.

        Args:
            top_n: Number of top genotypes to plot
            figsize: Figure size tuple
            sex: 'male' or 'female'

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        # Get top N genotypes
        top_genotypes = list(pdf_sorted.items())[:top_n]
        genotypes = [', '.join(map(str, peaks)) for peaks, _ in top_genotypes]
        probabilities = [prob for _, prob in top_genotypes]

        # Create bar plot
        bars = ax.bar(range(len(genotypes)), probabilities)
        ax.set_xlabel('Genotype (Peak Wavelengths)')
        ax.set_ylabel('Probability')
        ax.set_title(f'Top {top_n} {sex.capitalize()} Observer Genotypes by Probability')
        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=45, ha='right')

        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{prob:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_cdf(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8), sex: str = 'male') -> plt.Figure:
        """
        Plot the cumulative distribution function showing cumulative probability.

        Args:
            top_n: Number of top genotypes to plot
            figsize: Figure size tuple
            sex: 'male' or 'female'

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use the appropriate CDF based on sex
        cdf = self.get_cdf(sex)

        # Get top N genotypes
        top_genotypes = list(cdf.items())[:top_n]
        genotypes = [', '.join(map(str, peaks)) for peaks, _ in top_genotypes]
        cumulative_probs = [cum_prob for _, cum_prob in top_genotypes]

        # Create bar plot
        bars = ax.bar(range(len(genotypes)), cumulative_probs)
        ax.set_xlabel('Genotype (Peak Wavelengths)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Cumulative Distribution: Top {top_n} {sex.capitalize()} Genotypes')
        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=45, ha='right')
        ax.axhline(y=0.9, color='r', linestyle='--', label='90% Coverage')
        ax.legend()

        # Add cumulative probability values on bars
        for i, (bar, cum_prob) in enumerate(zip(bars, cumulative_probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cum_prob:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def threshold_observer(self, threshold_probability: float, sex: str = 'male') -> List[Tuple[float, ...]]:
        """
        Find all observers with cumulative probability <= threshold_probability.

        This returns genotypes in order until the cumulative probability reaches the threshold.

        Args:
            threshold_probability: Target cumulative probability threshold (e.g., 0.9 for 90% coverage)
            sex: 'male' or 'female'

        Returns:
            List of genotype tuples that meet the threshold
        """
        return self.get_genotypes_covering_probability(threshold_probability, sex)

    def get_observer_for_peaks(self, peaks: Tuple[float, ...], od: float = 0.5) -> Observer:
        """
        Create an Observer object for specific peak wavelengths, with sorted peaks.

        Args:
            peaks: List of peak wavelengths

        Returns:
            Observer object
        """
        # Add S cone if not present
        if 420 not in peaks:
            peaks = (420,) + peaks

        cones = []
        for peak in sorted(peaks):
            if peak == 420:
                od = od * 0.8
            cone = Cone.cone(peak, wavelengths=self.wavelengths, template='neitz', od=od)
            cones.append(cone)

        return Observer(cones, illuminant=None)

    def get_color_space_for_peaks(self, peaks: Tuple[float, ...], **kwargs) -> ColorSpace:
        """
        Create a ColorSpace object for specific peak wavelengths, with sorted peaks.

        Args:
            peaks: List of peak wavelengths
            **kwargs: Additional arguments to pass to ColorSpace constructor

        Returns:
            ColorSpace object
        """
        observer = self.get_observer_for_peaks(peaks)
        return ColorSpace(observer, **kwargs)

    def get_observers_by_probability(self, sex: str = 'male') -> List[Observer]:
        """
        Get Observer objects for each genotype ordered by probability (highest first).

        Args:
            sex: 'male' or 'female'

        Returns:
            List of Observer objects ordered by probability (highest first)
        """
        observers = []

        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        for genotype_peaks in pdf_sorted.keys():
            observer = self.get_observer_for_peaks(list(genotype_peaks))
            observers.append(observer)

        return observers

    def get_observer_for_genotype(self, genotype: Tuple[float, ...]) -> Observer:
        """
        Create an Observer object for a specific genotype.

        Args:
            genotype: Genotype tuple

        Returns:
            Observer object
        """
        return self.get_observer_for_peaks(list(genotype))

    def get_color_space_for_genotype(self, genotype: Tuple[float, ...],
                                     **kwargs) -> ColorSpace:
        """
        Create a ColorSpace object for a specific genotype.

        Args:
            genotype: Genotype tuple

        Returns:
            ColorSpace object
        """
        return self.get_color_space_for_peaks(list(genotype), **kwargs)

    def get_peaks_for_genotype(self, genotype: Tuple[float, ...]) -> List[float]:
        """
        Get the peak wavelengths for a specific genotype.

        Args:
            genotype: Genotype tuple

        Returns:
            List of peak wavelengths
        """
        return list(genotype)

    def get_dimension_for_genotype(self, genotype: Tuple[float, ...]) -> int:
        """
        Get the dimension (number of unique peaks) for a specific genotype.

        Args:
            genotype: Genotype tuple

        Returns:
            Number of unique peaks
        """
        return len(genotype)

    def get_probability_for_genotype(self, genotype: Tuple[float, ...], sex: str = 'male') -> float:
        """
        Get the probability for a specific genotype.

        Args:
            genotype: Genotype tuple
            sex: 'male' or 'female'

        Returns:
            Probability of this genotype
        """
        return self.get_pdf(sex).get(genotype, 0.0)

    def get_cumulative_probability(self, n_genotypes: int, sex: str = 'male') -> float:
        """
        Get the cumulative probability for the top N genotypes.

        Args:
            n_genotypes: Number of top genotypes to include
            sex: 'male' or 'female'

        Returns:
            Cumulative probability
        """
        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        cumulative = 0.0
        for i, (_, prob) in enumerate(pdf_sorted.items()):
            if i >= n_genotypes:
                break
            cumulative += prob
        return cumulative

    def get_genotypes_by_dimension(self, dimension: int, sex: str = 'male') -> List[Tuple[float, ...]]:
        """
        Get all genotypes with a specific dimension.

        Args:
            dimension: Number of unique peaks
            sex: 'male' or 'female'

        Returns:
            List of genotype tuples with the specified dimension
        """
        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        return [genotype for genotype in pdf_sorted.keys() if len(genotype) == dimension]

    def get_most_common_genotype(self, sex: str = 'male') -> Tuple[float, ...]:
        """
        Get the most common genotype.

        Args:
            sex: 'male' or 'female'

        Returns:
            Most common genotype tuple
        """
        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        return list(pdf_sorted.keys())[0]

    def get_most_common_probability(self, sex: str = 'male') -> float:
        """
        Get the probability of the most common genotype.

        Args:
            sex: 'male' or 'female'

        Returns:
            Probability of the most common genotype
        """
        # Use the appropriate sorted PDF based on sex
        pdf_sorted = self.get_pdf(sex)

        return list(pdf_sorted.values())[0]

    def get_summary_stats(self, sex: str = 'male') -> Dict[str, Union[int, float, List]]:
        """
        Get summary statistics for the genotype distribution.

        Args:
            sex: 'male' or 'female'

        Returns:
            Dictionary with summary statistics
        """
        # Use the appropriate data structures based on sex
        pdf_sorted = self.get_pdf(sex)

        probabilities = list(pdf_sorted.values())

        return {
            'sex': sex,
            'total_genotypes': len(pdf_sorted),
            'most_common_genotype': list(pdf_sorted.keys())[0],
            'most_common_probability': list(pdf_sorted.values())[0],
            'mean_probability': np.mean(probabilities),
            'std_probability': np.std(probabilities),
            'min_probability': np.min(probabilities),
            'max_probability': np.max(probabilities),
            'dimensions_present': sorted(list(set(len(genotype) for genotype in pdf_sorted.keys()))),
            'top_5_genotypes': list(pdf_sorted.keys())[:5],
            'top_5_probabilities': list(pdf_sorted.values())[:5]
        }

    def compare_sexes(self, top_n: int = 10) -> pd.DataFrame:
        """
        Compare male and female genotype distributions.

        Args:
            top_n: Number of top genotypes to compare

        Returns:
            DataFrame comparing male and female distributions
        """
        # Get top N genotypes for each sex
        male_top = list(self.get_pdf('male').items())[:top_n]
        female_top = list(self.get_pdf('female').items())[:top_n]

        # Create comparison DataFrame
        comparison_data = []

        # Get all unique genotypes from both sexes
        all_genotypes = set()
        for genotype, _ in male_top:
            all_genotypes.add(genotype)
        for genotype, _ in female_top:
            all_genotypes.add(genotype)

        for genotype in all_genotypes:
            male_prob = self.get_pdf('male').get(genotype, 0.0)
            female_prob = self.get_pdf('female').get(genotype, 0.0)

            comparison_data.append({
                'Genotype': ', '.join(map(str, genotype)),
                'Male_Probability': male_prob,
                'Female_Probability': female_prob,
                'Difference': female_prob - male_prob,
                'Dimension': len(genotype)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Male_Probability', ascending=False)

        return df


def test_observer_genotypes():
    """Test function for the ObserverGenotypes class."""
    print("Testing ObserverGenotypes class...")

    # Test 1: Initialize
    print("\n1. Testing initialization")
    og = ObserverGenotypes()
    print(f"Male PDF size: {len(og.get_pdf('male'))}")
    print(f"Female PDF size: {len(og.get_pdf('female'))}")
    print(f"Male CDF size: {len(og.get_cdf('male'))}")
    print(f"Female CDF size: {len(og.get_cdf('female'))}")

    # Test 2: Generate table for males
    print("\n2. Testing table generation for males")
    male_table = og.generate_table(n_simulations=10000, sex='male')
    print("Male table (top 5):")
    print(male_table.head(5))

    # Test 3: Generate table for females
    print("\n3. Testing table generation for females")
    female_table = og.generate_table(n_simulations=10000, sex='female')
    print("Female table (top 5):")
    print(female_table.head(5))

    # Test 4: Generate table for both
    print("\n4. Testing table generation for both sexes")
    both_table = og.generate_table(n_simulations=10000, sex='both')
    print("Both sexes table (top 5):")
    print(both_table.head(5))

    # Test 5: Compare sexes
    print("\n5. Testing sex comparison")
    comparison = og.compare_sexes(top_n=5)
    print("Sex comparison (top 5):")
    print(comparison)

    # Test 6: Summary stats for each sex
    print("\n6. Testing summary statistics")
    male_stats = og.get_summary_stats('male')
    female_stats = og.get_summary_stats('female')

    print("Male summary:")
    print(f"  Most common: {male_stats['most_common_genotype']} ({male_stats['most_common_probability']:.4f})")
    print(f"  Total genotypes: {male_stats['total_genotypes']}")
    print(f"  Dimensions: {male_stats['dimensions_present']}")

    print("Female summary:")
    print(f"  Most common: {female_stats['most_common_genotype']} ({female_stats['most_common_probability']:.4f})")
    print(f"  Total genotypes: {female_stats['total_genotypes']}")
    print(f"  Dimensions: {female_stats['dimensions_present']}")

    # Test 7: Test CDF functionality
    print("\n7. Testing CDF functionality")
    print(f"90th percentile male genotype: {og.get_quantile(0.9, 'male')}")
    print(f"90th percentile female genotype: {og.get_quantile(0.9, 'female')}")

    coverage_genotypes = og.get_genotypes_covering_probability(0.9, 'male')
    print(f"Number of genotypes covering 90% of male population: {len(coverage_genotypes)}")
    print(f"Actual coverage: {og.get_cumulative_probability(len(coverage_genotypes), 'male'):.4f}")

    # Test 8: Observer and ColorSpace creation (if available)
    print("\n8. Testing Observer and ColorSpace creation")

    # Test observer creation for specific peaks
    observer = og.get_observer_for_peaks((530, 559))
    print(f"Created observer with {observer.dimension} dimensions")
    print(f"Observer sensors: {[s.peak for s in observer.sensors]}")

    # Test color space creation
    color_space = og.get_color_space_for_peaks((530, 559))
    print(f"Created ColorSpace with {color_space.dim} dimensions")

    # Test getting observers by probability
    # male_observers = og.get_observers_by_probability(sex='male')
    # print(f"Created {len(male_observers)} male observers ordered by probability")

    # # Test getting color spaces by probability
    # male_color_spaces = og.get_color_spaces_by_probability(sex='male')
    # print(f"Created {len(male_color_spaces)} male ColorSpaces ordered by probability")

    # # Test 'both' sex option for color spaces
    # both_color_spaces = og.get_color_spaces_by_probability(sex='both', male_proportion=0.5)
    # print(f"Created {len(both_color_spaces)} ColorSpaces for combined population (50/50 male/female)")

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_observer_genotypes()
