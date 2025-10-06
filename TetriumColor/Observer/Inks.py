import heapq
import math
from collections import defaultdict
from itertools import chain
from itertools import islice, combinations
from itertools import product
from typing import Tuple, List, Union, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.typing as npt
from scipy.spatial import ConvexHull
from scipy.special import comb
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm
from colorsys import rgb_to_hsv

import os
import pickle
import hashlib
from importlib import resources
from .Spectra import Spectra
from .Observer import Observer
from .MaxBasis import MaxBasis

from sklearn.decomposition import PCA, TruncatedSVD
from scipy.special import comb
from scipy.spatial import ConvexHull


class Pigment(Spectra):
    # TODO: reformat with kwargs
    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 k: Optional[npt.NDArray] = None,
                 s: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None, **kwargs):
        """
        Either pass in @param reflectance or pass in
        @param k and @param s.

        k and s are stored as spectra rather than NDArrays.
        """
        if k is not None and s is not None and wavelengths is not None:
            # compute reflectance from k & s
            if not k.shape == s.shape or not k.shape == wavelengths.shape:
                raise ValueError("Coefficients k and s and wavelengths must be same shape.")

            r = 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))
            super().__init__(wavelengths=wavelengths, data=r, **kwargs)

        elif array is not None:
            if isinstance(array, Spectra):
                super().__init__(**array.__dict__, **kwargs)
            else:
                super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)
            k, s = self.compute_k_s()
        elif data is not None:
            super().__init__(wavelengths=wavelengths, data=data, **kwargs)

        else:
            raise ValueError("Must either specify reflectance or k and s coefficients and wavelengths.")

        self.k, self.s = k, s

    def compute_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # Walowit · 1987 specifies this least squares method
        # todo: GJK method as per Centore • 2015
        array = np.clip(self.array(), 1e-4, 1)
        k, s = [], []
        for wavelength, r in array:
            k_over_s = (1 - r) * (1 - r) / (2 * r)
            A = np.array([[-1, k_over_s], [1, 1]])
            b = np.array([0, 1])

            AtA_inv = np.linalg.inv(np.dot(A.T, A))
            Atb = np.dot(A.T, b)

            _k, _s = np.dot(AtA_inv, Atb)
            k.append(_k)
            s.append(_s)

        return np.clip(np.array(k), 0, 1), np.clip(np.array(s), 0, 1)

    def get_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # todo: pass in wavelength list for interpolation/sampling consistency with mixing
        return self.k, self.s


def get_metamers(points, target, threshold=1e-2, axis=2):
    # ok the idea here is to find a point in points
    # that is a confusion point wrt target
    metamers = []
    for idx, point in enumerate(points):
        # print(point)
        metamer_closeness = math.sqrt(
            sum(
                [
                    (point[i] - target[i]) ** 2
                    for i in range(len(point)) if i != axis])
        )
        if metamer_closeness < threshold:
            metamers.append((abs(target[axis] - point[axis]), idx))
    metamers.sort(reverse=True)
    return metamers


def lsh_buckets(points, ignored_axis=2):
    # Implementation of Locally Sensitive Hashing
    print(points.shape)
    n_dimensions = points.shape[1]

    weights = np.zeros(n_dimensions)
    for i in range(n_dimensions):
        if i != ignored_axis:
            weights[i] = 10 ** (2 * (i + 1))

    weights = np.zeros(n_dimensions)
    adjustments = []
    for i in range(n_dimensions):
        if i != ignored_axis:
            weights[i] = 10 ** (2 * (i + 1))
            exp_base = 10 ** (2 * (i + 1)) // 2
            adjustments.extend([exp_base, -exp_base])

    # Calculate base hash values excluding the ignored axis
    base_hashes = np.dot(points, weights)

    # Apply adjustments and calculate hash values
    hash_values = np.array([np.floor(base_hashes + adjustment).astype(int) for adjustment in adjustments])

    return hash_values


def bucket_points(points: npt.NDArray, axis=2, prec=0.005, exponent=8) -> Dict:
    # disjointed buckets
    buckets = defaultdict(list)
    N, d = points.shape

    # prec = 0.005
    # prec = 0.0005
    # prec = 0.0000005
    # 8 is large enough for prec = 0.005:
    # 8 > log_2 (1 / 0.005)
    # 22 > log_2(1/0.0000005)
    # 14 > log_2(1/0.00005)
    # weights = (2 ** (8 * np.arange(0, d)))
    weights = (2 ** (exponent * np.arange(0, d)))
    # weights = (2 ** (22 * np.arange(0, d)))
    weights[axis] = 0

    values = points // prec
    keys = values @ weights
    for i, (key, point) in enumerate(zip(keys, values)):
        buckets[key].append((point / 2, i))  # can replace 2 with 0.01 // prec

    return {k: v for k, v in buckets.items() if len(v) > 1}


def sort_buckets(buckets, axis=2) -> List:
    dist_buckets = []

    for metamers in buckets.values():
        if len(metamers) <= 1:
            continue

        axis_values = [metamer[0][axis] for metamer in metamers]

        min_val = min(axis_values)
        max_val = max(axis_values)

        distance = max_val - min_val

        min_index = axis_values.index(min_val)
        max_index = axis_values.index(max_val)
        best_indices = (metamers[min_index][1], metamers[max_index][1])

        dist_buckets.append((distance, best_indices))

    return sorted(dist_buckets, reverse=True)


def get_metamer_buckets(points, axis=2, prec=0.005, exponent=8) -> List:
    sorted_points = []

    buckets = sort_buckets(bucket_points(points, axis=axis, prec=prec, exponent=exponent), axis=axis)
    for dst, (i, j) in buckets:
        sorted_points.append((dst, (tuple(points[i]), tuple(points[j]))))

    sorted_points.sort(reverse=True)
    return sorted_points


def k_s_from_data(data: npt.NDArray):
    array = np.clip(data, 1e-4, 1)
    k, s = [], []
    k_over_s = (1 - array) ** 2 / (2 * array)
    b = np.array([0, 1])

    for f in k_over_s:
        A = np.array([[-1, f], [1, 1]])

        # should just be able to call np.linalg.inverse
        AtA_inv = np.linalg.inv(np.dot(A.T, A))
        Atb = np.dot(A.T, b)
        _k, _s = np.dot(AtA_inv, Atb)

        k.append(_k)
        s.append(_s)

    return np.clip(np.array(k), 0, 1), np.clip(np.array(s), 0, 1)


def data_from_k_s(k, s):
    return 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))


def k_s_from_pigments(pigments):
    k_s_pairs = [k_s_from_data(pigment.data) for pigment in pigments]
    k_matrix, s_matrix = np.array([k for k, _ in k_s_pairs]), np.array([s for _, s in k_s_pairs])

    return k_matrix.T, s_matrix.T


def km_mix(pigments, concentrations=None):
    K_matrix, S_matrix = k_s_from_pigments(pigments)
    wavelengths = pigments[0].wavelengths

    if not concentrations:
        concentrations = np.ones(len(pigments)) / len(pigments)

    K_mix = K_matrix @ concentrations
    S_mix = S_matrix @ concentrations

    return Spectra(data=data_from_k_s(K_mix, S_mix), wavelengths=wavelengths, normalized=False)


def load_neugebauer(inks, paper, n=50):
    num_inks = len(inks)
    primaries_dict = {'0' * num_inks: paper}

    for i in range(1, 2 ** num_inks):
        binary_str = format(i, f'0{num_inks}b')
        inks_to_mix = []

        for j, bit in enumerate(binary_str):
            if bit == '1':
                inks_to_mix.append(inks[j])

        if binary_str not in primaries_dict:  # should not need if statement
            mixed_ink = inks_to_mix[0]
            if len(inks_to_mix) > 1:
                mixed_ink = km_mix(inks_to_mix)
            primaries_dict[binary_str] = mixed_ink
    return Neugebauer(primaries_dict, n=n)


def observe_spectra(data, observer, illuminant):
    return np.divide(np.matmul(observer, (data * illuminant).T).squeeze(), np.matmul(observer, illuminant.T))


def observe_spectras(spectras: npt.NDArray, observer: npt.NDArray, illuminant: npt.NDArray) -> npt.NDArray:
    numerator = np.matmul(observer, (spectras * illuminant).T)
    denominator = np.matmul(observer, illuminant.T)
    result = np.divide(numerator, denominator[:, np.newaxis])
    return result.T


def find_best_n(primaries_dict, percentages: npt.NDArray, actual: Spectra, cellular=False):
    # Find best n array for a particular sample
    wavelengths = actual.wavelengths

    best_n = None
    best_error = float('inf')

    candidates = np.logspace(-2, 2, num=100, base=10)
    # print(candidates)
    # inefficient but idc
    for n in candidates:
        neug_n = Neugebauer(primaries_dict, n=n)
        if cellular:
            neug_n = CellNeugebauer(primaries_dict, n=n)
        result = neug_n.mix(percentages).reshape(-1)
        error = np.sum(np.square((actual.data - result)))
        if error < best_error:
            best_n = n
            best_error = error

    return best_n


class CellNeugebauer:
    # cell neugebauer with one division
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], n=50):
        """
        primaries_dict is (key, value) pairs where the key is either a
        string of ternary digits or a tuple of ternary values.
        """
        self.subcubes: Dict[Tuple, Neugebauer] = defaultdict(Neugebauer)

        weights = []
        spectras = []
        for key, spectra in primaries_dict.items():
            if isinstance(key, str):
                key = tuple(map(lambda x: int(x), key))
            weights.append(key)
            spectras.append(spectra)

        self.num_inks = round(math.log(len(primaries_dict), 3))
        self.wavelengths = list(primaries_dict.values())[0].wavelengths

        self.n = n

        for indices in product([0, 1], repeat=self.num_inks):
            primaries = {}
            for weight, spectra in zip(weights, spectras):
                if all(weight[d] in (indices[d], indices[d] + 1) for d in range(self.num_inks)):
                    adjusted_weight = tuple(w - i for w, i in zip(weight, indices))
                    primaries[adjusted_weight] = spectra
            self.subcubes[indices] = Neugebauer(primaries, n=n)

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        index = tuple((percentages > 0.5).astype(int))
        adjusted_percentages = 2 * (percentages - np.array(index) / 2)
        return self.subcubes[index].mix(adjusted_percentages)

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: Optional[npt.NDArray] = None):
        if illuminant is None:
            illuminant = np.ones_like(self.wavelengths)
        return observe_spectra(self.mix(percentages), observer, illuminant)


class Neugebauer:
    def __init__(self, primaries_dict: Optional[Dict[Union[Tuple, str], Spectra]], n=2):
        """
        primaries_dict is (key, value) pairs where the key is either a
        string of binary digits or a tuple of binary values, and value is a Spectra.
        """
        weights = []
        spectras = []
        for key, spectra in primaries_dict.items():
            if isinstance(key, str):
                key = tuple(map(lambda x: int(x), key))
            weights.append(np.array(key))
            spectras.append(spectra.data)
        self.wavelengths = list(primaries_dict.values())[0].wavelengths
        self.weights_array = np.array(weights)
        self.spectras_array = np.power(np.array(spectras), 1.0 / n)
        self.num_inks = round(math.log(self.weights_array.shape[0], 2))

        if isinstance(n, np.ndarray):
            assert len(n) == len(self.wavelengths)

        self.n = n

    def batch_mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = ((self.weights_array * percentages[:, np.newaxis, :]) +
               (1 - self.weights_array) * (1 - percentages[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        result = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), self.spectras_array), self.n).squeeze(axis=1)

        return result

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = (self.weights_array * percentages) + (1 - self.weights_array) * (1 - percentages)
        w_p_prod = np.prod(w_p, axis=1, keepdims=True)

        result = np.power(np.matmul(w_p_prod.T, self.spectras_array), self.n)

        return result

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: Optional[npt.NDArray] = None):
        if illuminant is None:
            illuminant = np.ones_like(self.wavelengths)
        if percentages.ndim > 1:
            return observe_spectras(self.batch_mix(percentages), observer, illuminant)
        return observe_spectra(self.mix(percentages), observer, illuminant)

    def get_pca_size(self, observe: npt.NDArray, illuminant: npt.NDArray):
        # built for very quick evaluation of gamut
        stepsize = 0.2
        values = np.arange(0, 1 + stepsize, stepsize)
        mesh = np.meshgrid(*([values] * self.num_inks))
        grid = np.stack(mesh, axis=-1).reshape(-1, self.num_inks)

        stimulus = self.observe(grid, observe, illuminant)
        pca = PCA(n_components=observe.shape[0])
        pca.fit(stimulus)
        return np.sqrt(pca.explained_variance_)[-1]


class InkGamut:
    def __init__(self, inks: Union[List[Spectra], Neugebauer, CellNeugebauer, Dict[Union[Tuple, str], Spectra]],
                 paper: Optional[Spectra] = None,
                 illuminant: Optional[Union[Spectra, npt.NDArray]] = None):
        if isinstance(inks, Dict):
            inks = Neugebauer(inks)
        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.wavelengths = inks.wavelengths
        else:
            self.wavelengths = inks[0].wavelengths

        self.illuminant = None
        if isinstance(illuminant, Spectra):
            self.illuminant = illuminant.interpolate_values(self.wavelengths).data
        elif isinstance(illuminant, np.ndarray):
            assert len(illuminant) == len(self.wavelengths)
            self.illuminant = illuminant
        else:
            self.illuminant = np.ones_like(self.wavelengths)

        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.neugebauer = inks
            return

        assert np.array_equal(self.wavelengths, paper.wavelengths), \
            "Must pass in paper spectra with consistent wavelengths"

        for ink in inks:
            assert np.array_equal(ink.wavelengths, self.wavelengths)

        self.neugebauer = load_neugebauer(inks, paper)  # KM interpolation

    def get_spectra(self, percentages: Union[List, npt.NDArray], clip=False):
        if not isinstance(percentages, np.ndarray):
            percentages = np.array(percentages)
        data = self.neugebauer.mix(percentages).T
        if clip:
            data = np.clip(data, 0, 1)
        return Spectra(data=data, wavelengths=self.wavelengths, normalized=clip)

    def batch_generator(self, iterable, batch_size):
        """Utility function to generate batches from an iterable."""
        iterator = iter(iterable)
        for first in iterator:
            yield np.array(list(islice(chain([first], iterator), batch_size - 1)))

    def get_spectral_point_cloud(self, stepsize=0.1, grid: Optional[npt.NDArray] = None, verbose=True, batch_size=1e5):

        point_cloud = []
        _percentages = []

        if grid is None:
            values = np.arange(0, 1 + stepsize, stepsize)
            total_combinations = len(values) ** self.neugebauer.num_inks
            grid = product(values, repeat=self.neugebauer.num_inks)
        else:
            total_combinations = grid.shape[0]

        desc = "Generating point cloud"
        verbose_progress = (lambda x: tqdm(x, total=int(total_combinations / batch_size), desc=desc)) if verbose else (
            lambda x: x)

        if isinstance(self.neugebauer, CellNeugebauer):
            for index, neugebauer in self.neugebauer.subcubes.items():
                subgamut = InkGamut(neugebauer, illuminant=self.illuminant)
                pc, perc = subgamut.get_spectral_point_cloud(stepsize=stepsize * 2, grid=None, verbose=verbose,
                                                             batch_size=batch_size)
                point_cloud.append(pc)
                _percentages.append(perc / 2 + np.array(index) / 2)

        else:
            for batch in verbose_progress(self.batch_generator(grid, int(batch_size))):
                valid_percentages = batch[np.all(batch <= 1, axis=1)]

                if valid_percentages.size == 0:
                    continue

                spectral_batch = self.neugebauer.batch_mix(valid_percentages)
                point_cloud.append(spectral_batch)
                _percentages.append(valid_percentages)

        # Concatenate the batched results
        point_cloud = np.concatenate(point_cloud, axis=0)
        _percentages = np.concatenate(_percentages, axis=0)

        return point_cloud, _percentages

    def get_point_cloud(self, observe: Union[Observer, npt.NDArray],
                        stepsize=0.1, grid: Optional[npt.NDArray] = None, verbose=True, batch_size=1e5):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)

        point_cloud = []
        _percentages = []

        if grid is None:
            values = np.arange(0, 1 + stepsize, stepsize)
            total_combinations = len(values) ** self.neugebauer.num_inks
            grid = product(values, repeat=self.neugebauer.num_inks)
        else:
            total_combinations = grid.shape[0]

        desc = "Generating point cloud"
        verbose_progress = (lambda x: tqdm(x, total=int(total_combinations / batch_size), desc=desc)) if verbose else (
            lambda x: x)

        if isinstance(self.neugebauer, CellNeugebauer):

            for index, neugebauer in self.neugebauer.subcubes.items():
                subgamut = InkGamut(neugebauer, illuminant=self.illuminant)
                pc, perc = subgamut.get_point_cloud(observe, stepsize=stepsize * 2, grid=None, verbose=verbose,
                                                    batch_size=batch_size)
                point_cloud.append(pc)
                _percentages.append(perc / 2 + np.array(index) / 2)

        else:
            for batch in verbose_progress(self.batch_generator(grid, int(batch_size))):
                valid_percentages = batch[np.all(batch <= 1, axis=1)]

                if valid_percentages.size == 0:
                    continue

                stimulus_batch = self.neugebauer.observe(valid_percentages, observe, self.illuminant)
                point_cloud.append(stimulus_batch)
                _percentages.append(valid_percentages)

        # Concatenate the batched results
        point_cloud = np.concatenate(point_cloud, axis=0)
        _percentages = np.concatenate(_percentages, axis=0)

        return point_cloud, _percentages

    def get_buckets(self, observe: Union[Observer, npt.NDArray],
                    axis=2, stepsize=0.1, verbose=True, save=False):
        point_cloud, percentages = self.get_point_cloud(observe, stepsize, verbose=verbose)
        if verbose:
            print("Point cloud generated.")

        if save:
            np.save(f"{save}_point_cloud{int(100 * stepsize)}", point_cloud)
            np.save(f"{save}_percentages{int(100 * stepsize)}", percentages)
            if verbose:
                print(f"Point cloud saved to {save}_point_cloud{int(100 * stepsize)}.")

        _percentages = []

        buckets = sort_buckets(bucket_points(point_cloud, axis=axis), axis=axis)
        for dst, (i, j) in buckets:
            _percentages.append((dst, (tuple(percentages[i]), tuple(percentages[j]))))

        _percentages.sort(reverse=True)
        return _percentages

    def get_buckets_in_hering(self, max_basis: MaxBasis,
                              axis=2, stepsize=0.1, verbose=True, save=False):
        maxbasis_observer = max_basis.get_max_basis_observer()
        point_cloud, percentages = self.get_point_cloud(maxbasis_observer, stepsize, verbose=verbose)
        if verbose:
            print("Point cloud generated.")

        if save:
            np.save(f"{save}_point_cloud{int(100 * stepsize)}", point_cloud)
            np.save(f"{save}_percentages{int(100 * stepsize)}", percentages)
            if verbose:
                print(f"Point cloud saved to {save}_point_cloud{int(100 * stepsize)}.")

        _percentages = []
        Tmat = max_basis.get_cone_to_maxbasis_transform()
        Q_vec = [Tmat @ np.array([0, 0, 1, 0]), np.array([1, 0, 0, 0]),
                 np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0])]  # Q direction

        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v - np.sum(np.dot(v, b)*b for b in basis)
                if (w > 1e-10).any():
                    basis.append(w/np.linalg.norm(w))
            return np.array(basis)
        A = gram_schmidt(Q_vec)
        new_point_cloud = (A @ point_cloud.T).T
        buckets = sort_buckets(bucket_points(new_point_cloud, axis=0), axis=0)
        for dst, (i, j) in buckets:
            _percentages.append((dst, (tuple(percentages[i]), tuple(percentages[j]))))

        _percentages.sort(reverse=True)
        return _percentages

    def get_pca_size(self, observe: Union[Observer, npt.NDArray], stepsize=0.1, verbose=False):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)
        point_cloud, _ = self.get_point_cloud(observe, stepsize, verbose=verbose)
        pca = PCA(n_components=observe.shape[0])
        pca.fit(point_cloud)

        return np.sqrt(pca.explained_variance_)[-1]

    def get_width(self, observe: Union[Observer, npt.NDArray],
                  axis=2, stepsize=0.1, verbose=True, save=False, refined=0):
        percentages = self.get_buckets(observe, axis=axis, stepsize=stepsize, verbose=verbose, save=save,
                                       )

        dst, (pi, pj) = percentages[0]

        if verbose:
            print(f"maximum distance is {dst} with percentages {pi} and {pj}")

        return dst


class InkLibrary:
    """
    InkLibrary is a class equipped with a method for "fast ink gamut search". It uses PCA to identify
    the best k-ink set from a larger library.
    """

    def __init__(self, library: Dict[str, Spectra], paper: Spectra):
        self.library = library
        self.names = list(library.keys())
        self.spectra_objs = list(library.values())
        self.wavelengths = self.spectra_objs[0].wavelengths
        self.spectras = np.array([s.data for s in self.spectra_objs])
        self.K = len(self.names)
        for s in self.spectra_objs:
            assert np.array_equal(s.wavelengths, self.wavelengths)
        assert np.array_equal(self.wavelengths, paper.wavelengths)
        self.paper = paper.data

    def get_paper(self) -> Spectra:
        """Get the paper spectra."""
        return Spectra(data=self.paper, wavelengths=self.wavelengths, normalized=False)

    @staticmethod
    def load_ink_library(filepath: str, filter_clogged: bool = True):
        """Load an ink library from a CSV file.

        Args:
            filepath (str): Path to the CSV file containing the ink library data.
            filter_clogged (bool): Whether to filter out clogged inks. Defaults to True.

        Returns:
            InkLibrary: An InkLibrary object containing the loaded inks and paper.
        """
        library, paper, _ = load_inkset(filepath, filter_clogged=filter_clogged)
        return InkLibrary(library, paper)

    def distance_search(self, observe: Union[Observer, npt.NDArray],
                        illuminant: Union[Spectra, npt.NDArray], top=100, k=None, stepsize=0.1):
        # slow unoptimized method, finds max q distance along point cloud
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if k is None:
            print(type(observe))
            k = observe.shape[0]

        top_scores = []
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):
            names = [self.names[i] for i in inkset_idx]

            spectras = [Spectra(wavelengths=self.wavelengths, data=self.spectras[idx]) for idx in inkset_idx]

            gamut = InkGamut(spectras, Spectra(wavelengths=self.wavelengths, data=self.paper), illuminant)
            score = gamut.get_width(observe, stepsize=stepsize, verbose=False)
            if len(top_scores) < top:
                heapq.heappush(top_scores, (score, names))
            else:
                if score > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (score, names))
        return sorted(top_scores, reverse=True)

    def convex_hull_search(self, observe: Union[Observer, npt.NDArray],
                           illuminant: Union[Spectra, npt.NDArray], top=100, k=None) -> List[Tuple[float, List[str]]]:
        """Find the best k-ink subset of the ink library using a convex hull approach.
        Args:
            observe (Union[Observer, npt.NDArray]): The observer or sensor matrix.
            illuminant (Union[Spectra, npt.NDArray]): The illuminant spectra
            top (int): The number of top results to return.
            k (Optional[int]): The number of inks to consider in the subset. If None, defaults to the number of observer channels.
        Returns:
            List[Tuple[float, List[str]]]: A list of tuples containing the volume and the names of the inks.
        """
        # super efficient way to find best k-ink subset of large K ink library
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if isinstance(illuminant, Spectra):
            illuminant = illuminant.interpolate_values(self.wavelengths).data
        if k is None:
            k = observe.shape[0]

        # Check cache first
        cache_data = self._load_from_cache('convex_hull_search',
                                           observe=observe.tolist() if isinstance(observe, np.ndarray) else str(observe),
                                           illuminant=illuminant.tolist() if isinstance(illuminant, np.ndarray) else str(illuminant),
                                           top=top, k=k)
        if cache_data is not None:
            return cache_data

        km_cache = {}

        total_iterations = sum(comb(self.K, i) for i in range(2, k + 1))

        ks_cache = []
        for ink in self.spectras:
            ks_cache.append(np.stack(k_s_from_data(ink)))

        with tqdm(total=total_iterations, desc="loading km cache") as pbar:
            for i in range(2, k + 1):
                concentrations = np.ones(i) / i
                for subset in combinations(range(self.K), i):
                    inks_to_mix = [ks_cache[idx] for idx in subset]
                    ks_batch = np.stack(inks_to_mix, axis=2)
                    ks_mix = ks_batch @ concentrations
                    data = data_from_k_s(ks_mix[0], ks_mix[1])
                    km_cache[subset] = data.astype(np.float16)
                    pbar.update(1)

        del ks_cache

        denominator = np.matmul(observe, illuminant.T)[:, np.newaxis]

        top_scores = []
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):
            names = [self.names[i] for i in inkset_idx]

            primaries_array = [self.paper]
            primaries_array.extend([self.spectras[idx] for idx in inkset_idx])
            primaries_array.extend(
                [km_cache[subset] for i in range(2, k + 1) for subset in combinations(inkset_idx, i)])

            primaries_array = np.array(primaries_array)

            numerator = np.matmul(observe, (primaries_array * illuminant).T)
            observe_mix = np.divide(numerator, denominator)  # 4 x 16
            vol = ConvexHull(observe_mix.T).volume

            if len(top_scores) < top:
                heapq.heappush(top_scores, (vol, names))
            else:
                if vol > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (vol, names))

        # Save to cache
        result = sorted(top_scores, reverse=True)
        self._save_to_cache('convex_hull_search', result,
                            observe=observe.tolist() if isinstance(observe, np.ndarray) else str(observe),
                            illuminant=illuminant.tolist() if isinstance(illuminant, np.ndarray) else str(illuminant),
                            top=top, k=k)
        return result

    def cached_pca_search(self, observe: Union[Observer, npt.NDArray],
                          illuminant: Union[Spectra, npt.NDArray], top=50, k=None):
        # super efficient way to find best k-ink subset of large K ink library
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if isinstance(illuminant, Spectra):
            illuminant = illuminant.interpolate_values(self.wavelengths).data
        if k is None:
            k = observe.shape[0]

        # Check cache first
        cache_data = self._load_from_cache('cached_pca_search',
                                           observe=observe.tolist() if isinstance(observe, np.ndarray) else str(observe),
                                           illuminant=illuminant.tolist() if isinstance(illuminant, np.ndarray) else str(illuminant),
                                           top=top, k=k)
        if cache_data is not None:
            return cache_data

        km_cache = {}
        # Populate cache
        total_iterations = sum(comb(self.K, i) for i in range(2, k + 1))

        top_scores = []

        with tqdm(total=total_iterations, desc="loading km cache") as pbar:
            ks_cache = []
            for ink in self.spectras:
                ks_cache.append(np.stack(k_s_from_data(ink)))

            for i in range(2, k + 1):
                concentrations = np.ones(i) / i
                for subset in combinations(range(self.K), i):
                    inks_to_mix = [ks_cache[idx] for idx in subset]
                    ks_batch = np.stack(inks_to_mix, axis=2)
                    ks_mix = ks_batch @ concentrations
                    data = data_from_k_s(ks_mix[0], ks_mix[1])
                    km_cache[subset] = data.astype(np.float16)
                    pbar.update(1)

            del ks_cache

        weights_array = []
        for i in range(k + 1):
            for subset in combinations(range(k), i):
                binary = [0] * k
                for index in subset:
                    binary[index] = 1
                weights_array.append(binary)
        weights_array = np.array(weights_array)

        # nothing
        top_scores = []

        stepsize = 0.2
        values = np.arange(0, 1 + stepsize, stepsize)
        mesh = np.meshgrid(*([values] * k))
        grid = np.stack(mesh, axis=-1).reshape(-1, k)

        w_p = ((weights_array * grid[:, np.newaxis, :]) +
               (1 - weights_array) * (1 - grid[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        pca = PCA(n_components=observe.shape[0])
        tsvd = TruncatedSVD(n_components=observe.shape[0], n_iter=5, random_state=42)

        denominator = np.matmul(observe, illuminant.T)[:, np.newaxis]

        inkset_iteration_tt = 0
        # Find best inkset
        iters = 0
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):

            names = [self.names[i] for i in inkset_idx]

            primaries_array = [self.paper]
            primaries_array.extend([self.spectras[idx] for idx in inkset_idx])
            primaries_array.extend(
                [km_cache[subset] for i in range(2, k + 1) for subset in combinations(inkset_idx, i)])

            primaries_array = np.array(primaries_array)
            data_array = np.power(primaries_array, 1.0 / 50)

            mix = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), data_array), 50).squeeze(axis=1)

            numerator = np.matmul(observe, (mix * illuminant).T)
            observe_mix = np.divide(numerator, denominator)
            pca.fit(observe_mix)
            score = np.sqrt(pca.explained_variance_)[-1]

            if len(top_scores) < top:
                heapq.heappush(top_scores, (score, names))
            else:
                if score > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (score, names))

            iters += 1
            if iters > 1000:
                print("pca iteration", inkset_iteration_tt / iters, "seconds")

                return

        # Save to cache
        result = sorted(top_scores, reverse=True)
        self._save_to_cache('cached_pca_search', result,
                            observe=observe.tolist() if isinstance(observe, np.ndarray) else str(observe),
                            illuminant=illuminant.tolist() if isinstance(illuminant, np.ndarray) else str(illuminant),
                            top=top, k=k)
        return result

    def _get_cache_filename(self, method_name: str, **kwargs) -> str:
        """
        Generate a unique filename for caching based on method parameters.

        Args:
            method_name (str): Name of the method being cached
            **kwargs: Method parameters to include in hash

        Returns:
            str: Cache filename
        """
        # Create hash input from method name and parameters
        hash_input = f"{method_name}|{str(self.K)}|{str(self.wavelengths.tolist())}|{str(sorted(self.names))}"

        # Add method-specific parameters
        for key, value in sorted(kwargs.items()):
            if isinstance(value, np.ndarray):
                hash_input += f"|{key}:{value.tolist()}"
            else:
                hash_input += f"|{key}:{str(value)}"

        # Create MD5 hash
        hash_obj = hashlib.md5(hash_input.encode())
        hash_str = hash_obj.hexdigest()

        return f"{method_name}_cache_{hash_str}.pkl"

    def _save_to_cache(self, method_name: str, data: any, **kwargs) -> None:
        """Save data to cache."""
        cache_filename = self._get_cache_filename(method_name, **kwargs)

        try:
            with resources.path("TetriumColor.Assets.Cache", cache_filename) as path:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                print(f"Saved {method_name} cache to {cache_filename}")
        except Exception as e:
            print(f"Failed to save {method_name} cache: {e}")

    def _load_from_cache(self, method_name: str, **kwargs) -> any:
        """
        Load data from cache.

        Returns:
            any: Cached data if available, None otherwise
        """
        cache_filename = self._get_cache_filename(method_name, **kwargs)

        try:
            with resources.path("TetriumColor.Assets.Cache", cache_filename) as path:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        cached_data = pickle.load(f)
                    print(f"Loaded {method_name} from cache: {cache_filename}")
                    return cached_data
        except Exception as e:
            print(f"Failed to load {method_name} from cache: {e}")

        return None


class FastNeugebauer:
    def __init__(self, weights_array, data_array, num_inks):
        self.weights_array = weights_array
        self.data_array = np.power(data_array, 1.0 / 50)
        self.num_inks = num_inks

    def batch_mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = ((self.weights_array * percentages[:, np.newaxis, :]) +
               (1 - self.weights_array) * (1 - percentages[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        result = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), self.data_array), 50).squeeze(axis=1)

        return result

    def batch_observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: npt.NDArray):
        spectras = self.batch_mix(percentages)
        numerator = np.matmul(observer, (spectras * illuminant).T)
        denominator = np.matmul(observer, illuminant.T)
        result = np.divide(numerator, denominator[:, np.newaxis])
        return result.T

    def get_pca_size(self, grid, observe: npt.NDArray, illuminant: npt.NDArray):
        stimulus = self.batch_observe(grid, observe, illuminant)

        pca = PCA(n_components=observe.shape[0])

        pca.fit(stimulus)

        return np.sqrt(pca.explained_variance_)[-1]


def load_inkset(filepath: str, filter_clogged: bool = True) -> Tuple[Dict[str, Spectra], Spectra, npt.NDArray]:
    """
    Load an inkset from a CSV file. CSV file should have the following structure:
    - First column: Index (e.g., 0, 1, 2, etc.)
    - Second column: Ink names (e.g., "paper", "ink1", "ink2", etc.)
    - Remaining columns: Reflectance data for each ink at the corresponding wavelengths. Must include numeric wavelengths
    - Optional column (anywhere after the name column): 'clogged' indicating unusable inks (true/false or 1/0)

    Parameters:
    - filepath: str, path to the CSV file containing the inkset data.
    - filter_clogged: bool, whether to filter out clogged inks. Defaults to True.

    Returns:
    - inks: dict, a dictionary of ink names and their corresponding Spectra objects.
    - paper: Optional[Spectra], the paper spectra if present in the CSV; otherwise None.
    - wavelengths: npt.NDArray, an array of wavelengths corresponding to the reflectance data.

    Raises:
    - ValueError: If the inkset does not contain a 'paper' ink.
    """
    df = pd.read_csv(filepath)

    # Identify optional 'clogged' column (case-insensitive match)
    clogged_col = next((c for c in df.columns if str(c).strip().lower() == 'clogged'), None)

    # Detect wavelength columns by extracting numeric parts of headers
    wave_cols = []
    wavelengths_list = []
    for col in df.columns[2:]:
        header = str(col)
        cleaned = ''.join(ch for ch in header if (ch.isdigit() or ch == '.'))
        if cleaned == '':
            continue
        try:
            wl = float(cleaned)
            wave_cols.append(col)
            wavelengths_list.append(wl)
        except ValueError:
            # Non-wavelength auxiliary column (e.g., 'clogged')
            continue

    if len(wave_cols) == 0:
        raise ValueError("No wavelength columns detected in inkset CSV.")

    # Ensure wavelength columns are sorted by numeric wavelength
    order = np.argsort(np.array(wavelengths_list))
    wave_cols = [wave_cols[i] for i in order]
    wavelengths = np.array([wavelengths_list[i] for i in order], dtype=float)

    inks: Dict[str, Spectra] = {}
    paper: Optional[Spectra] = None

    def parse_bool(val) -> bool:
        if isinstance(val, (int, float)):
            return bool(int(val))
        s = str(val).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    for i in range(len(df)):
        name = df.iloc[i, 1]
        # Read spectral data for this row in the sorted wavelength order
        data = df.loc[i, wave_cols].to_numpy(dtype=float)

        # Determine clogged status if column present
        is_clogged = False
        if clogged_col is not None and clogged_col in df.columns:
            is_clogged = parse_bool(df.loc[i, clogged_col])

        spectra_obj = Spectra(data=data, wavelengths=wavelengths)

        if isinstance(name, str) and name.strip().lower() == 'paper':
            paper = spectra_obj
            continue

        # Discard clogged inks at load time (if filter_clogged is True)
        if filter_clogged and is_clogged:
            continue

        inks[name] = spectra_obj

    if paper is None:
        raise ValueError("No paper spectra found in inkset CSV.")

    # Paper may be managed outside of this file (e.g., via registry). Do not error if missing.
    return inks, paper, wavelengths


def load_all_ink_libraries(ink_libraries: Dict[str, str], filter_clogged: bool = True) -> Dict[str, InkLibrary]:
    inksets = {}
    for name, path in ink_libraries.items():
        inksets[name] = InkLibrary.load_ink_library(path, filter_clogged=filter_clogged)
    return inksets


def combine_inksets(inkset_paths: List[str], output_path: str = None, filter_clogged: bool = True,
                    name_prefixes: List[str] = None, paper_source: str = "first") -> InkLibrary:
    """
    Combine multiple inkset CSV files into a single InkLibrary.

    Args:
        inkset_paths (List[str]): List of paths to inkset CSV files to combine
        output_path (str, optional): Path to save the combined CSV file. If None, no file is saved.
        filter_clogged (bool): Whether to filter out clogged inks. Defaults to True.
        name_prefixes (List[str], optional): Prefixes to add to ink names to avoid conflicts.
                                            If None, uses file basenames as prefixes.
        paper_source (str): Which inkset to use for paper spectra. Options: "first", "last", or specific inkset name.

    Returns:
        InkLibrary: Combined ink library containing all inks from all inksets.

    Raises:
        ValueError: If inksets have incompatible wavelength ranges or if paper_source is invalid.
    """
    if not inkset_paths:
        raise ValueError("At least one inkset path must be provided")

    # Load all inksets
    inksets = {}
    for path in inkset_paths:
        inkset_name = os.path.splitext(os.path.basename(path))[0]
        inksets[inkset_name] = InkLibrary.load_ink_library(path, filter_clogged=filter_clogged)

    # Determine which inkset to use for paper
    if paper_source == "first":
        paper_inkset = inksets[list(inksets.keys())[0]]
    elif paper_source == "last":
        paper_inkset = inksets[list(inksets.keys())[-1]]
    elif paper_source in inksets:
        paper_inkset = inksets[paper_source]
    else:
        raise ValueError(f"Invalid paper_source: {paper_source}. Must be 'first', 'last', or a valid inkset name.")

    # Check wavelength compatibility
    reference_wavelengths = paper_inkset.wavelengths
    for name, inkset in inksets.items():
        if not np.array_equal(inkset.wavelengths, reference_wavelengths):
            raise ValueError(f"Inkset {name} has incompatible wavelengths with reference inkset")

    # Combine all inks
    combined_library = {}
    name_prefixes = name_prefixes or list(inksets.keys())

    if len(name_prefixes) != len(inkset_paths):
        raise ValueError("Number of name_prefixes must match number of inkset_paths")

    for i, (inkset_name, inkset) in enumerate(inksets.items()):
        prefix = name_prefixes[i]
        for ink_name, ink_spectra in inkset.library.items():
            # Skip paper from non-reference inksets
            if ink_name.lower() == "paper" and inkset != paper_inkset:
                continue

            # Add prefix to avoid name conflicts
            prefixed_name = f"{prefix}_{ink_name}" if prefix else ink_name
            combined_library[prefixed_name] = ink_spectra

    # Create combined InkLibrary
    combined_inkset = InkLibrary(combined_library, paper_inkset.get_paper())

    # Save to CSV if output path is provided
    if output_path:
        save_combined_inkset_to_csv(combined_inkset, output_path)
        print(f"Combined inkset saved to: {output_path}")

    print(f"Combined {len(inksets)} inksets into {len(combined_library)} inks")
    return combined_inkset


def save_combined_inkset_to_csv(inkset: InkLibrary, output_path: str):
    """
    Save a combined inkset to a CSV file in the standard format.

    Args:
        inkset (InkLibrary): The inkset to save
        output_path (str): Path where to save the CSV file
    """
    # Create DataFrame with ink names as rows and wavelengths as columns
    ink_names = list(inkset.library.keys())
    wavelengths = inkset.wavelengths

    # Prepare data matrix
    reflectance_data = np.array([inkset.library[name].data for name in ink_names])

    # Create DataFrame
    df = pd.DataFrame(reflectance_data, index=ink_names, columns=wavelengths)
    df.index.name = "Name"

    # Reset index to make Name a column
    df = df.reset_index()

    # Add Index column
    df.insert(0, 'Index', range(len(df)))

    # Add clogged column (all False/0 for combined inksets)
    df['clogged'] = 0

    # Save to CSV
    df.to_csv(output_path, index=False)


def save_top_inks_as_csv(top_volumes, filename):
    import csv
    import json

    # Save top_volumes to a CSV file, serializing the inks list as a JSON string
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Volume", "Inks"])  # Header
        for volume, inks in top_volumes:
            writer.writerow([volume, json.dumps(inks)])


def load_top_inks(filename):
    import csv
    import json

    top_volumes = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            volume = float(row[0])
            inks = json.loads(row[1])
            top_volumes.append((volume, inks))
    return top_volumes


def plot_inks_by_hue(ink_dataset: Dict[str, Spectra], wavelengths: npt.NDArray, filename: Optional[str] = None):
    """
    Plots the inks in the dataset sorted by hue.

    Parameters:
    - ink_dataset: dict, a dictionary of ink names and their corresponding Spectra objects.
    - wavelengths: numpy.ndarray, array of wavelengths corresponding to the spectra data.
    - filename: optional filename to save the plot to
    """
    # Convert RGB to HSV and sort by hue
    def get_hue(spectra):
        r, g, b = spectra.to_rgb()
        h, _, _ = rgb_to_hsv(r, g, b)
        return h

    # Sort inks by hue
    sorted_inks = sorted(ink_dataset.items(), key=lambda item: get_hue(item[1]))

    num_inks = len(sorted_inks)

    # Dynamically determine grid size to fit all spectra
    # Try to make the grid as square as possible, but always enough to fit all
    if num_inks <= 4:
        cols = num_inks
        rows = 1
    else:
        cols = math.ceil(math.sqrt(num_inks))
        rows = math.ceil(num_inks / cols)
        # If the grid is too wide, limit columns to 8 and adjust rows accordingly
        if cols > 8:
            cols = 8
            rows = math.ceil(num_inks / cols)

    plt.figure(figsize=(cols * 3, rows * 2.5))

    for idx, (name, spectra) in enumerate(sorted_inks):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(wavelengths, spectra.data, c=spectra.to_rgb())
        plt.title(name[:10], fontsize=8)  # Show only the first 10 characters of the name
        plt.xlabel("Wavelength (nm)", fontsize=6)
        plt.ylabel("Reflectance", fontsize=6)
        plt.grid(True)
        plt.xlim(wavelengths[0], wavelengths[-1])
        plt.ylim(0, 1)
        plt.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def show_top_k_combinations(top_volumes: List[Tuple[float, List[str]]], inkset: Dict[str, Spectra], k=10, filename: Optional[str] = None):
    """
    Displays the top k ink combinations with their volumes.

    Parameters:
    - top_volumes: list of tuples (volume, [ink names])
    - k: number of top combinations to display
    - filename: optional filename to save the plot to
    """
    # Plot the spectra of the top inks for the first k entries
    plt.figure(figsize=(4 * 3, math.ceil(k / 4) * 3))

    for idx, (volume, ink_names) in enumerate(top_volumes[:k]):
        plt.subplot(math.ceil(k / 4), 4, idx + 1)  # Create a subplot for each entry
        for ink_name in ink_names:  # Plot the spectra of the first 4 inks
            spectra = inkset[ink_name]
            # Show only the first 10 characters of the name
            plt.plot(spectra.wavelengths, spectra.data, label=ink_name[:10], c=spectra.to_rgb())
        plt.title(f"Volume: {volume:.2e}", fontsize=10)
        plt.xlabel("Wavelength (nm)", fontsize=8)
        plt.ylabel("Reflectance", fontsize=8)
        plt.grid(True)
        plt.xlim(spectra.wavelengths[0], spectra.wavelengths[-1])
        plt.ylim(0, 1)
        plt.legend(fontsize=6)
        plt.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
