import hashlib
from importlib import resources
from itertools import combinations, product
from functools import reduce
from colour.appearance.ciecam02 import P
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import os
import pickle
import numpy.typing as npt
from typing import List

from . import Observer, GetHeringMatrix
from . import Spectra, Illuminant
from ..Utils.Hash import stable_hash
from ..Utils.BasisMath import get_transform_to_angle_basis, rotation_and_scale_to_point
from scipy.spatial import ConvexHull


def generate_parallelepiped(vectors):
    """
    Given d vectors in d-dimensional space, return the 2^d vertices of the parallelepiped.
    """
    vectors = np.array(vectors)  # Shape (d, d)
    d = vectors.shape[0]

    # All binary combinations of 0 and 1, for 2^d corners
    coeffs = np.array(list(product([0, 1], repeat=d)))  # Shape (2^d, d)

    # Multiply each coefficient vector with the matrix of vectors
    corners = coeffs @ vectors  # Shape (2^d, d)
    return corners


class MaxBasis:
    dim4SampleConst = 10
    dim3SampleConst = 2

    def __init__(self, observer: Observer, denom: float = 1,
                 verbose: bool = False) -> None:
        # perceptual points
        self.denom = denom
        self.exp = 1 / denom

        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension
        self.step_size = self.observer.wavelengths[1] - self.observer.wavelengths[0]
        self.dim_sample_const = self.dim4SampleConst if self.dimension == 4 else self.dim3SampleConst

        grid_axes = [np.linspace(0, 1, 50) for _ in range(self.dimension)]
        self.hypercube_points = np.array(list(product(*grid_axes)))
        self.HMatrix = GetHeringMatrix(observer.dimension)

        self.__getMaximalBasis()

    def __computeVolume(self, wavelengths):
        if self.denom > 1:
            vol = self.__computeVolumeViaPointCloudEstimation(wavelengths)
        else:
            transitions = self.GetCutpointTransitions(wavelengths)
            cone_vals = np.array([np.dot(self.matrix, Spectra.from_transitions(
                x, 1 if i == 0 else 0, self.wavelengths).data) for i, x in enumerate(transitions)])
            vol = np.abs(np.linalg.det(cone_vals))
        return vol

    # def __computeVolumeViaPointCloudEstimation(self, wavelengths):
    #     transitions = self.GetCutpointTransitions(wavelengths)
    #     cone_vals = np.array([np.dot(self.matrix, Spectra.from_transitions(
    #         x, 1 if i == 0 else 0, self.wavelengths).data) for i, x in enumerate(transitions)])
    #     # Generate a hypercube of points in [0, 1]^d

    #     if np.linalg.cond(cone_vals) > 1 / np.finfo(cone_vals.dtype).eps:
    #         return 0  # Singular matrix, volume

    #     all_points = cone_vals@self.hypercube_points.T
    #     # raise to the power
    #     transformed_points = np.power(all_points, 1/self.denom)

    #     # Find the bounding box of the transformed points
    #     min_coords = np.min(transformed_points, axis=1)
    #     max_coords = np.max(transformed_points, axis=1)

    #     # Generate random points in the bounding box
    #     num_samples = 1000
    #     random_points = np.random.uniform(
    #         min_coords[:, np.newaxis],
    #         max_coords[:, np.newaxis],
    #         (self.dimension, num_samples)
    #     )

    #     # To check if a point is inside the transformed parallelepiped,
    #     # we transform it back and check if it's in the original unit hypercube
    #     inv_points = np.power(random_points, self.denom)

    #     # Solve the linear system to find if they're in the original unit hypercube
    #     try:
    #         # Solve cone_vals * x = inv_points to find the coordinates in the original space
    #         # coefficients = np.linalg.solve(cone_vals, inv_points)
    #         cone_vals_inv = np.linalg.inv(cone_vals)
    #         coefficients = cone_vals_inv @ inv_points
    #         # Check if all coefficients are in [0,1] (within the unit hypercube)
    #         inside_mask = np.all((0 <= coefficients) & (coefficients <= 1), axis=0)
    #         points_inside = np.sum(inside_mask)

    #         # Calculate the volume
    #         bounding_box_volume = np.prod(max_coords - min_coords)
    #         volume = bounding_box_volume * (points_inside / num_samples)
    #     except np.linalg.LinAlgError:
    #         # Handle singular matrix case
    #         volume = 0
    #     print("volume is ", volume)
    #     return volume

    def __computeVolumeViaPointCloudEstimation(self, wavelengths):
        transitions = self.GetCutpointTransitions(wavelengths)
        cone_vals = np.array([np.dot(self.matrix, Spectra.from_transitions(
            x, 1 if i == 0 else 0, self.wavelengths).data) for i, x in enumerate(transitions)])

        try:
            cone_vals_inv = np.linalg.inv(cone_vals)
        except np.linalg.LinAlgError:
            return 0  # Singular matrix, volume is 0

        all_points = cone_vals @ self.hypercube_points.T

        if np.isclose(self.denom, 3):
            transformed_points = np.cbrt(all_points)
        else:
            transformed_points = np.power(all_points, self.exp)

        min_coords = np.min(transformed_points, axis=1)
        max_coords = np.max(transformed_points, axis=1)

        num_samples = 10000
        random_points = np.random.uniform(
            min_coords[:, np.newaxis],
            max_coords[:, np.newaxis],
            (self.dimension, num_samples)
        )

        inv_points = np.power(random_points, self.denom)
        coefficients = cone_vals_inv @ inv_points
        inside_mask = np.all((0 <= coefficients) & (coefficients <= 1), axis=0)
        points_inside = np.sum(inside_mask)

        bounding_box_volume = np.prod(max_coords - min_coords)
        volume = bounding_box_volume * (points_inside / num_samples)
        return volume

    def __findMaximalCMF(self, isReverse=True):
        sortedCutpoints = self.cutpoints[:self.dimension - 1]
        sortedCutpoints.sort()
        transitions = self.GetCutpointTransitions(sortedCutpoints)
        if isReverse:
            refs = np.array([Spectra.from_transitions(x, 1 if i == 0 else 0,
                            self.wavelengths).data for i, x in enumerate(transitions)])[::-1]
        else:
            refs = np.array([Spectra.from_transitions(x, 1 if i == 0 else 0,
                            self.wavelengths).data for i, x in enumerate(transitions)])
        white_point = np.power(np.dot(self.matrix, np.ones(len(self.wavelengths))), 1/self.denom)
        self.refs = refs
        basis_vectors = np.power(np.dot(self.matrix, refs.T), 1/self.denom)
        # self.cone_to_maxbasis = rotation_and_scale_to_point(
        # self.cone_to_maxbasis@white_point, np.ones(self.dimension))@self.cone_to_maxbasis
        self.cone_to_maxbasis = np.linalg.inv(basis_vectors)  # goes to the cube basis

        self.maximal_matrix = np.dot(self.cone_to_maxbasis, self.matrix)

        self.maximal_sensors = []
        for i in range(self.dimension):
            spectra = np.concatenate([self.observer.wavelengths[:, np.newaxis],
                                     self.maximal_matrix[i][:, np.newaxis]], axis=1)
            # false to stop clipping from 0 to 1
            self.maximal_sensors += [Spectra(spectra, self.observer.wavelengths, normalized=False)]
        self.maximal_observer = Observer(self.maximal_sensors, verbose=self.verbose)  # erased self.observer.illuminant.
        return self.maximal_sensors, self.maximal_observer

    def __findMaxCutpoints(self, rng=None):
        if self.dimension == 1:
            # No cutpoints in 1D; choose either end of the visible wavelength range
            # Compute volume with no cutpoints (transitions will be one endpoint)
            vol = self.__computeVolume([])
            self.listvol = []
            self.cutpoints = [vol]
            return self.cutpoints
        if self.dimension == 2:
            X = np.arange(self.observer.wavelengths[0] + self.step_size,
                          self.observer.wavelengths[-1] - self.step_size, self.step_size)
            Xidx = np.meshgrid(X)[0]
            Zidx = np.zeros_like(Xidx, dtype=float)

            for i in tqdm(range(len(X)), desc="Max Basis Calculation"):
                wavelength = [Xidx[i]]
                Zidx[i] = self.__computeVolume(wavelength)
            self.listvol = [Xidx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.cutpoints = [Xidx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints

        elif self.dimension == 3:
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size,
                              self.observer.wavelengths[-1] - self.step_size, self.step_size)
                Y = np.arange(self.observer.wavelengths[0] + self.step_size,
                              self.observer.wavelengths[-1] - self.step_size, self.step_size)
            else:
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)
            Xidx, Yidx = np.meshgrid(X, Y, indexing='ij')
            Zidx = np.zeros_like(Xidx, dtype=float)
            for i in tqdm(range(len(X)), desc="Max Basis Calculation"):
                for j in range(len(Y)):
                    if i <= j:
                        wavelengths = [Xidx[i, j], Yidx[i, j]]
                        wavelengths.sort()
                        Zidx[i, j] = self.__computeVolume(wavelengths)
            self.listvol = [Xidx, Yidx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.cutpoints = [Xidx[idxs][0], Yidx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints
        elif self.dimension == 4:
            # if not rng:
            #     X = np.arange(self.observer.wavelengths[0] + self.step_size,
            #                   self.observer.wavelengths[-1] - self.step_size, self.step_size)
            #     Y = np.arange(self.observer.wavelengths[0] + self.step_size,
            #                   self.observer.wavelengths[-1] - self.step_size, self.step_size)
            #     W = np.arange(self.observer.wavelengths[0] + self.step_size,
            #                   self.observer.wavelengths[-1] - self.step_size, self.step_size)
            # else:
            #     X = np.arange(rng[0][0], rng[0][1], self.step_size)
            #     Y = np.arange(rng[1][0], rng[1][1], self.step_size)
            #     W = np.arange(rng[2][0], rng[2][1], self.step_size)
            # Xidx, Yidx, Widx = np.meshgrid(X, Y, W, indexing='ij')

            # Zidx = np.zeros_like(Xidx, dtype=float)
            # for i in tqdm(range(len(X)), desc="Max Basis Calculation"):
            #     for j in range(len(Y)):
            #         for k in range(len(W)):
            #             if i <= j and j <= k:
            #                 wavelengths = [Xidx[i, j, k], Yidx[i, j, k], Widx[i, j, k]]
            #                 wavelengths.sort()
            #                 Zidx[i, j, k] = self.__computeVolume(wavelengths)
            # self.listvol = [Xidx, Yidx, Widx, Zidx]
            # maxvol = reduce(max, Zidx.flatten())
            # idxs = np.where(Zidx == maxvol)
            # self.idxs = [x.tolist()[0] for x in idxs]
            # self.cutpoints = [Xidx[idxs][0], Yidx[idxs][0], Widx[idxs][0], Zidx[idxs][0]]
            # return self.cutpoints
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size,
                              self.observer.wavelengths[-1] - self.step_size, self.step_size)
                Y = np.arange(self.observer.wavelengths[0] + self.step_size,
                              self.observer.wavelengths[-1] - self.step_size, self.step_size)
                W = np.arange(self.observer.wavelengths[0] + self.step_size,
                              self.observer.wavelengths[-1] - self.step_size, self.step_size)
            else:
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)
                W = np.arange(rng[2][0], rng[2][1], self.step_size)

            Xidx, Yidx, Widx = np.meshgrid(X, Y, W, indexing='ij')
            Zidx = np.zeros_like(Xidx, dtype=float)

            # Function to compute volume for a single (i, j, k)
            def compute_volume_for_indices(i, j, k):
                if i <= j and j <= k:
                    wavelengths = [Xidx[i, j, k], Yidx[i, j, k], Widx[i, j, k]]
                    wavelengths.sort()
                    return i, j, k, self.__computeVolume(wavelengths)
                return i, j, k, 0  # Return 0 for invalid indices

            # Flatten the indices for parallel processing
            indices = [(i, j, k) for i in range(len(X)) for j in range(len(Y)) for k in range(len(W))]

            # Parallelize the computation
            results = Parallel(n_jobs=-1)(delayed(compute_volume_for_indices)(i, j, k)
                                          for i, j, k in tqdm(indices, desc="Max Basis Calculation"))

            # Populate Zidx with the results
            for i, j, k, volume in results:
                Zidx[i, j, k] = volume

            self.listvol = [Xidx, Yidx, Widx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.idxs = [x.tolist()[0] for x in idxs]
            self.cutpoints = [Xidx[idxs][0], Yidx[idxs][0], Widx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints
        else:
            raise NotImplementedError

    def __getMaximalBasis(self, rng=None):
        range = []
        if self.step_size < 6 and self.dimension > 3:  # find a range to do fine-grained search to narrow down brute force
            rangbd = int(self.dim_sample_const * 2)
            coarse_wavelengths = np.arange(
                self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1] - self.step_size, self.dim_sample_const)
            coarse_sensors = [s.interpolate_values(coarse_wavelengths) for s in self.observer.sensors]
            coarseObserver = Observer(coarse_sensors, self.observer.illuminant)
            coarseMaxBasis = MaxBasis(coarseObserver, denom=self.denom, verbose=self.verbose)
            cutpoints = coarseMaxBasis.GetCutpoints()
            range = [[x - rangbd, x + rangbd] for x in cutpoints[:self.dimension-1]]

        self.__findMaxCutpoints(range)
        self.__findMaximalCMF(isReverse=False)

    def GetMaxBasisObserver(self):
        return self.maximal_observer

    def GetConeToMaxBasisTransform(self):
        return self.cone_to_maxbasis

    def GetCutpoints(self):
        return self.cutpoints

    def GetCMF(self):
        return self.maximal_sensors

    def GetCutpointTransitions(self, wavelengths):
        if len(wavelengths) == 0:
            # For 1D, no cutpoints; choose either end of the visible wavelength.
            # Selecting the start of the observer's wavelength range.
            return [[self.observer.wavelengths[0]]]
        transitions = [[wavelengths[0]], [wavelengths[len(wavelengths)-1]]]
        transitions += [[wavelengths[i], wavelengths[i+1]] for i in range(len(wavelengths)-1)]
        transitions.sort()
        return transitions

    def GetDiscreteRepresentation(self, reverse=False) -> tuple[List[Spectra], npt.NDArray, npt.NDArray, List[tuple[int, int]]]:
        """Get discrete representation of max basis

        Args:
            reverse (bool, optional): Reverse the order of the reflectances to be from bgr to rgb. Defaults to False.

        Returns:
            tuple[List[Spectra], npt.NDArray, npt.NDArray, List[tuple[int, int]]]: reflectances, points, rgbs, and lines
        """
        transitions = self.GetCutpointTransitions(self.cutpoints[:self.dimension-1])
        start_vals = [1 if i == 0 else 0 for i, x in enumerate(transitions)]
        allcombos = [[]]
        alllines = []
        allstart = [[0]]  # black starting value
        for i in range(self.dimension):
            alllines += list(combinations(range(self.dimension), i + 1))
            allcombos += [[elem for lst in x for elem in lst] for x in list(combinations(transitions, i + 1))]
            allstart += [list(x) for x in list(combinations(start_vals, i + 1))]
        final_start = [max(x) for x in allstart]
        final_combos = []
        for x in allcombos:
            lst_elems, counts = np.unique(x, return_counts=True)
            removeIdx = []
            lst_elems = list(lst_elems)
            num_elems = len(lst_elems)
            for i, cnt in enumerate(reversed(list(counts))):
                if cnt > 1:
                    del lst_elems[num_elems - i - 1]
            lst_elems.sort()
            final_combos += [lst_elems]
        lines = []
        for i, x in enumerate(alllines):
            if len(x) <= 1:
                lines += [[0, x[0] + 1]]  # connected to black
            else:  # > 1
                madeupof = list(combinations(x, len(x)-1))
                lines += [[alllines.index(elem) + 1, i + 1] for elem in madeupof]  # connected to each elem
        refs = [Spectra.from_transitions(x, final_start[i], self.wavelengths) for i, x in enumerate(final_combos)]
        cones = self.observer.observe_spectras(refs)
        power_cones = np.power(cones, 1/self.denom)
        points = power_cones@self.cone_to_maxbasis.T
        if reverse:  # need to reverse this..?
            points = points[::-1]
        rgbs = np.array([s.to_rgb(illuminant=Illuminant.get("E")) for s in refs])
        return refs, points, rgbs, lines

    def __hash__(self) -> int:
        return stable_hash(self.observer)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, MaxBasis):
            return stable_hash(self.observer) == stable_hash(value.observer)
        return False


class MaxBasisFactory:
    _cache_file = "max-basis-cache.pkl"

    @staticmethod
    def load_cache():
        # Load the cache from file if it exists
        with resources.path("TetriumColor.Assets.Cache", MaxBasisFactory._cache_file) as path:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        return {}

    @staticmethod
    def save_cache(cache):
        # Save the cache to disk
        with resources.path("TetriumColor.Assets.Cache", MaxBasisFactory._cache_file) as path:
            with open(path, "wb") as f:
                pickle.dump(cache, f)

    @staticmethod
    def get_object(*args, **kwargs):
        # Load existing cache or initialize it
        cache = MaxBasisFactory.load_cache()
        # Use the __hash__ of the first argument as the cache key
        normalize_float = f"{kwargs['denom']:.2f}"
        # join_lums_and_chromas = "".join(
        #     [f"{x:.2f}" for x in kwargs['lums_per_channel'] + kwargs['chromas_per_channel']])
        key = stable_hash(args[0]) + stable_hash(normalize_float)
        # stable_hash(join_lums_and_chromas) if args or kwargs else None
        if key is None:
            raise ValueError("The first argument must be hashable to act as a key.")

        # Check if object exists in the cache
        if key not in cache:
            # If not, create and cache it
            cache[key] = MaxBasis(*args, **kwargs)
            MaxBasisFactory.save_cache(cache)

        # Return the cached object
        return cache[key]
