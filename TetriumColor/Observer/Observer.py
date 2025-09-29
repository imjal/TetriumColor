from importlib import resources
from typing import List, Union, Optional
from scipy.spatial import ConvexHull
from colour import XYZ_to_RGB, wavelength_to_XYZ, MSDS_CMFS, MultiSpectralDistributions

import os
import pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from .Spectra import Spectra, Illuminant
from .Zonotope import getReflectance, getZonotopePoints
from ..Utils.Hash import stable_hash


def GetHeringMatrix(dim) -> npt.NDArray:
    """
    Get Hering Matrix for a given dimension
    """
    if dim == 1:
        return np.array([[1]])
    if dim == 2:
        return np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -(1/np.sqrt(2))]])
    elif dim == 3:
        return np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [np.sqrt(2/3), -(1/np.sqrt(6)), -(1/np.sqrt(6))], [0, 1/np.sqrt(2), -(1/np.sqrt(2))]])
    elif dim == 4:
        return np.array([[1/2, 1/2, 1/2, 1/2], [np.sqrt(3)/2, -(1/(2 * np.sqrt(3))), -(1/(2 * np.sqrt(3))), -(1/(2 * np.sqrt(3)))], [0, np.sqrt(2/3), -(1/np.sqrt(6)), -(1/np.sqrt(6))], [0, 0, 1/np.sqrt(2), -(1/np.sqrt(2))]])
    else:
        raise Exception("Can't implement orthogonalize without hardcoding")


def GetHeringMatrixLumYDir(dim: int) -> npt.NDArray:
    """Get the Hering Matrix with the Luminance Direction as the Y direction

    Args:
        dim (int): dimension of the matrix

    Returns:
        npt.NDArray: Hering Matrix with Luminance Direction as the Y direction
    """
    h_mat = GetHeringMatrix(dim)
    lum_dir = np.copy(h_mat[0])
    h_mat[0] = h_mat[1]
    h_mat[1] = lum_dir
    return h_mat


def GetPerceptualHering(dim: int, isLumY=False) -> npt.NDArray:
    if dim == 2:
        return np.array([[0, 1], [1/np.sqrt(2), -(1/np.sqrt(2))]])  # B Y
    elif dim == 3:
        # lum_vec = np.array([1/np.sqrt(2) / 8, 1/np.sqrt(2), 1/np.sqrt(2)])
        lum_vec = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        # blue_vec = np.array([-np.sqrt(2/3),  (1/np.sqrt(6)), (1/np.sqrt(6))])
        blue_vec = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        mat = np.array([lum_vec/np.linalg.norm(lum_vec), [0, -1/np.sqrt(2), (1/np.sqrt(2))],
                       blue_vec / np.linalg.norm(blue_vec)])  # B G R
        return mat[[1, 0, 2]] if isLumY else mat
    elif dim == 4:
        return np.array([[0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [np.sqrt(3)/2, -(1/(2 * np.sqrt(3))), -(1/(2 * np.sqrt(3))), -(1/(2 * np.sqrt(3)))], [0, np.sqrt(2/3), -(1/np.sqrt(6)), -(1/np.sqrt(6))], [0, 0, 1/np.sqrt(2), -(1/np.sqrt(2))]])
    else:
        raise Exception("Can't implement orthogonalize without hardcoding")


def BaylorNomogram(wls, lambdaMax: int):
    """
    Baylor, Nunn, and Schnapf, 1987.
    """
    # These are the coefficients for the polynomial approximation.
    aN = np.array([-5.2734, -87.403, 1228.4, -3346.3, -5070.3, 30881, -31607])

    wlsum = wls / 1000.0
    wlsVec = np.log10((1.0 / wlsum) * lambdaMax / 561)
    logS = aN[0] + aN[1] * wlsVec + aN[2] * wlsVec ** 2 + aN[3] * wlsVec ** 3 + \
        aN[4] * wlsVec ** 4 + aN[5] * wlsVec ** 5 + aN[6] * wlsVec ** 6
    T = 10 ** logS
    return Cone(data=T.T, wavelengths=wls, quantal=True)


def GovardovskiiNomogram(S, lambdaMax):
    """
    Victor I. Govardovskii et al., 2000.
    """
    # Valid range of wavelength for A1-based visual pigments
    Lmin, Lmax = 330, 700

    # Valid range of lambdaMax value
    lmaxLow, lmaxHigh = 350, 600

    # Alpha-band parameters
    A, B, C = 69.7, 28, -14.9
    D = 0.674
    b, c = 0.922, 1.104

    # Beta-band parameters
    Abeta = 0.26

    # Assuming S is directly the wavelengths array
    wls = np.array(S)

    nWls = len(wls)
    # nT is assumed to be 1 based on user note
    T_absorbance = np.zeros((1, nWls))

    if lmaxLow < lambdaMax < lmaxHigh:
        # alpha-band polynomial
        a = 0.8795 + 0.0459 * np.exp(-(lambdaMax - 300) ** 2 / 11940)

        x = lambdaMax / wls
        midStep1 = np.exp(
            np.array([A, B, C]) * np.array([a, b, c]) - x[:, None] * np.array([A, B, C]))
        midStep2 = np.sum(midStep1, axis=1) + D

        S_x = 1 / midStep2

        # Beta-band polynomial
        bbeta = -40.5 + 0.195 * lambdaMax
        lambdaMaxbeta = 189 + 0.315 * lambdaMax

        midStep1 = -((wls - lambdaMaxbeta) / bbeta) ** 2
        S_beta = Abeta * np.exp(midStep1)

        # alpha band and beta band together
        T_absorbance[0, :] = S_x + S_beta

        # Zero sensitivity outside valid range
        T_absorbance[0, wls < Lmin] = 0
        T_absorbance[0, wls > Lmax] = 0
    else:
        raise ValueError(f'Lambda Max {lambdaMax} not in range of nomogram')

    return Cone(data=np.clip(T_absorbance.T, 0, 1), wavelengths=wls, quantal=True)


def LambNomogram(wls, lambdaMax):
    """
    Lamb, 1995.
    """
    # Coefficients for Equation 2
    a, b, c = 70, 28.5, -14.1
    A, B, C, D = 0.880, 0.924, 1.104, 0.655

    wlarg = lambdaMax / wls
    T = 1 / (np.exp(a * (A - wlarg)) + np.exp(b * (B - wlarg)) +
             np.exp(c * (C - wlarg)) + D)
    T = T / max(T)  # Normalize the sensitivity to peak at 1

    return Cone(data=T, wavelengths=wls)


def StockmanSharpeNomogram(wls, lambdaMax):
    """
    Stockman and Sharpe nomogram.
    """
    # Polynomial coefficients
    a = -188862.970810906644
    b = 90228.966712600282
    c = -2483.531554344362
    d = -6675.007923501414
    e = 1813.525992411163
    f = -215.177888526334
    g = 12.487558618387
    h = -0.289541500599

    # Prepare the wavelengths normalization
    logWlsNorm = np.log10(wls) - np.log10(lambdaMax / 558)

    # Compute log optical density
    logDensity = (a + b * logWlsNorm ** 2 + c * logWlsNorm ** 4 +
                  d * logWlsNorm ** 6 + e * logWlsNorm ** 8 +
                  f * logWlsNorm ** 10 + g * logWlsNorm ** 12 +
                  h * logWlsNorm ** 14)

    # Convert log10 absorbance to absorbance
    T_absorbance = 10 ** logDensity

    return Cone(data=T_absorbance, wavelengths=wls, quantal=True)


def NeitzNomogram(wls, lambda_max=559):
    # Carroll, McMahon, Neitz, & Neitz (2000)

    wls = wls.astype(np.float32)

    A = 0.417050601
    B = 0.002072146
    C = 0.000163888
    D = -1.922880605
    E = -16.05774461
    F = 0.001575426
    G = 5.11376e-05
    H = 0.00157981
    I = 6.58428e-05
    J = 6.68402e-05
    K = 0.002310442
    L = 7.31313e-05
    M = 1.86269e-05
    N = 0.002008124
    O = 5.40717e-05
    P = 5.14736e-06
    Q = 0.001455413
    R = 4.217640000e-05
    S = 4.800000000e-06
    T = 0.001809022
    U = 3.86677000e-05
    V = 2.99000000e-05
    W = 0.001757315
    X = 1.47344000e-05
    Y = 1.51000000e-05

    A2 = (np.log10(1.00000000 / lambda_max) - np.log10(1.00000000 / 558.5))
    vector = np.log10(np.reciprocal(wls))
    const = 1 / np.sqrt(2 * np.pi)

    ex_temp1 = np.log10(-E + E * np.tanh(-((10 ** (vector - A2)) - F) / G)) + D
    ex_temp2 = A * np.tanh(-(((10 ** (vector - A2))) - B) / C)
    ex_temp3 = - \
        (J / I * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - H) / I) ** 2)))
    ex_temp4 = - \
        (M / L * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - K) / L) ** 2)))
    ex_temp5 = - \
        (P / O * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - N) / O) ** 2)))
    ex_temp6 = (
        S / R * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - Q) / R) ** 2)))
    ex_temp7 = (
        (V / U * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - T) / U) ** 2))) / 10)
    ex_temp8 = (
        (Y / X * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - W) / X) ** 2))) / 100)
    ex_temp = ex_temp1 + ex_temp2 + ex_temp3 + ex_temp4 + \
        ex_temp5 + ex_temp6 + ex_temp7 + ex_temp8

    return Cone(data=np.clip(10 ** ex_temp, 0, 1), wavelengths=wls, quantal=True)


def load_csv(name) -> Spectra:
    with resources.path("TetriumColor.Assets.Cones", name) as data_path:
        return Spectra(np.array(pd.read_csv(data_path, header=None)), normalized=False)


class Cone(Spectra):
    with resources.path("TetriumColor.Assets.Cones", "ss2deg_10lin.csv") as data_path:
        ss_data = pd.read_csv(data_path, header=None).iloc[:-130, :]
    lens_absorption = load_csv("lensss_1.csv")
    macular_absorption = load_csv("macss_1.csv")
    templates = {
        "neitz": NeitzNomogram,
        "stockman": StockmanSharpeNomogram,
        "baylor": BaylorNomogram,
        "govardovskii": GovardovskiiNomogram,
        "lamb": LambNomogram
    }

    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None,
                 quantal=False, **kwargs):
        self.quantal = quantal
        self.lens = None
        self.macular = None
        self.od = None
        self.peak = None
        if isinstance(array, Spectra):
            super().__init__(**array.__dict__, **kwargs)
            self.peak = int(array.wavelengths[np.argmax(array.data)].item())
        else:
            super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)
            self.peak = int(self.wavelengths[np.argmax(self.data)].item())

    def observe(self, spectra: Spectra, illuminant: Spectra):
        return np.divide(np.dot(self.data, spectra.data), np.dot(self.data, illuminant.data))

    def as_quantal(self):
        if self.quantal:
            return self
        log_data = np.log(self.data) - np.log(self.wavelengths)
        quantal_data = np.exp(log_data - np.max(log_data))
        attrs = self.__dict__.copy()
        attrs["data"] = quantal_data
        attrs["quantal"] = True
        return self.__class__(**attrs)

    def as_energy(self):
        if not self.quantal:
            return self
        log_data = np.log(self.wavelengths) + np.log(self.data)
        energy_data = np.exp(log_data - np.max(log_data))

        attrs = self.__dict__.copy()
        attrs["data"] = energy_data
        attrs["quantal"] = False
        return self.__class__(**attrs)

    def with_od(self, od):
        if not self.quantal:
            return self.as_quantal().with_od(od).as_energy()
        od_data = np.divide(
            1 - np.exp(np.log(10) * -od * self.data), 1 - (10 ** -od))
        attrs = self.__dict__.copy()
        attrs["data"] = od_data
        return self.__class__(**attrs)

    def with_preceptoral(self, od: float = 0.5, lens: float = 1, macular: float = 1):
        # There are other lens and macular pigment data sources,
        # which can be found in the cones/ subfolder.
        if not self.quantal:
            return self.as_quantal().with_preceptoral(od, lens, macular)
        self.od = od
        self.lens = lens
        self.macular = macular

        lens_spectra = lens * \
            Cone.lens_absorption.interpolate_values(self.wavelengths)
        macular_spectra = macular * \
            Cone.macular_absorption.interpolate_values(self.wavelengths)
        denom = (10 ** (lens_spectra + macular_spectra))
        C_r = self.with_od(od)
        return (~(C_r / denom)).as_energy()

    @staticmethod
    def cone(peak, template="govardovskii", od: float = 0.35, lens: float = 1.0, macular: float = 1.0, wavelengths=None):
        # TODO: want to add eccentricity and/or macular, lens control
        if not isinstance(peak, (int, float)):
            raise ValueError("Currently only numerical peaks are supported.")
        if wavelengths is None:
            wavelengths = np.arange(400, 701, 1)
        if template not in Cone.templates:
            raise ValueError(f"Choose a template from {Cone.templates.keys()}")
        return Cone.templates[template](wavelengths, peak).with_preceptoral(od=od, macular=macular, lens=lens)

    @staticmethod
    def l_cone(wavelengths=None, template=None):
        if template is None:
            reflectances = Cone.ss_data.iloc[:, [0, 1]].to_numpy()
            return Cone(reflectances).interpolate_values(wavelengths)
        return Cone.cone(559, template=template, od=0.50, wavelengths=wavelengths)

    @staticmethod
    def m_cone(wavelengths=None, template=None):
        if template is None:
            reflectances = Cone.ss_data.iloc[:, [0, 2]].to_numpy()
            return Cone(reflectances).interpolate_values(wavelengths)
        return Cone.cone(530, template=template, od=0.5, wavelengths=wavelengths)

    @staticmethod
    def s_cone(wavelengths=None, template=None):
        if template is None:
            reflectances = Cone.ss_data.iloc[:, [0, 3]].dropna().to_numpy()
            return Cone(reflectances).interpolate_values(wavelengths)
        # http://www.cvrl.org/database/text/intros/introod.htm
        # "There are no good estimates of pigment optical densities for the S-cones."
        return Cone.cone(419, template=template, od=0.4, wavelengths=wavelengths)

    @staticmethod
    def q_cone(wavelengths=None, template="neitz"):
        # 545 per nathan & merbs 92
        return Cone.cone(545, template=template, od=0.5, wavelengths=wavelengths)

    @staticmethod
    def old_q_cone(wavelengths=None):
        with resources.path("TetriumColor.Assets.Cones.old", "tetrachromat_cones.npy") as data_path:
            A = np.load(data_path)
        path_wavelengths = np.arange(390, 831, 1)
        arr = np.stack([path_wavelengths, A[3, :]])
        return Cone(arr.T).interpolate_values(wavelengths)


class Observer:
    def __init__(self, sensors: List[Cone], illuminant: Optional[Spectra] = Illuminant.get('D65'), verbose: bool = False):
        self.dimension = len(sensors)
        self.sensors = sensors

        total_wavelengths = np.unique(np.concatenate(
            [sensor.wavelengths for sensor in sensors]))
        self.wavelengths = total_wavelengths
        self.sensors = sensors
        self.sensor_matrix = self.get_sensor_matrix(total_wavelengths)

        # take the average of non S-cone sensors
        self.v_lambda = self.sensor_matrix[1:].sum(axis=0) / (len(self.sensors) - 1)

        if illuminant is not None:
            # illuminant = Illuminant.get('E').interpolate_values(self.wavelengths)
            illuminant = illuminant.interpolate_values(self.wavelengths)
        else:
            print("No illuminant provided, using Illuminant D65")
            # illuminant = Illuminant.get('E').interpolate_values(self.wavelengths)
            illuminant = Illuminant.get('D65').interpolate_values(self.wavelengths)
            # illuminant = Illuminant(
            # np.vstack([self.wavelengths, np.ones_like(self.wavelengths)]).T)

        self.illuminant = illuminant
        self.normalized_sensor_matrix = self.get_normalized_sensor_matrix(
            total_wavelengths)

        self.verbose = verbose

    def __eq__(self, other):
        if not isinstance(other, Observer):
            return False
        return (tuple([[s.peak, s.od, s.lens, s.macular] for s in self.sensors]) ==
                tuple([[s.peak, s.od, s.lens, s.macular] for s in other.sensors])) and \
            (self.wavelengths == other.wavelengths).all() and self.dimension == other.dimension

    def __hash__(self):
        return stable_hash((tuple(tuple([s.peak, s.od, s.lens, s.macular]) for s in self.sensors), tuple(self.wavelengths), self.dimension))

    def __str__(self):
        return f"Observer({[[s.peak, s.od, s.lens, s.macular] for s in self.sensors]})"

    def get_multispectral(self) -> MultiSpectralDistributions:
        if self.dimension > 3:
            subset = [0, 1, 3]
        else:
            subset = [0, 1, 2]
        mat = np.concatenate([self.wavelengths[np.newaxis, :], self.sensor_matrix[subset]]).T
        d = {}
        for m in mat:
            d[int(m[0])] = tuple(m[1:].tolist())
        return MultiSpectralDistributions(d)

    @staticmethod
    def dichromat(wavelengths=None, illuminant=None):
        s_cone = Cone.s_cone(wavelengths)
        m_cone = Cone.m_cone(wavelengths)
        return Observer([s_cone, m_cone], illuminant=illuminant)

    @staticmethod
    def trichromat(wavelengths=None, illuminant=None, template='neitz'):
        l_cone = Cone.l_cone(wavelengths, template=template)
        m_cone = Cone.m_cone(wavelengths, template=template)
        s_cone = Cone.s_cone(wavelengths, template=template)
        return Observer([s_cone, m_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def tetrachromat(wavelengths=None, illuminant=None, verbose=False):
        # This is a "maximally well spaced" tetrachromat
        # Cone.cone(555, wavelengths=wavelengths, template="neitz", od=0.35)
        l_cone = Cone.l_cone(wavelengths)
        q_cone = Cone.cone(545, wavelengths=wavelengths,
                           template="neitz", od=0.5)
        # Cone.cone(530, wavelengths=wavelengths, template="neitz", od=0.35)
        m_cone = Cone.m_cone(wavelengths)
        # Cone.s_cone(wavelengths=wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant, verbose=verbose)

    @staticmethod
    def old_tetrachromat(wavelengths=None, illuminant=None, verbose=False):
        # This is a "maximally well spaced" tetrachromat
        # Cone.cone(555, wavelengths=wavelengths, template="neitz", od=0.35)
        l_cone = Cone.l_cone(wavelengths)
        q_cone = Cone.old_q_cone(wavelengths=wavelengths)
        # Cone.cone(530, wavelengths=wavelengths, template="neitz", od=0.35)
        m_cone = Cone.m_cone(wavelengths)
        # Cone.s_cone(wavelengths=wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant, verbose=verbose)

    @staticmethod
    def neitz_tetrachromat(wavelengths=None, illuminant=None, verbose=False):
        # This is a "maximally well spaced" tetrachromat
        l_cone = Cone.cone(559, wavelengths=wavelengths,
                           template="neitz", od=0.35)
        q_cone = Cone.cone(545, wavelengths=wavelengths,
                           template="neitz", od=0.35)
        m_cone = Cone.cone(530, wavelengths=wavelengths,
                           template="neitz", od=0.35)
        s_cone = Cone.s_cone(wavelengths=wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant, verbose=verbose)

    @staticmethod
    def govardovskii_tetrachromat(wavelengths=None, illuminant=None, verbose=False):
        # This is a "maximally well spaced" tetrachromat
        l_cone = Cone.cone(559, wavelengths=wavelengths,
                           template="govardovskii", od=0.35)
        q_cone = Cone.cone(545, wavelengths=wavelengths,
                           template="govardovskii", od=0.35)
        m_cone = Cone.cone(530, wavelengths=wavelengths,
                           template="govardovskii", od=0.35)
        s_cone = Cone.s_cone(wavelengths=wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant, verbose=verbose)

    @staticmethod
    def gaussian_tetrachromat(wavelengths: npt.NDArray, illuminant=None, verbose=False):

        def gaussian(wavelengths, peak, width=0.1):
            x = wavelengths
            y = np.exp(-0.5 * ((x - peak) /
                               (width * (wavelengths[-1] - wavelengths[0])))**2)
            return y
        l_cone = Cone(np.concatenate([wavelengths[:, np.newaxis], gaussian(
            wavelengths, 559, width=0.15)[:, np.newaxis]], axis=1))
        q_cone = Cone(np.concatenate([wavelengths[:, np.newaxis], gaussian(
            wavelengths, 545, 0.13)[:, np.newaxis]], axis=1))
        m_cone = Cone(np.concatenate([wavelengths[:, np.newaxis], gaussian(
            wavelengths, 530, 0.11)[:, np.newaxis]], axis=1))
        s_cone = Cone(np.concatenate([wavelengths[:, np.newaxis], gaussian(
            wavelengths, 450, 0.1)[:, np.newaxis]], axis=1))

        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant, verbose=verbose)

    @staticmethod
    def protanope(wavelengths=None, illuminant=None):
        m_cone = Cone.m_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone], illuminant=illuminant)

    @staticmethod
    def deuteranope(wavelengths=None, illuminant=None):
        l_cone = Cone.l_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def tritanope(wavelengths=None, illuminant=None):
        m_cone = Cone.m_cone(wavelengths)
        l_cone = Cone.l_cone(wavelengths)
        return Observer([m_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def bird(name, wavelengths=None, illuminant=None, verbose=False):
        """
        bird: bird types are ['UVS-Average-Bird.csv', 'UVS-bluetit.csv', 'UVS-Starling.csv', 'VS-Average-Bird.csv', 'VS-Peafowl.csv']
        """
        with resources.path("TetriumColor.Assets.Cones", f"{name}.csv") as data_path:
            bird_data = pd.read_csv(data_path, header=None)
        all_cones = []
        for i in range(1, 5):
            reflectances = bird_data.iloc[:, [0, i]].to_numpy()
            all_cones += [Cone(reflectances).interpolate_values(wavelengths)]
        return Observer(all_cones, illuminant=illuminant, verbose=verbose)

    def from_peaks(peaks: List[float],
                   wavelengths: Optional[npt.NDArray] = None,
                   illuminant: Optional[Spectra] = None,
                   template: str = "neitz",
                   od: float = 0.5,
                   macular: float = 1,
                   lens: float = 1,
                   verbose: bool = False):
        """Create an observer from a list of peaks

        Args:
            peaks (List[float]): List of peaks
            wavelengths (Optional[npt.NDArray], optional): Array of wavelengths. Defaults to None.

        Returns:
            Observer: Observer object
        """
        if wavelengths is None:
            wavelengths = np.arange(390, 831, 1)

        cones = []
        for peak in sorted(peaks):
            cone = Cone.cone(peak, wavelengths=wavelengths, template=template,
                             od=od if peak > 450 else od * 0.8, macular=macular, lens=lens)
            cones.append(cone)
        return Observer(cones, illuminant=illuminant, verbose=verbose)

    @staticmethod
    def custom_observer(wavelengths: npt.NDArray,
                        od: float = 0.5,
                        dimension: int = 4,
                        s_cone_peak: float = 419,
                        m_cone_peak: float = 530,
                        q_cone_peak: float = 547,
                        l_cone_peak: float = 559,
                        macular: float = 1,
                        lens: float = 1,
                        template: str = "neitz",
                        illuminant: Spectra | None = None,
                        verbose: bool = False, subset: List[int] = [0, 1, 3]):
        """Given specific parameters, return an observer model with Q cone peaked at 547

        Args:
            wavelengths (_type_): Array of wavelengths
            od (float, optional): optical density of photopigment. Defaults to 0.5.
            m_cone_peak (int, optional): peak of the M-cone. Defaults to 530.
            l_cone_peak (int, optional): peak of the L-cone. Defaults to 560.
            template (str, optional): cone template function. Defaults to "neitz".

        Returns:
            _type_: Observer of specified paramters and 4 cone types
        """

        l_cone = Cone.cone(l_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        l_cone_555 = Cone.cone(555, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        l_cone_551 = Cone.cone(551, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        l_cone_547 = Cone.cone(547, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        q_cone = Cone.cone(q_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        m_cone = Cone.cone(m_cone_peak, wavelengths=wavelengths, template=template, od=od, macular=macular, lens=lens)
        s_cone = Cone.cone(s_cone_peak, wavelengths=wavelengths, template=template,
                           od=0.8 * od, macular=macular, lens=lens)
        # s_cone = Cone.s_cone(wavelengths=wavelengths)
        if dimension == 3:
            set_cones = [s_cone, m_cone, q_cone, l_cone]
            return Observer([set_cones[i] for i in subset], verbose=verbose, illuminant=illuminant)
        elif dimension == 2:
            return Observer([s_cone, m_cone], verbose=verbose)
        elif dimension == 4:
            return Observer([s_cone, m_cone, q_cone, l_cone], verbose=verbose)
        elif dimension == 6:
            return Observer([s_cone, m_cone, l_cone_547, l_cone_551, l_cone_555, l_cone], verbose=verbose)
        else:
            raise NotImplementedError

    def get_whitepoint(self, wavelengths: Optional[npt.NDArray] = None):
        sensor_matrix = self.get_sensor_matrix(wavelengths)

        return np.matmul(sensor_matrix, self.illuminant.data)

    def get_wavelength_sensitivity(self, wavelengths):
        return self.get_sensor_matrix(wavelengths)

    def get_sensor_matrix(self, wavelengths: Optional[npt.NDArray] = None) -> npt.NDArray:
        """Get sensor matrix at specific wavelengths

        Args:
            wavelengths (Optional[npt.NDArray], optional): Array corresponding to wavelengths to interpolate. Defaults to None.

        Returns:
            npt.NDArray: Sensor matrix at specific wavelengths as sensors x wavelengths
        """
        if wavelengths is None:
            return self.sensor_matrix
        # assert wavelengths is 1d etc
        sensor_matrix = np.zeros((self.dimension, wavelengths.shape[0]))
        for i, sensor in enumerate(self.sensors):
            for j, wavelength in enumerate(wavelengths):
                sensor_matrix[i, j] = sensor.interpolated_value(wavelength)

        return sensor_matrix

    def get_normalized_sensor_matrix(self, wavelengths: Optional[npt.NDArray] = None) -> npt.NDArray:
        """Get normalized sensor matrix at specific wavelengths

        Args:
            wavelengths (Optional[npt.NDArray], optional): Array corresponding to wavelengths. Defaults to None.

        Returns:
            npt.NDArray: Normalized sensor matrix at specific wavelengths as sensors x wavelengths
        """
        sensor_matrix = self.get_sensor_matrix(wavelengths)
        interp_illum_data = self.illuminant.interpolate_values(wavelengths).data
        whitepoint = np.matmul(sensor_matrix, interp_illum_data)
        return ((sensor_matrix * interp_illum_data).T / whitepoint).T

    def observe(self, data: Union[npt.NDArray, Spectra]) -> npt.NDArray:
        """Given a spectra or reflectance, observe the color

        Args:
            data (Union[npt.NDArray, Spectra]): Either a spectra or reflectance to observe

        Returns:
            npt.NDArray: A n-dimensional array of the observed color
        """
        if isinstance(data, Spectra):
            data = data.interpolate_values(self.wavelengths).data
        else:
            assert data.size == self.wavelengths.size, f"Data shape {data.shape} must match wavelengths shape {self.wavelengths.shape}"

        observed_color = np.matmul(
            self.sensor_matrix, data * self.illuminant.data)
        whitepoint = np.matmul(self.sensor_matrix, self.illuminant.data)

        return np.divide(observed_color, whitepoint)

    def observe_normalized(self, data: Union[npt.NDArray, Spectra]) -> npt.NDArray:
        """Given a spectra or reflectance, observe the color with normalized sensor matrix

        Args:
            data (Union[npt.NDArray, Spectra]): Either a spectra or reflectance to observe

        Returns:
            npt.NDArray: A n-dimensional array of the observed color
        """

        if isinstance(data, Spectra):
            data = data.interpolate_values(self.wavelengths).data

        return np.matmul(self.normalized_sensor_matrix, data)

    def observe_spectras(self, spectras: List[Spectra]) -> npt.NDArray:
        """Observe the list of spectras 

        Args:
            spectras (List[Spectra]): List of spectras to observe

        Returns:
            npt.NDArray: A n-dimensional array of the observed colors
        """
        primary_intensities = np.array(
            [self.observe_normalized(s) for s in spectras])
        return primary_intensities

    def observer_v_lambda(self, data: npt.NDArray | Spectra) -> npt.NDArray:
        """_summary_

        Args:
            data (npt.NDArray | Spectra): _description_

        Returns:
            npt.NDArray: _description_
        """
        if isinstance(data, Spectra):
            data = data.interpolate_values(self.wavelengths).data
        return np.matmul(self.v_lambda, data)

    def dist(self, color1: Union[npt.NDArray, Spectra], color2: Union[npt.NDArray, Spectra]):
        return np.linalg.norm(self.observe(color1) - self.observe(color2))

    def get_optimal_reflectances(self) -> List[Spectra]:
        spectras = []
        for cuts, start in self.facet_ids:
            ref = getReflectance(
                cuts, start, self.normalized_sensor_matrix, self.dimension)
            spectras += [Spectra(wavelengths=self.wavelengths, data=ref)]
        return spectras

    def get_optimal_rgbs(self) -> npt.NDArray:
        if hasattr(self, 'rgbs'):
            return self.rgbs
        rgbs = []
        for cuts, start in tqdm(self.facet_ids):
            ref = getReflectance(
                cuts, start, self.normalized_sensor_matrix, self.dimension)
            ref = np.concatenate(
                [self.wavelengths[:, np.newaxis], ref[:, np.newaxis]], axis=1)
            rgbs += [Spectra(ref).to_rgb(illuminant=self.illuminant)]
        self.rgbs = np.array(rgbs).reshape(-1, 3)
        return self.rgbs

    def __find_optimal_colors(self) -> None:
        facet_ids, facet_sums = getZonotopePoints(
            self.normalized_sensor_matrix, self.dimension, self.verbose)
        tmp = [[[x, 1], [x, 0]] for x in facet_ids]
        self.facet_ids = [y for x in tmp for y in x]
        self.point_cloud = np.array(
            list(facet_sums.values())).reshape(-1, self.dimension)
        self.get_optimal_rgbs()

    def get_enumeration_of_solid(self, step_size):  # TODO: In the works
        facet_ids, facet_sums = getZonotopePoints(
            self.normalized_sensor_matrix, self.dimension, self.verbose)
        tmp = [[[x, 1], [x, 0]] for x in facet_ids]
        self.facet_ids = [y for x in tmp for y in x]
        points = np.array(list(facet_sums.values())
                          ).reshape(-1, self.dimension)
        divisors = np.tile(np.repeat(np.arange(
            step_size, 1+step_size, step_size), 4).reshape(-1, 4), (len(points), 1))

        factor = 1 // step_size + (1 % step_size > 0)
        self.point_cloud = np.repeat(points, factor, axis=0)*divisors

        rgbs = []
        for cuts, start in tqdm(self.facet_ids):
            ref = getReflectance(
                cuts, start, self.normalized_sensor_matrix, self.dimension)
            ref = np.concatenate(
                [self.wavelengths[:, np.newaxis], ref[:, np.newaxis]], axis=1)
            rgbs += [Spectra(ref).to_rgb(illuminant=self.illuminant)]

        self.rgbs = np.array(rgbs).reshape(-1, 3)
        self.rgbs = np.repeat(self.rgbs, factor, axis=0)*divisors[:, :3]
        return self.point_cloud, self.rgbs

    def get_optimal_colors(self) -> Union[npt.NDArray, npt.NDArray]:
        if not hasattr(self, 'point_cloud'):
            self.__find_optimal_colors()
        ObserverFactory.save_object(self)
        return self.point_cloud, self.rgbs

    def get_optimal_point_cloud(self) -> npt.NDArray:
        if not hasattr(self, 'point_cloud'):
            self.__find_optimal_colors()
        return self.point_cloud

    def get_full_colors(self) -> Union[npt.NDArray, npt.NDArray]:
        if not hasattr(self, 'point_cloud'):
            self.__find_optimal_colors()
        chrom_points = transformToChromaticity(self.point_cloud)
        hull = ConvexHull(chrom_points)  # chromaticity coordinates now
        self._full_indices = hull.vertices
        ObserverFactory.save_object(self)
        return chrom_points[self._full_indices], np.array(self.rgbs)[self._full_indices]

    def get_full_colors_in_activations(self) -> Union[npt.NDArray, npt.NDArray]:
        if not hasattr(self, 'point_cloud'):
            self.__find_optimal_colors()
        chrom_points = transformToChromaticity(self.point_cloud)
        hull = ConvexHull(chrom_points)  # chromaticity coordinates now
        self._full_indices = hull.vertices
        return self.point_cloud[self._full_indices], np.array(self.rgbs)[self._full_indices]

    def getFacetIdxFromIdx(self, idx) -> tuple:
        newidx = idx % int(len(self.point_cloud)/2)
        tmp = [int(i) for i in self.facet_ids[newidx]]
        start = 0 if idx > int(len(self.point_cloud)/2) else 1
        return tmp, start

    def get_trans_ref_from_idx(self, index: int) -> npt.NDArray:
        cuts, start = self.getFacetIdxFromIdx(index)
        return getReflectance(cuts, start, self.get_sensor_matrix(), self.dimension)


class ObserverFactory:
    _cache_file = "observer-cache.pkl"

    @staticmethod
    def load_cache():
        # Load the cache from file if it exists
        with resources.path("TetriumColor.Assets.Cache", ObserverFactory._cache_file) as path:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            return {}

    @staticmethod
    def save_cache(cache):
        # Save the cache to disk
        with resources.path("TetriumColor.Assets.Cache", ObserverFactory._cache_file) as path:
            with open(path, "wb") as f:
                pickle.dump(cache, f)

    @staticmethod
    def save_object(obs: Observer):
        key = stable_hash(obs)
        cache = ObserverFactory.load_cache()
        cache[key] = obs
        # Save the cache to disk
        ObserverFactory.save_cache(cache)

    @staticmethod
    def get_object(*args, **kwargs) -> Observer:
        # Load existing cache or initialize it
        cache = ObserverFactory.load_cache()
        obs = Observer(*args)
        # Use the __hash__ of the first argument as the cache key
        key = stable_hash(obs)

        # Check if object exists in the cache
        if key not in cache:
            cache[key] = obs
            ObserverFactory.save_cache(cache)

        # Return the cached object
        return cache[key]


def getsRGBfromWavelength(wavelength):
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    return XYZ_to_RGB(wavelength_to_XYZ(wavelength), "sRGB")


def getSampledHyperCube(step_size, dimension, outer_range=[[0, 1], [0, 1], [0, 1], [0, 1]]):
    """
    Get a hypercube sample of the space
    """
    g = np.meshgrid(*[np.arange(outer_range[i][0], outer_range[i]
                                [1] + 0.0000001, step_size) for i in range(dimension)])
    return np.array(list(zip(*(x.flat for x in g))))


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def transformToChromaticity(matrix) -> npt.NDArray:
    """
    Transform Coordinates (n_rows x dim) into Hering Chromaticity Coordinates
    """
    HMatrix = GetHeringMatrix(matrix.shape[1])
    return (HMatrix@matrix.T).T[:, 1:]


def transformToDisplayChromaticity(matrix, T, idxs=None) -> npt.NDArray:
    """
    Transform Coordinates (dim x n_rows) into Display Chromaticity Coordinates (divide by Luminance)
    """
    return (T@(matrix / np.sum(matrix, axis=0)))[idxs]
