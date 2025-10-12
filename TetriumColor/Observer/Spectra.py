from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import warnings
import copy

from colour import SDS_ILLUMINANTS, SDS_LIGHT_SOURCES, sRGB_to_XYZ, sd_to_XYZ, XYZ_to_xy, XYZ_to_sRGB, XYZ_to_Lab, delta_E, SpectralDistribution, notation, MultiSpectralDistributions
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class Spectra:
    def __init__(self, array: Optional[Union[npt.NDArray, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None,
                 normalized: Optional[bool] = True, **kwargs):
        """
        Either provide `reflectance` as a two column NDArray or provide both
        `wavelengths` and `data` as single column NDArrays.
        """
        if array is not None:
            if not isinstance(array, np.ndarray):
                raise TypeError("Input should be a numpy array")
            if array.shape[1] != 2:
                raise ValueError("Array should have two columns")
            wavelengths = array[:, 0]
            data = array[:, 1]
        if wavelengths is None:
            raise ValueError("Wavelengths must be provided or in array parameter.")

        if not (np.all(wavelengths >= 0)):
            raise ValueError("Wavelengths must be positive.")

        if not np.all(wavelengths == np.sort(wavelengths)):
            raise ValueError("Wavelengths should be in ascending order")

        if data is None:
            raise ValueError("Data must be provided with the wavelengths parameter or all included in array parameter.")
        if normalized and not (np.all(data >= 0) and np.all(data <= 1)):
            warnings.warn("Data has values not between 0 and 1. Clipping.")
            data = np.clip(data, 0, 1)

        self.wavelengths = wavelengths.reshape(-1)
        self.data = data.reshape(-1)
        self.normalized = normalized

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def array(self) -> npt.NDArray:
        """Get the array representation of the Spectra object. Useful for replicating the spectra's data in a new object. 

        Returns:
            npt.NDArray: the column stack of the wavelengths and spectral data.
        """
        return np.column_stack((self.wavelengths, self.data))

    def create_monochromatic(self, wavelengths: npt.NDArray, peak: float) -> 'Spectra':
        """
        Create a vector of the same shape as wavelengths where the value at the corresponding peak is 1.

        Parameters:
        wavelengths (numpy.ndarray): Array of wavelength values.
        peak (float): The peak wavelength value.

        Returns:
        numpy.ndarray: A vector with 1 at the peak position and 0 elsewhere.
        """
        peak_vector = np.zeros_like(wavelengths, dtype=float)
        peak_index = np.argmin(np.abs(wavelengths - peak))
        peak_vector[peak_index] = 1
        return Spectra(wavelengths=wavelengths, data=peak_vector)

    @staticmethod
    def from_transitions(transitions: List[float], start: int, wavelengths: npt.NDArray, maxVal: float = 1) -> 'Spectra':
        """Faster version of from_transitions using vectorized masking."""
        step = wavelengths[1] - wavelengths[0]
        minwave = wavelengths[0]
        maxwave = wavelengths[-1] + step

        # Ensure boundaries are included
        transitions = [minwave] + transitions + [maxwave]
        transitions = np.round(transitions, 2)

        ref = np.zeros_like(wavelengths, dtype=float)

        # Vectorized block filling
        current_value = start
        for i in range(0, len(transitions) - 1, 1):
            mask = (wavelengths >= transitions[i]) & (wavelengths < transitions[i + 1])
            ref[mask] = current_value
            current_value = 1 - current_value

        data = ref * maxVal
        return Spectra(wavelengths=wavelengths, data=data)

    # @staticmethod
    # def from_transitions(transitions: List[int], start: int, wavelengths: npt.NDArray, maxVal: float = 1) -> 'Spectra':
    #     """Return a Spectra that has the transitions of the given list, and starts with 0 or 1 depending on start parameter.

    #     Args:
    #         transitions (List[int]): list of wavelengths at which the reflectance goes from 0 to 1 or vice versa.
    #         start (int): 0 or 1, the starting value of the reflectance.
    #         wavelengths (npt.NDArray): the array of wavelengths.
    #         maxVal (float, optional): The maximum value of the reflectance. Defaults to 1.

    #     Returns:
    #         Spectra: the reflectance spectra with the transitions.
    #     """

    #     step = wavelengths[1] - wavelengths[0]
    #     minwave = wavelengths[0]
    #     maxwave = wavelengths[-1] + step

    #     transitions = copy.deepcopy(transitions)
    #     transitions.insert(0, minwave)
    #     transitions.insert(len(transitions), maxwave)
    #     transitions = [round(t, 2) for t in transitions]
    #     ref = []
    #     for i in range(len(transitions)-1):
    #         ref += [np.full(int(round((transitions[i+1] - transitions[i]) / step)), (start + i) % 2)]
    #     ref = np.concatenate(ref)
    #     assert (len(ref) == len(wavelengths))
    #     data = ref * maxVal
    #     return Spectra(wavelengths=wavelengths, data=data)

    def to_colour(self) -> SpectralDistribution:
        """Converts the Spectra object to a SpectralDistribution object from the Colour library.

        Returns:
            SpectralDistribution: the SpectralDistribution object represented by the Spectra object.  
        """
        wvs = np.arange(self.wavelengths[0], self.wavelengths[-1] + 1, 1)
        data = self.interpolate_values(wvs).data
        return SpectralDistribution(data=data, domain=wvs)

    def to_xyz(self, illuminant: Optional["Spectra"] = None, cmfs: Optional["MultiSpectralDistributions"] = None) -> npt.NDArray:
        """Converts the spectra to the XYZ color space value.

        Args:
            illuminant (Optional[&quot;Spectra&quot;], optional): Spectra object corresponding to wavelengths. Defaults to None.
            cmfs (Optional[&quot;MultiSpectralDistributions&quot;], optional): color matching functions to use for the XYZ transformation. Defaults to None.

        Returns:
            npt.NDArray: the XYZ color space value of the spectra. 
        """
        i = illuminant.to_colour() if illuminant else None

        return sd_to_XYZ(self.to_colour(), illuminant=i, cmfs=cmfs) / 100

    def to_xyz_d65(self, cmfs: Optional["MultiSpectralDistributions"] = None) -> npt.NDArray:
        """Converts the spectra to the XYZ color space value.

        Args:
            illuminant (Optional[&quot;Spectra&quot;], optional): Spectra object corresponding to wavelengths. Defaults to None.
            cmfs (Optional[&quot;MultiSpectralDistributions&quot;], optional): color matching functions to use for the XYZ transformation. Defaults to None.

        Returns:
            npt.NDArray: the XYZ color space value of the spectra. 
        """
        d65 = Illuminant.get("D65").to_xyz()
        return sd_to_XYZ(self.to_colour(), cmfs=cmfs) / d65[1] / 100

    def to_rgb(self, illuminant: Optional["Spectra"] = None, cmfs: Optional["MultiSpectralDistributions"] = None) -> npt.NDArray:
        """Converts the spectra to the sRGB color space value.

        Args:
            illuminant (Optional[&quot;Spectra&quot;], optional): Spectra corresponding to the illuminant. Defaults to None.
            cmfs (Optional[&quot;MultiSpectralDistributions&quot;], optional): color matching functions to use for the RGB transformation. Defaults to None.

        Returns:
            npt.NDArray: the sRGB color space value of the spectra.
        """
        i = illuminant.to_colour() if illuminant is not None else Illuminant.get("D65").to_colour()

        chromaticity_coord = XYZ_to_xy(sd_to_XYZ(i, cmfs=cmfs) / 100)

        return np.clip(XYZ_to_sRGB(self.to_xyz(illuminant), illuminant=chromaticity_coord), 0, 1)

    def to_lab(self, illuminant: Optional["Spectra"] = None, cmfs: Optional["MultiSpectralDistributions"] = None) -> npt.NDArray:
        """Converts the spectra to the Lab color space value.
        """
        return XYZ_to_Lab(self.to_xyz(illuminant))

    @staticmethod
    def delta_e(spectra1: "Spectra", spectra2: "Spectra") -> float:
        """Calculates the delta E between two spectra.
        """
        return delta_E(spectra1.to_lab(), spectra2.to_lab())

    def avg_delta_e_unit(self, illuminant: "Spectra", num_samples: int = 100, ball_radius: float = 0.01) -> float:
        chromaticity_coord = XYZ_to_xy(sd_to_XYZ(illuminant.to_colour()) / 100)
        rgb = self.to_rgb(illuminant)

        peturbations = np.random.normal(size=(num_samples, 3))
        peturbations /= np.linalg.norm(peturbations, axis=1, keepdims=True)
        peturbations = peturbations * ball_radius + rgb

        deltas = []
        for p in peturbations:
            peturb_lab = XYZ_to_Lab(sRGB_to_XYZ(p, illuminant=chromaticity_coord))
            deltas.append(delta_E(self.to_lab(), peturb_lab))
        return np.mean(deltas), np.std(deltas)

    @staticmethod
    def spectral_rmse(spectra1: "Spectra", spectra2: "Spectra") -> float:
        """Calculates the spectral RMSE between two spectra.
        """
        return np.sqrt(np.mean((spectra1.data - spectra2.data) ** 2))

    def to_hex(self, illuminant: Optional["Spectra"] = None):
        return notation.RGB_to_HEX(np.clip(self.to_rgb(illuminant), 0, 1))

    def plot(self, name: str | None = None, color: npt.NDArray | List[float] | tuple | None = None, ax: plt.Axes | None = None, alpha: float = 1.0, normalize: bool = False) -> None:
        """Plot the spectra.

        Args:
            name (str | None, optional): Name of the spectra to plot. Defaults to None.
            color (npt.NDArray | List[float] | tuple | None, optional): color tuple. Defaults to None.
            ax (plt.Axes | None, optional): matplotlib axes to plot on. Will create a new one if there is none provided. Defaults to None.
            alpha (float, optional): transparency of the spectra plotted. Defaults to 1.0.
            normalize (bool, optional): whether to normalize the spectra. Defaults to False.
        """
        if color is None and name is None:
            color = self.to_rgb()
            # name = self.__class__.__name__
        factor = np.max(self.data) if normalize else 1
        if not ax:
            plt.scatter(self.wavelengths, self.data/factor, color=color, s=10)
            plt.plot(self.wavelengths, self.data/factor, label=name, color=color, alpha=alpha)
        else:
            ax.scatter(self.wavelengths, self.data/factor, color=color, s=10)
            ax.plot(self.wavelengths, self.data/factor, label=name, color=color, alpha=alpha)

    def interpolate_values(self, wavelengths: Union[npt.NDArray, None]):
        """Interpolate the spectra to the given wavelengths.

        Args:
            wavelengths (Union[npt.NDArray, None]): the wavelengths to linearly interpolate the spectra to.

        Returns:
            Spectra: the interpolated spectra.
        """
        if wavelengths is None:
            return self
        if np.array_equal(wavelengths, self.wavelengths):
            return self
        interpolated_data = []
        for wavelength in wavelengths:
            d = self.interpolated_value(wavelength)
            interpolated_data.append(d)
        attrs = self.__dict__.copy()

        attrs["data"] = np.array(interpolated_data)
        attrs["wavelengths"] = wavelengths
        return self.__class__(**attrs)

    def interpolated_value(self, wavelength: float) -> float:
        """Interpolates the value of the spectra at the given wavelength.

        Args:
            wavelength (float): the wavelength to interpolate the spectra to.

        Returns:
            float: the interpolated value of the spectra at the given wavelength.
        """
        idx = np.searchsorted(self.wavelengths, wavelength)

        if idx == 0:
            return float(self.data[0])
        if idx == len(self.wavelengths):
            return float(self.data[-1])

        if self.wavelengths[idx] == wavelength:
            return float(self.data[idx])

        # Linearly interpolate between the nearest wavelengths
        x1, y1 = self.wavelengths[idx - 1], self.data[idx - 1]
        x2, y2 = self.wavelengths[idx], self.data[idx]

        return y1 + (y2 - y1) * (wavelength - x1) / (x2 - x1)

    def interpolate(self, new_wavelengths: npt.NDArray, method: str = 'linear') -> 'Spectra':
        """Interpolate the spectra to the given wavelengths using the specified method.

        Args:
            new_wavelengths (npt.NDArray): The wavelengths to interpolate the spectra to.
            method (str, optional): The interpolation method to use. Defaults to 'linear'. 
            Options include:
            - 'linear': Linear interpolation.
            - 'nearest': Nearest-neighbor interpolation.
            - 'zero': Zero-order spline interpolation.
            - 'slinear': First-order spline interpolation.
            - 'quadratic': Second-order spline interpolation.
            - 'cubic': Third-order spline interpolation.
            - 'previous': Previous value interpolation.
            - 'next': Next value interpolation.

        Returns:
            Spectra: The interpolated spectra.
        """

        interpolator = interp1d(self.wavelengths, self.data, kind=method, bounds_error=False)
        new_data = interpolator(new_wavelengths)

        return Spectra(wavelengths=new_wavelengths, data=new_data)

    def smooth(self, poly_degree=8) -> 'Spectra':
        """Smooth the spectra data using Savitzky-Golay filtering.

        Args:
            poly_degree (int, optional): The degree of the polynomial to fit. Defaults to 5.
        Returns:
            Spectra: The spectra with the fitted smooth function.
        """

        # Apply Savitzky-Golay filter
        smoothed_data = savgol_filter(self.data, window_length=11, polyorder=poly_degree)

        return Spectra(wavelengths=self.wavelengths, data=smoothed_data)

    @staticmethod
    def multiply(primaries: List['Spectra'], bgor: npt.NDArray[np.float64]) -> List['Spectra']:
        new_specs = []
        for p, b in zip(primaries, bgor):
            new_specs.append(Spectra(wavelengths=p.wavelengths, data=p.data * b))
        return new_specs

    @staticmethod
    def add(primaries: List['Spectra']) -> 'Spectra':
        data = np.sum(p.data for p in primaries)
        return Spectra(wavelengths=primaries[0].wavelengths, data=data)

    def __getitem__(self, wavelength: float) -> float:
        return self.interpolated_value(wavelength)

    def __add__(self, other: Union['Spectra', float, int]) -> 'Spectra':
        # todo: can add ndarray support
        # todo: can add wavelength interpolation
        attrs = self.__dict__.copy()

        if isinstance(other, float) or isinstance(other, int):
            attrs["data"] = self.data + other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths, self.wavelengths):
                raise ValueError(f"Wavelengths must match for addition.")
            attrs["data"] = self.data + other.data
        else:
            raise TypeError("This addition not supported.")

        return self.__class__(**attrs)

    def __rsub__(self, other: Union[float, int]) -> 'Spectra':
        attrs = self.__dict__.copy()

        if isinstance(other, (float, int)):
            attrs["data"] = other - self.data
        else:
            raise TypeError("This subtraction not supported from the left side with a non-numeric type.")

        return self.__class__(**attrs)

    def __mul__(self, other: Union[int, float, 'Spectra']):
        attrs = self.__dict__.copy()
        if isinstance(other, (int, float)):
            attrs["data"] = other * self.data
        else:
            attrs["data"] = other.data * self.data
        return self.__class__(**attrs)

    def __rmul__(self, scalar: Union[int, float]):
        # This method allows scalar * Spectra to work
        return self.__mul__(scalar)

    def __rpow__(self, base: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(base, self.data)
        return self.__class__(**attrs)

    def __pow__(self, exponent: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(self.data, exponent)
        return self.__class__(**attrs)

    def __truediv__(self, other: Union["Spectra", float, int]):
        attrs = self.__dict__.copy()
        if isinstance(other, (int, float)):
            attrs["data"] = self.data / other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths, self.wavelengths):
                raise ValueError("Wavelengths must match for division")
            denom = np.clip(other.data, 1e-7, None)
            attrs["data"] = self.data / denom
        return self.__class__(**attrs)

    """Normalize operator, overwriting invert ~ operator."""

    def __invert__(self):
        # Division by maximum element
        attrs = self.__dict__.copy()
        attrs["data"] = self.data / np.max(self.data)
        attrs["normalized"] = True
        return self.__class__(**attrs)

    def __str__(self):
        # TODO: can be smarter
        return str(self.wavelengths)


class Illuminant(Spectra):
    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None, **kwargs):
        if isinstance(array, Spectra):
            super().__init__(**array.__dict__, **kwargs)
        else:
            super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)

    @staticmethod
    def get(name: str) -> 'Illuminant':
        """Get the Illuminant object corresponding to the given name.

        Args:
            name (str): the name of the illuminant from the SDS_ILLUMINANTS or SDS_LIGHT_SOURCES in the Colour library. 

        Returns:
            Illuminant: the Illuminant object corresponding to the given name.
        """

        light = SDS_ILLUMINANTS.get(name)
        if light is None:
            light = SDS_LIGHT_SOURCES.get(name)
            if light is None:
                raise ValueError(f"Illuminant {name} not found.")
        return Illuminant(data=light.values / np.max(light.values), wavelengths=light.wavelengths)


def convert_refs_to_spectras(refs: npt.NDArray, wavelengths: npt.NDArray) -> List[Spectra]:
    """Converts the array of reflectance values to Spectra objects.

    Args:
        refs (List[Spectra] | npt.NDArray): List of reflectances or an npt.NDArray of reflectances, as a num_of_refs x num_of_wavelengths array.
        wavelengths (npt.NDArray): Array of wavelengths.

    Returns:
        List[Spectra]: List of Spectra objects corresponding to the reflectances.
    """
    new_refs = [np.concatenate([wavelengths[:, np.newaxis], ref[:, np.newaxis]], axis=1) for ref in refs]
    return [Spectra(ref) for ref in new_refs]


def convert_refs_to_rgbs(refs: npt.NDArray, wavelengths: npt.NDArray) -> List[npt.NDArray]:
    """Converts the array of reflectance values to Spectra objects.

    Args:
        refs (List[Spectra] | npt.NDArray): List of reflectances or an npt.NDArray of reflectances, as a num_of_refs x num_of_wavelengths array.
        wavelengths (npt.NDArray): Array of wavelengths.

    Returns:
        List[npt.NDArray]: List of sRGB values corresponding to the reflectances.
    """
    return [Spectra(np.concatenate([wavelengths[:, np.newaxis], ref[:, np.newaxis]], axis=1)).to_rgb() for ref in refs]
