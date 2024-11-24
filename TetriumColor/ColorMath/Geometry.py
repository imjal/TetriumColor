import numpy as np
import numpy.typing as npt
import math


def SampleAnglesEqually(samples, dim) -> npt.NDArray:
    """
    For a given dimension, sample the sphere equally
    """
    if dim == 2:
        return SampleCircle(samples)
    elif dim == 3:
        return SampleFibonacciSphere(samples)
    else:
        raise NotImplementedError("Only 2D and 3D Spheres are supported")


def SampleCircle(samples=1000) -> npt.NDArray:
    return np.array([[2 * math.pi * (i / float(samples)) for i in range(samples)]]).T


def SampleFibonacciSphere(samples=1000) -> npt.NDArray:
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # stupid but i do not care right now
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(z / r)
        theta = np.arctan2(y, x)
        points.append((theta, phi))

    return np.array(points)


def ConvertXYZToCubeUV(x, y, z):
    # Compute absolute values
    absX = np.abs(x)
    absY = np.abs(y)
    absZ = np.abs(z)

    # Determine the positive and dominant axes
    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    # Initialize arrays for index, u, v
    index = np.zeros_like(x, dtype=int)
    maxAxis = np.zeros_like(x, dtype=float)
    uc = np.zeros_like(x, dtype=float)
    vc = np.zeros_like(x, dtype=float)

    # POSITIVE X
    mask = isXPositive & (absX >= absY) & (absX >= absZ)
    maxAxis[mask] = absX[mask]
    uc[mask] = -z[mask]
    vc[mask] = y[mask]
    index[mask] = 0

    # NEGATIVE X
    mask = ~isXPositive & (absX >= absY) & (absX >= absZ)
    maxAxis[mask] = absX[mask]
    uc[mask] = z[mask]
    vc[mask] = y[mask]
    index[mask] = 1

    # POSITIVE Y
    mask = isYPositive & (absY >= absX) & (absY >= absZ)
    maxAxis[mask] = absY[mask]
    uc[mask] = x[mask]
    vc[mask] = -z[mask]
    index[mask] = 2

    # NEGATIVE Y
    mask = ~isYPositive & (absY >= absX) & (absY >= absZ)
    maxAxis[mask] = absY[mask]
    uc[mask] = x[mask]
    vc[mask] = z[mask]
    index[mask] = 3

    # POSITIVE Z
    mask = isZPositive & (absZ >= absX) & (absZ >= absY)
    maxAxis[mask] = absZ[mask]
    uc[mask] = x[mask]
    vc[mask] = y[mask]
    index[mask] = 4

    # NEGATIVE Z
    mask = ~isZPositive & (absZ >= absX) & (absZ >= absY)
    maxAxis[mask] = absZ[mask]
    uc[mask] = -x[mask]
    vc[mask] = y[mask]
    index[mask] = 5

    # Convert range from -1 to 1 to 0 to 1
    u = 0.5 * (uc / maxAxis + 1.0)
    v = 0.5 * (vc / maxAxis + 1.0)

    return index, u, v


def ConvertCubeUVToXYZ(index, u, v, normalize=None) -> npt.NDArray:
    """
    Convert cube UV coordinates back to XYZ with all points at a specified radius from the origin.
    """
    # Convert range 0 to 1 to -1 to 1
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    # Initialize x, y, z
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)

    # POSITIVE X
    mask = index == 0
    x[mask], y[mask], z[mask] = 1.0, vc[mask], -uc[mask]

    # NEGATIVE X
    mask = index == 1
    x[mask], y[mask], z[mask] = -1.0, vc[mask], uc[mask]

    # POSITIVE Y
    mask = index == 2
    x[mask], y[mask], z[mask] = uc[mask], 1.0, -vc[mask]

    # NEGATIVE Y
    mask = index == 3
    x[mask], y[mask], z[mask] = uc[mask], -1.0, vc[mask]

    # POSITIVE Z
    mask = index == 4
    x[mask], y[mask], z[mask] = uc[mask], vc[mask], 1.0

    # NEGATIVE Z
    mask = index == 5
    x[mask], y[mask], z[mask] = -uc[mask], vc[mask], -1.0

    # Normalize to unit sphere
    if normalize is not None:
        norm = np.sqrt(x**2 + y**2 + z**2)
        x = (x / norm) * normalize
        y = (y / norm) * normalize
        z = (z / norm) * normalize

    return np.array([x, y, z]).T
