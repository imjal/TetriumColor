import numpy.typing as npt
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog, minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

from TetriumColor.Observer.Zonotope import getZonotopePoints


def ScalarProjection(a: npt.NDArray, b: npt.NDArray) -> float | npt.NDArray:
    """
    Compute the scalar projection of vector a onto vector b.

    Parameters:
    - a: (N,) array, the vector to be projected
    - b: (N,) array, the vector onto which a is projected

    Returns:
    - scalar_proj: float, the scalar projection of a onto b
    """
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        raise ValueError("The vector b must not be the zero vector.")
    scalar_proj = np.dot(a, b) / b_norm
    return scalar_proj


def ProjectPointOntoPlane(p: npt.NDArray, q: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """
    Projects a point p onto a plane defined by a point q and a normal vector n.

    Parameters:
    - p: (N,) array, point to project
    - q: (N,) array, a point on the plane
    - n: (N,) array, normal vector of the plane (should be normalized)

    Returns:
    - p_proj: (N,) array, projected point on the plane
    - v_orthogonal: (N,) array, orthogonal vector from p to the plane
    """
    # Ensure n is normalized
    n = n / np.linalg.norm(n)

    # Compute the vector from p to q
    v = p - q

    # Compute the scalar projection
    d = np.dot(v, n)

    # Compute the projected point
    p_proj = p - d * n

    # Compute the orthogonal vector
    # v_orthogonal = d * n

    return p_proj


def ParallelepipedToInequalities(p0, V):
    """
    Convert a parallelepiped defined by an origin and spanning vectors into linear inequalities.

    Parameters:
    - p0: (N,) array, the origin of the parallelepiped
    - V: (N x N) array, spanning vectors (columns)

    Returns:
    - A: (M x N) array of inequality coefficients
    - b: (M,) array of inequality bounds
    """
    # Generate all vertices of the parallelepiped
    num_vectors = V.shape[1]
    vertices = np.array([
        p0 + np.dot(V, np.array(t))
        # Generate all 2^n combinations of 0 and 1
        for t in np.ndindex(*(2,) * num_vectors)
    ])

    # Compute the convex hull of the vertices
    hull = ConvexHull(vertices)

    # Extract inequalities A, b from the convex hull
    A = hull.equations[:, :-1]  # Coefficients of the facets
    b = -hull.equations[:, -1]  # Offsets (note the sign)

    return A, b


def ZonotopeToInequalities(generators: npt.NDArray):
    """
    Convert generating vectors of a zonotope into a list of inequalities.

    Parameters:
        generators (np.ndarray): A 2D array of shape (n, d), where n is the number of generating vectors
                                 and d is the dimension of the space.

    Returns:
        A (np.ndarray): Coefficients of the inequalities, shape (m, d).
        b (np.ndarray): Right-hand side of the inequalities, shape (m,).
    """
    n, d = generators.shape

    # Compute all vertices of the zonotope as Minkowski sum of segments
    vertices = getZonotopePoints(generators.T, d, verbose=True)
    vertices = np.array(list(vertices[1].values())).reshape(-1, d)
    # Compute the convex hull of the vertices

    hull = ConvexHull(vertices)

    # Extract inequalities (half-space representation) from the convex hull
    A = hull.equations[:, :-1]  # Coefficients of inequalities
    b = -hull.equations[:, -1]  # RHS of inequalities

    return A, b


def MaximizeDimensionOnSubspace(A, b, p0, V, k):
    """
    find the maximum value along the k-th dimension of a plane's parameter space
    that intersects with an n-dimensional convex hull of inequalities.

    Parameters:
    - A: (P x N) matrix of constraints (A @ x <= b)
    - b: (P,) vector of bounds
    - p0: (N,) vector, the origin of the subspace (plane)
    - V: (N x M) matrix of spanning vectors for the M-dimensional subspace
    - k: Index of the dimension in the subspace to maximize

    Returns:
    - max_value: Maximum value of t_k
    - result.x: Values of [t1, t2, ..., tM] that maximize t_k
    """
    # Number of spanning vectors
    M = V.shape[1]  # Dimensionality of the subspace

    # Transform parallelepiped constraints: A @ (p0 + V @ t) <= b
    A_plane = np.dot(A, V)  # Transform spanning vectors
    b_plane = b - np.dot(A, p0)  # Shift bounds by the plane's origin

    # Objective function: Maximize t_k (k-th parameter of the subspace)
    c = np.zeros(M)
    c[k] = -1  # Negate to maximize (linprog minimizes by default)

    # Solve linear program
    result = linprog(c, A_ub=A_plane, b_ub=b_plane,
                     bounds=(None, None), method="highs")

    if result.success:
        max_value = -result.fun  # Negate to get the maximum
        return max_value, result.x
    else:
        raise ValueError("Optimization failed. No feasible solution found.")


def FindMaximalSaturation(hue_direction: npt.NDArray, generating_vecs: npt.NDArray) -> npt.NDArray | None:
    """
    Finds the point in the zonotope that has maximal saturation in the given hue direction,
    ensuring it remains within the subspace.

    Parameters:
    hue_direction   - (d,) array representing the hue direction vector.
    generating_vecs - (n, d) array where each row is a generating vector of the zonotope.

    Returns:
    boundary_point - The furthest point in the zonotope along the hue axis.
    """

    d = hue_direction.shape[0]
    n = generating_vecs.shape[0]  # Number of generators
    p0 = np.zeros(d)  # Origin

    # Compute the subspace basis
    v1 = np.ones(d) / np.linalg.norm(np.ones(d))  # Luminance vector
    v2 = ProjectPointOntoPlane(hue_direction, p0, v1)  # Hue vector
    V = np.column_stack((v1, v2))  # Subspace spanning vectors (d, 2)

    # Compute the projection matrix onto the subspace
    P_V = V @ np.linalg.inv(V.T @ V) @ V.T  # Projection matrix onto V
    I = np.eye(d)

    # Constraint to enforce the slice condition
    A_eq = (I - P_V) @ generating_vecs.T  # Ensure Gλ remains in the slice
    b_eq = np.zeros(d)  # Enforce the slice constraint

    # Objective: maximize v2^T G λ (hue axis)
    c = -v2.T @ generating_vecs.T  # Since linprog minimizes, negate the objective

    # Bounds: lambda values between 0 and 1
    bounds = [(0, 1) for _ in range(n)]

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        # Compute the full N-dimensional point
        boundary_point = generating_vecs.T @ res.x
        return boundary_point
    else:
        return None


def FindMaximumIn1DimDirection(point: npt.NDArray, metameric_direction: npt.NDArray, generating_vecs: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Find maximal point that lies in one direction inside of a linear constraint set

    Args:
        display_space_direction (npt.NDArray): _description_
        paralleletope_vecs (npt.NDArray): _description_

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Returns in both positive and negative directions the maximal point
    """
    M = generating_vecs.shape[0]  # Number of vectors defining the parallelepiped
    N = generating_vecs.shape[1]  # Dimension of parallelepiped

    p0 = np.zeros(N)
    if M > N:
        A, b = ZonotopeToInequalities(generating_vecs)
    else:
        A, b = ParallelepipedToInequalities(p0, generating_vecs)
    V = metameric_direction[np.newaxis, :].T
    # Maximize in the only dimension of the subspace
    k = 0
    # positive direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, point, V, k)
    x_max = point + np.dot(V, optimal_t)

    # negative direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, point, -1 * V, k)
    x_min = point + np.dot(-1 * V, optimal_t)

    return x_max, x_min


def GeneratePointsOnSubspace(vectors, num_points=100, distance=1):
    """
    Generate points on the subspace spanned by the given vectors, equidistant from the origin.

    Parameters:
    - vectors: A list of linearly independent vectors spanning the subspace.
    - num_points: Number of points to generate.
    - distance: Desired distance from the origin.

    Returns:
    - points: An array of points on the subspace.
    """
    k = len(vectors)
    points = []

    for _ in range(num_points):
        # Random coefficients
        coefficients = np.random.uniform(-1, 1, k)

        # Compute the point on the subspace
        point = np.sum([coefficients[i] * vectors[i] for i in range(k)], axis=0)

        # Normalize the point to lie on the unit sphere
        point_normalized = point / np.linalg.norm(point)

        # Scale the point to the desired distance
        point_scaled = distance * point_normalized
        points.append(point_scaled)

    return np.array(points)


def MaximizeAllDirectionsOnSubspace(A, b, V, center, num_directions=100):
    directions = GeneratePointsOnSubspace(V.T, num_directions)
    intersections = []

    for direction in directions:
        direction = direction[:, np.newaxis]  # Convert to column vector
        k = 0
        # positive direction optimization
        max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, center, direction, k)
        x_max = center + np.dot(direction, optimal_t)
        intersections.append(x_max)
    return np.array(intersections)


def excitations_to_contrast(
    resp: npt.NDArray[np.float64],
    background: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute contrast between response and background.
    """
    return (resp - background) / background


def receptor_isolate_spectral(
    T_receptors: npt.NDArray[np.float64],          # shape: (n_receptors, n_wavelengths)
    desired_contrasts: npt.NDArray[np.float64],    # shape: (n_receptors,)
    B_primary: npt.NDArray[np.float64],            # shape: (n_wavelengths, n_primaries)
    background_primary: npt.NDArray[np.float64],   # shape: (n_primaries,)
    initial_primary: npt.NDArray[np.float64],      # shape: (n_primaries,)
    primary_headroom: float,
    target_basis: npt.NDArray[np.float64],         # shape: (n_wavelengths, n_basis)
    project_indices: Optional[npt.NDArray[np.int_]],  # shape: (n_selected_wavelengths,)
    target_lambda: float,
    ambient_spd: npt.NDArray[np.float64],          # shape: (n_wavelengths,)
    POSITIVE_ONLY: bool = False,
    EXCITATIONS: bool = False
) -> Tuple[
    npt.NDArray[np.float64],  # modulating_primary
    npt.NDArray[np.float64],  # upper_primary
    npt.NDArray[np.float64]   # lower_primary
]:
    """
    Compute primary modulation that achieves desired receptor contrast,
    with optional smoothness constraints via projection to basis functions.
    """

    # Use all wavelengths if none selected
    if project_indices is None or len(project_indices) == 0:
        project_indices = np.arange(target_basis.shape[0])

    # Check receptor/contrast dimensions
    if T_receptors.shape[0] != len(desired_contrasts):
        raise ValueError("T_receptors and desired_contrasts dimension mismatch")

    # Initial guess (modulation vector)
    x0 = initial_primary - background_primary
    n_primaries = B_primary.shape[1]

    # Headroom sanity check
    tol = 1e-7
    if np.any(background_primary < primary_headroom - tol) or \
       np.any(background_primary > 1 - primary_headroom + tol):
        raise ValueError("Background primary out of allowed headroom range")

    # Bounds in primary space
    vub = np.zeros(n_primaries)
    vlb = np.zeros(n_primaries)
    for i in range(n_primaries):
        if background_primary[i] > 0.5:
            vub[i] = 1 - primary_headroom
            vlb[i] = background_primary[i] - (1 - primary_headroom - background_primary[i])
        elif background_primary[i] < 0.5:
            vub[i] = background_primary[i] + (background_primary[i] - primary_headroom)
            vlb[i] = primary_headroom
        else:
            vub[i] = 1 - primary_headroom
            vlb[i] = primary_headroom

    bounds = [(vlb[i] - background_primary[i], vub[i] - background_primary[i]) for i in range(n_primaries)]

    # Objective function
    def isolate_objective(x: npt.NDArray[np.float64]) -> float:
        background_spd = B_primary @ background_primary + ambient_spd
        modulation_spd = B_primary @ x + background_spd

        if EXCITATIONS:
            response = T_receptors @ modulation_spd
        else:
            response = excitations_to_contrast(T_receptors @ modulation_spd,
                                               T_receptors @ background_spd)

        f1 = np.linalg.norm(response - desired_contrasts) / np.linalg.norm(desired_contrasts)

        if target_lambda > 0:
            coeffs = np.linalg.lstsq(target_basis[project_indices, :],
                                     modulation_spd[project_indices], rcond=None)[0]
            projection_spd = target_basis @ coeffs
            f2 = np.linalg.norm(modulation_spd[project_indices] - projection_spd[project_indices]) / \
                np.linalg.norm(modulation_spd[project_indices])
            return f1 + target_lambda * f2
        else:
            return f1

    result = minimize(
        isolate_objective,
        x0,
        bounds=bounds,
        method='SLSQP',
        options={'disp': False, 'ftol': 1e-10}
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    modulating_primary = result.x
    upper_primary = background_primary + modulating_primary
    lower_primary = background_primary - modulating_primary

    # Final bounds check
    primary_tol = 2e-2
    if np.any(upper_primary > 1 - primary_headroom + primary_tol) or \
       np.any(upper_primary < 0 + primary_headroom - primary_tol):
        raise ValueError("upperPrimary out of gamut")

    if not POSITIVE_ONLY:
        if np.any(lower_primary > 1 - primary_headroom + primary_tol) or \
           np.any(lower_primary < 0 + primary_headroom - primary_tol):
            raise ValueError("lowerPrimary out of gamut")

    return modulating_primary, upper_primary, lower_primary


def FindMaximumWidthAlongDirection(metameric_direction: npt.NDArray, generating_vecs: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Find the pair of points in the gamut that have the maximum distance along a given metameric direction.

    Parameters:
    - metameric_direction: (d,) array representing the direction to find maximum width along
    - generating_vecs: (n, d) array where each row is a generating vector of the zonotope/parallelepiped

    Returns:
    - point1: (d,) array, first point of maximum width along the direction
    - point2: (d,) array, second point of maximum width along the direction
    """
    M = generating_vecs.shape[0]  # Number of vectors defining the parallelepiped
    N = generating_vecs.shape[1]  # Dimension of parallelepiped

    p0 = np.zeros(N)
    if M > N:
        A, b = ZonotopeToInequalities(generating_vecs)
    else:
        A, b = ParallelepipedToInequalities(p0, generating_vecs)

    # Objective function: maximize distance along the metameric direction
    def objective(center):
        V = metameric_direction[np.newaxis, :].T
        k = 0
        # positive direction optimization
        max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, center, V, k)
        x_max = center + np.dot(V, optimal_t)

        # negative direction optimization
        max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, center, -1 * V, k)
        x_min = center + np.dot(-1 * V, optimal_t)

        return -np.linalg.norm(x_max - x_min)

    # Use differential evolution to find the optimal center point
    result = differential_evolution(objective, bounds=[(0, 1)] * N)

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    center = result.x
    V = metameric_direction[np.newaxis, :].T
    k = 0
    # positive direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, center, V, k)
    point1 = center + np.dot(V, optimal_t)

    # negative direction optimization
    max_t1, optimal_t = MaximizeDimensionOnSubspace(A, b, center, -1 * V, k)
    point2 = center + np.dot(-1 * V, optimal_t)

    max_dist = np.linalg.norm(point1 - point2)
    print(f"Maximal Metameric Color Pair: {point1}, {point2}, Maximal Distance: {max_dist}")

    return point1, point2
