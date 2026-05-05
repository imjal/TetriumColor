try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None
import numpy as np
import numpy.typing as npt
import math

from scipy.spatial import ConvexHull
from TetriumColor.Observer.Zonotope import getZonotopePoints


def GetSimplexOrigin(side_length, dimension) -> npt.NDArray:
    """Get the origin of the simplex

    Args:
        side_length (_type_): side length of the simplex
        dimension (_type_): dimension of the simplex

    Returns:
        npt.NDArray: The origin of the simplex
    """
    return np.sum(GetSimplex(side_length, dimension), axis=0) / dimension


def GetSimplex(side_length, dimension) -> npt.NDArray:
    """Get Canonical Simplex

    Args:
        side_length (_type_): side length of the simplex
        dimension (_type_): dimension of the simplex

    Raises:
        ValueError: Only dimensions 2 & 3 & 4 are supported

    Returns:
        npt.NDArray: The canonical simplex
    """
    # Calculate the height of the tetrahedron
    height = (3 ** 0.5) * side_length / 2

    # Calculate the coordinates of the vertices
    vertex1 = [-side_length/2, 0, 0]
    vertex2 = [side_length/2, 0, 0]
    vertex3 = [0, height, 0]
    vertex4 = [0, height/3, (2/3)**0.5 * side_length]

    arr = np.array([vertex1, vertex2, vertex3, vertex4])

    if dimension == 3:
        return arr[:3, :2]
    elif dimension == 4:
        return arr
    elif dimension == 2:
        return arr[:2, :2]
    else:
        raise ValueError("Only dimensions 2 & 3 & 4 are supported")


def ComputeBarycentricCoordinates(coordinates: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """Compute Barycentric Coordinates of a point relative to coordinates of another simplex

    Args:
        coordinates (npt.NDArray): coordinates of the simplex
        p (npt.NDArray): point to locate relative to the simplex

    Raises:
        NotImplementedError: Only 3D & 4D Simplex are supported

    Returns:
        npt.NDArray: Barycentric Coordinates of the point relative to the simplex
    """
    if len(coordinates) == 4:
        a = coordinates[0]
        b = coordinates[1]
        c = coordinates[2]
        d = coordinates[3]

        vap = p - a
        vbp = p - b

        vab = b - a
        vac = c - a
        vad = d - a

        vbc = c - b
        vbd = d - b

        va6 = np.dot(np.cross(vbp, vbd), vbc)
        vb6 = np.dot(np.cross(vap, vac), vad)
        vc6 = np.dot(np.cross(vap, vad), vab)
        vd6 = np.dot(np.cross(vap, vab), vac)
        v6 = 1 / np.dot(np.cross(vab, vac), vad)
        return np.array([va6*v6, vb6*v6, vc6*v6, vd6*v6])
    elif len(coordinates) == 3:
        v0 = coordinates[1] - coordinates[0]
        v1 = coordinates[2] - coordinates[0]
        v2 = p - coordinates[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01

        barycentric_coords = np.zeros(3)
        barycentric_coords[1] = (d11 * d20 - d01 * d21) / denom
        barycentric_coords[2] = (d00 * d21 - d01 * d20) / denom
        barycentric_coords[0] = 1 - \
            barycentric_coords[1] - barycentric_coords[2]

        return barycentric_coords
    else:  # 2D
        v0 = coordinates[1] - coordinates[0]
        v1 = p - coordinates[0]
        return np.array([np.dot(v1, v0) / np.dot(v0, v0), 1 - np.dot(v1, v0) / np.dot(v0, v0)])


def GetSimplexBarycentricCoords(dimension: int, simplex_points: npt.NDArray, points_to_locate: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Get the barycentric coordinates of a point in a simplex

    Args:
        dimension (int): dimension of the simplex (4 is tetrahedron, 3 is triangle, 2 is line)
        simplex_points (npt.NDArray): points that define the simplex (will be used as the basis to transform to canonical)
        points_to_locate (npt.NDArray): points that we want to locate in the simplex (traditionally, the spectral locus)

    Returns:
        tuple[npt.NDArray, npt.NDArray]: The canonical simplex coordinates and the barycentric coordinates relative to it
    """
    simplex_coords = GetSimplex(1, dimension)
    coords = np.zeros((points_to_locate.shape[0], dimension))
    for i in range(points_to_locate.shape[0]):
        coords[i] = ComputeBarycentricCoordinates(simplex_points, points_to_locate[i])

    barycentric_coords = coords@simplex_coords
    return simplex_coords, barycentric_coords


def CheckIfPointInConvexHull(point: npt.NDArray, hull, tol=1e-12):
    A = hull.equations[:, :-1]  # Coefficients for the inequality (Ax + b <= 0)
    b = hull.equations[:, -1]  # Constant term in the inequality
    return np.all(np.dot(A, point) + b <= tol)


def ComputeVolumeOfPolytope(points: npt.NDArray) -> float:
    return ConvexHull(points).volume


def ComputeParalleletope(basis):
    d = basis.shape[1]
    _, facet_sums = getZonotopePoints(basis, d)
    return np.array(list(facet_sums.values())).reshape(-1, d)


def ConvertPolarToCartesian(SH: npt.NDArray) -> npt.NDArray:
    """
    Convert Polar to Cartesian Coordinates
    Args:
        SH (npt.ArrayLike, N x 2): The SH coordinates that we want to transform. Saturation and Hue are transformed
    """
    S, H = SH[:, 0], SH[:, 1]
    return np.array([S * np.cos(H), S * np.sin(H)]).T


def ConvertCartesianToPolar(CC: npt.NDArray) -> npt.NDArray:
    """
    Convert Cartesian to Polar Coordinates (SH)
    Args:
        Cartesian (npt.ArrayLike, N x 2): The Cartesian coordinates that we want to transform
    """
    x, y = CC[:, 0], CC[:, 1]
    rTheta = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)]).T
    rTheta[:, 1] = np.where(rTheta[:, 1] < -1e-9, rTheta[:, 1] + 2 * np.pi, rTheta[:, 1])  # Ensure θ is in [0, 2π]
    return rTheta


def ConvertSphericalToCartesian(rThetaPhi) -> npt.NDArray:
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

    Parameters:
    - r: Radial distance (array or scalar).
    - theta: Azimuthal angle in radians (array or scalar).
    - phi: Polar angle in radians (array or scalar).

    Returns:
    - A numpy array of shape (n, 3), where each row is [x, y, z].
    """
    r, theta, phi = rThetaPhi[:, 0], rThetaPhi[:, 1], rThetaPhi[:, 2]

    # Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.vstack((x, y, z)).T


def ConvertCartesianToSpherical(xyz) -> npt.NDArray:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters:
    - x, y, z: Arrays or scalars representing Cartesian coordinates.

    Returns:
    - A numpy array of shape (n, 3), where each row is [r, theta, phi].
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuthal angle (theta)
    theta = np.arctan2(y, x)

    # Polar angle (phi)
    phi = np.arccos(np.clip(z / r, -1.0, 1.0))  # Clip for numerical stability

    return np.vstack((r, theta, phi)).T


def SampleAnglesEqually(samples, dim) -> npt.NDArray:
    """
    For a given dimension, sample the sphere equally
    """
    if dim == 2:
        return SampleCircle(samples)
    elif dim == 3:
        xyz = SampleFibonacciSphereCartesian(samples)
        spherical = ConvertCartesianToSpherical(xyz)
        return spherical[:, 1:]
    else:
        raise NotImplementedError("Only 2D and 3D Spheres are supported")


def SampleCircle(samples=1000) -> npt.NDArray:
    return np.array([[2 * math.pi * (i / float(samples)) for i in range(samples)]]).T


def SampleFibonacciSphereCartesian(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def RotateToZAxis(vector: npt.NDArray) -> npt.NDArray:
    """
    Returns a rotation matrix that rotates the given vector to align with the Z-axis.

    Parameters:
        vector (array-like): The input vector to align with the Z-axis.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # Normalize the input vector
    v = np.array(vector, dtype=float)
    v = v / np.linalg.norm(v)

    # Z-axis unit vector
    z_axis = np.array([0, 0, 1], dtype=float)

    # Compute the axis of rotation (cross product)
    axis = np.cross(v, z_axis)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        # The vector is already aligned with the Z-axis
        return np.eye(3)

    axis = axis / axis_norm  # Normalize the axis

    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.dot(v, z_axis))

    # Compute the skew-symmetric cross-product matrix for the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Use the Rodrigues' rotation formula
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


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


def CartesianToUV(vertices):
    """
    Compute UV mapping for vertices on a sphere.

    Args:
        vertices (ndarray): Nx3 array of vertex positions in Cartesian coordinates.

    Returns:
        ndarray: Nx2 array of UV coordinates (u, v) for each vertex.
    """
    # Normalize vertices to ensure they're on the unit sphere
    normalized = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    # Compute spherical coordinates
    theta = np.arctan2(z, x)  # Longitude: angle in the x-z plane
    phi = np.arcsin(y)        # Latitude: angle from the y-axis

    # Normalize to [0, 1]
    u = (theta + np.pi) / (2 * np.pi)  # Map theta from [-π, π] to [0, 1]
    v = (phi + np.pi / 2) / np.pi      # Map phi from [-π/2, π/2] to [0, 1]

    return np.column_stack((u, v))


def UVToCartesian(uv_coords, radius=1.0):
    """
    Convert UV coordinates back to Cartesian coordinates on a sphere.

    Args:
        uv_coords (ndarray): Nx2 array of UV coordinates (u, v).
        radius (float, optional): Radius of the sphere. Defaults to 1.0.

    Returns:
        ndarray: Nx3 array of Cartesian coordinates (x, y, z) on the sphere.
    """
    u, v = uv_coords[:, 0], uv_coords[:, 1]

    # Convert UV to spherical coordinates
    theta = u * 2 * np.pi - np.pi  # Map u from [0, 1] to [-π, π]
    phi = v * np.pi - np.pi / 2    # Map v from [0, 1] to [-π/2, π/2]

    # Convert spherical to Cartesian coordinates with radius
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.sin(phi)
    z = radius * np.cos(phi) * np.sin(theta)

    return np.column_stack((x, y, z))


def GenerateGeometryFromVertices(vertices: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate CCW-oriented triangles for a convex hull of the given vertices.

    Args:
        vertices (NDArray): An array of vertices with shape (N, 3).

    Returns:
        tuple[NDArray, NDArray, NDArray]: 
            - Vertices used for the convex hull.
            - CCW-oriented triangle indices (simplices).
            - Indices of the vertices used in the convex hull.
    """
    hull = ConvexHull(vertices)
    indices = hull.vertices
    vertices = vertices[indices]  # Extract only vertices on the convex hull
    hull = ConvexHull(vertices)  # Recompute the hull on the reduced vertex set

    # Ensure triangles have CCW orientation
    simplices = hull.simplices
    for i, simplex in enumerate(simplices):
        # Compute two edges of the triangle
        edge1 = vertices[simplex[1]] - vertices[simplex[0]]
        edge2 = vertices[simplex[2]] - vertices[simplex[0]]
        # Compute the normal using the cross product
        normal = np.cross(edge1, edge2)
        # Check if the normal is inward-facing (dot product with centroid vector)
        centroid = vertices[simplex].mean(axis=0)
        outward = np.dot(normal, centroid) > 0
        if not outward:
            # Swap two vertices to flip the triangle
            simplices[i] = [simplex[0], simplex[2], simplex[1]]

    if o3d is None:
        raise ModuleNotFoundError(
            "open3d is required for GenerateGeometryFromVertices but is not installed"
        )

    # Create Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(simplices)

    # Compute normals
    mesh.compute_vertex_normals()

    return vertices, simplices, mesh.vertex_normals, indices


def ExportGeometryToObjFile(vertices, triangles, normals, uv_coords, colors, obj_filename):
    """
    Save a geometry to an .obj file with vertex normals, texture coordinates,
    and separate textures for RGB and OCV.

    Args:
        vertices (list): List of 3D vertices (Nx3 array).
        triangles (list): List of triangle indices (Mx3 array).
        color_tuples (list): List of 6D color tuples [(R, G, B, O, C, V)].
        obj_filename (str): Path to save the .obj file.
        rgb_texture_filename (str): Path to save the RGB texture image.
        ocv_texture_filename (str): Path to save the OCV texture image.
    """
    # Generate UV coordinates (for simplicity, map vertex indices to a linear grid)
    # Write .obj file
    with open(obj_filename, "w") as obj_file:
        # Write vertices
        for v, color in zip(vertices, colors):
            # color = [int(c.RGB[i] * 255) for i in range(3)]
            obj_file.write(
                f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f} {color[0]} {color[1]} {color[2]}\n")

        # Write texture coordinates
        for uv in uv_coords:
            obj_file.write(f"vt {uv[0]} {uv[1]}\n")

        # Write normals
        for n in np.asarray(normals):
            obj_file.write(f"vn {n[0]: .3f} {n[1]:.3f} {n[2]:.3f}\n")

        # Write faces
        for t in triangles:
            # obj_file.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
            obj_file.write(f"f {t[0]+1}/{t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}/{t[2]+1}\n")
