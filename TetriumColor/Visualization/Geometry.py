import numpy as np
from typing import List
import numpy.typing as npt

import open3d as o3d
import glm
from itertools import combinations
import tetrapolyscope as ps
from scipy.spatial import ConvexHull


def GetCylinderTransform(endpoints: List | tuple | npt.NDArray) -> glm.mat4:
    a = endpoints[1]-endpoints[0]
    a = glm.vec3(a[0], a[1], a[2])
    b = glm.vec3(0, 0, 1)

    mat = glm.mat4()
    # translate
    mat = glm.translate(mat, glm.vec3(
        endpoints[0][0], endpoints[0][1], endpoints[0][2]))
    # rotate
    v = glm.cross(b, a)
    if v != glm.vec3(0, 0, 0):
        angle = glm.acos(glm.dot(b, a) / (glm.length(b) * glm.length(a)))
        mat = glm.rotate(mat, angle, v)

    # scale
    scale_factor = glm.length(a)
    mat = glm.scale(mat, glm.vec3(scale_factor, scale_factor, scale_factor))

    return mat


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


class GeometryPrimitives:

    def __init__(self) -> None:
        """Initializes the GeometryPrimitives class.
        """
        self.objects = []

    def add_obj(self, obj: o3d.geometry.TriangleMesh) -> None:
        """Add an object to an instance of the class.

        Args:
            obj (o3d.geometry.TriangleMesh): triangle mesh object to be added.
        """
        self.objects.append(obj)

    @staticmethod
    def CollapseMeshObjects(objects: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        """Collapse all objects in list into a single mesh.

        Args:
            objects (List[o3d.geometry.TriangleMesh]): List of objects

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh object
        """
        mesh = o3d.geometry.TriangleMesh()
        for obj in objects:
            mesh += obj
        return mesh

    @staticmethod
    def CreateSphere(radius: float = 0.025, center: List | tuple | npt.NDArray = [0, 0, 0],
                     resolution: float = 20, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        """Create a sphere mesh.

        Args:
            radius (float, optional): radius of sphere. Defaults to 0.025.
            center (List | tuple | npt.NDArray, optional): position of center of sphere. Defaults to [0, 0, 0].
            resolution (float, optional): resolution of the mesh. Defaults to 20.
            color (List | tuple | npt.NDArray, optional): color of mesh. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of sphere
        """
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, resolution=resolution)
        mesh.translate(center)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def CreateCylinder(endpoints: List | tuple | npt.NDArray, radius: float = 0.025/2,
                       resolution: float = 20, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        """Cylinder Mesh

        Args:
            endpoints (List | tuple | npt.NDArray): endpoints of the cylinder
            radius (float, optional): radius of the cylinder. Defaults to 0.025/2.
            resolution (float, optional): resolution of mesh for cylinder. Defaults to 20.
            color (List | tuple | npt.NDArray, optional): color of cylinder. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of cylinder
        """
        # canonical cylinder is along z axis with height 1 and centered
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=1, resolution=resolution)
        mesh.translate([0, 0, 1/2])

        matrix = GetCylinderTransform(endpoints)
        mesh.transform(matrix)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def calculate_zy_rotation_for_arrow(vec: npt.NDArray) -> tuple:
        """Generate 3d rotation matrix for an z axis defined arrow given a vector

        Args:
            vec (npt.NDArray): vector defining the endpoints of transform

        Returns:
            tuple: rotation matrices for z and y axisR
        """
        gamma = np.arctan2(vec[1], vec[0])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        vec = Rz.T @ vec

        beta = np.arctan2(vec[0], vec[2])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        return Rz, Ry

    @staticmethod
    def _getArrow(endpoint: List | tuple | npt.NDArray, origin: List | tuple | npt.NDArray = np.array([0, 0, 0]),
                  ratio_cone_cylinder: float = 0.15, radius: float = 1/60, ratio_cone_bottom_to_cylinder: float = 2,
                  resolution: float = 20, scale: float = 1):
        """Get Arrow Mesh

        Args:
            endpoint (List | tuple | npt.NDArray): end point of arrow
            origin (List | tuple | npt.NDArray, optional): origin of the arrow. Defaults to np.array([0, 0, 0]).
            scale (float, optional): scale of arrow. Defaults to 1.
            ratio_cone_cylinder (float, optional): ratio of cone to cylinder. Defaults to 0.15.
            radius (float, optional): radius of arrow. Defaults to 60.
            ratio_cone_bottom_to_cylinder (float, optional): ratio of cone bottom to cylinder. Defaults to 2.

        Returns:
            _type_: Arrow Mesh in open3d format
        """
        assert (not np.all(endpoint == origin))
        vec = np.array(endpoint) - np.array(origin)
        size = np.sqrt(np.sum(vec**2))

        Rz, Ry = GeometryPrimitives.calculate_zy_rotation_for_arrow(vec)
        mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=radius * ratio_cone_bottom_to_cylinder * scale,
                                                      cone_height=size * ratio_cone_cylinder * scale,
                                                      cylinder_radius=radius * scale,
                                                      cylinder_height=size * (1 - ratio_cone_cylinder * scale),
                                                      resolution=resolution)
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        return (mesh)

    @staticmethod
    def CreateArrow(endpoints: npt.NDArray, radius: float = 0.025/2, resolution: float = 20, scale: float = 1, color: List | tuple | npt.NDArray = np.array([0, 0, 0])) -> o3d.geometry.TriangleMesh:
        """Get Arrow Mesh

        Args:
            endpoints (npt.NDArray): endpoints of the mesh
            radius (float, optional): radius of arrow. Defaults to 0.025/2.
            resolution (float, optional): resolution of the mesh. Defaults to 20.
            scale (float, optional): scale of the mesh. Defaults to 1.
            color (List | tuple | npt.NDArray, optional): color of the mesh. Defaults to np.array([0, 0, 0]).

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of arrow 
        """
        mesh = GeometryPrimitives._getArrow(
            endpoints[1], endpoints[0], scale=scale, radius=radius, resolution=resolution)
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([color]*len(mesh.vertices)))
        return mesh

    @staticmethod
    def CreateCoordinateBasis(basis: npt.NDArray, radius: float = 0.025/2, resolution: float = 20,
                              scale: float = 1, color: List | tuple | npt.NDArray = [0, 0, 0]) -> o3d.geometry.TriangleMesh:
        """Create a coordinate basis mesh

        Args:
            basis (npt.NDArray): basis vectors (can be n x 3)
            radius (float, optional): radius of mesh. Defaults to 0.025/2.
            resolution (float, optional): resolution of mesh. Defaults to 20.
            scale (float, optional): scale of mesh. Defaults to 1.
            color (List | tuple | npt.NDArray, optional): color of mesh. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of coordinate basis
        """
        meshes = []
        for i, b in enumerate(basis):
            mesh = GeometryPrimitives.CreateArrow(
                endpoints=np.array([[0, 0, 0], b]), radius=radius, resolution=resolution, color=color[i], scale=scale)
            meshes.append(mesh)
        mesh = GeometryPrimitives.CollapseMeshObjects(meshes)
        return mesh

    @staticmethod
    def CreateMaxBasis(points: npt.NDArray, rgbs: npt.NDArray | List[List],
                       lines: npt.NDArray | List[List | tuple], line_color: List[List] | npt.NDArray = [0, 0, 0],
                       ball_radius: float = 0.025) -> o3d.geometry.TriangleMesh:
        """Create a mesh of points and lines to represent the max basis

        Args:
            points (npt.NDArray): array of 3d points
            rgbs (npt.NDArray | List[List]): array of rgb values associated with points
            lines (npt.NDArray | List[List  |  tuple]): list of pairs of indices to draw lines with each other
            line_color (List[List] | npt.NDArray, optional): color of lines. Defaults to [0, 0, 0].
            ball_radius (float, optional): radisu of ball. Defaults to 0.025.

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of max basis
        """
        objs = []
        for rgb, point in zip(rgbs, points):
            objs += [GeometryPrimitives.CreateSphere(
                center=point, color=rgb, radius=ball_radius)]
        for line in lines:
            objs += [GeometryPrimitives.CreateCylinder(
                endpoints=[points[line[0]], points[line[1]]], color=line_color)]
        return GeometryPrimitives.CollapseMeshObjects(objs)

    @staticmethod
    def CreateParallelotopeEdges(basis: npt.NDArray, color: List | tuple | npt.NDArray = [0, 0, 0],
                                 line_color: List | npt.NDArray = [0, 0, 0],
                                 T: npt.NDArray = np.eye(3)) -> o3d.geometry.TriangleMesh:
        """Paralleletope Edges

        Args:
            basis (npt.NDArray): basis vectors for the paralleletope (equal to dimension)
            color (List | tuple | npt.NDArray, optional): color of points. Defaults to [0, 0, 0].
            line_color (List | npt.NDArray, optional): color of lines. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.TriangleMesh: returns the open3d triangle mesh of the paralleletope edges
        """
        dim = basis.shape[1]
        if dim < 3:
            raise ValueError("Basis must be at least 3D")
        elif dim > 4:
            raise ValueError("Basis must be at most 4D")
        alllines = []
        for i in range(dim):
            alllines += list(combinations(range(dim), i + 1))

        lines = []
        for i, x in enumerate(alllines):
            if len(x) <= 1:
                lines += [[0, x[0] + 1]]  # connected to black
            else:  # > 1
                madeupof = list(combinations(x, len(x)-1))
                lines += [[alllines.index(elem) + 1, i + 1]
                          for elem in madeupof]  # connected to each elem

        if dim == 3:
            # # Create points of the parallelepiped
            points = np.array([[0, 0, 0]] + [basis[i].tolist() for i in range(3)] +
                              [(basis[i] + basis[j]).tolist() for i in range(3) for j in range(i+1, 3)] +
                              [(basis[0] + basis[1] + basis[2]).tolist()])
        else:
            # Create points of the 4D paralleletope
            points = np.array([[0, 0, 0, 0]] + [basis[i].tolist() for i in range(4)] +
                              [(basis[i] + basis[j]).tolist() for i in range(4) for j in range(i+1, 4)] +
                              [(basis[i] + basis[j] + basis[k]).tolist() for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                              [(basis[0] + basis[1] + basis[2] + basis[3]).tolist()])
        rgbs = np.array([color]*len(points))

        points = points@T.T

        return GeometryPrimitives.CreateMaxBasis(points, rgbs, lines, line_color=line_color, ball_radius=0.025)

    @staticmethod
    def CreateParallelotopeMesh(basis: npt.NDArray, color: List | tuple | npt.NDArray = [0, 0, 0], T: npt.NDArray = np.eye(3)) -> o3d.geometry.TriangleMesh:
        """Create Mesh of Paralleletope

        Args:
            basis (npt.NDArray): basis vectors that define paralleletope (equal to dimension of space)
            color (List | tuple | npt.NDArray, optional): color of mesh. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.TriangleMesh: open3d triangle mesh of paralleletope
        """
        dim = basis.shape[1]
        if dim < 3:
            raise ValueError("Basis must be at least 3D")
        elif dim > 4:
            raise ValueError("Basis must be at most 4D")

        if dim == 3:
            # # Create points of the parallelepiped
            points = np.array([[0, 0, 0]] + [basis[i].tolist() for i in range(3)] +
                              [(basis[i] + basis[j]).tolist() for i in range(3) for j in range(i+1, 3)] +
                              [(basis[0] + basis[1] + basis[2]).tolist()])
        else:
            # Create points of the 4D paralleletope
            points = np.array([[0, 0, 0, 0]] + [basis[i].tolist() for i in range(4)] +
                              [(basis[i] + basis[j]).tolist() for i in range(4) for j in range(i+1, 4)] +
                              [(basis[i] + basis[j] + basis[k]).tolist() for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                              [(basis[0] + basis[1] + basis[2] + basis[3]).tolist()])
        rgbs = np.array([color]*len(points))
        points = points@T.T

        return GeometryPrimitives.Create3DMesh(points, rgbs)

    @staticmethod
    def Create3DMeshfromNonConvexPoints(points: npt.NDArray, rgbs: npt.NDArray) -> o3d.geometry.TriangleMesh:
        """Create a 3D Mesh from a point cloud that is not convex.

        Args:
            points (npt.NDArray): points in 3D space

        Returns:
            o3d.geometry.TriangleMesh: Point cloud mesh
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Estimate appropriate radius:
        dists = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)

        # Try several radii (start small)
        radii = [avg_dist * f for f in [0.5, 1.0, 1.5]]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([rgbs]*len(mesh.vertices)))
        return mesh

    @staticmethod
    def Create3DMesh(points, rgbs) -> o3d.geometry.TriangleMesh:
        """Create a 3D Mesh from a point cloud and associated rgbs.

        Args:
            points (_type_): points in 3D space
            rgbs (_type_): associated rgbs

        Returns:
            o3d.geometry.TriangleMesh: Point cloud mesh
        """
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        mesh, point_indices = pcl.compute_convex_hull()
        mesh.compute_vertex_normals()

        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs[point_indices])
        return mesh

    @staticmethod
    def ConvertTriangleMeshToPolyscope(name: str, mesh: o3d.geometry.TriangleMesh) -> None:
        """Convert open3d triangle mesh to polyscope mesh

        Args:
            name (str): name of the mesh to be registered with polyscope
            mesh (o3d.geometry.TriangleMesh): open3d geometry triangle mesh to be converted
        """
        ps_mesh = ps.register_surface_mesh(f"{name}", np.asarray(mesh.vertices), np.asarray(
            mesh.triangles), material='wax', smooth_shade=True)
        ps_mesh.add_color_quantity(f"{name}_colors", np.asarray(
            mesh.vertex_colors), defined_on='vertices', enabled=True)
