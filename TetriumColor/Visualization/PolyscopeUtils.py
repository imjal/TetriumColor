import numpy as np
import numpy.typing as npt
from typing import List

import tetrapolyscope as ps

from TetriumColor import ColorSampler, ColorSpace, ColorSpaceType, PolyscopeDisplayType

from .Geometry import GeometryPrimitives
from ..Observer import Observer, MaxBasisFactory
from .Animation import AnimationUtils
from scipy.spatial import ConvexHull
from itertools import combinations


def OpenVideo(filename: str):
    """Open a video file for writing.

    Args:
        filename (str): filename

    Returns:
        file descriptor: file descriptor for the video file
    """
    return ps.open_video_file(filename, fps=30)


def CloseVideo(fd) -> None:
    """Close the video file descriptor

    Args:
        fd (dunno): file descriptor
    """
    ps.close_video_file(fd)
    return


def RenderVideo(fd, total_frames: int, target_fps: int = 30) -> None:
    """
    Renders a video by updating animations frame by frame.

    :param fd: File descriptor or path to save the video.
    :param total_frames: Total number of frames to render.
    :param target_fps: Target frames per second for the video (default: 30).
    """
    delta_time: float = 1 / target_fps

    for i in range(total_frames):
        # Update animations via AnimationUtils
        AnimationUtils.UpdateObjects(delta_time)

        # Render the current frame to the video
        ps.write_video_frame(fd, transparent_bg=False)


def Render2DMesh(name: str, points: npt.NDArray, rgb: npt.NDArray) -> float:
    """Create a 2D mesh from a list of vertices (N x 2) and RGB colors (1 x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 2 array of vertices
        rgbs (npt.ArrayLike): 1 x 3 array of RGB colors
    """
    # Compute the convex hull of the points
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]

    # rerun to get the right triangle indices
    hull = ConvexHull(hull_vertices)
    hull_vertices = np.hstack((hull_vertices[hull.vertices], np.zeros((len(hull_vertices), 1))))
    hull_triangles = np.hstack((hull.simplices, np.ones((hull.simplices.shape[0], 1)) * len(hull_vertices)))
    # add center coordinate to make it a triangle fan
    hull_vertices = np.vstack((hull_vertices, [0, 0, 0]))
    # Register the convex hull mesh with Polyscope
    ps_hull_mesh = ps.register_surface_mesh(f"{name}", hull_vertices, hull_triangles, back_face_policy='identical')
    ps_hull_mesh.add_color_quantity(f"{name}_colors",
                                    np.tile(rgb, (len(hull_vertices), 1)), defined_on='vertices', enabled=True)
    return hull.volume


def Render3DMesh(name: str, points: npt.ArrayLike, rgbs: npt.ArrayLike, back_face_policy: str = 'identical') -> float:
    """Create a 3D mesh from a list of vertices (N x 3) and RGB colors (N x 3)

    Args:
        name (str): Name of the mesh
        points (npt.ArrayLike): N x 3 array of vertices
        rgbs (npt.ArrayLike): N x 3 array of RGB colors
        back_face_policy (str): Polyscope back face policy ('identical', 'cull', 'different'). Defaults to 'identical'.
    """
    mesh = GeometryPrimitives.Create3DMesh(points, rgbs)
    ps_mesh = ps.register_surface_mesh(name, np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                       back_face_policy=back_face_policy, material='wax', smooth_shade=True)
    ps_mesh.add_color_quantity(f"{name}_colors", np.asarray(mesh.vertex_colors), defined_on='vertices', enabled=True)
    hull = ConvexHull(points)
    return hull.volume


def Render3DCone(name: str, points: npt.NDArray, line_colors: npt.NDArray, mesh_color: npt.NDArray, mesh_alpha: float = 1, arrow_alpha: float = 1) -> None:
    """Create a 3D cone from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the cone asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color or N x 3 array of RGB colors
        mesh_alpha (float, optional): Transparency of the Mesh. Defaults to 1.
        arrow_alpha (float, optional): Transparency of the Arrows. Defaults to 1.
    """
    mesh_colors = np.tile(mesh_color, (len(points)+1, 1))
    if len(line_colors) == 3:
        line_colors = np.tile(line_colors, (len(points), 1))

    center_vertex = np.zeros(points.shape[1])
    vertices = np.concatenate([[center_vertex], points])
    triangles = [[0, i, i + 1] for i in range(1, len(vertices) - 1)]
    ps_mesh = ps.register_surface_mesh(f"{name}", np.asarray(vertices), np.asarray(
        triangles), transparency=mesh_alpha, material='wax', smooth_shade=True)
    ps_mesh.add_color_quantity(f"{name}_colors", mesh_colors, defined_on='vertices', enabled=True)

    arrow_mesh = []
    for i in range(len(points)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [[0, 0, 0], points[i]]), color=np.array([0, 0, 0]), radius=0.025/100)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    ps_arrows = ps.register_surface_mesh(f"{name}_arrows", np.asarray(
        arrow_mesh.vertices), np.asarray(arrow_mesh.triangles), transparency=arrow_alpha, smooth_shade=True)
    ps_arrows.add_color_quantity(f"{name}_arrows_colors", np.asarray(
        arrow_mesh.vertex_colors), defined_on='vertices', enabled=True)

    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points))])
    ps_net = ps.register_curve_network(f"{name}_curve", points, edges)
    ps_net.add_color_quantity(f"{name}_curve_colors", line_colors, defined_on='nodes', enabled=True)


def Render3DLine(name: str, points: npt.NDArray, color: npt.NDArray, radius=None) -> None:
    """Create a 3D line from a list of vertices (N x 3) and a color (3)

    Args:
        name (str): basename for the line asset
        points (npt.NDArray): N x 3 array of vertices
        color (npt.ArrayLike): 1 x 3 Array of RGB color
        line_alpha (float, optional): Transparency of the Line. Defaults to 1.
    """
    if len(color) == 3:
        color = np.tile(color, (len(points), 1))
    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points)-1)])
    ps_net = ps.register_curve_network(f"{name}", points, edges, radius=radius)
    ps_net.add_color_quantity(f"{name}_colors", color, defined_on='nodes', enabled=True)


def RenderSimplexElements(name: str, dim: int, simplex_coords: npt.NDArray, simplex_colors: npt.NDArray, mesh_color=np.array([0.25, 0, 1]) * 0.5, isColored=False):
    """Create a Simplex from the Elements

    Args:
        name (str): Name of the simplex
        dim (int): Dimension of the simplex
        simplex_coords (npt.NDArray): N x 3 array of vertices
        simplex_colors (npt.NDArray): N x 3 array of RGB colors
        isColored (bool, optional): Whether the simplex is colored. Defaults to False.
    """
    names = []
    # GENERATE POINTS
    if dim < 4:
        simplex_coords = np.hstack((simplex_coords, np.zeros((simplex_coords.shape[0], 1))))
    RenderPointCloud(f"{name}_simplex_points", simplex_coords, simplex_colors, radius=0.1)
    names.append((f"{name}_simplex_points", "point_cloud"))
    # GENERATE EDGES
    edges = list(combinations(range(len(simplex_coords)), 2))

    for i, edge in enumerate(edges):
        if isColored:
            color = np.sum(simplex_colors[list(edge)], axis=0)
        else:
            color = np.zeros(3)
        Render3DLine(f'{name}_simplex_edge_{i}', simplex_coords[list(edge)], color=color)
        names.append((f"{name}_simplex_edge_{i}", "curve_network"))

    # GENERATE FACES
    faces = list(combinations(range(len(simplex_coords)), 3))
    if dim == 3:
        if isColored:
            color = np.sum(simplex_colors[list(faces[0])], axis=0)
        else:
            color = mesh_color
        RenderTriangle(f"{name}_simplex_face", simplex_coords[list(faces[0])], color)
        names.append((f"{name}_simplex_face", "surface_mesh"))
        ps.get_surface_mesh(f'{name}_simplex_face').set_transparency(0.4)

    elif dim == 4:
        for i, face in enumerate(faces):
            if isColored:
                color = np.sum(simplex_colors[list(faces[0])], axis=0)
            else:
                color = mesh_color
            color = np.sum(simplex_colors[list(face)], axis=0)
            RenderTriangle(f"{name}_simplex_face_{i}", simplex_coords[list(face)], color)
            ps.get_surface_mesh(f"{name}_simplex_face_{i}").set_transparency(0.4)
            names.append((f"{name}_simplex_face_{i}", "surface_mesh"))
    return names


def RenderSphere(name: str, radius):
    mesh = GeometryPrimitives.CreateSphere(radius)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def RenderSimplexGamut(name: str, dim: int, points: npt.NDArray, colors: npt.NDArray, mesh_color: npt.NDArray):
    """Generate a Simplex Gamut
    Args:
        name (str): Name of the simplex
        dim (int): Dimension of the observer
        points (npt.NDArray): N x 3 array of vertices
        colors (npt.NDArray): N x 3 array of RGB colors
        mesh_color (npt.NDArray): 1 x 3 array of RGB color
    """
    if dim < 4:
        simplex_coords = np.hstack((points, np.zeros((points.shape[0], 1))))

    def render_edges():
        edges = list(combinations(range(len(points)), 2))
        for i, edge in enumerate(edges):
            Render3DLine(f'{name}_edge_{i}', points[list(edge)], color=np.zeros(3))

    if dim == 3:
        RenderPointCloud(name, points, colors, radius=0.1)
        render_edges()
        RenderTriangle(f'{name}_face', points, mesh_color)
    else:
        RenderPointCloud(name, points, colors, radius=0.1)
        render_edges()
        Render3DMesh(f'{name}_mesh', points, np.tile(mesh_color, (len(points), 1)))
        ps.get_surface_mesh(f'{name}_mesh').set_transparency(0.4)


def RenderTriangle(name: str, points: npt.NDArray, color: npt.NDArray) -> None:
    """Render a 3 point element as a triangle surface mesh

    Args:
        name (str): Name of the triangle
        points (npt.NDArray): 3 x 3 array of vertices
        color (npt.NDArray): 1 x 3 array of RGB color
    """
    if points.shape[0] != 3 or points.shape[1] != 3:
        raise ValueError("Points array must be of shape (3, 3)")

    # Create the triangle mesh
    triangles = np.array([[0, 1, 2]])
    colors = np.tile(color, (3, 1))

    # Register the triangle mesh with Polyscope
    ps_triangle_mesh = ps.register_surface_mesh(name, points, triangles)
    ps_triangle_mesh.add_color_quantity(f"{name}_colors", colors, defined_on='vertices', enabled=True)


def RenderMetamericDirection(name: str, observer: Observer, display_basis: PolyscopeDisplayType,
                             metameric_axis: int, color: npt.NDArray, radius: float = 1, scale: float = 1) -> None:
    """Render a line showing the metameric direction through the origin.

    Args:
        name (str): Name for the line
        observer (Observer): Observer object
        display_basis (PolyscopeDisplayType): Display basis to render in
        metameric_axis (int): Which cone axis is metameric (0=L, 1=M, 2=S, 3=Q)
        color (npt.NDArray): RGB color for the line
        radius (float, optional): Line radius. Defaults to 1.
        scale (float, optional): Length scale. Defaults to 1.
    """
    length = 1 * 0.05
    basisLMSQ = np.zeros((1, observer.dimension))
    basisLMSQ[:, metameric_axis] = 1
    basisLMSQ = basisLMSQ * length
    basisLMSQ = ColorSpace(observer).convert_to_polyscope(basisLMSQ, ColorSpaceType.CONE, display_basis)
    normalizedLMSQ = basisLMSQ[0] / np.linalg.norm(basisLMSQ[0]) * scale
    Render3DLine(name, np.array([-normalizedLMSQ, normalizedLMSQ]), color, radius)


def RenderMetamericPairWithDirection(name: str, cst: ColorSpace, display_basis: PolyscopeDisplayType,
                                     point_disp: npt.NDArray, metameric_axis: int,
                                     color: npt.NDArray, show_cone_direction: bool = True,
                                     proportion: float = 0.8) -> None:
    """Render a metameric pair and optionally show the pure cone direction for comparison.

    Args:
        name (str): Base name for rendered objects
        cst (ColorSpace): ColorSpace with display primaries
        display_basis (PolyscopeDisplayType): Display basis to render in
        point_disp (npt.NDArray): Center point in DISP space
        metameric_axis (int): Which cone axis to vary (0=L, 1=M, 2=S, 3=Q)
        color (npt.NDArray): RGB color for the pair
        show_cone_direction (bool, optional): Whether to show pure cone direction. Defaults to True.
        proportion (float, optional): Proportion of maximum distance. Defaults to 0.8.
    """
    # Get metameric pair
    result = cst.get_maximal_pair_in_disp_from_pt(
        pt=point_disp,
        metameric_axis=metameric_axis,
        output_space=ColorSpaceType.CONE,
        proportion=proportion
    )

    if result is None:
        print(f"Warning: Could not find metameric pair for {name}")
        return

    cone1, cone2, metamer_diff = result

    # Convert to display space
    disp1 = cst.convert_to_polyscope(cone1.reshape(1, -1), ColorSpaceType.CONE, display_basis)[0]
    disp2 = cst.convert_to_polyscope(cone2.reshape(1, -1), ColorSpaceType.CONE, display_basis)[0]

    # Render the metameric pair
    RenderPointCloud(f"{name}_point1", disp1.reshape(1, -1), color.reshape(1, -1), radius=0.025)
    RenderPointCloud(f"{name}_point2", disp2.reshape(1, -1), (color * 0.7).reshape(1, -1), radius=0.025)
    Render3DLine(f"{name}_line", np.array([disp1, disp2]), color, radius=0.005)

    # Optionally show the pure cone direction at the midpoint
    if show_cone_direction:
        midpoint_disp = (disp1 + disp2) / 2

        # Create a small vector in the pure cone direction
        cone_dir = np.zeros(cst.dim)
        cone_dir[metameric_axis] = 0.1  # Small step
        cone_dir_display = cst.convert_to_polyscope(
            np.array([np.zeros(cst.dim), cone_dir]),
            ColorSpaceType.CONE,
            display_basis
        )
        direction = cone_dir_display[1] - cone_dir_display[0]
        direction = direction / np.linalg.norm(direction) * np.linalg.norm(disp2 - disp1) * 0.5

        # Render the pure cone direction line
        Render3DLine(
            f"{name}_cone_dir",
            np.array([midpoint_disp - direction, midpoint_disp + direction]),
            color * 0.5,  # Darker color to distinguish
            radius=0.003
        )


def RenderOBS(name: str, cst: ColorSpace, display_basis: PolyscopeDisplayType, num_samples=10000) -> None:
    """Render Object Color Solid in Specified Basis

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        display_basis (PolyscopeDisplayType): Basis to render the object in
    """
    if cst.observer.dimension == 4:
        csampler = ColorSampler(cst, cubemap_size=128)
        boundary_points = csampler.sample_full_colors(num_samples)
        # boundary_points = cst.convert(boundary_points, ColorSpaceType.HERING, ColorSpaceType.CONE)
        sRGBs = np.clip(cst.convert(boundary_points, ColorSpaceType.HERING, ColorSpaceType.SRGB), 0, 1)
        Render3DMesh(f"{name}", boundary_points[:, 1:], sRGBs)
        # points = cst.convert(boundary_points, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
        # Render3DMesh(f"{name}", points, sRGBs)
        return
    else:
        boundary_points, sRGBs = cst.observer.get_optimal_colors()

    new_points = cst.convert_to_polyscope(boundary_points, ColorSpaceType.CONE, display_basis)
    if cst.observer.dimension == 2:
        Render2DMesh(f"{name}", new_points, np.zeros((1, 3)))
        # Order the boundary points using the convex hull
        hull = ConvexHull(new_points)
        ordered_points = new_points[hull.vertices]
        ordered_colors = sRGBs[hull.vertices]
        # Append the beginning of ordered_points and ordered_colors to the back to form a "circle"
        ordered_points = np.vstack((ordered_points, ordered_points[0]))
        ordered_colors = np.vstack((ordered_colors, ordered_colors[0]))
        # Draw the line with the ordered points
        Render3DLine(f"{name}_ordered_line", ordered_points, ordered_colors)
        # Render3DLine(f"{name}_line", boundary_points, sRGBs)
    else:
        Render3DMesh(f"{name}", new_points, sRGBs)


def RenderMaxBasis(name: str, cst: ColorSpace, display_basis: PolyscopeDisplayType) -> None:
    """Render Max Basis Objects of Points and Lines - A Luminance Projected Parallelotope.

    Args:
        name (str): name of object to register with polyscope
        observer (Observer): Observer object to render
        display_basis (PolyscopeDisplayType, optional): Display Basis to Render in. Defaults to PolyscopeDisplayType.MaxBasis.
    """
    points, rgbs, lines = cst.get_maxbasis_parallelepiped(display_basis)
    mesh = GeometryPrimitives.CreateMaxBasis(points, rgbs, lines)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def RenderDisplayGamut(name: str, basis_vectors: npt.NDArray, T: npt.NDArray = np.eye(3)) -> None:
    """Render Display Gamut in Polyscope

    Args:
        name (str): Name of the gamut to be registered with polyscope
        basis_vectors (npt.NDArray): basis vectors of the paralleletope gamut
    """

    gamut_edges = GeometryPrimitives.CreateParallelotopeEdges(basis_vectors, color=[1, 1, 1], T=T)
    gamut = GeometryPrimitives.CreateParallelotopeMesh(basis_vectors, color=[1, 1, 1], T=T)

    two_mesh = GeometryPrimitives.CollapseMeshObjects([gamut_edges, gamut])
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, two_mesh)


def RenderPointCloud(name: str, points: npt.NDArray, rgb: npt.NDArray | None = None, radius: float = 0.01, mode: str = 'sphere') -> None:
    """Render a point cloud in Polyscope

    Args:
        name (str): Name of the point cloud
        points (npt.Array): N x 3 array of vertices
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
        radius (float, optional): Radius of the points. Defaults to 0.01.
    """
    pcl = ps.register_point_cloud(name, points, radius=0.01, point_render_mode=mode)
    if rgb is not None:
        pcl.add_color_quantity(f"{name}_colors", rgb, enabled=True)


def RenderSetOfArrows(name: str, endpoints: List[tuple], rgb: npt.NDArray | None = None, radius: float = 0.025/20) -> None:
    """Render a set of basis vectors as arrows in Polyscope

    Args:
        name (str): Name of the basis
        basis (npt.NDArray): N x 3 array of basis vectors
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    if rgb is None:
        rgb = np.zeros((len(endpoints), 3))

    arrow_mesh = []
    for i in range(len(endpoints)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [endpoints[i][0], endpoints[i][1]]), color=rgb[i], radius=radius)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)


def RenderBasisArrows(name: str, basis: npt.NDArray, rgb: npt.NDArray | None = None, radius: float = 0.025/20) -> None:
    """Render a set of basis vectors as arrows in Polyscope

    Args:
        name (str): Name of the basis
        basis (npt.NDArray): N x 3 array of basis vectors
        rgb (npt.NDArray | None, optional): N x 3 array of RGB colors. Defaults to None.
    """
    if rgb is None:
        rgb = np.zeros((len(basis), 3))

    arrow_mesh = []
    for i in range(len(basis)):
        arrow_mesh += [GeometryPrimitives.CreateArrow(endpoints=np.array(
            [[0, 0, 0], basis[i]]), color=rgb[i], radius=radius)]
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(arrow_mesh)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)


def RenderGridOfArrows(name: str):
    """Render a grid of arrows in Polyscope

    Args:
        name (str): Name to be registered with polyscope
    """
    grid_size = 10
    arrow_length = 1.0
    grid_range = np.linspace(-arrow_length/2, arrow_length/2, grid_size)
    arrow_mesh = []
    objs = GeometryPrimitives()
    for x in grid_range:
        for y in grid_range:
            objs.add_obj(GeometryPrimitives.CreateArrow(np.array([[x, y, -arrow_length/2], [x, y, arrow_length/2]])))
    arrow_mesh = GeometryPrimitives.CollapseMeshObjects(objs.objects)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, arrow_mesh)


def RenderMeshFromNonConvexPointCloud(name: str, points: npt.NDArray, rgb: npt.NDArray | None = None) -> None:
    """Render a 3D mesh from a point cloud in Polyscope
    Args:
        name (str): Name of the mesh
        points (npt.NDArray): N x 3 array of vertices
    """
    if rgb is None:
        rgb = np.ones((len(points), 3)) / 2
    mesh = GeometryPrimitives.Create3DMeshfromNonConvexPoints(points, rgb)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, mesh)


def RenderNoiseBall(name: str, center: npt.NDArray, noise_std: npt.NDArray,
                    rotation: npt.NDArray | None = None,  # NEW: columns are principal axes
                    color: npt.NDArray | None = None, num_samples: int = 1000,
                    alpha: float = 0.3, min_std: float = 1e-4) -> None:
    """Render a noise ball (ellipsoid) representing observer uncertainty.

    Args:
        name: Name of the noise ball to register with polyscope
        center: Center point of the noise ball (3D)
        noise_std: Semi-axis lengths (eigenvalues of covariance, sqrt'd)
        rotation: 3x3 rotation matrix whose columns are principal directions.
                  If None, assumes axis-aligned.
        ...
    """
    if color is None:
        color = np.array([0.7, 0.7, 0.7])

    # Generate points on a unit sphere
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    theta = np.random.uniform(0, np.pi, num_samples)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere_points = np.column_stack([x, y, z])  # (N, 3)

    # Ensure minimum std to prevent degeneracy
    noise_std_safe = np.maximum(np.abs(noise_std[:3]), min_std)

    # Scale by semi-axes
    ellipsoid_points = sphere_points * noise_std_safe  # (N, 3)

    # Rotate if non-axis-aligned
    if rotation is not None:
        ellipsoid_points = ellipsoid_points @ rotation.T  # (N, 3) @ (3, 3).T

    # Translate to center
    points = ellipsoid_points + center[:3]

    # Create mesh and render
    colors = np.tile(color, (len(points), 1))
    print(points.shape, colors.shape)
    try:
        Render3DMesh(name, points, colors)
        ps.get_surface_mesh(name).set_transparency(alpha)
    except RuntimeError as e:
        if "coplanar" in str(e).lower() or "flat" in str(e).lower():
            # If points are still coplanar, render as point cloud instead
            print(f"Warning: Noise ball {name} is degenerate, rendering as point cloud")
            RenderPointCloud(name, points, colors, radius=0.005)
        else:
            raise


def RenderRYGBGamut(name: str, cst: ColorSpace, display_basis: PolyscopeDisplayType,
                    color: npt.NDArray | None = None, alpha: float = 0.4,
                    vertex_radius: float = 0.015, edge_radius: float = 0.003,
                    scale: float = 1.0, show_hull: bool = True) -> None:
    """Render the RYGB gamut as a projected hypercube (single unified mesh).

    The RYGB gamut is defined by the Red-Yellow-Green-Blue basis with cutpoints
    at [493, 563, 608] nm. This forms a 4D hypercube that is projected into 3D.

    Args:
        name (str): Name of the gamut to register with polyscope
        cst (ColorSpace): ColorSpace object containing the observer
        display_basis (PolyscopeDisplayType): Basis to display in
        color (npt.NDArray | None, optional): RGB color for edges. Defaults to white.
        alpha (float, optional): Transparency of the mesh. Defaults to 0.4.
        vertex_radius (float, optional): Radius of vertex spheres. Defaults to 0.015.
        edge_radius (float, optional): Radius of edge cylinders. Defaults to 0.003.
        scale (float, optional): Scale factor for the entire gamut. Defaults to 1.0.
    """
    if color is None:
        color = np.array([1, 1, 1])

    # Generate all vertices of the unit hypercube in RYGB space
    vertices_rygb = np.array([[r, y, g, b] for r in [0, 1]
                              for y in [0, 1]
                              for g in [0, 1]
                              for b in [0, 1]])

    # Convert to target display basis
    vertices_display = cst.convert_to_polyscope(vertices_rygb, ColorSpaceType.RYGB, display_basis)

    # Apply scaling
    vertices_display = vertices_display * scale

    # Compute colors for vertices (convert to sRGB)
    vertices_cone = cst.convert(vertices_rygb, ColorSpaceType.RYGB, ColorSpaceType.CONE)
    vertex_colors = np.clip(cst.convert(vertices_cone, ColorSpaceType.CONE, ColorSpaceType.SRGB), 0, 1)

    # Create mesh objects list
    mesh_objects = []

    # Scale the radii proportionally
    scaled_vertex_radius = vertex_radius * scale
    scaled_edge_radius = edge_radius * scale

    # Create sphere meshes for vertices
    for i, (vertex, vertex_color) in enumerate(zip(vertices_display, vertex_colors)):
        sphere = GeometryPrimitives.CreateSphere(
            radius=scaled_vertex_radius,
            center=vertex,
            color=vertex_color,
            resolution=10
        )
        mesh_objects.append(sphere)

    # Find edges of the hypercube (vertices that differ in exactly one coordinate)
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            diff = np.sum(vertices_rygb[i] != vertices_rygb[j])
            if diff == 1:
                edges.append((i, j))

    # Create cylinder meshes for edges
    for i, j in edges:
        edge_color = (vertex_colors[i] + vertex_colors[j]) / 2
        cylinder = GeometryPrimitives.CreateCylinder(
            endpoints=[vertices_display[i], vertices_display[j]],
            radius=scaled_edge_radius,
            color=edge_color,
            resolution=8
        )
        mesh_objects.append(cylinder)

    combined_mesh = GeometryPrimitives.CollapseMeshObjects(mesh_objects)
    GeometryPrimitives.ConvertTriangleMeshToPolyscope(name, combined_mesh)

    # Create convex hull mesh (optional)
    if show_hull:
        try:
            hull_mesh = GeometryPrimitives.Create3DMesh(vertices_display, vertex_colors)
            GeometryPrimitives.ConvertTriangleMeshToPolyscope(name + "_hull", hull_mesh)
            ps.get_surface_mesh(name + "_hull").set_transparency(alpha)
        except Exception as e:
            print(f"Warning: Could not create hull mesh for {name}: {e}")


def RenderGamutSlices(name: str, cst: ColorSpace, display_space: ColorSpaceType, display_basis: PolyscopeDisplayType,
                      luminance_values: List[float] = [0.25, 0.5, 0.75],
                      grid_resolution: int = 20,
                      tolerance: float = 0.05,
                      alpha: float = 0.3) -> None:
    """Render slices of the 4D display gamut at different Hering luminance levels.

    This helps visualize the 4D gamut structure by showing how the chromatic gamut
    changes at different luminance values.

    Args:
        name (str): Base name for the slices
        cst (ColorSpace): ColorSpace with display primaries
        display_basis (PolyscopeDisplayType): Display basis to render in
        luminance_values (List[float], optional): Hering luminance values to slice at. Defaults to [0.25, 0.5, 0.75].
        grid_resolution (int, optional): Resolution of sampling grid in DISP space. Defaults to 20.
        tolerance (float, optional): Tolerance for luminance matching. Defaults to 0.05.
        alpha (float, optional): Transparency of slice meshes. Defaults to 0.3.
    """
    from TetriumColor.Observer import GetHeringMatrix
    from itertools import product

    # Sample the DISP space [0,1]^4 on a grid
    grid_1d = np.linspace(0, 1, grid_resolution)
    disp_points = np.array(list(product(grid_1d, grid_1d, grid_1d, grid_1d)))

    # Convert to CONE space
    cone_points = cst.convert(disp_points, display_space, ColorSpaceType.CONE)

    # Get Hering transform and compute luminance for each point
    H = GetHeringMatrix(cst.dim)
    hering_points = cone_points @ H.T
    luminances = hering_points[:, 0]  # First coordinate is luminance

    slice_colors = [
        np.array([1.0, 0.3, 0.3]),  # Red
        np.array([0.3, 1.0, 0.3]),  # Green
        np.array([0.3, 0.3, 1.0]),  # Blue
        np.array([1.0, 1.0, 0.3]),  # Yellow
        np.array([1.0, 0.3, 1.0]),  # Magenta
    ]

    # For each luminance level, extract points and render
    for slice_idx, target_lum in enumerate(luminance_values):
        # Find points close to this luminance level
        mask = np.abs(luminances - target_lum) < tolerance
        slice_points_cone = cone_points[mask]

        if len(slice_points_cone) < 10:
            print(f"Warning: Only {len(slice_points_cone)} points found at luminance {target_lum:.2f}")
            continue

        # Convert to visualization space
        slice_points_viz = cst.convert_to_polyscope(slice_points_cone, ColorSpaceType.CONE, display_basis)

        # Render the slice
        color = slice_colors[slice_idx % len(slice_colors)]

        # Try to create a convex hull mesh
        try:
            colors = np.tile(color, (len(slice_points_viz), 1))
            Render3DMesh(f"{name}_slice_L{target_lum:.2f}", slice_points_viz, colors, back_face_policy='cull')

            # Use varying transparency to reduce z-fighting between slices
            slice_alpha = alpha + (slice_idx * 0.05)  # Each slice slightly more/less transparent
            ps.get_surface_mesh(f"{name}_slice_L{target_lum:.2f}").set_transparency(slice_alpha)

            # Set material to reduce artifacts
            ps.get_surface_mesh(f"{name}_slice_L{target_lum:.2f}").set_material('wax')

            print(f"Slice at L={target_lum:.2f}: {len(slice_points_viz)} points, alpha={slice_alpha:.2f}")
        except Exception as e:
            # If mesh fails, render as point cloud
            print(f"Slice at L={target_lum:.2f}: mesh failed ({e}), rendering {len(slice_points_viz)} points")
            RenderPointCloud(f"{name}_slice_L{target_lum:.2f}", slice_points_viz,
                             np.tile(color, (len(slice_points_viz), 1)), radius=0.008)


def RenderLMSQVectors(name: str, cst: ColorSpace, display_basis: PolyscopeDisplayType,
                      scale: float = 0.3, arrow_radius: float = 0.005) -> None:
    """Render the LMSQ cone basis vectors as arrows.

    Args:
        name (str): Base name for the vectors to register with polyscope
        cst (ColorSpace): ColorSpace object containing the observer
        display_basis (PolyscopeDisplayType): Basis to display in
        scale (float, optional): Scale factor for vector length. Defaults to 0.3.
        arrow_radius (float, optional): Radius of arrow shafts. Defaults to 0.005.
    """
    dim = cst.observer.dimension

    # Define LMSQ basis vectors in cone space
    cone_basis = np.eye(dim)

    # Standard colors for LMSQ: Red, Green, Blue, Violet
    colors = [
        np.array([0, 0, 1]),    # S - Red
        np.array([0, 1, 0]),    # M - Green
        np.array([0.5, 0.5, 0]),  # Q - Violet (if 4D)
        np.array([1, 0, 0])    # L - Blue
    ]

    # Names for each cone type
    cone_names = ['L', 'M', 'S', 'Q']

    endpoints = []
    endpoint_colors = []

    for i in range(dim):
        # Scale the basis vector
        scaled_vector = cone_basis[i] * scale

        # Convert to display basis
        vector_display = cst.convert_to_polyscope(
            np.array([np.zeros(dim), scaled_vector]),
            ColorSpaceType.CONE,
            display_basis
        )

        # Normalize the DIRECTION (not the endpoint itself)
        # vector_display[0] is origin, vector_display[1] is endpoint
        direction = vector_display[1] - vector_display[0]
        normalized_direction = direction / np.linalg.norm(direction)

        # Scale to desired length
        vector_display[1] = vector_display[0] + normalized_direction * scale

        endpoints.append((vector_display[0], vector_display[1]))
        endpoint_colors.append(colors[i])

    # Render all arrows
    RenderSetOfArrows(name, endpoints, np.array(endpoint_colors), radius=arrow_radius)

    # Also render individual lines for labels
    for i in range(dim):
        Render3DLine(f"{name}_{cone_names[i]}",
                     np.array(endpoints[i]),
                     endpoint_colors[i],
                     radius=arrow_radius)
