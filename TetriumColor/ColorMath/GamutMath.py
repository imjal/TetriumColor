"""
Goal: Sample Directions in Color Space.
3.5 -- probably want to precompute the bounds on each direction so quest doesn't try to keep testing useless saturations
"""
import numpy.typing as npt
from typing import List

from tqdm import tqdm
import numpy as np

from TetriumColor.ColorMath.SubSpaceIntersection import FindMaximalSaturation, FindMaximumIn1DimDirection
import TetriumColor.ColorMath.Geometry as Geometry
import TetriumColor.ColorMath.Conversion as Conversion
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor
from TetriumColor.Visualization.cubeMapViz import SetUp3DPlot


def ConvertVSHToHering(vsh: npt.NDArray) -> npt.NDArray:
    """
    Converts from HSV to the Max Basis. Returns an Nxdim array of points in the Max Basis
    Args:
        HSV (npt.ArrayLike, Nxdim): The HSV coordinates that we want to transform
    """
    if vsh.shape[1] == 4:
        return np.hstack([vsh[:, [0]], Geometry.ConvertSphericalToCartesian(vsh[:, 1:])])
    elif vsh.shape[1] == 3:
        return np.hstack([vsh[:, [0]], Geometry.ConvertPolarToCartesian(vsh[:, 1:])])
    else:
        raise NotImplementedError(
            "Not implemented for dimensions other than 3 or 4")


def ConvertHeringToVSH(hering: npt.NDArray) -> npt.NDArray:
    """
    Converts from the Hering Basis to HSV. Returns an Nxdim array of points in HSV #TODO: decide on ordering.
    Args:
        Hering (npt.ArrayLike, Nxdim): The Max Basis coordinates that we want to transform
    """
    if hering.shape[1] == 4:
        return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToSpherical(hering[:, 1:])])
    elif hering.shape[1] == 3:
        return np.hstack([hering[:, [0]], Geometry.ConvertCartesianToPolar(hering[:, 1:])])
    else:
        raise NotImplementedError("Not implemented for dimensions other than 3 or 4")


def ConvertVSHtoTetraColor(vsh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> List[TetraColor]:
    """
    Convert VSH to TetraColor
    Args:
        vsh (npt.NDArray): The VSH coordinates to convert
    """
    hering = ConvertVSHToHering(vsh)
    disp = (color_space_transform.hering_to_disp@hering.T).T
    six_d_color = Conversion.Map4DTo6D(disp, color_space_transform)
    return [TetraColor(six_d_color[i, :3], six_d_color[i, 3:]) for i in range(six_d_color.shape[0])]


def ConvertVSHToPlateColor(vsh: npt.NDArray, luminance: float, color_space_transform: ColorSpaceTransform) -> PlateColor:
    """
    Convert VSH to PlateColor
    Args:
        vsh (npt.NDArray): The VSH coordinates to convert
        luminance (float): The luminance value of the plane
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to use for the conversion
    """
    pair_colors = np.concatenate([vsh[np.newaxis, :], np.array([luminance, 0, 0, 0])[np.newaxis, :]])
    hering = ConvertVSHToHering(pair_colors)
    disp = (color_space_transform.hering_to_disp@hering.T).T
    six_d_color = Conversion.Map4DTo6D(disp, color_space_transform)
    return PlateColor(TetraColor(six_d_color[0][:3], six_d_color[0][3:]), TetraColor(six_d_color[1][:3], six_d_color[1][3:]))


def FindMaxSaturationForVSH(vsh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> tuple[float, float]:
    cartesian = (color_space_transform.hering_to_disp@ConvertVSHToHering(vsh).T).T
    max_sat_pt_in_display = np.array([FindMaximalSaturation(cartesian[0], np.eye(color_space_transform.dim))])

    # convert display points back to VSH, and set parameters
    invMat = np.linalg.inv(color_space_transform.hering_to_disp)
    max_sat_per_angle = ConvertHeringToVSH((invMat@max_sat_pt_in_display.T).T)[0]
    return tuple([max_sat_per_angle[0], max_sat_per_angle[1]])


def GenerateGamutLUT(all_vshh: npt.NDArray, color_space_transform: ColorSpaceTransform) -> dict:
    """
    Generate a Look-Up Table for the Gamut of the Given ColorSpaceTransform
    Args:
        all_vshh (npt.NDArray): The VSHH points to generate the LUT for
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to generate the LUT for
    """
    dim = color_space_transform.cone_to_disp.shape[0]
    all_cartesian_points = (color_space_transform.hering_to_disp@ConvertVSHToHering(all_vshh).T).T

    # get max sat points for each hue direction
    map_angle_to_sat = {}
    pts = []
    for pt in tqdm(all_cartesian_points):
        pts += [FindMaximalSaturation(pt, np.eye(dim))]
    max_sat_cartesian_per_angle = np.array(pts)

    # convert display points back to VSH, and set parameters
    invMat = np.linalg.inv(color_space_transform.hering_to_disp)
    max_sat_per_angle = ConvertHeringToVSH((invMat@max_sat_cartesian_per_angle.T).T)
    for angle, sat in zip(all_vshh[:, 2:], max_sat_per_angle):
        map_angle_to_sat[tuple(angle)] = tuple([sat[0], sat[1]])
    return map_angle_to_sat


def SolveForBoundary(L: float, max_L: float, lum_cusp: float, sat_cusp: float) -> float:
    """
    Solve for the boundary of the gamut
    Args:
        L (float): The Luminance Value to solve for
        max_L (float): The Maximum Luminance Value
        lum_cusp (float): The Luminance Value at the Cusp
        sat_cusp (float): The Saturation Value at the Cusp

    Returns:
        float: The Saturation Value that corresponds to the boundary point at L
    """
    # get the cusp point for the given angle -- either presolved or solved on the fly atm
    if L >= lum_cusp:
        slope = -(max_L - lum_cusp) / sat_cusp
        return (L - max_L) / (slope)
    else:
        slope = lum_cusp / sat_cusp
        return L / slope


def GetEquiluminantPlane(luminance: float, color_space_transform: ColorSpaceTransform, map_angle_sat: dict) -> dict:
    """Get the saturation plane for the given VSHH points

    Args:
        luminance (float): luminance value
        color_space_transform (ColorSpaceTransform): color space transform object
        map_angle_sat (dict): dictionary that solves for the boundary of the gamut

    Returns:
        map_angle_lum_sat(dict): Mapping from angle to constant luminance varying saturation
    """
    map_angle_lum_sat = {}
    max_L = (np.linalg.inv(color_space_transform.hering_to_disp) @
             np.ones(color_space_transform.cone_to_disp.shape[0]))[0]
    for angle, (lum_cusp, sat_cusp) in map_angle_sat.items():
        sat = SolveForBoundary(luminance, max_L, lum_cusp, sat_cusp)
        map_angle_lum_sat[angle] = (luminance, sat)
    return map_angle_lum_sat


def RemapGamutPoints(VSHH: npt.NDArray, color_space_transform: ColorSpaceTransform, map_angle_sat: dict) -> npt.NDArray:
    """Given a set of VSHH points, remap the saturation values to be within the gamut

    Args:
        VSHH (npt.NDArray): value, saturation, hue(dim-2)
        color_space_transform (ColorSpaceTransform): color space transform object
        map_angle_sat (dict): dictionary that solves for the boundary of the gamut

    Returns:
        npt.NDArray: Remapped VSHH
    """
    max_L = (np.linalg.inv(color_space_transform.hering_to_disp) @
             np.ones(color_space_transform.cone_to_disp.shape[0]))[0]
    for i in range(len(VSHH)):
        angle = tuple(VSHH[i, 2:])
        if angle not in map_angle_sat:
            lum_cusp, sat_cusp = FindMaxSaturationForVSH(np.array([[0, 1, *angle]]), color_space_transform)
        else:
            lum_cusp, sat_cusp = map_angle_sat[angle]
        sat = SolveForBoundary(VSHH[i][0], max_L, lum_cusp, sat_cusp)
        VSHH[i, 1] = min(sat, VSHH[i][1])
    return VSHH


def SampleHueManifold(luminance: float, saturation: float, dim: int, num_points: int) -> npt.NDArray:
    """
    Generate a sphere of hue values
    Args:
        luminance (float): The luminance value to generate the sphere at
        saturation (float): The saturation value to generate the sphere at
    Returns: 
        npt.NDArray: The sphere of values in vshh space
    """
    all_angles = Geometry.SampleAnglesEqually(num_points, dim-1)
    all_vshh = np.zeros((len(all_angles), dim))
    all_vshh[:, 0] = luminance
    all_vshh[:, 1] = saturation
    all_vshh[:, 2:] = all_angles
    return all_vshh


def __rotateToZAxis(vector: npt.NDArray) -> npt.NDArray:
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


def GetMetamericAxisInDispSpace(color_space_transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Get the Metameric Axis in Hering
    Args:
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to get the Metameric Axis for
    """
    metameric_axis = np.zeros(color_space_transform.cone_to_disp.shape[0])
    metameric_axis[color_space_transform.metameric_axis] = 1
    direction = np.dot(color_space_transform.cone_to_disp, metameric_axis)
    normalized_direction = direction / np.linalg.norm(direction)  # return normalized direction
    return normalized_direction


def GetMetamericAxisInVSH(color_space_transform: ColorSpaceTransform) -> npt.NDArray:
    """
    Get the Metameric Axis in VSH
    Args:
        color_space_transform (ColorSpaceTransform): The ColorSpaceTransform to get the Metameric Axis for
    """
    normalized_direction_in_hering: npt.NDArray = np.linalg.inv(
        color_space_transform.hering_to_disp)@GetMetamericAxisInDispSpace(color_space_transform)
    return ConvertHeringToVSH(normalized_direction_in_hering[np.newaxis, :])


def GetTransformChromToQDir(transform: ColorSpaceTransform):
    """
    Get the transformation matrix from chromaticity to the metameric direction.
    """
    shh = GetMetamericAxisInVSH(transform)[0, 1:]  # remove luminance
    return __rotateToZAxis(shh)


def GetGridPoints(dist_from_axis: float, cube_idx: int, grid_size: int, color_space_transform: ColorSpaceTransform):
    all_us = (np.arange(grid_size) + 0.5) / grid_size
    all_vs = (np.arange(grid_size) + 0.5) / grid_size
    cube_u, cube_v = np.meshgrid(all_us, all_vs)
    flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

    qDirMat = GetTransformChromToQDir(color_space_transform)
    invQDirMat = np.linalg.inv(qDirMat)

    xyz_in_cube = Geometry.ConvertCubeUVToXYZ(cube_idx, cube_u, cube_v, dist_from_axis).reshape(-1, 3)
    xyz_in_chrom = np.dot(invQDirMat, xyz_in_cube.T).T

    return xyz_in_chrom, flattened_u, flattened_v


def GetMaximalMetamerPointsOnGrid(luminance: float, saturation: float, cube_idx: int,
                                  grid_size: int, color_space_transform: ColorSpaceTransform) -> npt.NDArray:
    """ Get the metamer points for a given luminance and cube index
    Args:
        luminance (float): luminance value
        saturation (float): saturation value
        cube_idx (int): cube index
        grid_size (int): grid size
        color_space_transform (ColorSpaceTransform): color space transform object

    Returns:
        npt.NDArray: The metamer points
    """
    xyz_in_chrom, _, _ = GetGridPoints(saturation, cube_idx, grid_size, color_space_transform)
    lum_vector = luminance * np.ones(grid_size * grid_size, dtype=float)

    vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz_in_chrom))
    disp_points = (color_space_transform.hering_to_disp@vxyz.T).T

    # metamer_vec = np.zeros(color_space_transform.dim)
    # metamer_vec[color_space_transform.metameric_axis] = 1
    # metamer_in_disp = color_space_transform.cone_to_disp@metamer_vec
    metamer_dir_in_disp = GetMetamericAxisInDispSpace(color_space_transform)

    disp_to_cone = np.linalg.inv(color_space_transform.cone_to_disp)
    disp_to_hering = np.linalg.inv(color_space_transform.hering_to_disp)

    metamers_in_disp = np.zeros((vxyz.shape[0], 2, color_space_transform.dim))
    cone_responses = []
    hering_responses = []
    for i in range(metamers_in_disp.shape[0]):
        # points in contention in disp space, bounded by unit cube scaled by vectors, direction is the metameric axis
        metamers_in_disp[i] = FindMaximumIn1DimDirection(
            disp_points[i], metamer_dir_in_disp, np.eye(color_space_transform.dim))

        hering_responses += [(disp_to_hering@metamers_in_disp[i].T).T]
        cone_responses = (disp_to_cone@metamers_in_disp[i].T).T
        # print(np.round(np.abs(cone_responses[1] - cone_responses[0]), 4))
        # print(metamers_in_disp[i])
    np.printoptions(precision=5, suppress=True)
    hering_responses = np.array(hering_responses).reshape(-1, 4)[:, 1:]
    output = Conversion.Map4DTo6D(metamers_in_disp.reshape(-1, 4), color_space_transform)
    display = np.rint(output[:, :4] * 255 / color_space_transform.white_weights).astype(np.uint8) / 255

    cone_responses = (disp_to_cone@display.T).T.reshape(-1, 2, 4)
    without_discretization = (disp_to_cone@(output[:, :4] / color_space_transform.white_weights).T).T.reshape(-1, 2, 4)

    diff = np.abs(cone_responses[:, 0] - cone_responses[:, 1])
    diff_without_discretization = np.abs(without_discretization[:, 0] - without_discretization[:, 1])

    print(np.array_str(diff, suppress_small=True, precision=5))
    print(np.round(diff_without_discretization, 5))
    return output.reshape(grid_size, grid_size, 2, 6)