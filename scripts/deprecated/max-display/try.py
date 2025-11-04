from colour import luminance
import numpy as np
from scipy.optimize import minimize

from TetriumColor.Observer import MaxBasisFactory, Observer, GetPerceptualHering
import TetriumColor.Utils.BasisMath as BasisMath
from TetriumColor import ColorSpace, ColorSpaceType


import numpy as np


def yz_plane_equiangular_error(vectors):
    """
    Given a list of 3D vectors, compute how evenly spaced they are in the YZ-plane.
    Assumes vectors are originating from the same point.

    Parameters:
    - vectors: np.ndarray of shape (n, 3)

    Returns:
    - angular_error: float (lower is better, 0 is perfect)
    """
    # Project to YZ-plane
    yz_vectors = vectors[:, 1:3]

    # Normalize to unit vectors (in YZ-plane)
    norms = np.linalg.norm(yz_vectors, axis=1, keepdims=True)
    yz_unit_vectors = yz_vectors / np.clip(norms, 1e-8, None)

    # Compute angles in YZ-plane (atan2 gives angle from Y axis)
    angles = np.arctan2(yz_unit_vectors[:, 1], yz_unit_vectors[:, 0])  # angle from Y (y=0) counter-clockwise
    angles = np.mod(angles, 2 * np.pi)  # Wrap to [0, 2pi)

    # Sort angles
    angles = np.sort(angles)

    # Compute angle differences between adjacent vectors
    angle_diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))  # Close the circle

    # Ideal angle spacing
    n = len(vectors)
    ideal_angle = 2 * np.pi / n

    # Total squared deviation from ideal spacing
    error = np.sum((angle_diffs - ideal_angle) ** 2)

    return error


def solve_transformation_matrix(source_points, target_points):
    """
    Solve for a transformation matrix that maps source points to target points
    while ensuring (1,0,0) maps to (1,0,0) and preserving angular relationships.

    Parameters:
    - source_points: numpy array of shape (n, 3), each row is a source point
    - target_points: numpy array of shape (n, 3), each row is a target point

    Returns:
    - transformation_matrix: numpy array of shape (3, 3)
    - error: final optimization error
    """
    # Initial guess for the 6 free parameters (a12, a13, a22, a23, a32, a33)
    initial_guess = np.zeros(6)

    def objective_function(params):
        a21, a31, a22, a23, a32, a33 = params

        # Build matrix A where (1,0,0) stays fixed
        A = np.array([
            [1/2, 1/2, 0],
            [a21, a22, a23],
            [a31, a32, a33]
        ])

        transformed_points = np.array([A @ p for p in source_points])
        error = np.mean(np.sum((transformed_points - target_points) ** 2))
        luminance_error = np.mean(np.sum((target_points[:, 0] - transformed_points[:, 0])**2))

        # compute the perpendicular distance from the line (1, 0, 0)
        # Compute the perpendicular distance from the line (1, 0, 0)
        chromas = np.linalg.norm(transformed_points[:, 1:], axis=1)  # distance from (0, 0)
        actual_chromas = np.linalg.norm(target_points[:, 1:], axis=1)
        print("Actual chromas: ", actual_chromas)
        print("Derived chromas: ", chromas)
        chroma_error = np.mean((actual_chromas - chromas) ** 2)

        # Compute the y-z angle between vectors
        should_be_low = yz_plane_equiangular_error(target_points[1:])
        angular_error = yz_plane_equiangular_error(transformed_points[1:])
        print("Should be low: ", should_be_low)
        print("Actual error: ", angular_error)

        print(
            f"Current params: {params}, Error: {error:.4f}, Angular error: {angular_error:.4f},  Luminance error: {luminance_error:.4f}, Chroma error: {chroma_error:.4f}")
        combined_error = 100 * error + angular_error + luminance_error + chroma_error
        return combined_error * 100
    # # Solve the optimization problems
    # result = minimize(objective_function, initial_guess, method='BFGS', options={'maxiter': 100000})

    from scipy.optimize import differential_evolution

    bounds = [(-2, 2)] * 6  # Reasonable bounds for a12, a13, a22, ..., a33
    result = differential_evolution(objective_function, bounds, strategy='best1bin',
                                    maxiter=10000, polish=True, disp=True)

    # Extract the optimized parameters
    a21, a31, a22, a23, a32, a33 = result.x

    # Construct the final transformation matrix
    # Build matrix A where (1,0,0) stays fixed
    transformation_matrix = np.array([
        [1/2, 1/2, 0],
        [a21, a22, a23],
        [a31, a32, a33]
    ])

    print("Optimization result status:", result.message)
    print("Optimization error:", result.fun)

    # Validate the transformation
    validate_transformation(source_points, target_points, transformation_matrix)

    return transformation_matrix, result.fun


def validate_transformation(source_points, target_points, transformation_matrix):
    """
    Validate the transformation by comparing transformed source points with target points
    and evaluating angular preservation.
    """
    print("\nValidation Results:")

    # Transform source points
    transformed_points = np.array([transformation_matrix @ point for point in source_points])

    # Point-to-point error
    total_distance_error = 0
    print("Point mapping errors:")
    for i, (src, tgt, transformed) in enumerate(zip(source_points, target_points, transformed_points)):
        error = np.linalg.norm(transformed - tgt)
        total_distance_error += error
        print(f"  Point {i}: Source {src} → Target {tgt} → Transformed {transformed.round(4)} (Error: {error:.4f})")

    print(f"\nTotal distance error: {total_distance_error:.4f}")

    # Angular preservation metrics
    print("\nAngular preservation metrics:")

    # Get pairwise vectors
    source_vectors = []
    target_vectors = []
    transformed_vectors = []

    n = len(source_points)
    for i in range(n):
        for j in range(i+1, n):
            source_vectors.append(source_points[j] - source_points[i])
            target_vectors.append(target_points[j] - target_points[i])
            transformed_vectors.append(transformed_points[j] - transformed_points[i])

    # Calculate angles between corresponding vectors
    angle_errors = []
    for i, (src_vec, tgt_vec, trans_vec) in enumerate(zip(source_vectors, target_vectors, transformed_vectors)):
        # Calculate angles
        src_norm = np.linalg.norm(src_vec)
        tgt_norm = np.linalg.norm(tgt_vec)
        trans_norm = np.linalg.norm(trans_vec)

        if src_norm > 1e-6 and tgt_norm > 1e-6 and trans_norm > 1e-6:
            # Calculate the angle between transformed vector and target vector (in degrees)
            trans_tgt_dot = np.clip(np.dot(trans_vec/trans_norm, tgt_vec/tgt_norm), -1.0, 1.0)
            angle_error_degrees = np.arccos(trans_tgt_dot) * 180 / np.pi
            angle_errors.append(angle_error_degrees)
            print(f"  Vector pair {i}: Angle error: {angle_error_degrees:.2f}°")

    if angle_errors:
        print(f"\nMean angle error: {np.mean(angle_errors):.2f}°")
        print(f"Max angle error: {np.max(angle_errors):.2f}°")

# Example usage:


if __name__ == "__main__":
    # Example source and target points
    # These should be chosen to reflect the desired mapping
    # (1,0,0) maps to (1,0,0), and other points are chosen to test angular preservation

    white_point = np.array([1, 1, 1])  # in cone space
    observer = Observer.trichromat(wavelengths=np.arange(400, 701, 10))
    cst = ColorSpace(observer)  # luminance_per_channel=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    denom_of_nonlin = 2.43

    # 0.5 Get all of the basis vectors, and transform to this space
    max_basis = MaxBasisFactory.get_object(observer, denom=denom_of_nonlin)
    refs, _, _, _ = max_basis.GetDiscreteRepresentation()
    maxbasis_points = observer.observe_spectras(refs[1:4])

    # 2. apply non-linearity in cone space
    maxbasis_points = np.power(maxbasis_points, 1/denom_of_nonlin)
    lums = maxbasis_points[:, 0]  # luminance value
    lums = [lums[0], lums[1], lums[2]]
    chromas = np.ones(3) * np.sqrt(2/3) * np.array([1.0, 0.5, 0.5])  # chromas as fractions of a basis
    vshh = cst.convert(maxbasis_points, ColorSpaceType.HERING, ColorSpaceType.VSH)
    vshh[:, 0] = lums
    vshh[:, 1] = chromas
    angle_basis = BasisMath.construct_angle_basis(maxbasis_points.shape[1], np.array([1, 0, 0]), lums, chromas)

    white_pt = np.array([1, 1, 1])

    source_points = np.concatenate([[white_pt], maxbasis_points])
    target_points = np.concatenate([[[1, 0, 0]], angle_basis])

    transform_mat, _ = solve_transformation_matrix(source_points, target_points[:, [0, 2, 1]])

    print("Max basis points:", source_points)
    print("Target points:", target_points[:, [0, 2, 1]])
    print("Transform matrix:", transform_mat)
    print("Transformed points:", source_points@transform_mat.T)

    BasisMath.visualize_transformation(
        source_points, target_points, source_points@transform_mat.T)
