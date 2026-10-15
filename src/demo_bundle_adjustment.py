from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def unpack_parameters(parameters, fixed_translation2, fixed_axis):
    """
    Camera 1 is fixed.

    Camera 2:
        3 rotation parameters + 2 free translation components.

    Camera 3:
        3 rotation parameters + 3 translation components.

    Remaining parameters:
        XYZ positions of all map points.
    """
    free_axes = [axis for axis in range(3) if axis != fixed_axis]

    R2, _ = cv2.Rodrigues(parameters[:3])

    t2 = fixed_translation2.copy()
    t2[free_axes] = parameters[3:5]

    R3, _ = cv2.Rodrigues(parameters[5:8])
    t3 = parameters[8:11]

    points = parameters[11:].reshape(-1, 3)

    rotations = np.stack((np.eye(3), R2, R3))
    translations = np.stack((np.zeros(3), t2, t3))

    return rotations, translations, points


def project_observations(
    parameters,
    fixed_translation2,
    fixed_axis,
    K,
    camera_indices,
    point_indices,
):
    """Project every observed 3D point into its observing camera."""
    rotations, translations, points = unpack_parameters(
        parameters, fixed_translation2, fixed_axis
    )

    camera_points = np.einsum(
        "nij,nj->ni",
        rotations[camera_indices],
        points[point_indices],
    ) + translations[camera_indices]

    projected_h = camera_points @ K.T
    denominators = projected_h[:, 2]

    # Protect intermediate solver evaluations against division by zero.
    safe_denominators = np.where(
        np.abs(denominators) < 1e-8,
        np.where(denominators < 0, -1e-8, 1e-8),
        denominators,
    )

    projected_pixels = (
        projected_h[:, :2] / safe_denominators[:, None]
    )

    return projected_pixels, camera_points[:, 2]


def residuals(
    parameters,
    fixed_translation2,
    fixed_axis,
    K,
    camera_indices,
    point_indices,
    observed_pixels,
):
    """Return horizontal and vertical pixel errors for all observations."""
    projected_pixels, _ = project_observations(
        parameters,
        fixed_translation2,
        fixed_axis,
        K,
        camera_indices,
        point_indices,
    )

    return (projected_pixels - observed_pixels).ravel()


def make_jacobian_sparsity(camera_indices, point_indices, point_count):
    """Describe which parameters affect each residual."""
    observation_count = len(camera_indices)

    sparsity = lil_matrix(
        (2 * observation_count, 11 + 3 * point_count),
        dtype=np.int8,
    )

    for observation_index, (camera_index, point_index) in enumerate(
        zip(camera_indices, point_indices)
    ):
        rows = slice(2 * observation_index, 2 * observation_index + 2)

        if camera_index == 1:
            sparsity[rows, 0:5] = 1
        elif camera_index == 2:
            sparsity[rows, 5:11] = 1

        point_start = 11 + 3 * point_index
        sparsity[rows, point_start:point_start + 3] = 1

    return sparsity.tocsr()


def print_statistics(label, errors, camera_indices):
    """Report Euclidean pixel errors on the same observation set."""
    print(f"\n{label}")

    for camera_index in range(3):
        values = errors[camera_indices == camera_index]

        if len(values) == 0:
            continue

        print(
            f"Camera {camera_index + 1}: "
            f"median={np.median(values):.3f} px, "
            f"90th percentile={np.percentile(values, 90):.3f} px, "
            f"maximum={np.max(values):.3f} px"
        )

    print(f"Overall RMS pixel error: {np.sqrt(np.mean(errors**2)):.3f} px")


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    tracks_path = output_directory / "tracks.npz"

    with np.load(tracks_path, allow_pickle=False) as data:
        tracks = {key: data[key].copy() for key in data.files}

    points = tracks["points3d"]
    rotations = tracks["rotations"]
    translations = tracks["translations"]
    K = tracks["K"]

    camera_indices = tracks["camera_indices"]
    point_indices = tracks["point_indices"]
    observed_pixels = tracks["pixels"]

    # Fix the largest-magnitude camera-2 translation component.
    # This anchors scale while avoiding a near-zero component.
    fixed_translation2 = translations[1].copy()
    fixed_axis = int(np.argmax(np.abs(fixed_translation2)))
    free_axes = [axis for axis in range(3) if axis != fixed_axis]

    if abs(fixed_translation2[fixed_axis]) < 1e-8:
        raise RuntimeError("Camera 2 cannot provide a scale anchor.")

    rvec2, _ = cv2.Rodrigues(rotations[1])
    rvec3, _ = cv2.Rodrigues(rotations[2])

    initial_parameters = np.concatenate((
        rvec2.ravel(),
        translations[1, free_axes],
        rvec3.ravel(),
        translations[2],
        points.ravel(),
    ))

    arguments = (
        fixed_translation2,
        fixed_axis,
        K,
        camera_indices,
        point_indices,
        observed_pixels,
    )

    initial_residuals = residuals(initial_parameters, *arguments)
    initial_errors = np.linalg.norm(
        initial_residuals.reshape(-1, 2), axis=1
    )

    _, initial_depths = project_observations(
        initial_parameters, *arguments[:-1]
    )

    if not np.all(initial_depths > 1e-8):
        raise RuntimeError("Some initial observations have invalid depth.")

    print("Points:", len(points))
    print("Observations:", len(observed_pixels))
    print("Optimization parameters:", len(initial_parameters))
    print("Fixed camera: camera 1")
    print(
        "Scale anchor: camera-2 translation component",
        "XYZ"[fixed_axis],
        "=",
        fixed_translation2[fixed_axis],
    )

    print_statistics(
        "Before bundle adjustment",
        initial_errors,
        camera_indices,
    )

    sparsity = make_jacobian_sparsity(
        camera_indices, point_indices, len(points)
    )

    result = least_squares(
        residuals,
        initial_parameters,
        args=arguments,
        jac_sparsity=sparsity,
        method="trf",
        tr_solver="lsmr",
        tr_options={
            "atol": 1e-10,
            "btol": 1e-10,
            "maxiter": 500,
        },
        x_scale="jac",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=200,
        verbose=1,
    )

    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Optimization returned non-finite parameters.")

    final_residuals = residuals(result.x, *arguments)
    final_errors = np.linalg.norm(
        final_residuals.reshape(-1, 2), axis=1
    )

    # SciPy's soft_l1 cost with f_scale=1.
    initial_cost = np.sum(
        np.sqrt(1.0 + initial_residuals**2) - 1.0
    )

    if not np.isfinite(result.cost) or result.cost > initial_cost + 1e-8:
        raise RuntimeError("Optimization did not improve the objective.")

    print("\nSolver converged:", result.success)
    print("Termination:", result.message)
    print(f"Initial robust cost: {initial_cost:.6f}")
    print(f"Final robust cost: {result.cost:.6f}")

    if not result.success:
        print(
            "The solver stopped before convergence. "
            "The saved result is an intermediate estimate."
        )

    print_statistics(
        "After bundle adjustment — same observations",
        final_errors,
        camera_indices,
    )

    final_rotations, final_translations, final_points = (
        unpack_parameters(result.x, fixed_translation2, fixed_axis)
    )

    _, final_depths = project_observations(
        result.x, *arguments[:-1]
    )

    # Reject an entire track if any observation fails the final checks.
    observation_ok = (
        np.isfinite(final_errors)
        & (final_errors < 3.0)
        & (final_depths > 1e-8)
    )

    keep_points = np.ones(len(points), dtype=bool)
    keep_points[np.unique(point_indices[~observation_ok])] = False

    print("\nPoints retained after final checks:", np.count_nonzero(keep_points))
    print("Points rejected:", np.count_nonzero(~keep_points))

    if not np.any(keep_points):
        raise RuntimeError("No points survived the final checks.")

    keep_observations = keep_points[point_indices]

    # Compact point indices after removing rejected tracks.
    index_mapping = np.full(len(points), -1, dtype=np.int64)
    index_mapping[keep_points] = np.arange(np.count_nonzero(keep_points))

    original_count = int(tracks["original_point_count"])
    retained_original_count = np.count_nonzero(
        keep_points[:original_count]
    )

    for camera_index in range(3):
        center = (
            -final_rotations[camera_index].T
            @ final_translations[camera_index]
        )
        print(f"Camera {camera_index + 1} center:", center)

    reconstruction_path = (
        output_directory / "bundle_adjusted_reconstruction.npz"
    )

    np.savez(
        reconstruction_path,
        points3d=final_points[keep_points],
        colors=tracks["colors"][keep_points],
        original_point_count=retained_original_count,
        K=K,
        R2=final_rotations[1],
        t2=final_translations[1],
        R3=final_rotations[2],
        t3=final_translations[2],
        image_shape=tracks["image_shape"],
        solver_converged=result.success,
    )

    adjusted_tracks_path = output_directory / "bundle_adjusted_tracks.npz"

    np.savez(
        adjusted_tracks_path,
        points3d=final_points[keep_points],
        colors=tracks["colors"][keep_points],
        camera_indices=camera_indices[keep_observations],
        point_indices=index_mapping[point_indices[keep_observations]],
        keypoint_indices=tracks["keypoint_indices"][keep_observations],
        pixels=observed_pixels[keep_observations],
        reprojection_errors=final_errors[keep_observations],
        rotations=final_rotations,
        translations=final_translations,
        K=K,
        image_shape=tracks["image_shape"],
        source_point_indices=tracks["source_point_indices"][keep_points],
    )

    print(f"\nSaved reconstruction: {reconstruction_path}")
    print(f"Saved adjusted tracks: {adjusted_tracks_path}")


if __name__ == "__main__":
    main()