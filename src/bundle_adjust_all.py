import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from incremental_sfm import (
    build_associations,
    load_checkpoint,
    save_checkpoint,
    show_reconstruction,
)


def pack_parameters(rotations, translations, points, fixed_axis):
    """Pack camera parameters, followed by all XYZ point coordinates."""
    free_axes = [axis for axis in range(3) if axis != fixed_axis]
    blocks = []

    # Camera 2 has five free parameters.
    rvec2, _ = cv2.Rodrigues(rotations[1])
    blocks.extend((rvec2.ravel(), translations[1, free_axes]))

    # Every additional camera has six free parameters.
    for camera_index in range(2, len(rotations)):
        rvec, _ = cv2.Rodrigues(rotations[camera_index])
        blocks.extend((rvec.ravel(), translations[camera_index]))

    blocks.append(points.ravel())
    return np.concatenate(blocks)


def unpack_parameters(
    parameters,
    camera_count,
    fixed_translation2,
    fixed_axis,
):
    """Recover camera extrinsics and map points from the parameter vector."""
    free_axes = [axis for axis in range(3) if axis != fixed_axis]

    rotations = np.repeat(np.eye(3)[None, :, :], camera_count, axis=0)
    translations = np.zeros((camera_count, 3))

    rotations[1], _ = cv2.Rodrigues(parameters[:3])

    translations[1] = fixed_translation2
    translations[1, free_axes] = parameters[3:5]

    offset = 5

    for camera_index in range(2, camera_count):
        rotations[camera_index], _ = cv2.Rodrigues(
            parameters[offset:offset + 3]
        )
        translations[camera_index] = parameters[offset + 3:offset + 6]
        offset += 6

    points = parameters[offset:].reshape(-1, 3)
    return rotations, translations, points


def project_observations(
    parameters,
    camera_count,
    fixed_translation2,
    fixed_axis,
    K,
    camera_indices,
    point_indices,
):
    """Project each point using the camera that observed it."""
    rotations, translations, points = unpack_parameters(
        parameters,
        camera_count,
        fixed_translation2,
        fixed_axis,
    )

    camera_points = np.einsum(
        "nij,nj->ni",
        rotations[camera_indices],
        points[point_indices],
    ) + translations[camera_indices]

    projected_h = camera_points @ K.T
    denominators = projected_h[:, 2]

    # Protect temporary solver evaluations near zero depth.
    safe_denominators = np.where(
        np.abs(denominators) < 1e-8,
        np.where(denominators < 0, -1e-8, 1e-8),
        denominators,
    )

    pixels = projected_h[:, :2] / safe_denominators[:, None]
    return pixels, camera_points[:, 2]


def residuals(parameters, *arguments):
    observed_pixels = arguments[-1]
    projected_pixels, _ = project_observations(
        parameters, *arguments[:-1]
    )
    return (projected_pixels - observed_pixels).ravel()


def make_sparsity(camera_indices, point_indices, camera_count, point_count):
    """Each observation depends on one camera and one 3D point."""
    camera_parameter_count = 5 + 6 * (camera_count - 2)

    sparsity = lil_matrix(
        (
            2 * len(camera_indices),
            camera_parameter_count + 3 * point_count,
        ),
        dtype=np.int8,
    )

    for observation_index, (camera_index, point_index) in enumerate(
        zip(camera_indices, point_indices)
    ):
        rows = slice(2 * observation_index, 2 * observation_index + 2)

        if camera_index == 1:
            sparsity[rows, :5] = 1
        elif camera_index >= 2:
            start = 5 + 6 * (camera_index - 2)
            sparsity[rows, start:start + 6] = 1

        point_start = camera_parameter_count + 3 * point_index
        sparsity[rows, point_start:point_start + 3] = 1

    return sparsity.tocsr()


def print_statistics(label, errors, camera_indices, image_names):
    """Report errors before filtering, using the same observations."""
    print(f"\n{label}")

    for camera_index, name in enumerate(image_names):
        values = errors[camera_indices == camera_index]

        if len(values) == 0:
            print(f"{name}: no observations")
            continue

        print(
            f"{name}: observations={len(values)}, "
            f"median={np.median(values):.3f} px, "
            f"p90={np.percentile(values, 90):.3f} px, "
            f"max={np.max(values):.3f} px"
        )

    print(
        "Overall RMS pixel error:",
        f"{np.sqrt(np.mean(errors**2)):.3f} px",
    )


def track_angle_mask(points, rotations, translations, observations):
    """
    Require at least one well-separated pair of viewing rays per track.

    Near-parallel and near-opposite rays provide weak triangulation.
    """
    centers = np.array([
        -R.T @ t for R, t in zip(rotations, translations)
    ])

    observing_cameras = [[] for _ in range(len(points))]

    for camera_index, _, point_index, _ in observations:
        observing_cameras[point_index].append(camera_index)

    keep = np.zeros(len(points), dtype=bool)

    for point_index, cameras in enumerate(observing_cameras):
        if len(cameras) < 2:
            continue

        rays = points[point_index] - centers[cameras]
        lengths = np.linalg.norm(rays, axis=1)

        if (
            not np.all(np.isfinite(rays))
            or np.any(lengths < 1e-10)
        ):
            continue

        unit_rays = rays / lengths[:, None]

        for first in range(len(cameras)):
            for second in range(first + 1, len(cameras)):
                cosine = np.dot(unit_rays[first], unit_rays[second])
                angle = np.degrees(
                    np.arccos(np.clip(cosine, -1.0, 1.0))
                )

                if 1.0 <= angle <= 179.0:
                    keep[point_index] = True
                    break

            if keep[point_index]:
                break

    return keep


def cameras_connected(observations, camera_count):
    """Check that shared tracks connect every camera to camera 1."""
    point_cameras = {}

    for camera_index, _, point_index, _ in observations:
        point_cameras.setdefault(point_index, set()).add(camera_index)

    neighbors = [set() for _ in range(camera_count)]

    for cameras in point_cameras.values():
        for camera_index in cameras:
            neighbors[camera_index].update(cameras - {camera_index})

    visited = {0}
    pending = [0]

    while pending:
        camera_index = pending.pop()

        for neighbor in neighbors[camera_index]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    return len(visited) == camera_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Display the saved optimized map without rerunning BA.",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    input_path = output_directory / "incremental_reconstruction.npz"
    output_path = output_directory / "global_bundle_adjusted.npz"

    if args.view_only:
        state = load_checkpoint(output_path)
        show_reconstruction(state, output_directory)
        return

    state = load_checkpoint(input_path)
    build_associations(state)

    rotations = np.asarray(state["rotations"], dtype=float)
    translations = np.asarray(state["translations"], dtype=float)
    points = np.asarray(state["points"], dtype=float)
    K = state["K"]

    camera_count = len(rotations)
    observations = state["observations"]

    if camera_count < 3:
        raise ValueError("This script expects at least three cameras.")

    if not cameras_connected(observations, camera_count):
        raise RuntimeError("The camera-track graph is disconnected.")

    camera_indices = np.array(
        [item[0] for item in observations], dtype=np.int64
    )
    point_indices = np.array(
        [item[2] for item in observations], dtype=np.int64
    )
    observed_pixels = np.array(
        [item[3] for item in observations], dtype=float
    )

    if not all(
        np.all(np.isfinite(array))
        for array in (rotations, translations, points, K, observed_pixels)
    ):
        raise ValueError("Reconstruction data must be finite.")

    if (
        not np.allclose(rotations[0], np.eye(3), atol=1e-8)
        or not np.allclose(translations[0], 0, atol=1e-8)
    ):
        raise ValueError("Camera 1 must define the world frame.")

    track_lengths = np.bincount(
        point_indices, minlength=len(points)
    )

    if np.any(track_lengths < 2):
        raise ValueError("Every point must have at least two observations.")

    fixed_translation2 = translations[1].copy()
    fixed_axis = int(np.argmax(np.abs(fixed_translation2)))

    if abs(fixed_translation2[fixed_axis]) < 1e-8:
        raise RuntimeError("Invalid camera-2 scale anchor.")

    initial_parameters = pack_parameters(
        rotations, translations, points, fixed_axis
    )

    arguments = (
        camera_count,
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

    print("Registered cameras:", camera_count)
    print("Points:", len(points))
    print("Observations:", len(observations))
    print("Optimization parameters:", len(initial_parameters))
    print(
        "Scale anchor: camera-2 translation",
        "XYZ"[fixed_axis],
        "=",
        fixed_translation2[fixed_axis],
    )

    for length in np.unique(track_lengths):
        print(
            f"Tracks with {length} observations:",
            np.count_nonzero(track_lengths == length),
        )

    print_statistics(
        "Before global bundle adjustment",
        initial_errors,
        camera_indices,
        state["image_names"],
    )

    sparsity = make_sparsity(
        camera_indices, point_indices, camera_count, len(points)
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
        verbose=2,
    )

    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Optimization returned non-finite parameters.")

    initial_cost = np.sum(
        np.sqrt(1.0 + initial_residuals**2) - 1.0
    )

    if (
        not np.isfinite(result.cost)
        or result.cost > initial_cost + 1e-8
    ):
        raise RuntimeError("The robust objective did not improve.")

    final_residuals = residuals(result.x, *arguments)
    final_errors = np.linalg.norm(
        final_residuals.reshape(-1, 2), axis=1
    )

    final_rotations, final_translations, final_points = unpack_parameters(
        result.x,
        camera_count,
        fixed_translation2,
        fixed_axis,
    )

    _, final_depths = project_observations(
        result.x, *arguments[:-1]
    )

    print("\nSolver converged:", result.success)
    print("Termination:", result.message)
    print(f"Initial robust cost: {initial_cost:.6f}")
    print(f"Final robust cost: {result.cost:.6f}")

    if not result.success:
        print("The saved result will be labeled as an intermediate estimate.")

    print_statistics(
        "After global BA — same observations, before filtering",
        final_errors,
        camera_indices,
        state["image_names"],
    )

    # Remove poor observations first, rather than rejecting a whole
    # multi-view track because one observation is inaccurate.
    observation_ok = (
        np.isfinite(final_errors)
        & (final_errors < 3.0)
        & (final_depths > 1e-8)
    )

    valid_observations = [
        observation
        for observation, valid in zip(observations, observation_ok)
        if valid
    ]

    valid_track_lengths = np.bincount(
        point_indices[observation_ok],
        minlength=len(points),
    )

    angle_ok = track_angle_mask(
        final_points,
        final_rotations,
        final_translations,
        valid_observations,
    )

    keep_points = (
        (valid_track_lengths >= 2)
        & angle_ok
        & np.all(np.isfinite(final_points), axis=1)
    )

    if not np.any(keep_points):
        raise RuntimeError("No points survived final filtering.")

    index_mapping = np.full(len(points), -1, dtype=np.int64)
    index_mapping[keep_points] = np.arange(np.count_nonzero(keep_points))

    retained_observations = [
        (
            camera_index,
            feature_index,
            int(index_mapping[point_index]),
            pixel.copy(),
        )
        for camera_index, feature_index, point_index, pixel
        in valid_observations
        if keep_points[point_index]
    ]

    if not cameras_connected(retained_observations, camera_count):
        raise RuntimeError(
            "Final filtering disconnected the camera-track graph. "
            "The original checkpoint has been preserved."
        )

    state["points"] = list(final_points[keep_points])
    state["colors"] = list(
        np.asarray(state["colors"])[keep_points]
    )
    state["rotations"] = list(final_rotations)
    state["translations"] = list(final_translations)
    state["observations"] = retained_observations

    build_associations(state)
    save_checkpoint(output_path, state)

    # Save optimization status and the unfiltered error comparison.
    diagnostics_path = output_directory / "global_ba_diagnostics.npz"

    np.savez(
        diagnostics_path,
        converged=result.success,
        termination_message=str(result.message),
        function_evaluations=result.nfev,
        initial_cost=initial_cost,
        final_cost=result.cost,
        optimality=result.optimality,
        before_errors=initial_errors,
        after_errors=final_errors,
        camera_indices=camera_indices,
    )

    print("\nRetained points:", len(state["points"]))
    print("Retained observations:", len(retained_observations))

    for camera_index, name in enumerate(state["image_names"]):
        center = (
            -final_rotations[camera_index].T
            @ final_translations[camera_index]
        )
        observation_count = sum(
            observation[0] == camera_index
            for observation in retained_observations
        )

        print(
            f"{name}: center={center}, "
            f"retained observations={observation_count}"
        )

    print(f"\nSaved refined map: {output_path}")
    print(f"Saved diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()