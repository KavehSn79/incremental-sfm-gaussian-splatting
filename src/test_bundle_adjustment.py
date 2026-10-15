import cv2
import numpy as np
from scipy.optimize import least_squares

from demo_bundle_adjustment import (
    make_jacobian_sparsity,
    project_observations,
    residuals,
    unpack_parameters,
)


def pack_parameters(rotations, translations, points, fixed_axis):
    """Use the same parameter ordering as our real bundle adjustment."""
    free_axes = [axis for axis in range(3) if axis != fixed_axis]

    rvec2, _ = cv2.Rodrigues(rotations[1])
    rvec3, _ = cv2.Rodrigues(rotations[2])

    return np.concatenate((
        rvec2.ravel(),
        translations[1, free_axes],
        rvec3.ravel(),
        translations[2],
        points.ravel(),
    ))


def camera_centers(rotations, translations):
    """Convert world-to-camera extrinsics into world-space centers."""
    return np.array([
        -R.T @ t
        for R, t in zip(rotations, translations)
    ])


def rotation_error_degrees(estimated, reference):
    """Return the angle of the relative rotation."""
    relative = estimated @ reference.T
    cosine = (np.trace(relative) - 1.0) / 2.0

    return np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )


def make_problem(noise_sigma):
    """Create known geometry, observations, and a perturbed initialization."""
    rng = np.random.default_rng(42)

    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ])

    point_count = 30

    true_points = rng.uniform(
        low=[-1.5, -0.8, 4.0],
        high=[1.5, 0.8, 9.0],
        size=(point_count, 3),
    )

    R2, _ = cv2.Rodrigues(
        np.array([0.01, -0.03, 0.005], dtype=float)
    )
    R3, _ = cv2.Rodrigues(
        np.array([-0.01, 0.02, -0.005], dtype=float)
    )

    true_rotations = np.stack((np.eye(3), R2, R3))

    true_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.05, 0.1],
    ])

    true_translations = np.array([
        -R @ center
        for R, center in zip(true_rotations, true_centers)
    ])

    # Twenty points have three observations.
    # Five points have only cameras 1 and 2.
    # Five points have only cameras 2 and 3.
    camera_indices = []
    point_indices = []
    clean_pixels = []

    for point_index, point in enumerate(true_points):
        if point_index < 20:
            observing_cameras = (0, 1, 2)
        elif point_index < 25:
            observing_cameras = (0, 1)
        else:
            observing_cameras = (1, 2)

        for camera_index in observing_cameras:
            point_camera = (
                true_rotations[camera_index] @ point
                + true_translations[camera_index]
            )

            if point_camera[2] <= 0:
                raise RuntimeError("Synthetic point is behind a camera.")

            projected_h = K @ point_camera
            pixel = projected_h[:2] / projected_h[2]

            camera_indices.append(camera_index)
            point_indices.append(point_index)
            clean_pixels.append(pixel)

    camera_indices = np.array(camera_indices, dtype=np.int64)
    point_indices = np.array(point_indices, dtype=np.int64)
    clean_pixels = np.array(clean_pixels, dtype=float)

    observed_pixels = clean_pixels + rng.normal(
        0.0,
        noise_sigma,
        size=clean_pixels.shape,
    )

    # Perturb the initial geometry.
    initial_rotations = true_rotations.copy()

    for camera_index in (1, 2):
        perturbation, _ = cv2.Rodrigues(
            rng.normal(0.0, 0.005, size=3)
        )
        initial_rotations[camera_index] = (
            perturbation @ true_rotations[camera_index]
        )

    initial_translations = true_translations.copy()
    initial_translations[1:] += rng.normal(
        0.0, 0.02, size=(2, 3)
    )

    initial_points = true_points + rng.normal(
        0.0, 0.08, size=true_points.shape
    )

    fixed_axis = int(np.argmax(np.abs(true_translations[1])))

    # Keep the scale anchor equal to ground truth.
    # This allows direct comparison without similarity alignment.
    initial_translations[1, fixed_axis] = (
        true_translations[1, fixed_axis]
    )

    fixed_translation2 = initial_translations[1].copy()

    initial_parameters = pack_parameters(
        initial_rotations,
        initial_translations,
        initial_points,
        fixed_axis,
    )

    true_parameters = pack_parameters(
        true_rotations,
        true_translations,
        true_points,
        fixed_axis,
    )

    return {
        "K": K,
        "true_points": true_points,
        "true_rotations": true_rotations,
        "true_translations": true_translations,
        "camera_indices": camera_indices,
        "point_indices": point_indices,
        "clean_pixels": clean_pixels,
        "observed_pixels": observed_pixels,
        "fixed_axis": fixed_axis,
        "fixed_translation2": fixed_translation2,
        "initial_parameters": initial_parameters,
        "true_parameters": true_parameters,
    }


def evaluate_geometry(parameters, problem):
    """Measure errors relative to the known synthetic geometry."""
    rotations, translations, points = unpack_parameters(
        parameters,
        problem["fixed_translation2"],
        problem["fixed_axis"],
    )

    centers = camera_centers(rotations, translations)
    true_centers = camera_centers(
        problem["true_rotations"],
        problem["true_translations"],
    )

    point_errors = np.linalg.norm(
        points - problem["true_points"],
        axis=1,
    )

    center_errors = np.linalg.norm(
        centers - true_centers,
        axis=1,
    )

    rotation_errors = np.array([
        rotation_error_degrees(R, R_true)
        for R, R_true in zip(
            rotations, problem["true_rotations"]
        )
    ])

    return point_errors, center_errors, rotation_errors


def run_experiment(noise_sigma):
    problem = make_problem(noise_sigma)

    arguments = (
        problem["fixed_translation2"],
        problem["fixed_axis"],
        problem["K"],
        problem["camera_indices"],
        problem["point_indices"],
        problem["observed_pixels"],
    )

    # Verify projection and parameter packing independently of optimization.
    projected_true, depths_true = project_observations(
        problem["true_parameters"],
        *arguments[:-1],
    )

    assert np.all(depths_true > 0)
    assert np.allclose(
        projected_true,
        problem["clean_pixels"],
        rtol=0,
        atol=1e-9,
    )

    initial_residuals = residuals(
        problem["initial_parameters"], *arguments
    )

    initial_pixel_errors = np.linalg.norm(
        initial_residuals.reshape(-1, 2), axis=1
    )
    initial_rms = np.sqrt(np.mean(initial_pixel_errors**2))

    initial_point_errors, _, _ = evaluate_geometry(
        problem["initial_parameters"], problem
    )

    sparsity = make_jacobian_sparsity(
        problem["camera_indices"],
        problem["point_indices"],
        len(problem["true_points"]),
    )

    # Each solver starts from exactly the same initial parameters.
    configurations = [
        (
            "Sparse / default inner tolerances",
            {
                "jac_sparsity": sparsity,
                "tr_solver": "lsmr",
            },
        ),
        (
            "Sparse / tighter inner tolerances",
            {
                "jac_sparsity": sparsity,
                "tr_solver": "lsmr",
                "tr_options": {
                    "atol": 1e-10,
                    "btol": 1e-10,
                    "maxiter": 500,
                },
            },
        ),
        (
            "Dense reference",
            {
                "tr_solver": "exact",
            },
        ),
    ]

    print("\n" + "=" * 65)
    print(f"Pixel noise standard deviation: {noise_sigma:.2f} px")
    print("Points:", len(problem["true_points"]))
    print("Observations:", len(problem["observed_pixels"]))
    print(f"Initial RMS reprojection error: {initial_rms:.6f} px")
    print(
        "Initial mean 3D point error: "
        f"{np.mean(initial_point_errors):.6f} map units"
    )

    results = {}

    for label, options in configurations:
        result = least_squares(
            residuals,
            problem["initial_parameters"].copy(),
            args=arguments,
            method="trf",
            x_scale="jac",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
            verbose=0,
            **options,
        )

        final_residuals = residuals(result.x, *arguments)
        pixel_errors = np.linalg.norm(
            final_residuals.reshape(-1, 2), axis=1
        )
        rms = np.sqrt(np.mean(pixel_errors**2))

        point_errors, center_errors, rotation_errors = (
            evaluate_geometry(result.x, problem)
        )

        _, final_depths = project_observations(
            result.x, *arguments[:-1]
        )

        print(f"\n{label}")
        print("Converged:", result.success)
        print("Function evaluations:", result.nfev)
        print(f"Robust cost: {result.cost:.8f}")
        print(f"First-order optimality: {result.optimality:.6g}")
        print(f"RMS reprojection error: {rms:.6f} px")
        print(
            f"Mean 3D point error: "
            f"{np.mean(point_errors):.6f} map units"
        )
        print(
            f"Maximum camera-center error: "
            f"{np.max(center_errors):.6f} map units"
        )
        print(
            f"Maximum rotation error: "
            f"{np.max(rotation_errors):.6f} degrees"
        )
        print("All observed points have positive depth:", np.all(final_depths > 0))

        assert np.all(np.isfinite(result.x))
        assert np.all(final_depths > 0)
        assert rms < initial_rms

        results[label] = {
            "result": result,
            "rms": rms,
            "point_errors": point_errors,
            "center_errors": center_errors,
            "rotation_errors": rotation_errors,
        }

    # Use the dense solver as a small-problem numerical reference.
    reference = results["Dense reference"]

    if noise_sigma == 0.0:
        assert reference["rms"] < 1e-5, (
            "Exact observations should be recovered to numerical precision."
        )
        assert np.max(reference["point_errors"]) < 1e-4
        assert np.max(reference["center_errors"]) < 1e-4

        print("\nExact-data geometry recovery checks passed.")
    else:
        # Check image fitting separately from true geometry recovery.
        assert reference["rms"] < 2.0 * noise_sigma

        initial_3d_error = np.mean(initial_point_errors)
        final_3d_error = np.mean(reference["point_errors"])

        print("\nNoisy-data reprojection checks passed.")
        print(f"Initial mean ground-truth 3D error: {initial_3d_error:.6f}")
        print(f"Final mean ground-truth 3D error: {final_3d_error:.6f}")

        if final_3d_error > initial_3d_error:
            print(
                "Geometry accuracy worsened despite improved pixel fitting. "
                "This experiment does not pass a geometry-accuracy check."
            )

    precise_cost = results[
        "Sparse / tighter inner tolerances"
    ]["result"].cost
    dense_cost = reference["result"].cost

    print(
        "Tighter-sparse versus dense cost difference:",
        abs(precise_cost - dense_cost),
    )


def main():
    run_experiment(noise_sigma=0.0)
    run_experiment(noise_sigma=0.5)


if __name__ == "__main__":
    main()