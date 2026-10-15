import numpy as np

from camera import world_to_camera_points, project_points
from epipolar import (
    estimate_fundamental_matrix,
    fundamental_to_essential,
    decompose_essential_matrix,
    select_pose_by_cheirality,
    reprojection_errors,
)


def main():
    K = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ])

    rng = np.random.default_rng(42)
    points_world = rng.uniform(
        low=[-2.0, -1.0, 4.0],
        high=[2.0, 1.0, 10.0],
        size=(20, 3),
    )

    # Camera 1 is at the origin; camera 2 is one unit right.
    R_cw = np.eye(3)
    t_cw1 = np.zeros(3)
    t_cw2 = np.array([-1.0, 0.0, 0.0])

    points_camera1 = world_to_camera_points(
        points_world, R_cw, t_cw1
    )
    points_camera2 = world_to_camera_points(
        points_world, R_cw, t_cw2
    )

    pixels1, valid1 = project_points(points_camera1, K)
    pixels2, valid2 = project_points(points_camera2, K)

    valid = valid1 & valid2
    pixels1 = pixels1[valid]
    pixels2 = pixels2[valid]

    # Verify fundamental-matrix estimation.
    F = estimate_fundamental_matrix(pixels1, pixels2)

    points1_h = np.column_stack((pixels1, np.ones(len(pixels1))))
    points2_h = np.column_stack((pixels2, np.ones(len(pixels2))))

    lines2 = (F @ points1_h.T).T
    residuals = np.sum(points2_h * lines2, axis=1)
    singular_values = np.linalg.svd(F, compute_uv=False)

    F_reference = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    F_reference /= np.linalg.norm(F_reference)

    reference_error = min(
        np.linalg.norm(F - F_reference),
        np.linalg.norm(F + F_reference),
    )

    print("Number of correspondences:", len(pixels1))
    print("Estimated F:\n", F)
    print("F singular values:", singular_values)
    print("Maximum absolute residual:", np.max(np.abs(residuals)))
    print("Reference matrix error:", reference_error)

    assert np.allclose(pixels1[:, 1], pixels2[:, 1])
    assert singular_values[-1] < 1e-10
    assert np.max(np.abs(residuals)) < 1e-8
    assert reference_error < 1e-8

    # Verify pose recovery and triangulation.
    E = fundamental_to_essential(F, K, K)
    candidates = decompose_essential_matrix(E)

    R, t, points3d, positive_depth_mask, counts = (
        select_pose_by_cheirality(
            pixels1, pixels2, K, K, candidates
        )
    )

    camera_center2 = -R.T @ t

    print("\nPositive-depth counts:", counts)
    print("Selected R:\n", R)
    print("Selected t:", t)
    print("Camera 2 center:", camera_center2)

    assert np.allclose(R, np.eye(3), atol=1e-8)
    assert np.allclose(t, [-1.0, 0.0, 0.0], atol=1e-8)
    assert np.allclose(camera_center2, [1.0, 0.0, 0.0], atol=1e-8)
    assert np.all(positive_depth_mask)

    reconstruction_error = np.linalg.norm(
        points3d - points_world[valid], axis=1
    )

    print("Maximum 3D error:", np.max(reconstruction_error))
    assert np.max(reconstruction_error) < 1e-8

    # Verify reprojection errors against synthetic observations.
    P1 = K @ np.column_stack((np.eye(3), np.zeros(3)))
    P2 = K @ np.column_stack((R, t))

    errors1 = reprojection_errors(points3d, pixels1, P1)
    errors2 = reprojection_errors(points3d, pixels2, P2)

    for camera_index, errors in enumerate((errors1, errors2), start=1):
        print(f"\nCamera {camera_index} reprojection errors:")
        print(f"  Median: {np.median(errors):.3e} px")
        print(f"  Maximum: {np.max(errors):.3e} px")

        assert np.all(np.isfinite(errors))
        assert np.max(errors) < 1e-8

    print("\nAll synthetic checks passed.")


if __name__ == "__main__":
    main()