from pathlib import Path

import cv2
import numpy as np

from epipolar import (
    ransac_fundamental_matrix,
    fundamental_to_essential,
    decompose_essential_matrix,
    select_pose_by_cheirality,
    reprojection_errors,
)
from sift_features import detect_sift_features


def match_descriptors(descriptors1, descriptors2):
    """Match image-1 descriptors to image 2 using Lowe's ratio test."""
    matches = []

    for query_index, descriptor in enumerate(descriptors1):
        differences = descriptors2 - descriptor
        distances = np.linalg.norm(differences, axis=1)
        nearest = np.argsort(distances)[:2]

        best_index = int(nearest[0])
        second_index = int(nearest[1])

        if distances[best_index] < 0.75 * distances[second_index]:
            matches.append(
                cv2.DMatch(
                    _queryIdx=int(query_index),
                    _trainIdx=best_index,
                    _distance=float(distances[best_index]),
                )
            )

    return matches


def print_error_statistics(label, errors1, errors2, mask):
    """Print pixel reprojection errors for the selected points."""
    print(f"\n{label}")
    print("Evaluated points:", np.count_nonzero(mask))

    if not np.any(mask):
        print("No valid points to evaluate.")
        return

    for camera_index, errors in enumerate((errors1, errors2), start=1):
        values = errors[mask]
        print(f"Camera {camera_index}:")
        print(f"  Median: {np.median(values):.3f} px")
        print(
            f"  90th percentile: "
            f"{np.percentile(values, 90):.3f} px"
        )
        print(f"  Maximum: {np.max(values):.3f} px")


def triangulation_angles(points3d, camera_center2):
    """Compute angles between viewing rays in the camera-1 world frame."""
    rays1 = points3d
    rays2 = points3d - camera_center2

    lengths1 = np.linalg.norm(rays1, axis=1)
    lengths2 = np.linalg.norm(rays2, axis=1)

    valid = (
        np.all(np.isfinite(points3d), axis=1)
        & (lengths1 > 1e-10)
        & (lengths2 > 1e-10)
    )

    angles = np.full(len(points3d), np.nan)

    unit_rays1 = rays1[valid] / lengths1[valid, None]
    unit_rays2 = rays2[valid] / lengths2[valid, None]

    cosines = np.sum(unit_rays1 * unit_rays2, axis=1)
    angles[valid] = np.degrees(
        np.arccos(np.clip(cosines, -1.0, 1.0))
    )

    return angles, valid


def save_xyz_ply(path, points3d):
    """Write an ASCII PLY containing XYZ coordinates."""
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points3d)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        np.savetxt(file, points3d, fmt="%.8f")


def main():
    project_root = Path(__file__).resolve().parent.parent
    image_directory = project_root / "data" / "images"
    output_directory = project_root / "outputs"
    output_directory.mkdir(exist_ok=True)

    image1, keypoints1, descriptors1 = detect_sift_features(
        image_directory / "image1.jpg"
    )
    image2, keypoints2, descriptors2 = detect_sift_features(
        image_directory / "image2.jpg"
    )

    print("Image 1 shape:", image1.shape)
    print("Image 2 shape:", image2.shape)

    expected_shape = (5712, 4284)

    if (
        image1.shape[:2] != expected_shape
        or image2.shape[:2] != expected_shape
    ):
        raise ValueError(
            "The current approximate K expects shape (5712, 4284)."
        )

    # Approximate EXIF-based intrinsics, not calibrated intrinsics.
    K = np.array([
        [3960.6, 0.0, 2142.0],
        [0.0, 3960.6, 2856.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    if descriptors1 is None or len(descriptors1) == 0:
        raise ValueError("No descriptors found in image 1.")

    if descriptors2 is None or len(descriptors2) < 2:
        raise ValueError("Image 2 needs at least two descriptors.")

    accepted_matches = match_descriptors(descriptors1, descriptors2)

    if len(accepted_matches) < 8:
        raise RuntimeError("At least eight candidate matches are required.")

    points1 = np.array([
        keypoints1[match.queryIdx].pt
        for match in accepted_matches
    ], dtype=float)

    points2 = np.array([
        keypoints2[match.trainIdx].pt
        for match in accepted_matches
    ], dtype=float)

    P1 = K @ np.column_stack((np.eye(3), np.zeros(3)))

    # ------------------------------------------------------------
    # Custom F-based pipeline: retain it as an evaluation baseline.
    # ------------------------------------------------------------
    F, f_inlier_mask = ransac_fundamental_matrix(points1, points2)
    f_inlier_mask = np.asarray(f_inlier_mask, dtype=bool).reshape(-1)

    f_points1 = points1[f_inlier_mask]
    f_points2 = points2[f_inlier_mask]

    E_initial = K.T @ F @ K
    initial_values = np.linalg.svd(E_initial, compute_uv=False)

    print("\n--- Custom F-based estimation ---")
    print("Total queries:", len(descriptors1))
    print("Candidate matches:", len(accepted_matches))
    print("F-RANSAC inliers:", np.count_nonzero(f_inlier_mask))
    print("Estimated F:\n", F)
    print("Initial E singular values:", initial_values)
    print(
        "Second / first singular-value ratio:",
        initial_values[1] / initial_values[0],
    )

    E_custom = fundamental_to_essential(F, K, K)
    custom_candidates = decompose_essential_matrix(E_custom)

    (
        R_custom,
        t_custom,
        custom_points3d,
        custom_front,
        custom_counts,
    ) = select_pose_by_cheirality(
        f_points1,
        f_points2,
        K,
        K,
        custom_candidates,
    )

    P2_custom = K @ np.column_stack((R_custom, t_custom))
    custom_errors1 = reprojection_errors(
        custom_points3d, f_points1, P1
    )
    custom_errors2 = reprojection_errors(
        custom_points3d, f_points2, P2_custom
    )

    custom_evaluation = (
        custom_front
        & np.isfinite(custom_errors1)
        & np.isfinite(custom_errors2)
    )

    print("Positive-depth counts:", custom_counts)
    print("Custom camera 2 center:", -R_custom.T @ t_custom)

    print_error_statistics(
        "Custom pose reprojection errors",
        custom_errors1,
        custom_errors2,
        custom_evaluation,
    )

    # ------------------------------------------------------------
    # Final map: directly estimate E, then use our pose recovery.
    # ------------------------------------------------------------
    cv2.setRNGSeed(42)

    E, essential_mask = cv2.findEssentialMat(
        points1,
        points2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5,
    )

    if E is None or essential_mask is None:
        raise RuntimeError("Essential-matrix estimation failed.")

    if E.shape != (3, 3):
        raise RuntimeError(
            f"Expected one essential matrix, received {E.shape}."
        )

    essential_inliers = essential_mask.ravel() != 0

    reconstruction_pixels1 = points1[essential_inliers]
    reconstruction_pixels2 = points2[essential_inliers]

    if len(reconstruction_pixels1) < 8:
        raise RuntimeError("Too few essential-matrix inliers.")

    candidates = decompose_essential_matrix(E)

    R, t, points3d, front_mask, counts = select_pose_by_cheirality(
        reconstruction_pixels1,
        reconstruction_pixels2,
        K,
        K,
        candidates,
    )

    # Verify the returned rotation and translation direction.
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-8)

    camera_center2 = -R.T @ t
    P2 = K @ np.column_stack((R, t))

    errors1 = reprojection_errors(
        points3d, reconstruction_pixels1, P1
    )
    errors2 = reprojection_errors(
        points3d, reconstruction_pixels2, P2
    )

    angles_deg, angle_valid = triangulation_angles(
        points3d, camera_center2
    )

    finite_front = (
        front_mask
        & angle_valid
        & np.isfinite(errors1)
        & np.isfinite(errors2)
    )

    max_reprojection_px = 3.0
    min_angle_deg = 1.0

    reprojection_ok = (
        finite_front
        & (errors1 < max_reprojection_px)
        & (errors2 < max_reprojection_px)
    )

    keep = (
        reprojection_ok
        & (angles_deg >= min_angle_deg)
        & (angles_deg <= 180.0 - min_angle_deg)
    )

    print("\n--- Final reconstruction ---")
    print("Essential-matrix inliers:", len(reconstruction_pixels1))
    print("Positive-depth counts:", counts)
    print("Finite positive-depth points:", np.count_nonzero(finite_front))
    print("After reprojection filtering:", np.count_nonzero(reprojection_ok))
    print("After angle filtering:", np.count_nonzero(keep))
    print("Camera 2 center:", camera_center2)

    if not np.any(keep):
        raise RuntimeError("No points survived reconstruction filtering.")

    print_error_statistics(
        "Retained point reprojection errors",
        errors1,
        errors2,
        keep,
    )

    print(
        "Retained triangulation angles: "
        f"minimum={np.min(angles_deg[keep]):.3f} degrees, "
        f"median={np.median(angles_deg[keep]):.3f} degrees"
    )

    cloud_points = points3d[keep]

    # Preserve exactly the same ordering through both filters:
    # accepted matches -> E inliers -> retained 3D points.
    essential_matches = [
        match
        for match, is_inlier in zip(accepted_matches, essential_inliers)
        if is_inlier
    ]

    query_indices = np.array(
        [match.queryIdx for match in essential_matches],
        dtype=np.int64,
    )[keep]

    train_indices = np.array(
        [match.trainIdx for match in essential_matches],
        dtype=np.int64,
    )[keep]

    map_descriptors = descriptors1[query_indices]

    if len(map_descriptors) != len(cloud_points):
        raise RuntimeError("Map descriptors and 3D points are misaligned.")

    reconstruction_path = output_directory / "reconstruction.npz"

    np.savez(
        reconstruction_path,
        points3d=cloud_points,
        pixels1=reconstruction_pixels1[keep],
        pixels2=reconstruction_pixels2[keep],
        errors1=errors1[keep],
        errors2=errors2[keep],
        angles_deg=angles_deg[keep],
        K=K,
        R=R,
        t=t,
        camera_center2=camera_center2,
        image_shape=np.array(expected_shape),
    )

    map_path = output_directory / "map_features.npz"

    np.savez(
        map_path,
        points3d=cloud_points,
        descriptors=map_descriptors,
        keypoint_indices1=query_indices,
        keypoint_indices2=train_indices,
        K=K,
        image_shape=np.array(expected_shape),
    )

    cloud_path = output_directory / "sparse_cloud.ply"
    save_xyz_ply(cloud_path, cloud_points)

    # Visualize a sample of the final retained correspondences.
    retained_indices = np.flatnonzero(keep)
    rng = np.random.default_rng(42)

    selected_indices = rng.choice(
        retained_indices,
        size=min(50, len(retained_indices)),
        replace=False,
    )

    selected_matches = [
        essential_matches[index] for index in selected_indices
    ]

    visualization = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    visualization_path = output_directory / "inlier_matches.jpg"

    if not cv2.imwrite(str(visualization_path), visualization):
        raise IOError(
            f"Could not save visualization: {visualization_path}"
        )

    print(f"\nSaved reconstruction: {reconstruction_path}")
    print(f"Saved map features: {map_path}")
    print(f"Saved point cloud: {cloud_path}")
    print(f"Saved retained matches: {visualization_path}")


if __name__ == "__main__":
    main()