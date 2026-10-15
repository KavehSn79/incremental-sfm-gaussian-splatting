from pathlib import Path

import cv2
import numpy as np

from epipolar import reprojection_errors
from sift_features import detect_sift_features


def match_map_to_image(map_descriptors, image_descriptors):
    """
    Return (map_index, image_index, descriptor_distance) tuples.

    Apply Lowe's ratio test and prevent multiple map points
    from being assigned to the same image feature.
    """
    candidates = []

    for map_index, descriptor in enumerate(map_descriptors):
        differences = image_descriptors - descriptor
        distances = np.linalg.norm(differences, axis=1)
        nearest = np.argsort(distances)[:2]

        best_index = int(nearest[0])
        second_index = int(nearest[1])

        if distances[best_index] < 0.75 * distances[second_index]:
            candidates.append(
                (
                    map_index,
                    best_index,
                    float(distances[best_index]),
                )
            )

    candidates.sort(key=lambda match: match[2])

    matches = []
    used_image_indices = set()

    for match in candidates:
        image_index = match[1]

        if image_index not in used_image_indices:
            matches.append(match)
            used_image_indices.add(image_index)

    return matches


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    output_directory.mkdir(exist_ok=True)

    map_path = output_directory / "map_features.npz"
    image_path = project_root / "data" / "images" / "image3.jpg"

    if not map_path.is_file():
        raise FileNotFoundError(
            "Run demo_single_match.py first to create map_features.npz."
        )

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing third image: {image_path}")

    with np.load(map_path, allow_pickle=False) as data:
        map_points = data["points3d"]
        map_descriptors = data["descriptors"]
        K = data["K"]
        expected_shape = tuple(data["image_shape"].tolist())

    if map_descriptors.ndim != 2 or map_descriptors.shape[1] != 128:
        raise ValueError("Map descriptors must have shape (N, 128).")

    if map_points.shape != (len(map_descriptors), 3):
        raise ValueError("Map points and descriptors must align.")

    if len(map_points) < 6:
        raise ValueError("This demo needs at least six map points.")

    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3).")

    if not all(
        np.all(np.isfinite(array))
        for array in (map_points, map_descriptors, K)
    ):
        raise ValueError("Map data must be finite.")

    image3, keypoints3, descriptors3 = detect_sift_features(image_path)

    print("Image 3 shape:", image3.shape)
    print("Map points:", len(map_points))
    print("Image 3 keypoints:", len(keypoints3))

    # Same resolution is required here; lens and zoom must also match.
    if image3.shape[:2] != expected_shape:
        raise ValueError(
            f"The saved K expects image shape {expected_shape}."
        )

    if descriptors3 is None or len(descriptors3) < 2:
        raise ValueError("Image 3 needs at least two descriptors.")

    matches = match_map_to_image(map_descriptors, descriptors3)

    print("Candidate 2D-3D matches:", len(matches))

    # Six is our demo's practical minimum.
    if len(matches) < 6:
        raise RuntimeError(
            "Too few map matches. Use a third image with more overlap."
        )

    map_indices = np.array(
        [match[0] for match in matches],
        dtype=np.int64,
    )

    image_indices = np.array(
        [match[1] for match in matches],
        dtype=np.int64,
    )

    # Corresponding rows represent the same physical feature.
    object_points = np.ascontiguousarray(
        map_points[map_indices],
        dtype=np.float64,
    )

    image_points = np.ascontiguousarray(
        [keypoints3[index].pt for index in image_indices],
        dtype=np.float64,
    )

    # Continue with the same zero-distortion approximation.
    distortion = np.zeros((5, 1), dtype=np.float64)
    threshold_px = 3.0

    cv2.setRNGSeed(42)

    success, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=distortion,
        iterationsCount=2000,
        reprojectionError=threshold_px,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or inlier_indices is None:
        raise RuntimeError("PnP-RANSAC could not estimate camera 3.")

    inlier_indices = inlier_indices.ravel()

    print("PnP-RANSAC inliers:", len(inlier_indices))

    if len(inlier_indices) < 6:
        raise RuntimeError("Too few PnP inliers for this demo.")

    # Refine only the camera pose; the map points stay fixed.
    rvec, tvec = cv2.solvePnPRefineLM(
        np.ascontiguousarray(object_points[inlier_indices]),
        np.ascontiguousarray(image_points[inlier_indices]),
        K,
        distortion,
        rvec,
        tvec,
    )

    # Convert axis-angle rotation into a rotation matrix.
    R_cw3, _ = cv2.Rodrigues(rvec)
    t_cw3 = tvec.ravel()

    if not (
        np.all(np.isfinite(R_cw3))
        and np.all(np.isfinite(t_cw3))
    ):
        raise RuntimeError("PnP returned non-finite camera parameters.")

    # Translation is in map units. Do not normalize it.
    camera_center3 = -R_cw3.T @ t_cw3

    P3 = K @ np.column_stack((R_cw3, t_cw3))

    errors = reprojection_errors(
        object_points,
        image_points,
        P3,
    )

    points_camera3 = (
        R_cw3 @ object_points.T
    ).T + t_cw3

    ransac_mask = np.zeros(len(matches), dtype=bool)
    ransac_mask[inlier_indices] = True

    retained_mask = (
        ransac_mask
        & np.isfinite(errors)
        & (points_camera3[:, 2] > 1e-8)
        & (errors < threshold_px)
    )

    if np.count_nonzero(retained_mask) < 6:
        raise RuntimeError(
            "Too few correspondences remain after pose refinement."
        )

    retained_errors = errors[retained_mask]

    print("\n--- Camera 3 registration ---")
    print("World-to-camera rotation:\n", R_cw3)
    print("Extrinsic translation:", t_cw3)
    print("Camera center in the map frame:", camera_center3)
    print("Retained correspondences:", len(retained_errors))
    print(
        f"Median reprojection error: "
        f"{np.median(retained_errors):.3f} px"
    )
    print(
        f"90th percentile reprojection error: "
        f"{np.percentile(retained_errors, 90):.3f} px"
    )
    print(
        f"Maximum reprojection error: "
        f"{np.max(retained_errors):.3f} px"
    )

    pose_path = output_directory / "camera3_pose.npz"

    np.savez(
        pose_path,
        K=K,
        R_cw=R_cw3,
        t_cw=t_cw3,
        camera_center=camera_center3,
        map_indices=map_indices[retained_mask],
        image_keypoint_indices=image_indices[retained_mask],
        pixels=image_points[retained_mask],
        reprojection_errors=retained_errors,
        image_shape=np.array(expected_shape),
    )

    # Draw observed pixels in green and projected points in red.
    if image3.ndim == 2:
        visualization = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
    else:
        visualization = image3.copy()

    retained_points = object_points[retained_mask]
    observed_pixels = image_points[retained_mask]

    homogeneous_points = np.column_stack((
        retained_points,
        np.ones(len(retained_points)),
    ))

    projected_h = (P3 @ homogeneous_points.T).T
    projected_pixels = (
        projected_h[:, :2] / projected_h[:, 2:3]
    )

    for observed, projected in zip(observed_pixels, projected_pixels):
        observed_xy = tuple(np.rint(observed).astype(int))
        projected_xy = tuple(np.rint(projected).astype(int))

        cv2.circle(
            visualization, observed_xy, 12, (0, 255, 0), 2
        )
        cv2.circle(
            visualization, projected_xy, 5, (0, 0, 255), -1
        )

    visualization_path = output_directory / "camera3_reprojection.jpg"

    if not cv2.imwrite(str(visualization_path), visualization):
        raise IOError(
            f"Could not save visualization: {visualization_path}"
        )

    print(f"\nSaved camera 3 pose: {pose_path}")
    print(f"Saved reprojection visualization: {visualization_path}")
    print("Green circles: observed features; red dots: projected map points.")


if __name__ == "__main__":
    main()