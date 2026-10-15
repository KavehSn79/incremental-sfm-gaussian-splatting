from pathlib import Path

import cv2
import numpy as np

from demo_single_match import match_descriptors
from epipolar import triangulate_points, reprojection_errors
from sift_features import detect_sift_features


def sample_colors(path, pixels, expected_shape):
    """Sample RGB colors using oriented image coordinates."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load {path}")

    if image.shape[:2] != expected_shape:
        raise ValueError(f"Unexpected image dimensions: {path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    indices = np.rint(pixels).astype(np.int64)
    u, v = indices[:, 0], indices[:, 1]

    height, width = expected_shape

    if not np.all(
        (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ):
        raise ValueError("Color coordinates are outside the image.")

    return rgb[v, u].astype(float) / 255.0


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    image_directory = project_root / "data" / "images"

    with np.load(
        output_directory / "reconstruction.npz",
        allow_pickle=False,
    ) as data:
        old_points = data["points3d"]
        old_pixels1 = data["pixels1"]
        old_pixels2 = data["pixels2"]
        K = data["K"]
        R2 = data["R"]
        t2 = data["t"].reshape(3)
        expected_shape = tuple(data["image_shape"].tolist())

    with np.load(
        output_directory / "map_features.npz",
        allow_pickle=False,
    ) as data:
        map_points = data["points3d"]
        map_keypoints2 = data["keypoint_indices2"]

    with np.load(
        output_directory / "camera3_pose.npz",
        allow_pickle=False,
    ) as data:
        R3 = data["R_cw"]
        t3 = data["t_cw"].reshape(3)
        registered_keypoints3 = data["image_keypoint_indices"]
        K3 = data["K"]

    if (
        map_points.shape != old_points.shape
        or not np.allclose(map_points, old_points)
    ):
        raise RuntimeError(
            "The reconstruction and feature map do not match. "
            "Rerun demo_single_match.py and demo_pnp.py."
        )

    if not np.allclose(K, K3):
        raise RuntimeError("Camera 3 uses different intrinsics.")

    image2, keypoints2, descriptors2 = detect_sift_features(
        image_directory / "image2.jpg"
    )
    image3, keypoints3, descriptors3 = detect_sift_features(
        image_directory / "image3.jpg"
    )

    if (
        image2.shape[:2] != expected_shape
        or image3.shape[:2] != expected_shape
    ):
        raise ValueError("Images must match the reconstruction dimensions.")

    if (
        descriptors2 is None
        or len(descriptors2) == 0
        or descriptors3 is None
        or len(descriptors3) < 2
    ):
        raise RuntimeError("Insufficient descriptors for matching.")

    # We redetect features, so verify that saved image-2 feature IDs
    # still refer to the same observed pixels.
    if np.any(
        (map_keypoints2 < 0) | (map_keypoints2 >= len(keypoints2))
    ):
        raise RuntimeError("Saved image-2 feature IDs are invalid.")

    redetected_pixels2 = np.array([
        keypoints2[int(index)].pt for index in map_keypoints2
    ])

    if not np.allclose(
        redetected_pixels2, old_pixels2, rtol=0, atol=1e-3
    ):
        raise RuntimeError(
            "Image-2 features changed. Rebuild the map before expansion."
        )

    matches = match_descriptors(descriptors2, descriptors3)

    # Greedily enforce unique image-3 assignments.
    matches.sort(key=lambda match: match.distance)
    unique_matches = []
    used_train_indices = set()

    for match in matches:
        if match.trainIdx not in used_train_indices:
            unique_matches.append(match)
            used_train_indices.add(match.trainIdx)

    # Existing observations should not create duplicate map entries.
    occupied2 = set(map_keypoints2.tolist())
    occupied3 = set(registered_keypoints3.tolist())

    new_matches = [
        match
        for match in unique_matches
        if (
            match.queryIdx not in occupied2
            and match.trainIdx not in occupied3
        )
    ]

    print("Image-2/image-3 ratio-test matches:", len(matches))
    print("Matches after unique assignment:", len(unique_matches))
    print("Candidates for new map points:", len(new_matches))

    if not new_matches:
        raise RuntimeError("No unused feature pairs to triangulate.")

    pixels2 = np.array([
        keypoints2[match.queryIdx].pt for match in new_matches
    ], dtype=float)

    pixels3 = np.array([
        keypoints3[match.trainIdx].pt for match in new_matches
    ], dtype=float)

    P2 = K @ np.column_stack((R2, t2))
    P3 = K @ np.column_stack((R3, t3))

    new_points, finite = triangulate_points(
        pixels2, pixels3, P2, P3
    )

    camera_points2 = (R2 @ new_points.T).T + t2
    camera_points3 = (R3 @ new_points.T).T + t3

    front = (
        finite
        & (camera_points2[:, 2] > 1e-8)
        & (camera_points3[:, 2] > 1e-8)
    )

    errors2 = reprojection_errors(new_points, pixels2, P2)
    errors3 = reprojection_errors(new_points, pixels3, P3)

    center2 = -R2.T @ t2
    center3 = -R3.T @ t3

    # Viewing rays must be expressed in the same world frame.
    rays2 = new_points - center2
    rays3 = new_points - center3

    lengths2 = np.linalg.norm(rays2, axis=1)
    lengths3 = np.linalg.norm(rays3, axis=1)

    angle_valid = (
        finite
        & np.all(np.isfinite(new_points), axis=1)
        & (lengths2 > 1e-10)
        & (lengths3 > 1e-10)
    )

    angles = np.full(len(new_points), np.nan)

    unit2 = rays2[angle_valid] / lengths2[angle_valid, None]
    unit3 = rays3[angle_valid] / lengths3[angle_valid, None]

    cosine = np.sum(unit2 * unit3, axis=1)
    angles[angle_valid] = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    reprojection_ok = (
        front
        & np.isfinite(errors2)
        & np.isfinite(errors3)
        & (errors2 < 3.0)
        & (errors3 < 3.0)
    )

    keep = (
        reprojection_ok
        & angle_valid
        & (angles >= 1.0)
        & (angles <= 179.0)
    )

    print("Positive-depth candidates:", np.count_nonzero(front))
    print("After reprojection filtering:", np.count_nonzero(reprojection_ok))
    print("New retained points:", np.count_nonzero(keep))

    if np.any(keep):
        print(
            "New point median errors: "
            f"camera 2={np.median(errors2[keep]):.3f} px, "
            f"camera 3={np.median(errors3[keep]):.3f} px"
        )
        print(
            "New point triangulation angles: "
            f"minimum={np.min(angles[keep]):.3f} degrees, "
            f"median={np.median(angles[keep]):.3f} degrees"
        )

    old_colors = sample_colors(
        image_directory / "image1.jpg",
        old_pixels1,
        expected_shape,
    )

    new_colors = sample_colors(
        image_directory / "image2.jpg",
        pixels2[keep],
        expected_shape,
    )

    expanded_points = np.vstack((old_points, new_points[keep]))
    expanded_colors = np.vstack((old_colors, new_colors))

    # Preserve feature associations for the newly added points.
    new_ids2 = np.array([
        match.queryIdx for match in new_matches
    ], dtype=np.int64)[keep]

    new_ids3 = np.array([
        match.trainIdx for match in new_matches
    ], dtype=np.int64)[keep]

    expanded_path = output_directory / "expanded_reconstruction.npz"

    np.savez(
        expanded_path,
        points3d=expanded_points,
        colors=expanded_colors,
        original_point_count=len(old_points),
        new_keypoint_indices2=new_ids2,
        new_keypoint_indices3=new_ids3,
        new_pixels2=pixels2[keep],
        new_pixels3=pixels3[keep],
        new_descriptors2=descriptors2[new_ids2],
        new_errors2=errors2[keep],
        new_errors3=errors3[keep],
        new_angles_deg=angles[keep],
        K=K,
        R2=R2,
        t2=t2,
        R3=R3,
        t3=t3,
        image_shape=np.array(expected_shape),
    )

    print("Original map points:", len(old_points))
    print("Expanded map points:", len(expanded_points))
    print(f"Saved expanded reconstruction: {expanded_path}")


if __name__ == "__main__":
    main()