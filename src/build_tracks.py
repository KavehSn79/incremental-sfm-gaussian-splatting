from pathlib import Path

import numpy as np


def load_npz(path):
    """Load arrays independently of the open NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"

    reconstruction = load_npz(
        output_directory / "reconstruction.npz"
    )
    feature_map = load_npz(
        output_directory / "map_features.npz"
    )
    pose3 = load_npz(
        output_directory / "camera3_pose.npz"
    )
    expanded = load_npz(
        output_directory / "expanded_reconstruction.npz"
    )

    old_points = reconstruction["points3d"]
    original_count = int(expanded["original_point_count"])

    if original_count != len(old_points):
        raise RuntimeError("Original point counts do not match.")

    if not np.allclose(
        expanded["points3d"][:original_count], old_points
    ):
        raise RuntimeError("Expanded and original maps do not match.")

    if (
        feature_map["points3d"].shape != old_points.shape
        or not np.allclose(feature_map["points3d"], old_points)
    ):
        raise RuntimeError("Feature map and reconstruction do not match.")

    if not (
        np.allclose(expanded["R3"], pose3["R_cw"])
        and np.allclose(expanded["t3"], pose3["t_cw"])
    ):
        raise RuntimeError("Expansion uses a different camera-3 pose.")

    # Associate registered image-3 observations with original map IDs.
    observations3 = {}

    for map_index, keypoint_index, pixel, error in zip(
        pose3["map_indices"],
        pose3["image_keypoint_indices"],
        pose3["pixels"],
        pose3["reprojection_errors"],
    ):
        map_index = int(map_index)

        if not 0 <= map_index < original_count:
            raise RuntimeError("Invalid registered map-point index.")

        if map_index in observations3:
            raise RuntimeError("Repeated image-3 observation for a map point.")

        observations3[map_index] = (
            2,
            int(keypoint_index),
            pixel,
            float(error),
        )

    # Camera indices are zero-based: 0, 1, 2.
    candidates = []

    for index in range(original_count):
        observations = [
            (
                0,
                int(feature_map["keypoint_indices1"][index]),
                reconstruction["pixels1"][index],
                float(reconstruction["errors1"][index]),
            ),
            (
                1,
                int(feature_map["keypoint_indices2"][index]),
                reconstruction["pixels2"][index],
                float(reconstruction["errors2"][index]),
            ),
        ]

        if index in observations3:
            observations.append(observations3[index])

        candidates.append((index, observations))

    new_count = len(expanded["points3d"]) - original_count

    for index in range(new_count):
        observations = [
            (
                1,
                int(expanded["new_keypoint_indices2"][index]),
                expanded["new_pixels2"][index],
                float(expanded["new_errors2"][index]),
            ),
            (
                2,
                int(expanded["new_keypoint_indices3"][index]),
                expanded["new_pixels3"][index],
                float(expanded["new_errors3"][index]),
            ),
        ]

        candidates.append((original_count + index, observations))

    # Prefer more observations, then lower mean reprojection error.
    candidates.sort(
        key=lambda item: (
            -len(item[1]),
            np.mean([observation[3] for observation in item[1]]),
        )
    )

    used_features = set()
    selected = []
    conflict_count = 0

    for source_index, observations in candidates:
        feature_keys = [
            (camera_index, feature_index)
            for camera_index, feature_index, _, _ in observations
        ]

        if any(key in used_features for key in feature_keys):
            conflict_count += 1
            continue

        used_features.update(feature_keys)
        selected.append((source_index, observations))

    # Preserve original-map-first ordering.
    selected.sort(key=lambda item: item[0])

    if not selected:
        raise RuntimeError("No feature tracks survived.")

    source_indices = np.array(
        [source_index for source_index, _ in selected],
        dtype=np.int64,
    )

    camera_indices = []
    point_indices = []
    keypoint_indices = []
    pixels = []

    for point_index, (_, observations) in enumerate(selected):
        for camera_index, feature_index, pixel, _ in observations:
            camera_indices.append(camera_index)
            point_indices.append(point_index)
            keypoint_indices.append(feature_index)
            pixels.append(pixel)

    camera_indices = np.array(camera_indices, dtype=np.int64)
    point_indices = np.array(point_indices, dtype=np.int64)
    keypoint_indices = np.array(keypoint_indices, dtype=np.int64)
    pixels = np.array(pixels, dtype=float)

    track_lengths = np.bincount(
        point_indices, minlength=len(selected)
    )

    rotations = np.stack((
        np.eye(3),
        expanded["R2"],
        expanded["R3"],
    ))

    translations = np.stack((
        np.zeros(3),
        expanded["t2"],
        expanded["t3"],
    ))

    path = output_directory / "tracks.npz"

    np.savez(
        path,
        points3d=expanded["points3d"][source_indices],
        colors=expanded["colors"][source_indices],
        source_point_indices=source_indices,
        original_point_count=np.count_nonzero(
            source_indices < original_count
        ),
        camera_indices=camera_indices,
        point_indices=point_indices,
        keypoint_indices=keypoint_indices,
        pixels=pixels,
        rotations=rotations,
        translations=translations,
        K=expanded["K"],
        image_shape=expanded["image_shape"],
    )

    print("Candidate tracks:", len(candidates))
    print("Conflicting tracks rejected:", conflict_count)
    print("Retained tracks:", len(selected))
    print("Two-view tracks:", np.count_nonzero(track_lengths == 2))
    print("Three-view tracks:", np.count_nonzero(track_lengths == 3))
    print("Total observations:", len(pixels))
    print(f"Saved tracks: {path}")


if __name__ == "__main__":
    main()