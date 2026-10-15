import argparse
from pathlib import Path

import cv2
import numpy as np

from demo_single_match import match_descriptors
from epipolar import triangulate_points, reprojection_errors
from sift_features import detect_sift_features

import subprocess
import sys

from cache_features import cache_signature

def unique_matches(matches):
    """Keep at most one assignment to each destination feature."""
    result = []
    used = set()

    for match in sorted(matches, key=lambda item: item.distance):
        if match.trainIdx not in used:
            result.append(match)
            used.add(match.trainIdx)

    return result


def projection_matrix(K, R, t):
    return K @ np.column_stack((R, t))


def fits_observation(point, pixel, K, R, t, threshold=3.0):
    """Check whether an existing map point fits a new observation."""
    point = np.asarray(point, dtype=float)
    point_camera = R @ point + t

    if not np.all(np.isfinite(point_camera)):
        return False

    if point_camera[2] <= 1e-8:
        return False

    projected_h = K @ point_camera
    projected = projected_h[:2] / projected_h[2]

    return np.linalg.norm(projected - pixel) < threshold


def sample_color(image_rgb, pixel):
    """Read the nearest RGB pixel as an Open3D-compatible color."""
    u, v = np.rint(pixel).astype(int)
    height, width = image_rgb.shape[:2]

    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Color-sampling coordinate is outside the image.")

    return image_rgb[v, u].astype(float) / 255.0


def save_checkpoint(path, state):
    """Save a checkpoint using an atomic replacement."""
    observations = state["observations"]

    camera_indices = np.array(
        [item[0] for item in observations], dtype=np.int64
    )
    keypoint_indices = np.array(
        [item[1] for item in observations], dtype=np.int64
    )
    point_indices = np.array(
        [item[2] for item in observations], dtype=np.int64
    )
    pixels = np.array(
        [item[3] for item in observations], dtype=float
    )

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        np.savez(
            file,
            points3d=np.asarray(state["points"], dtype=float),
            colors=np.asarray(state["colors"], dtype=float),
            rotations=np.asarray(state["rotations"], dtype=float),
            translations=np.asarray(state["translations"], dtype=float),
            camera_indices=camera_indices,
            keypoint_indices=keypoint_indices,
            point_indices=point_indices,
            pixels=pixels,
            image_names=np.asarray(state["image_names"]),
            K=state["K"],
            image_shape=np.asarray(state["image_shape"]),
        )

    temporary_path.replace(path)


def load_initial_state(output_directory):
    """Start from our refined three-camera tracks."""
    path = output_directory / "bundle_adjusted_tracks.npz"

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}

    observations = [
        (
            int(camera_index),
            int(keypoint_index),
            int(point_index),
            pixel.copy(),
        )
        for camera_index, keypoint_index, point_index, pixel in zip(
            arrays["camera_indices"],
            arrays["keypoint_indices"],
            arrays["point_indices"],
            arrays["pixels"],
        )
    ]

    return {
        "points": list(arrays["points3d"]),
        "colors": list(arrays["colors"]),
        "rotations": list(arrays["rotations"]),
        "translations": list(arrays["translations"]),
        "observations": observations,
        "image_names": ["image1.jpg", "image2.jpg", "image3.jpg"],
        "K": arrays["K"],
        "image_shape": tuple(arrays["image_shape"].tolist()),
    }


def load_checkpoint(path):
    """Resume a previously saved incremental reconstruction."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}

    observations = [
        (
            int(camera_index),
            int(keypoint_index),
            int(point_index),
            pixel.copy(),
        )
        for camera_index, keypoint_index, point_index, pixel in zip(
            arrays["camera_indices"],
            arrays["keypoint_indices"],
            arrays["point_indices"],
            arrays["pixels"],
        )
    ]

    return {
        "points": list(arrays["points3d"]),
        "colors": list(arrays["colors"]),
        "rotations": list(arrays["rotations"]),
        "translations": list(arrays["translations"]),
        "observations": observations,
        "image_names": arrays["image_names"].tolist(),
        "K": arrays["K"],
        "image_shape": tuple(arrays["image_shape"].tolist()),
    }


def get_features(name, image_directory, expected_shape, cache):
    """Load cached features or extract them in a separate process."""
    if name in cache:
        return cache[name]

    image_path = image_directory / name

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    project_root = image_directory.parent.parent
    cache_directory = project_root / "outputs" / "feature_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)

    cache_path = cache_directory / f"{name}.npz"
    signature = cache_signature(image_path)
    cache_valid = False

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            cache_valid = str(data["signature"].item()) == signature

    if not cache_valid:
        worker_path = Path(__file__).resolve().parent / "cache_features.py"

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(worker_path),
                str(image_path),
                str(cache_path),
            ],
            check=True,
        )

    with np.load(cache_path, allow_pickle=False) as data:
        image_shape = tuple(data["image_shape"].tolist())
        descriptors = data["descriptors"].copy()
        pixels = data["pixels"].copy()

    if image_shape != expected_shape:
        raise ValueError(
            f"{name}: expected shape {expected_shape}, "
            f"received {image_shape}."
        )

    # The incremental pipeline only needs descriptors and pixel positions.
    cache[name] = (None, None, descriptors, pixels)

    print(f"Loaded cached features: {name}", flush=True)
    return cache[name]


def build_associations(state):
    """Build lookup tables enforcing observation ownership."""
    feature_to_point = {}
    point_camera_to_feature = {}

    for camera_index, feature_index, point_index, _ in state["observations"]:
        feature_key = (camera_index, feature_index)
        point_key = (point_index, camera_index)

        if feature_key in feature_to_point:
            raise RuntimeError("An image feature belongs to multiple tracks.")

        if point_key in point_camera_to_feature:
            raise RuntimeError("A track has multiple features in one camera.")

        feature_to_point[feature_key] = point_index
        point_camera_to_feature[point_key] = feature_index

    return feature_to_point, point_camera_to_feature


def add_observation(
    state,
    feature_to_point,
    point_camera_to_feature,
    camera_index,
    feature_index,
    point_index,
    pixel,
):
    """Add an observation only if it preserves track uniqueness."""
    feature_key = (camera_index, feature_index)
    point_key = (point_index, camera_index)

    if (
        feature_key in feature_to_point
        or point_key in point_camera_to_feature
    ):
        return False

    state["observations"].append(
        (camera_index, feature_index, point_index, pixel.copy())
    )
    feature_to_point[feature_key] = point_index
    point_camera_to_feature[point_key] = feature_index

    return True


def map_descriptors(state, cache):
    """Represent each map point using its most recent observation."""
    latest = {}

    for camera_index, feature_index, point_index, _ in state["observations"]:
        if (
            point_index not in latest
            or camera_index > latest[point_index][0]
        ):
            latest[point_index] = (camera_index, feature_index)

    descriptors = []

    for point_index in range(len(state["points"])):
        camera_index, feature_index = latest[point_index]
        name = state["image_names"][camera_index]
        descriptors.append(cache[name][2][feature_index])

    return np.asarray(descriptors, dtype=np.float32)


def register_camera(state, descriptors_new, pixels_new, cache):
    """Estimate a new world-to-camera pose from map-to-image matches."""
    descriptors_map = map_descriptors(state, cache)

    matches = unique_matches(
        match_descriptors(descriptors_map, descriptors_new)
    )

    print("Candidate map-to-image matches:", len(matches))

    if len(matches) < 6:
        print("Registration skipped: fewer than six matches.")
        return None

    map_indices = np.array(
        [match.queryIdx for match in matches], dtype=np.int64
    )
    feature_indices = np.array(
        [match.trainIdx for match in matches], dtype=np.int64
    )

    object_points = np.ascontiguousarray(
        np.asarray(state["points"])[map_indices],
        dtype=np.float64,
    )
    image_points = np.ascontiguousarray(
        pixels_new[feature_indices], dtype=np.float64
    )

    K = state["K"]
    distortion = np.zeros((5, 1), dtype=float)

    cv2.setRNGSeed(42)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        distortion,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < 6:
        print("Registration skipped: insufficient PnP consensus.")
        return None

    inliers = inliers.ravel()

    rvec, tvec = cv2.solvePnPRefineLM(
        np.ascontiguousarray(object_points[inliers]),
        np.ascontiguousarray(image_points[inliers]),
        K,
        distortion,
        rvec,
        tvec,
    )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()

    if not (
        np.all(np.isfinite(R))
        and np.all(np.isfinite(t))
    ):
        print("Registration skipped: non-finite pose.")
        return None

    errors = reprojection_errors(
        object_points,
        image_points,
        projection_matrix(K, R, t),
    )
    camera_points = (R @ object_points.T).T + t

    mask = np.zeros(len(matches), dtype=bool)
    mask[inliers] = True
    mask &= (
        np.isfinite(errors)
        & (errors < 3.0)
        & (camera_points[:, 2] > 1e-8)
    )

    if np.count_nonzero(mask) < 6:
        print("Registration skipped: too few valid refined observations.")
        return None

    print("PnP inliers:", len(inliers))
    print("Retained map observations:", np.count_nonzero(mask))
    print(f"Median PnP reprojection error: {np.median(errors[mask]):.3f} px")
    print("New camera center:", -R.T @ t)

    return R, t, map_indices[mask], feature_indices[mask]


def expand_map(state, previous_camera, new_camera, cache, image_rgb):
    """Extend tracks and triangulate against the preceding camera."""
    previous_name = state["image_names"][previous_camera]
    new_name = state["image_names"][new_camera]

    _, _, descriptors_old, pixels_old = cache[previous_name]
    _, _, descriptors_new, pixels_new = cache[new_name]

    matches = unique_matches(
        match_descriptors(descriptors_old, descriptors_new)
    )

    feature_to_point, point_camera_to_feature = build_associations(state)

    K = state["K"]
    R_old = state["rotations"][previous_camera]
    t_old = state["translations"][previous_camera]
    R_new = state["rotations"][new_camera]
    t_new = state["translations"][new_camera]

    new_pairs = []
    extended_count = 0

    for match in matches:
        old_id = match.queryIdx
        new_id = match.trainIdx

        old_point = feature_to_point.get((previous_camera, old_id))
        new_point = feature_to_point.get((new_camera, new_id))

        if old_point is not None and new_point is not None:
            # Matching tracks already agree, or conflict.
            # Neither situation creates a new point.
            continue

        if old_point is not None:
            if fits_observation(
                state["points"][old_point],
                pixels_new[new_id],
                K,
                R_new,
                t_new,
            ):
                extended_count += add_observation(
                    state,
                    feature_to_point,
                    point_camera_to_feature,
                    new_camera,
                    new_id,
                    old_point,
                    pixels_new[new_id],
                )
            continue

        if new_point is not None:
            if fits_observation(
                state["points"][new_point],
                pixels_old[old_id],
                K,
                R_old,
                t_old,
            ):
                extended_count += add_observation(
                    state,
                    feature_to_point,
                    point_camera_to_feature,
                    previous_camera,
                    old_id,
                    new_point,
                    pixels_old[old_id],
                )
            continue

        new_pairs.append((old_id, new_id))

    print("Existing-track observations added:", extended_count)
    print("Unused pairs for triangulation:", len(new_pairs))

    if not new_pairs:
        return

    old_ids = np.array([pair[0] for pair in new_pairs], dtype=np.int64)
    new_ids = np.array([pair[1] for pair in new_pairs], dtype=np.int64)

    observed_old = pixels_old[old_ids]
    observed_new = pixels_new[new_ids]

    P_old = projection_matrix(K, R_old, t_old)
    P_new = projection_matrix(K, R_new, t_new)

    points, finite = triangulate_points(
        observed_old, observed_new, P_old, P_new
    )

    camera_points_old = (R_old @ points.T).T + t_old
    camera_points_new = (R_new @ points.T).T + t_new

    errors_old = reprojection_errors(points, observed_old, P_old)
    errors_new = reprojection_errors(points, observed_new, P_new)

    center_old = -R_old.T @ t_old
    center_new = -R_new.T @ t_new

    rays_old = points - center_old
    rays_new = points - center_new

    lengths_old = np.linalg.norm(rays_old, axis=1)
    lengths_new = np.linalg.norm(rays_new, axis=1)

    angle_valid = (
        finite
        & np.all(np.isfinite(points), axis=1)
        & (lengths_old > 1e-10)
        & (lengths_new > 1e-10)
    )

    angles = np.full(len(points), np.nan)
    unit_old = rays_old[angle_valid] / lengths_old[angle_valid, None]
    unit_new = rays_new[angle_valid] / lengths_new[angle_valid, None]

    cosine = np.sum(unit_old * unit_new, axis=1)
    angles[angle_valid] = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    keep = (
        angle_valid
        & (camera_points_old[:, 2] > 1e-8)
        & (camera_points_new[:, 2] > 1e-8)
        & np.isfinite(errors_old)
        & np.isfinite(errors_new)
        & (errors_old < 3.0)
        & (errors_new < 3.0)
        & (angles >= 1.0)
        & (angles <= 179.0)
    )

    for index in np.flatnonzero(keep):
        point_index = len(state["points"])
        state["points"].append(points[index])
        state["colors"].append(
            sample_color(image_rgb, observed_new[index])
        )

        add_observation(
            state,
            feature_to_point,
            point_camera_to_feature,
            previous_camera,
            int(old_ids[index]),
            point_index,
            observed_old[index],
        )
        add_observation(
            state,
            feature_to_point,
            point_camera_to_feature,
            new_camera,
            int(new_ids[index]),
            point_index,
            observed_new[index],
        )

    print("New triangulated points retained:", np.count_nonzero(keep))

    if np.any(keep):
        print(
            "New-point median errors: "
            f"previous={np.median(errors_old[keep]):.3f} px, "
            f"new={np.median(errors_new[keep]):.3f} px"
        )


def show_reconstruction(state, output_directory):
    """Display any number of registered cameras."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(state["points"])
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(state["colors"])
    )

    cloud_path = output_directory / "incremental_cloud_colored.ply"

    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise IOError(f"Could not save {cloud_path}")

    height, width = state["image_shape"]
    K = state["K"]
    baseline = np.linalg.norm(state["translations"][1])

    geometries = [cloud]
    centers = []

    palette = [
        [0.1, 0.4, 1.0],
        [1.0, 0.5, 0.0],
        [0.2, 1.0, 0.3],
        [1.0, 0.2, 0.5],
        [0.3, 0.9, 0.9],
        [0.8, 0.5, 1.0],
    ]

    for index, (R, t) in enumerate(
        zip(state["rotations"], state["translations"])
    ):
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = t

        camera = o3d.geometry.LineSet.create_camera_visualization(
            width, height, K, T_cw, scale=0.3 * baseline
        )
        camera.paint_uniform_color(palette[index % len(palette)])
        geometries.append(camera)
        centers.append(-R.T @ t)

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(np.array(centers))
    trajectory.lines = o3d.utility.Vector2iVector([
        [index, index + 1] for index in range(len(centers) - 1)
    ])
    trajectory.paint_uniform_color([0.8, 0.2, 0.8])
    geometries.append(trajectory)

    viewer = o3d.visualization.Visualizer()

    if not viewer.create_window(
        window_name="Incremental SfM",
        width=1280,
        height=800,
    ):
        raise RuntimeError("Could not create the Open3D window.")

    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)

        options = viewer.get_render_option()
        options.background_color = np.array([0.12, 0.12, 0.12])
        options.point_size = 5.0
        viewer.run()
    finally:
        viewer.destroy_window()

    print(f"Saved colored cloud: {cloud_path}")


def main():
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images",
        nargs="*",
        help="Additional filenames inside data/images, in desired order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load incremental_reconstruction.npz instead of the initial map.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display the resulting reconstruction.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    image_directory = project_root / "data" / "images"
    output_directory = project_root / "outputs"
    checkpoint_path = (
        output_directory / "incremental_reconstruction.npz"
    )

    state = (
        load_checkpoint(checkpoint_path)
        if args.resume
        else load_initial_state(output_directory)
    )

    cache = {}

    # Load existing registered images and verify feature-ID consistency.
    for name in state["image_names"]:
        get_features(name, image_directory, state["image_shape"], cache)

    for camera_index, feature_index, _, pixel in state["observations"]:
        name = state["image_names"][camera_index]
        pixels = cache[name][3]

        if (
            not 0 <= feature_index < len(pixels)
            or not np.allclose(
                pixels[feature_index], pixel, rtol=0, atol=1e-3
            )
        ):
            raise RuntimeError(
                f"Saved feature IDs no longer match {name}. "
                "Keep images and SIFT settings unchanged."
            )

    build_associations(state)

    if not args.images and not args.view:
        parser.error("Provide additional images or use --view.")

    for name in args.images:
        if Path(name).name != name:
            raise ValueError("Provide filenames inside data/images.")

        if name in state["image_names"]:
            print(f"\nSkipping already registered image: {name}")
            continue

        print(f"\n--- Registering {name} ---")

        _, _, descriptors, pixels = get_features(
            name, image_directory, state["image_shape"], cache
        )

        # Load colors before modifying reconstruction state.
        image_bgr = cv2.imread(
            str(image_directory / name), cv2.IMREAD_COLOR
        )

        if (
            image_bgr is None
            or image_bgr.shape[:2] != state["image_shape"]
        ):
            raise RuntimeError(f"Could not load oriented color image: {name}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        registration = register_camera(
            state, descriptors, pixels, cache
        )

        if registration is None:
            continue

        R, t, map_indices, feature_indices = registration

        previous_camera = len(state["rotations"]) - 1
        new_camera = len(state["rotations"])

        state["image_names"].append(name)
        state["rotations"].append(R)
        state["translations"].append(t)

        feature_to_point, point_camera_to_feature = build_associations(state)

        for point_index, feature_index in zip(map_indices, feature_indices):
            add_observation(
                state,
                feature_to_point,
                point_camera_to_feature,
                new_camera,
                int(feature_index),
                int(point_index),
                pixels[feature_index],
            )

        expand_map(
            state, previous_camera, new_camera, cache, image_rgb
        )

        save_checkpoint(checkpoint_path, state)

        print("Registered cameras:", len(state["rotations"]))
        print("Map points:", len(state["points"]))
        print("Observations:", len(state["observations"]))
        print(f"Saved checkpoint: {checkpoint_path}")

    if args.view:
        show_reconstruction(state, output_directory)


if __name__ == "__main__":
    main()
    