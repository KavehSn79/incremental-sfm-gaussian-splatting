import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def write_initialization_ply(path, points, colors):
    """Write the XYZ, RGB, and normal fields expected by the trainer."""
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)

    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]

    for name in ("nx", "ny", "nz"):
        vertices[name] = 0

    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, channel]

    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=True,
    ).write(str(path))


def main():
    cv2.setNumThreads(1)

    project_root = Path(__file__).resolve().parent.parent
    reconstruction_path = (
        project_root / "outputs" / "global_bundle_adjusted.npz"
    )
    destination = project_root / "data" / "gs_our_sfm"
    images_directory = destination / "images"
    model_directory = destination / "sparse" / "0"

    with np.load(reconstruction_path, allow_pickle=False) as data:
        points = data["points3d"].copy()
        colors = data["colors"].copy()
        rotations = data["rotations"].copy()
        translations = data["translations"].copy()
        camera_indices = data["camera_indices"].copy()
        point_indices = data["point_indices"].copy()
        feature_indices = data["keypoint_indices"].copy()
        pixels = data["pixels"].copy()
        names = data["image_names"].tolist()
        K = data["K"].copy()
        height, width = data["image_shape"].tolist()

    camera_count = len(names)
    point_count = len(points)

    if (
        rotations.shape != (camera_count, 3, 3)
        or translations.shape != (camera_count, 3)
        or points.shape != (point_count, 3)
        or colors.shape != points.shape
        or pixels.shape != (len(point_indices), 2)
    ):
        raise ValueError("Unexpected reconstruction array shapes.")

    if not (
        len(camera_indices)
        == len(point_indices)
        == len(feature_indices)
        == len(pixels)
    ):
        raise ValueError("Observation arrays must align.")

    if camera_count < 2 or point_count == 0:
        raise ValueError("Insufficient reconstruction data.")

    if not all(
        np.all(np.isfinite(array))
        for array in (points, colors, rotations, translations, pixels, K)
    ):
        raise ValueError("Reconstruction contains non-finite values.")

    if (
        np.any(camera_indices < 0)
        or np.any(camera_indices >= camera_count)
        or np.any(point_indices < 0)
        or np.any(point_indices >= point_count)
    ):
        raise ValueError("Invalid camera or point indices.")

    if np.any(
        np.bincount(point_indices, minlength=point_count) < 2
    ):
        raise ValueError("Each exported point needs at least two observations.")

    if not np.allclose(
        rotations.transpose(0, 2, 1) @ rotations,
        np.eye(3),
        atol=1e-6,
    ):
        raise ValueError("Camera rotations must be orthogonal.")

    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6):
        raise ValueError("Camera rotations must have determinant +1.")

    export_names = [f"{Path(name).stem}.png" for name in names]

    if len(set(export_names)) != camera_count:
        raise ValueError("Export image filenames must be unique.")

    # Use original images directly, independently of COLMAP's output.
    for name in names:
        source = project_root / "data" / "images" / name
        if not source.is_file():
            raise FileNotFoundError(source)

    # This dataset is dedicated to text-format export.
    # Binary models would take precedence over these text files.
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        if (model_directory / filename).exists():
            raise RuntimeError(
                f"Unexpected binary model in destination: {filename}. "
                "Use a separate destination for this text export."
            )

    images_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    # Verify observation ownership and collect per-image records.
    image_observations = [[] for _ in range(camera_count)]
    used_features = set()
    used_point_cameras = set()

    for camera_index, point_index, feature_index, pixel in zip(
        camera_indices, point_indices, feature_indices, pixels
    ):
        camera_index = int(camera_index)
        point_index = int(point_index)
        feature_index = int(feature_index)

        feature_key = (camera_index, feature_index)
        point_camera_key = (point_index, camera_index)

        if (
            feature_key in used_features
            or point_camera_key in used_point_cameras
        ):
            raise ValueError("Conflicting feature-track associations.")

        used_features.add(feature_key)
        used_point_cameras.add(point_camera_key)

        image_observations[camera_index].append(
            (feature_index, point_index, pixel)
        )

    for observations in image_observations:
        observations.sort(key=lambda item: item[0])

    # Build the bidirectional image-observation / point-track references.
    tracks = [[] for _ in range(point_count)]

    for camera_index, observations in enumerate(image_observations):
        for exported_feature_index, (_, point_index, _) in enumerate(
            observations
        ):
            tracks[point_index].append(
                (camera_index + 1, exported_feature_index)
            )

    # Compute a mean pixel reprojection error for each point.
    camera_points = np.einsum(
        "nij,nj->ni",
        rotations[camera_indices],
        points[point_indices],
    ) + translations[camera_indices]

    if np.any(camera_points[:, 2] <= 1e-8):
        raise ValueError("An exported observation has invalid depth.")

    projected_h = camera_points @ K.T
    projected_pixels = projected_h[:, :2] / projected_h[:, 2:3]
    observation_errors = np.linalg.norm(
        projected_pixels - pixels, axis=1
    )

    error_sums = np.bincount(
        point_indices,
        weights=observation_errors,
        minlength=point_count,
    )
    observation_counts = np.bincount(
        point_indices, minlength=point_count
    )
    mean_point_errors = error_sums / observation_counts

    # Convert integer-centered OpenCV pixels to COLMAP's pixel convention.
    # Shift observations and principal point together.
    K_export = K.copy()
    K_export[0, 2] += 0.5
    K_export[1, 2] += 0.5

    cameras_path = model_directory / "cameras.txt"

    with cameras_path.open("w", encoding="utf-8") as file:
        file.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n")
        file.write(
            f"1 PINHOLE {width} {height} "
            f"{K_export[0, 0]:.17g} "
            f"{K_export[1, 1]:.17g} "
            f"{K_export[0, 2]:.17g} "
            f"{K_export[1, 2]:.17g}\n"
        )

    images_path = model_directory / "images.txt"

    with images_path.open("w", encoding="utf-8") as file:
        file.write(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
            "# POINTS2D: X Y POINT3D_ID\n"
        )

        for camera_index, name in enumerate(export_names):
            # SciPy returns quaternion components in XYZW order.
            qx, qy, qz, qw = Rotation.from_matrix(
                rotations[camera_index]
            ).as_quat()

            tx, ty, tz = translations[camera_index]

            pose_values = " ".join(
                f"{value:.17g}"
                for value in (qw, qx, qy, qz, tx, ty, tz)
            )

            file.write(
                f"{camera_index + 1} {pose_values} 1 {name}\n"
            )

            feature_records = []

            for _, point_index, pixel in image_observations[camera_index]:
                u, v = pixel + 0.5
                feature_records.append(
                    f"{u:.17g} {v:.17g} {point_index + 1}"
                )

            file.write(" ".join(feature_records) + "\n")

    points_path = model_directory / "points3D.txt"
    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with points_path.open("w", encoding="utf-8") as file:
        file.write(
            "# POINT3D_ID X Y Z R G B ERROR TRACK\n"
            "# TRACK: IMAGE_ID POINT2D_IDX pairs\n"
        )

        for point_index, point in enumerate(points):
            xyz = " ".join(f"{value:.17g}" for value in point)
            red, green, blue = map(int, rgb[point_index])

            track_text = " ".join(
                f"{image_id} {feature_index}"
                for image_id, feature_index in tracks[point_index]
            )

            file.write(
                f"{point_index + 1} {xyz} "
                f"{red} {green} {blue} "
                f"{mean_point_errors[point_index]:.17g} "
                f"{track_text}\n"
            )

    # Regenerate this file on every export to prevent stale initialization.
    write_initialization_ply(
        model_directory / "points3D.ply",
        points,
        colors,
    )

    for source_name, export_name in zip(names, export_names):
        source = project_root / "data" / "images" / source_name
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"Could not load matching oriented image: {source}")

        destination_image = images_directory / export_name

        if not cv2.imwrite(str(destination_image), image):
            raise IOError(f"Could not save {destination_image}")

        del image

    manifest = {
        "initialization_source": "our custom SfM pipeline",
        "source_reconstruction": str(reconstruction_path),
        "registered_images": camera_count,
        "points": point_count,
        "observations": len(pixels),
        "camera_model": "PINHOLE",
        "coordinate_convention": "world-to-camera R and t",
        "scale": "arbitrary reconstruction units",
        "image_names": export_names,
        "pixel_convention_conversion": (
            "Added 0.5 to pixel coordinates and principal point."
        ),
        "mean_observation_reprojection_error_px": float(
            np.mean(observation_errors)
        ),
    }

    (destination / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Export source:", reconstruction_path)
    print("Exported cameras:", camera_count)
    print("Exported points:", point_count)
    print("Exported observations:", len(pixels))
    print("Dataset:", destination)
    print("COLMAP reconstruction was not used.")


if __name__ == "__main__":
    main()
    