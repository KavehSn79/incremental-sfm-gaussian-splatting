import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def read_colmap_images(path):
    """Read camera poses from COLMAP's two-line-per-image text format."""
    cameras = {}

    with path.open(encoding="utf-8") as file:
        while True:
            line = file.readline()

            if not line:
                break

            if not line.strip() or line.lstrip().startswith("#"):
                continue

            fields = line.strip().split(maxsplit=9)

            if len(fields) != 10:
                raise ValueError(f"Invalid pose line in {path}")

            # COLMAP: qw, qx, qy, qz.
            # SciPy: qx, qy, qz, qw.
            qw, qx, qy, qz = map(float, fields[1:5])

            R_cw = Rotation.from_quat(
                [qx, qy, qz, qw]
            ).as_matrix()

            t_cw = np.array(fields[5:8], dtype=float)
            name = fields[9]
            stem = Path(name).stem

            if stem in cameras:
                raise ValueError(f"Duplicate image stem: {stem}")

            cameras[stem] = {
                "name": name,
                "R_cw": R_cw,
                "center": -R_cw.T @ t_cw,
            }

            # Consume the feature-observation line, even if empty.
            file.readline()

    return cameras


def fit_similarity(source, target):
    """
    Fit target ≈ scale * R @ source + t.

    Points are stored as rows, so application is:
        aligned = scale * source @ R.T + t
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape:
        raise ValueError("Corresponding point arrays must match.")

    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Need at least three corresponding 3D points.")

    if not (
        np.all(np.isfinite(source))
        and np.all(np.isfinite(target))
    ):
        raise ValueError("Camera centers must be finite.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Collinear centers cannot uniquely determine alignment rotation.
    if (
        np.linalg.matrix_rank(source_centered) < 2
        or np.linalg.matrix_rank(target_centered) < 2
    ):
        raise ValueError("Camera centers are collinear or degenerate.")

    covariance = (
        target_centered.T @ source_centered / len(source)
    )

    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3)

    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1

    R_align = U @ correction @ Vt

    source_variance = np.mean(
        np.sum(source_centered**2, axis=1)
    )

    scale = (
        np.sum(singular_values * np.diag(correction))
        / source_variance
    )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale.")

    t_align = target_mean - scale * R_align @ source_mean

    return scale, R_align, t_align


def rotation_difference_degrees(R1, R2):
    """Angle of the relative rotation."""
    relative = R1 @ R2.T
    cosine = (np.trace(relative) - 1.0) / 2.0

    return float(
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    )


def check_alignment_implementation():
    """Verify alignment on a known synthetic similarity transform."""
    rng = np.random.default_rng(7)
    source = rng.normal(size=(10, 3))

    R_reference = Rotation.from_rotvec(
        [0.2, -0.1, 0.3]
    ).as_matrix()

    scale_reference = 2.5
    translation_reference = np.array([1.0, -2.0, 0.5])

    target = (
        scale_reference * source @ R_reference.T
        + translation_reference
    )

    scale, R, translation = fit_similarity(source, target)

    assert np.isclose(scale, scale_reference, atol=1e-10)
    assert np.allclose(R, R_reference, atol=1e-10)
    assert np.allclose(translation, translation_reference, atol=1e-10)


def set_equal_axes(axis, points):
    """Use equal spatial ranges on the three plot axes."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    half_range = max(np.max(maximum - minimum) / 2.0, 0.1) * 1.15

    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1, 1, 1))


def main():
    check_alignment_implementation()

    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    colmap_directory = output_directory / "colmap_fixed_k"

    with np.load(
        output_directory / "global_bundle_adjusted.npz",
        allow_pickle=False,
    ) as data:
        our_names = data["image_names"].tolist()
        our_rotations = data["rotations"].copy()
        our_translations = data["translations"].copy()

    our_centers = np.array([
        -R.T @ t
        for R, t in zip(our_rotations, our_translations)
    ])

    our_stems = [Path(name).stem for name in our_names]

    # Select the text model with the most corresponding registered images.
    candidates = []

    for path in colmap_directory.glob("text_*/images.txt"):
        cameras = read_colmap_images(path)
        overlap = sum(stem in cameras for stem in our_stems)
        candidates.append((overlap, path, cameras))

    if not candidates:
        raise FileNotFoundError(
            "No COLMAP text model found. Run run_colmap.py first."
        )

    overlap, model_path, colmap_cameras = max(
        candidates, key=lambda item: item[0]
    )

    if overlap < 3:
        raise RuntimeError("Too few shared cameras for alignment.")

    shared_indices = [
        index
        for index, stem in enumerate(our_stems)
        if stem in colmap_cameras
    ]

    names = [our_names[index] for index in shared_indices]
    target_centers = our_centers[shared_indices]
    target_rotations = our_rotations[shared_indices]

    source_centers = np.array([
        colmap_cameras[our_stems[index]]["center"]
        for index in shared_indices
    ])
    source_rotations = np.array([
        colmap_cameras[our_stems[index]]["R_cw"]
        for index in shared_indices
    ])

    scale, R_align, t_align = fit_similarity(
        source_centers, target_centers
    )

    aligned_centers = (
        scale * source_centers @ R_align.T + t_align
    )

    # The world frame changes by R_align.
    # World-to-camera rotations therefore multiply by its inverse.
    aligned_rotations = np.array([
        R_cw @ R_align.T for R_cw in source_rotations
    ])

    center_differences = np.linalg.norm(
        aligned_centers - target_centers,
        axis=1,
    )

    orientation_differences = np.array([
        rotation_difference_degrees(R_ours, R_colmap)
        for R_ours, R_colmap in zip(
            target_rotations, aligned_rotations
        )
    ])

    # Normalize positional differences by our initial camera baseline.
    baseline = np.linalg.norm(our_centers[1] - our_centers[0])

    if baseline < 1e-10:
        raise RuntimeError("Invalid initial baseline.")

    center_percentages = 100.0 * center_differences / baseline

    print("Selected COLMAP model:", model_path.parent)
    print("Shared cameras:", len(names))
    print("COLMAP-to-our-map scale:", scale)
    print("Alignment rotation:\n", R_align)
    print("Alignment translation:", t_align)
    print("Our initial baseline:", baseline)

    print("\nPer-camera differences after alignment:")

    for name, position, percentage, angle in zip(
        names,
        center_differences,
        center_percentages,
        orientation_differences,
    ):
        print(
            f"{name}: center={position:.6f} map units "
            f"({percentage:.3f}% of initial baseline), "
            f"orientation={angle:.3f} degrees"
        )

    center_rms = float(
        np.sqrt(np.mean(center_differences**2))
    )

    print("\nSummary — alignment fitted using these same cameras:")
    print(f"Camera-center RMS difference: {center_rms:.6f} map units")
    print(
        "Camera-center RMS / initial baseline:",
        f"{100.0 * center_rms / baseline:.3f}%",
    )
    print(
        "Median orientation difference:",
        f"{np.median(orientation_differences):.3f} degrees",
    )
    print(
        "Maximum orientation difference:",
        f"{np.max(orientation_differences):.3f} degrees",
    )

    # Fit without each camera, then evaluate that withheld camera.
    # This is a diagnostic of alignment stability, not ground truth.
    loo_differences = np.full(len(names), np.nan)

    if len(names) >= 4:
        for index in range(len(names)):
            training_mask = np.arange(len(names)) != index

            try:
                s, R, t = fit_similarity(
                    source_centers[training_mask],
                    target_centers[training_mask],
                )
            except ValueError:
                continue

            predicted = s * R @ source_centers[index] + t
            loo_differences[index] = np.linalg.norm(
                predicted - target_centers[index]
            )

    finite_loo = loo_differences[np.isfinite(loo_differences)]

    if len(finite_loo):
        print(
            "Leave-one-camera-out center RMS difference:",
            f"{np.sqrt(np.mean(finite_loo**2)):.6f} map units",
        )

    # Report whether the camera-center distribution is nearly collinear.
    center_singular_values = np.linalg.svd(
        source_centers - source_centers.mean(axis=0),
        compute_uv=False,
    )

    print("COLMAP center-distribution singular values:", center_singular_values)

    if center_singular_values[1] / center_singular_values[0] < 0.01:
        print(
            "Camera centers are nearly collinear; "
            "alignment rotation may be sensitive to small position changes."
        )

    csv_path = output_directory / "camera_pose_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "center_difference_map_units",
            "center_difference_percent_initial_baseline",
            "orientation_difference_degrees",
            "leave_one_out_center_difference_map_units",
        ])

        for index, name in enumerate(names):
            writer.writerow([
                name,
                center_differences[index],
                center_percentages[index],
                orientation_differences[index],
                (
                    float(loo_differences[index])
                    if np.isfinite(loo_differences[index])
                    else ""
                ),
            ])

    summary = {
        "shared_cameras": len(names),
        "colmap_text_model": str(model_path.parent),
        "alignment_direction": "COLMAP to our reconstruction",
        "alignment_scale": float(scale),
        "alignment_rotation": R_align.tolist(),
        "alignment_translation": t_align.tolist(),
        "initial_baseline_map_units": float(baseline),
        "camera_center_rms_map_units": center_rms,
        "camera_center_rms_percent_initial_baseline": (
            100.0 * center_rms / baseline
        ),
        "median_orientation_difference_degrees": float(
            np.median(orientation_differences)
        ),
        "maximum_orientation_difference_degrees": float(
            np.max(orientation_differences)
        ),
        "leave_one_out_center_rms_map_units": (
            float(np.sqrt(np.mean(finite_loo**2)))
            if len(finite_loo)
            else None
        ),
    }

    json_path = output_directory / "camera_pose_comparison.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    np.savez(
        output_directory / "colmap_alignment.npz",
        scale=scale,
        R_align=R_align,
        t_align=t_align,
        image_names=np.asarray(names),
        our_centers=target_centers,
        aligned_colmap_centers=aligned_centers,
        aligned_colmap_rotations=aligned_rotations,
    )

    figure = plt.figure(figsize=(13, 5))
    axis = figure.add_subplot(121, projection="3d")

    axis.scatter(
        *target_centers.T,
        color="tab:blue",
        label="Our pipeline",
    )
    axis.scatter(
        *aligned_centers.T,
        color="tab:orange",
        marker="x",
        label="COLMAP aligned",
    )

    for index, (ours, colmap) in enumerate(
        zip(target_centers, aligned_centers)
    ):
        segment = np.vstack((ours, colmap))
        axis.plot(*segment.T, color="gray", linewidth=1)
        axis.text(*ours, str(shared_indices[index] + 1), fontsize=8)

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("Corresponding camera centers")
    axis.legend()

    set_equal_axes(
        axis, np.vstack((target_centers, aligned_centers))
    )

    angle_axis = figure.add_subplot(122)
    angle_axis.bar(
        np.arange(len(names)),
        orientation_differences,
        color="tab:purple",
    )
    angle_axis.set_xticks(
        np.arange(len(names)),
        [Path(name).stem for name in names],
        rotation=45,
    )
    angle_axis.set_ylabel("Orientation difference (degrees)")
    angle_axis.set_title("Camera orientations after center alignment")
    angle_axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    plot_path = output_directory / "camera_pose_comparison.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved table: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(f"Saved comparison plot: {plot_path}")


if __name__ == "__main__":
    main()