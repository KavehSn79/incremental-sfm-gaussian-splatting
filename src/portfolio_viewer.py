import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Display every point, including distant points.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    path = output_directory / "global_bundle_adjusted.npz"

    with np.load(path, allow_pickle=False) as data:
        points = data["points3d"].copy()
        colors = data["colors"].copy()
        rotations = data["rotations"].copy()
        translations = data["translations"].copy()
        names = data["image_names"].tolist()
        K = data["K"].copy()
        height, width = data["image_shape"].tolist()

    if (
        len(points) == 0
        or points.shape != colors.shape
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("Invalid point cloud.")

    display_mask = np.ones(len(points), dtype=bool)

    if not args.show_all:
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        cutoff = np.percentile(distances, 98.0)
        display_mask = distances <= cutoff

    displayed_points = points[display_mask]
    displayed_colors = colors[display_mask]
    hidden_count = int(np.count_nonzero(~display_mask))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(displayed_points)
    cloud.colors = o3d.utility.Vector3dVector(displayed_colors)

    centers = np.array([
        -R.T @ t for R, t in zip(rotations, translations)
    ])

    baseline = np.linalg.norm(centers[1] - centers[0])

    if baseline < 1e-10:
        raise ValueError("Invalid camera baseline.")

    palette = [
        [0.2, 0.5, 1.0],
        [1.0, 0.6, 0.1],
        [0.3, 1.0, 0.4],
        [1.0, 0.3, 0.5],
        [0.3, 0.9, 0.9],
        [0.8, 0.5, 1.0],
        [1.0, 0.9, 0.3],
        [0.9, 0.9, 0.9],
    ]

    geometries = [cloud]

    for index, (R, t) in enumerate(zip(rotations, translations)):
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = t

        camera = o3d.geometry.LineSet.create_camera_visualization(
            width,
            height,
            K,
            T_cw,
            scale=0.2 * baseline,
        )
        camera.paint_uniform_color(palette[index % len(palette)])
        geometries.append(camera)

        print(
            f"Camera {index + 1}: {names[index]}, "
            f"center={centers[index]}"
        )

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(centers)
    trajectory.lines = o3d.utility.Vector2iVector([
        [index, index + 1]
        for index in range(len(centers) - 1)
    ])
    trajectory.paint_uniform_color([0.8, 0.3, 0.8])
    geometries.append(trajectory)

    screenshot_path = output_directory / "reconstruction_portfolio.png"
    metadata_path = output_directory / "reconstruction_portfolio.json"

    def save_screenshot(viewer):
        viewer.capture_screen_image(
            str(screenshot_path),
            do_render=True,
        )

        metadata = {
            "reconstruction": path.name,
            "registered_cameras": len(centers),
            "total_points": len(points),
            "displayed_points": len(displayed_points),
            "hidden_points": hidden_count,
            "display_crop": (
                "none"
                if args.show_all
                else "nearest 98% by distance from coordinate-wise median"
            ),
            "note": "Display-only crop; source reconstruction unchanged.",
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"Saved screenshot: {screenshot_path}")
        print(f"Saved display metadata: {metadata_path}")
        return False

    viewer = o3d.visualization.VisualizerWithKeyCallback()

    title = (
        f"Sparse SfM | {len(centers)} cameras | "
        f"{len(displayed_points)}/{len(points)} points displayed"
    )

    if not viewer.create_window(
        window_name=title,
        width=1600,
        height=1000,
    ):
        raise RuntimeError("Could not create the Open3D window.")

    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)

        options = viewer.get_render_option()
        options.background_color = np.array([0.10, 0.10, 0.12])
        options.point_size = 4.0

        viewer.register_key_callback(ord("S"), save_screenshot)

        print("\nTotal map points:", len(points))
        print("Displayed points:", len(displayed_points))
        print("Hidden for display only:", hidden_count)
        print("Rotate and zoom to frame the scene.")
        print("Press S while the viewer has focus to save a screenshot.")
        print("Close the window to exit.")

        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    main()