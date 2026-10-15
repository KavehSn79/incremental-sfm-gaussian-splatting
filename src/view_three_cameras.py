from pathlib import Path

import numpy as np
import open3d as o3d


def make_extrinsic(R_cw, t_cw):
    """Build a world-to-camera transformation."""
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return T_cw


def make_pose(R_cw, t_cw):
    """Build the inverse camera-to-world transformation."""
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_cw.T
    T_wc[:3, 3] = -R_cw.T @ t_cw
    return T_wc


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    reconstruction_path = (
        output_directory / "bundle_adjusted_reconstruction.npz"
        )

    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            "Run expand_reconstruction.py before opening this viewer."
        )

    with np.load(reconstruction_path, allow_pickle=False) as data:
        points = data["points3d"]
        colors = data["colors"]
        original_count = int(data["original_point_count"])
        K = data["K"]
        R2 = data["R2"]
        t2 = data["t2"].reshape(3)
        R3 = data["R3"]
        t3 = data["t3"].reshape(3)
        height, width = data["image_shape"].tolist()

    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or len(points) == 0
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(colors))
    ):
        raise ValueError("Invalid point cloud or color data.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    cloud_path = output_directory / "bundle_adjusted_cloud_colored.ply"

    if not o3d.io.write_point_cloud(
        str(cloud_path), cloud, write_ascii=True
    ):
        raise IOError(f"Could not save {cloud_path}")

    camera_parameters = [
        (np.eye(3), np.zeros(3)),
        (R2, t2),
        (R3, t3),
    ]

    camera_colors = [
        [0.1, 0.4, 1.0],  # Camera 1: blue
        [1.0, 0.5, 0.0],  # Camera 2: orange
        [0.2, 1.0, 0.3],  # Camera 3: green
    ]

    baseline = np.linalg.norm(-R2.T @ t2)

    if baseline < 1e-10:
        raise ValueError("Invalid initial camera baseline.")

    geometries = [cloud]
    centers = []

    for index, ((R, t), color) in enumerate(
        zip(camera_parameters, camera_colors),
        start=1,
    ):
        T_cw = make_extrinsic(R, t)
        T_wc = make_pose(R, t)
        center = T_wc[:3, 3]
        centers.append(center)

        camera = o3d.geometry.LineSet.create_camera_visualization(
            int(width),
            int(height),
            K,
            T_cw,
            scale=0.3 * baseline,
        )
        camera.paint_uniform_color(color)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2 * baseline
        )
        axes.transform(T_wc)

        geometries.extend((camera, axes))
        print(f"Camera {index} center:", center)

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(
        np.asarray(centers)
    )
    trajectory.lines = o3d.utility.Vector2iVector([
        [0, 1],
        [1, 2],
    ])
    trajectory.paint_uniform_color([0.8, 0.2, 0.8])
    geometries.append(trajectory)

    print("\nOriginal points:", original_count)
    print("New points:", len(points) - original_count)
    print("Total points:", len(points))
    print("Camera colors: 1 blue, 2 orange, 3 green.")
    print("Purple segments connect the camera centers.")
    print(f"Saved colored cloud: {cloud_path}")

    viewer = o3d.visualization.Visualizer()

    if not viewer.create_window(
        window_name="Three-camera expanded reconstruction",
        width=1280,
        height=800,
    ):
        raise RuntimeError("Could not create the Open3D window.")

    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)

        options = viewer.get_render_option()
        options.background_color = np.array([0.12, 0.12, 0.12])
        options.point_size = 6.0

        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    main()