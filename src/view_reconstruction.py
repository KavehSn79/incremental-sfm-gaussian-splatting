from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def make_extrinsic(R_cw, t_cw):
    """Build a 4x4 world-to-camera transformation."""
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return T_cw


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "outputs"
    reconstruction_path = output_directory / "reconstruction.npz"

    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            f"Reconstruction file not found: {reconstruction_path}"
        )

    with np.load(reconstruction_path, allow_pickle=False) as data:
        points3d = data["points3d"]
        pixels1 = data["pixels1"]
        K = data["K"]
        R = data["R"]
        t = data["t"].reshape(3)

    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("points3d must have shape (N, 3).")

    if len(points3d) == 0 or not np.all(np.isfinite(points3d)):
        raise ValueError("The point cloud must contain finite points.")

    if pixels1.shape != (len(points3d), 2):
        raise ValueError(
            "Each reconstructed point needs one image-1 pixel."
        )

    if not np.all(np.isfinite(pixels1)):
        raise ValueError("Pixel coordinates must be finite.")

    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError("K and R must have shape (3, 3).")

    if not all(np.all(np.isfinite(a)) for a in (K, R, t)):
        raise ValueError("Camera parameters must be finite.")

    # These dimensions match the oriented reconstruction images.
    image_width = 4284
    image_height = 5712

    # Load image 1 in color. OpenCV applies its EXIF orientation.
    image_path = project_root / "data" / "images" / "image1.jpg"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    if image_bgr.shape[:2] != (image_height, image_width):
        raise ValueError(
            "Color image dimensions do not match the reconstruction."
        )

    # OpenCV uses BGR; Open3D expects RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Sample the nearest pixel at each saved feature position.
    pixel_indices = np.rint(pixels1).astype(np.int64)
    u = pixel_indices[:, 0]
    v = pixel_indices[:, 1]

    inside_image = (
        (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )

    if not np.all(inside_image):
        raise ValueError("Some color-sampling pixels are outside image 1.")

    # Images are indexed [row, column] = [v, u].
    # Open3D expects floating-point colors in [0, 1].
    colors = image_rgb[v, u].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points3d)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    colored_cloud_path = output_directory / "sparse_cloud_colored.ply"

    if not o3d.io.write_point_cloud(
        str(colored_cloud_path),
        cloud,
        write_ascii=True,
    ):
        raise IOError(
            f"Could not save colored cloud: {colored_cloud_path}"
        )

    print(f"Saved colored point cloud: {colored_cloud_path}")

    # Camera 1 defines the world coordinate system.
    T_cw1 = make_extrinsic(np.eye(3), np.zeros(3))
    T_cw2 = make_extrinsic(R, t)

    # Camera-to-world poses position local coordinate axes.
    T_wc1 = np.eye(4)

    T_wc2 = np.eye(4)
    T_wc2[:3, :3] = R.T
    T_wc2[:3, 3] = -R.T @ t

    camera_center1 = T_wc1[:3, 3]
    camera_center2 = T_wc2[:3, 3]
    baseline = np.linalg.norm(camera_center2 - camera_center1)

    if baseline < 1e-10:
        raise ValueError("Camera centers are too close to visualize.")

    # Camera wireframes use world-to-camera extrinsics.
    camera1 = o3d.geometry.LineSet.create_camera_visualization(
        image_width,
        image_height,
        K,
        T_cw1,
        scale=0.3 * baseline,
    )
    camera1.paint_uniform_color([0.1, 0.4, 1.0])

    camera2 = o3d.geometry.LineSet.create_camera_visualization(
        image_width,
        image_height,
        K,
        T_cw2,
        scale=0.3 * baseline,
    )
    camera2.paint_uniform_color([1.0, 0.5, 0.0])

    # Coordinate axes: X red, Y green, Z blue.
    axes1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2 * baseline
    )
    axes1.transform(T_wc1)

    axes2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2 * baseline
    )
    axes2.transform(T_wc2)

    # Draw the line connecting the two camera centers.
    baseline_line = o3d.geometry.LineSet()
    baseline_line.points = o3d.utility.Vector3dVector(
        np.vstack((camera_center1, camera_center2))
    )
    baseline_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    baseline_line.paint_uniform_color([0.6, 0.1, 0.7])

    print("Loaded points:", len(points3d))
    print("Camera 1 center:", camera_center1)
    print("Camera 2 center:", camera_center2)
    print("Baseline in reconstruction units:", baseline)
    print("\nBlue camera: image 1")
    print("Orange camera: image 2")
    print("Purple line: camera baseline")
    print("Drag to rotate; scroll to zoom.")
    print("Close the window to stop the viewer.")

    geometries = [
        cloud,
        camera1,
        camera2,
        axes1,
        axes2,
        baseline_line,
    ]

    viewer = o3d.visualization.Visualizer()

    if not viewer.create_window(
        window_name="Colored two-view sparse reconstruction",
        width=1280,
        height=800,
    ):
        raise RuntimeError(
            "Could not create the Open3D window. "
            "Check whether GUI applications work in your WSL environment."
        )

    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)

        options = viewer.get_render_option()
        options.background_color = np.array([0.12, 0.12, 0.12])
        options.point_size = 7.0

        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    main()