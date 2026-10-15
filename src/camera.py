import numpy as np

def world_to_camera(point_world, R, t):
    """
    Transform a 3D point from world coordinates
    to camera coordinates.
    """
    return R @ point_world + t

def project_point(point_3d, K):
    """
    Project a 3D point in camera coordinates onto the image plane.

    Args:
        point_3d: [X, Y, Z]
        K: 3x3 camera intrinsic matrix

    Returns:
        Pixel coordinates [u, v]
    """

    X, Y, Z = point_3d

    if Z <= 0:
        raise ValueError("Point must be in front of the camera.")

    # Perspective division
    normalized_point = np.array([
        X / Z,
        Y / Z,
        1.0
    ])

    # Apply camera intrinsics
    pixel_homogeneous = K @ normalized_point

    u = pixel_homogeneous[0]
    v = pixel_homogeneous[1]

    return np.array([u, v])


def rotation_y(angle_degrees):
    theta = np.deg2rad(angle_degrees)

    return np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0,           1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)]
    ])


def pose_to_extrinsics(R_wc, camera_center):
    """
    Convert camera pose (camera-to-world)
    to extrinsics (world-to-camera).
    """
    R_cw = R_wc.T
    t_cw = -R_cw @ camera_center

    return R_cw, t_cw


def world_to_camera(point_world, R_cw, t_cw):
    return R_cw @ point_world + t_cw


def world_to_camera_points(points_world, R_cw, t_cw):
    """
    Transform multiple points from world coordinates to camera coordinates.

    points_world: shape (N, 3)
    returns: shape (N, 3)
    """
    points_world = np.asarray(points_world, dtype=float)

    return (R_cw @ points_world.T).T + t_cw


def project_points(points_camera, K):
    """
    Project camera-space 3D points onto the image plane.

    Returns:
        pixels: shape (N, 2)
        valid: points with positive depth
    """
    points_camera = np.asarray(points_camera, dtype=float)

    depths = points_camera[:, 2]
    valid = depths > 1e-8

    # Invalid points receive NaN coordinates
    pixels = np.full((len(points_camera), 2), np.nan)

    # Project only points located in front of the camera
    homogeneous_pixels = (K @ points_camera[valid].T).T

    pixels[valid] = (
        homogeneous_pixels[:, :2]
        / homogeneous_pixels[:, 2:3]
    )

    return pixels, valid


if __name__ == "__main__":
    # -------------------------
    # Camera intrinsics
    # -------------------------

    fx = 800
    fy = 800
    cx = 640
    cy = 360

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=float)


    # -------------------------
    # Camera extrinsics
    # -------------------------

    R = np.eye(3)

    t = np.array([
        -1.0,
        0.0,
        0.0
    ])


    # -------------------------
    # World point
    # -------------------------

    point_world = np.array([
        2.0,
        1.0,
        10.0
    ])


    point_camera = world_to_camera(
        point_world,
        R,
        t
    )

    pixel = project_point(
        point_camera,
        K
    )


    print("World point:")
    print(point_world)

    print("\nCamera point:")
    print(point_camera)

    print("\nPixel:")
    print(pixel)


    C = np.array([0.0, 0.0, 0.0])
    P_w = np.array([10.0, 0.0, 0.0])

    R_wc = rotation_y(90)
    R_cw, t_cw = pose_to_extrinsics(R_wc, C)

    P_c = world_to_camera(P_w, R_cw, t_cw)

    print("R_cw:\n", np.round(R_cw, decimals=6))
    print("t_cw:", np.round(t_cw, decimals=6))
    print("Camera point:", np.round(P_c, decimals=6))
    pass