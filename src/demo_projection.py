import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from camera import (
    rotation_y,
    pose_to_extrinsics,
    world_to_camera_points,
    project_points,
)


# Image size
image_width = 1280
image_height = 720


# Camera intrinsics
K = np.array([
    [800.0, 0.0, 640.0],
    [0.0, 800.0, 360.0],
    [0.0, 0.0, 1.0],
])


# Multiple world-space points
points_world = np.array([
    [0.0,  0.0, 10.0],
    [2.0,  1.0, 10.0],
    [-2.0, -1.0, 10.0],
])


# Camera pose
camera_center = np.array([0.0, 0.0, 0.0])
R_wc = rotation_y(0)

# Camera pose -> world-to-camera extrinsics
R_cw, t_cw = pose_to_extrinsics(
    R_wc,
    camera_center
)


# World -> Camera
points_camera = world_to_camera_points(
    points_world,
    R_cw,
    t_cw
)


# Camera -> Image
pixels, valid = project_points(
    points_camera,
    K
)


print("World points:\n", points_world)
print("\nCamera points:\n", points_camera)
print("\nPixels:\n", pixels)
print("\nValid points:", valid)


# Check whether projected points are inside the image
inside_image = (
    valid
    & (pixels[:, 0] >= 0)
    & (pixels[:, 0] < image_width)
    & (pixels[:, 1] >= 0)
    & (pixels[:, 1] < image_height)
)


# Visualization
plt.figure(figsize=(10, 6))

plt.scatter(
    pixels[inside_image, 0],
    pixels[inside_image, 1],
    color="red",
    s=70
)

for index, pixel in enumerate(pixels):
    if inside_image[index]:
        plt.annotate(
            f"P{index + 1}",
            (pixel[0] + 10, pixel[1] - 10)
        )

# Draw the principal point
plt.scatter(
    K[0, 2],
    K[1, 2],
    color="blue",
    marker="+",
    s=150,
    label="Principal point"
)

plt.xlim(0, image_width)
plt.ylim(image_height, 0)  # Image y-axis points downward

plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.title("Projection of 3D Points")
plt.grid()
plt.legend()
plt.gca().set_aspect("equal")

output_directory = (
    Path(__file__).resolve().parent.parent / "outputs"
)
output_directory.mkdir(exist_ok=True)

output_path = output_directory / "projection.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to: {output_path}")