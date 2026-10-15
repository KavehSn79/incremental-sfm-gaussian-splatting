from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam


def main():
    parser = ArgumentParser(description="Render an interpolated camera-path video.")

    model_parameters = ModelParams(parser, sentinel=True)
    pipeline_parameters = PipelineParams(parser)

    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--path_images",
        nargs="+",
        default=["image1", "image2", "image3"],
        help="Image stems defining the camera path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output MP4 filename.",
    )

    args = get_combined_args(parser)

    if args.seconds <= 0 or args.fps < 1:
        raise ValueError("Duration and frame rate must be positive.")

    if len(args.path_images) < 2:
        raise ValueError("Choose at least two cameras.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render this model.")

    dataset = model_parameters.extract(args)
    pipeline = pipeline_parameters.extract(args)

    # Keep reference photos in CPU memory.
    dataset.data_device = "cpu"

    if getattr(dataset, "train_test_exp", False):
        raise ValueError(
            "This script expects a model trained without train_test_exp."
        )

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path(dataset.model_path) / "camera_path.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
        )

        # Include the held-out camera when building the path.
        cameras = scene.getTrainCameras() + scene.getTestCameras()

        cameras_by_name = {}
        for camera in cameras:
            name = Path(camera.image_name).stem

            if name in cameras_by_name:
                raise ValueError(f"Duplicate camera name: {name}")

            cameras_by_name[name] = camera

        selected_cameras = []

        for name in args.path_images:
            stem = Path(name).stem

            if stem not in cameras_by_name:
                raise ValueError(
                    f"Camera {stem!r} was not found. "
                    f"Available: {sorted(cameras_by_name)}"
                )

            selected_cameras.append(cameras_by_name[stem])

        reference_camera = selected_cameras[0]
        width = reference_camera.image_width
        height = reference_camera.image_height

        for camera in selected_cameras:
            if (
                camera.image_width != width
                or camera.image_height != height
                or not np.isclose(camera.FoVx, reference_camera.FoVx)
                or not np.isclose(camera.FoVy, reference_camera.FoVy)
            ):
                raise ValueError(
                    "The selected cameras must share resolution and field of view."
                )

        # Recover conventional camera-to-world matrices.
        # The renderer stores world-to-camera matrices transposed.
        centers = []
        rotations_wc = []

        for camera in selected_cameras:
            transform_cw = (
                camera.world_view_transform
                .detach()
                .cpu()
                .numpy()
                .T
                .astype(np.float64)
            )

            transform_wc = np.linalg.inv(transform_cw)

            centers.append(transform_wc[:3, 3])
            rotations_wc.append(transform_wc[:3, :3])

        centers = np.asarray(centers)
        key_times = np.arange(len(selected_cameras), dtype=float)

        rotation_interpolator = Slerp(
            key_times,
            Rotation.from_matrix(np.asarray(rotations_wc)),
        )

        background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

        # Padding preserves the rendered image while giving the video
        # encoder even dimensions.
        video_width = width + width % 2
        video_height = height + height % 2

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (video_width, video_height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

        frame_count = max(2, round(args.seconds * args.fps))
        last_key_time = key_times[-1]
        projection = reference_camera.projection_matrix

        print("Camera path:", " -> ".join(args.path_images))
        print("Frames:", frame_count)
        print("Rendered dimensions:", (width, height))
        print("Trained Gaussians:", len(gaussians.get_xyz))

        try:
            for frame_index in tqdm(range(frame_count), desc="Rendering video"):
                phase = frame_index / (frame_count - 1)

                # An eased forward-and-back sweep through the path.
                path_time = (
                    0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
                    * last_key_time
                )
                path_time = float(np.clip(path_time, 0.0, last_key_time))

                segment = min(
                    int(np.floor(path_time)),
                    len(selected_cameras) - 2,
                )
                fraction = path_time - segment

                center = (
                    (1.0 - fraction) * centers[segment]
                    + fraction * centers[segment + 1]
                )

                R_wc = rotation_interpolator([path_time]).as_matrix()[0]
                R_cw = R_wc.T
                t_cw = -R_cw @ center

                transform_cw = np.eye(4, dtype=np.float32)
                transform_cw[:3, :3] = R_cw
                transform_cw[:3, 3] = t_cw

                world_view = (
                    torch.from_numpy(transform_cw)
                    .to(device="cuda")
                    .transpose(0, 1)
                    .contiguous()
                )

                camera = MiniCam(
                    width,
                    height,
                    reference_camera.FoVy,
                    reference_camera.FoVx,
                    reference_camera.znear,
                    reference_camera.zfar,
                    world_view,
                    world_view @ projection,
                )

                rendered = render(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                )["render"]

                rgb = (
                    rendered.clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if video_width != width or video_height != height:
                    bgr = cv2.copyMakeBorder(
                        bgr,
                        0,
                        video_height - height,
                        0,
                        video_width - width,
                        cv2.BORDER_REPLICATE,
                    )

                writer.write(bgr)

        finally:
            writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("The video was not saved successfully.")

    # Verify that the saved video can be decoded.
    capture = cv2.VideoCapture(str(output_path))
    readable, _ = capture.read()
    capture.release()

    if not readable:
        raise RuntimeError(f"Could not decode the saved video: {output_path}")

    print(f"\nSaved video: {output_path}")


if __name__ == "__main__":
    main()