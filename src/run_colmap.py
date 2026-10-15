import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def command_help(executable, command, environment):
    result = subprocess.run(
        [executable, command, "-h"],
        capture_output=True,
        text=True,
        env=environment,
    )
    return result.stdout + result.stderr


def supported_option(help_text, candidates):
    """Choose the option name supported by the installed COLMAP version."""
    for option in candidates:
        if f"--{option}" in help_text:
            return f"--{option}"

    raise RuntimeError(
        f"None of these COLMAP options are available: {candidates}"
    )


def run_command(arguments, log_path, environment):
    """Display progress and save the complete command output."""
    print("\nRunning:", " ".join(map(str, arguments)), flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(map(str, arguments)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"COLMAP exited with code {return_code}. See {log_path}"
        )


def count_registered_images(images_path):
    """Count image records in COLMAP's two-line-per-image text format."""
    count = 0

    with images_path.open(encoding="utf-8") as file:
        while True:
            line = file.readline()

            if not line:
                break

            if not line.strip() or line.lstrip().startswith("#"):
                continue

            # First line: image ID, pose, camera ID, and filename.
            fields = line.split()

            if len(fields) < 10:
                raise RuntimeError("Unexpected COLMAP images.txt format.")

            count += 1

            # Second line: feature observations; it may be empty.
            file.readline()

    return count


def main():
    cv2.setNumThreads(1)

    project_root = Path(__file__).resolve().parent.parent
    original_directory = project_root / "data" / "images"

    # Dedicated output directory for this fixed-intrinsics experiment.
    work_directory = project_root / "outputs" / "colmap_fixed_k"
    image_directory = work_directory / "images"
    sparse_directory = work_directory / "sparse"
    logs_directory = work_directory / "logs"

    for directory in (
        work_directory,
        image_directory,
        sparse_directory,
        logs_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("colmap")

    if executable is None:
        raise RuntimeError("COLMAP is not installed or is not on PATH.")

    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"

    reconstruction_path = (
        project_root / "outputs" / "global_bundle_adjusted.npz"
    )

    with np.load(reconstruction_path, allow_pickle=False) as data:
        K = data["K"]
        expected_shape = tuple(data["image_shape"].tolist())
        our_names = data["image_names"].tolist()
        our_point_count = len(data["points3d"])

    if len(our_names) != 8:
        raise RuntimeError("Expected our eight-image reconstruction.")

    # Write the same oriented pixels as lossless PNGs.
    # This avoids different EXIF-orientation handling between readers.
    for name in our_names:
        source = original_directory / name
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not load {source}")

        if image.shape[:2] != expected_shape:
            raise ValueError(f"Unexpected dimensions for {name}")

        destination = image_directory / f"{Path(name).stem}.png"

        if not cv2.imwrite(str(destination), image):
            raise IOError(f"Could not save {destination}")

        del image

    database_path = work_directory / "database.db"

    extraction_help = command_help(
        executable, "feature_extractor", environment
    )
    matching_help = command_help(
        executable, "exhaustive_matcher", environment
    )
    mapper_help = command_help(
        executable, "mapper", environment
    )

    extraction_gpu = supported_option(
        extraction_help,
        ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"],
    )
    extraction_threads = supported_option(
        extraction_help,
        ["FeatureExtraction.num_threads", "SiftExtraction.num_threads"],
    )
    extraction_size = supported_option(
        extraction_help,
        ["FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"],
    )
    matching_gpu = supported_option(
        matching_help,
        ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"],
    )

    # PINHOLE parameter order: fx, fy, cx, cy.
    camera_parameters = ",".join(
        str(value)
        for value in (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    )

    extraction_command = [
        executable,
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_directory,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", camera_parameters,
        extraction_gpu, "0",
        extraction_threads, "1",
        extraction_size, "3200",
        "--SiftExtraction.max_num_features", "2000",
        "--SiftExtraction.first_octave", "0",
    ]

    run_command(
        extraction_command,
        logs_directory / "feature_extraction.log",
        environment,
    )

    matching_command = [
        executable,
        "exhaustive_matcher",
        "--database_path", database_path,
        matching_gpu, "0",
    ]

    for option in (
        "FeatureMatching.num_threads",
        "SiftMatching.num_threads",
    ):
        if f"--{option}" in matching_help:
            matching_command.extend([f"--{option}", "1"])
            break

    run_command(
        matching_command,
        logs_directory / "matching.log",
        environment,
    )

    mapper_command = [
        executable,
        "mapper",
        "--database_path", database_path,
        "--image_path", image_directory,
        "--output_path", sparse_directory,
    ]

    # Match our fixed-intrinsics bundle-adjustment assumptions.
    for option in (
        "Mapper.ba_refine_focal_length",
        "Mapper.ba_refine_principal_point",
        "Mapper.ba_refine_extra_params",
    ):
        if f"--{option}" not in mapper_help:
            raise RuntimeError(
                f"Installed COLMAP does not expose --{option}. "
                "Send the output of colmap mapper -h."
            )

        mapper_command.extend([f"--{option}", "0"])

    if "--Mapper.num_threads" in mapper_help:
        mapper_command.extend(["--Mapper.num_threads", "1"])

    run_command(
        mapper_command,
        logs_directory / "mapping.log",
        environment,
    )

    # COLMAP may produce multiple disconnected reconstructions.
    model_directories = sorted(
        directory
        for directory in sparse_directory.iterdir()
        if directory.is_dir()
        and (directory / "images.bin").is_file()
    )

    if not model_directories:
        raise RuntimeError(
            "COLMAP did not produce a sparse reconstruction. "
            "Inspect mapping.log."
        )

    model_records = []

    for model_directory in model_directories:
        text_directory = work_directory / f"text_{model_directory.name}"
        text_directory.mkdir(exist_ok=True)

        run_command(
            [
                executable,
                "model_converter",
                "--input_path", model_directory,
                "--output_path", text_directory,
                "--output_type", "TXT",
            ],
            logs_directory / f"convert_{model_directory.name}.log",
            environment,
        )

        registered_count = count_registered_images(
            text_directory / "images.txt"
        )

        model_records.append(
            (registered_count, model_directory, text_directory)
        )

    registered_count, best_model, best_text = max(
        model_records,
        key=lambda record: record[0],
    )

    ply_path = work_directory / "sparse_cloud.ply"

    run_command(
        [
            executable,
            "model_converter",
            "--input_path", best_model,
            "--output_path", ply_path,
            "--output_type", "PLY",
        ],
        logs_directory / "ply_export.log",
        environment,
    )

    run_command(
        [
            executable,
            "model_analyzer",
            "--path", best_model,
        ],
        logs_directory / "model_analysis.log",
        environment,
    )

    print("\n--- COLMAP experiment complete ---")
    print("Our registered images:", len(our_names))
    print("Our retained points:", our_point_count)
    print("COLMAP registered images in largest model:", registered_count)
    print("Number of COLMAP models:", len(model_records))
    print("Selected model:", best_model)
    print("Text model:", best_text)
    print("Point cloud:", ply_path)
    print("Analysis:", logs_directory / "model_analysis.log")


if __name__ == "__main__":
    main()