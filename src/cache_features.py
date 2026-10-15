import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from sift_features import detect_sift_features


def cache_signature(image_path):
    """Invalidate the cache when the image or detector implementation changes."""
    detector_path = Path(__file__).resolve().parent / "sift_features.py"

    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(detector_path.read_bytes())
    digest.update(cv2.__version__.encode())

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("cache_path", type=Path)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    print(f"Extracting features: {args.image_path.name}", flush=True)

    image, keypoints, descriptors = detect_sift_features(
        args.image_path
    )

    if descriptors is None or len(descriptors) < 2:
        raise RuntimeError("Insufficient SIFT descriptors.")

    pixels = np.array(
        [keypoint.pt for keypoint in keypoints],
        dtype=float,
    )
    image_shape = np.array(image.shape[:2], dtype=np.int64)

    # Do not save the full-resolution image.
    del image

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.cache_path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        np.savez(
            file,
            descriptors=descriptors,
            pixels=pixels,
            image_shape=image_shape,
            signature=cache_signature(args.image_path),
        )

    temporary_path.replace(args.cache_path)

    print(
        f"Cached {len(keypoints)} features: {args.image_path.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()