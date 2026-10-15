from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIRECTORY = PROJECT_ROOT / "data" / "images"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

OUTPUT_DIRECTORY.mkdir(exist_ok=True)


def detect_sift_features(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image_gray is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Create the SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=2000
    )

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    return image_gray, keypoints, descriptors


def save_keypoint_visualization(
    image_gray,
    keypoints,
    output_path
):
    visualization = cv2.drawKeypoints(
        image_gray,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(
        str(output_path),
        visualization
    )


def main():
    image_paths = [
        IMAGE_DIRECTORY / "image1.jpg",
        IMAGE_DIRECTORY / "image2.jpg",
    ]

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):
        image, keypoints, descriptors = (
            detect_sift_features(image_path)
        )

        output_path = (
            OUTPUT_DIRECTORY
            / f"sift_keypoints_{index}.jpg"
        )

        save_keypoint_visualization(
            image,
            keypoints,
            output_path
        )

        print(f"\nImage {index}")
        print(f"Image shape: {image.shape}")
        print(f"Number of keypoints: {len(keypoints)}")

        if descriptors is not None:
            print(
                f"Descriptor shape: "
                f"{descriptors.shape}"
            )

        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()