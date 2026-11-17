
import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

from gpu_utils import resolve_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_test_directory(data_yaml):
    data_yaml = Path(data_yaml).resolve()

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    root = Path(config.get("path", data_yaml.parent))

    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    test_path = Path(config["test"])

    if not test_path.is_absolute():
        test_path = root / test_path

    return test_path.resolve()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="runs/detect/train/weights/best.pt"
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml"
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    test_dir = get_test_directory(args.data)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    images = [
        p for p in test_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError("No test images found.")

    selected = random.sample(
        images,
        min(args.samples, len(images))
    )

    print(f"Test images available: {len(images)}")
    print(f"Selected images: {len(selected)}")

    device = resolve_device(args.device)
    model = YOLO(str(model_path))

    output_dir = Path("runs/test_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=[str(p) for p in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        half=device != "cpu",
        save=False,
        verbose=False
    )

    fig, axes = plt.subplots(
        2, 5, figsize=(20, 9)
    )
    axes = axes.flatten()

    for index, (image_path, result) in enumerate(
        zip(selected, results)
    ):
        # Ultralytics returns an annotated BGR image
        annotated = result.plot()

        # Save individual prediction image
        output_path = output_dir / (
            image_path.stem + "_prediction.jpg"
        )

        cv2.imwrite(str(output_path), annotated)

        # Convert BGR to RGB for Matplotlib
        annotated_rgb = cv2.cvtColor(
            annotated, cv2.COLOR_BGR2RGB
        )

        axes[index].imshow(annotated_rgb)
        axes[index].set_title(image_path.name)
        axes[index].axis("off")

        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0

        print(
            f"{image_path.name}: "
            f"{count} detections"
        )

    # Hide unused plots if fewer than requested
    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()

    contact_sheet = output_dir / "test_results.png"
    plt.savefig(
        contact_sheet,
        dpi=150,
        bbox_inches="tight"
    )

    print("\nTesting completed!")
    print(f"Individual images: {output_dir}")
    print(f"Contact sheet: {contact_sheet}")

    plt.show()


if __name__ == "__main__":
    main()
