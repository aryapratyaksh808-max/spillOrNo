from pathlib import Path
import argparse

import torch

from spillguard_inference import collect_images, load_model, predict_single_image

DEFAULT_MODEL_PATH = Path("processed_dataset/best_efficientnet_b0_spillguard.pth")
DEFAULT_INPUT_PATH = Path("datasets/data/Class_1")
DEFAULT_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description="Run inference with the saved SpillGuard EfficientNet-B0 checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved .pth checkpoint")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Specific image file(s) or folder(s) to predict. Repeat this option for multiple files.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold for Oil vs No-Oil")
    args = parser.parse_args()

    if args.input is None:
        args.input = [DEFAULT_INPUT_PATH]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    collected_images = []
    for item in args.input:
        collected_images.extend(collect_images(item))

    # remove duplicates if the same file is provided multiple times
    unique_images = list(dict.fromkeys(collected_images))

    print(f"Found {len(unique_images)} image(s) to process")

    for image in unique_images:
        result = predict_single_image(model, image, device, threshold=args.threshold)
        print(result)


if __name__ == "__main__":
    main()
