from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_model():
    """Build the same EfficientNet-B0 architecture used in training."""
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_model(model_path: str | Path, device: torch.device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = build_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_single_image(model, image_path: str | Path, device: torch.device, threshold: float = 0.5):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze(0).item()

    pred_label = "Oil" if prob >= threshold else "No-Oil"
    pred_idx = 1 if prob >= threshold else 0

    return {
        "image": str(image_path),
        "probability": round(prob, 6),
        "predicted_label": pred_label,
        "predicted_index": pred_idx,
        "threshold": threshold,
    }


def collect_images(input_path: str | Path):
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Unsupported file type: {input_path}")

    if input_path.is_dir():
        files = []
        for file in sorted(input_path.rglob("*")):
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                files.append(file)
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")
