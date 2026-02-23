from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch import nn
from torchvision import models

from car_listing_visual_verification.config import (
    HF_RELEASE_CLASS_MAPPING_FILE,
    MODELS_DIR,
)
from car_listing_visual_verification.train import device, val_transforms


def infer(
    image_path: Path,
    checkpoint_path: Path = MODELS_DIR / "model.pkl",
    class_mapping_path: Path = HF_RELEASE_CLASS_MAPPING_FILE,
):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    class_mapping_df = pd.read_parquet(class_mapping_path).sort_values("class_id")
    class_mapping = dict(
        zip(
            class_mapping_df["class_id"].astype(int).tolist(),
            class_mapping_df["class_name"].astype(str).tolist(),
        )
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 99)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    class_id = int(pred_class.item())
    class_name = class_mapping.get(class_id, str(class_id))
    score = float(confidence.item())

    logger.success(f"Prediction: {class_name} (class_id={class_id}, prob={score:.4f})")


def parse_args():
    parser = ArgumentParser(description="Single image inference")
    parser.add_argument("image_path", type=Path, help="Path to image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=HF_RELEASE_CLASS_MAPPING_FILE,
        help="Path to class_mapping.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        class_mapping_path=args.class_mapping,
    )
