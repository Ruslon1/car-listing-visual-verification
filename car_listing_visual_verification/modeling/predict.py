from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import typer
from loguru import logger

from car_listing_visual_verification.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_class_names(train_csv_path: Path):
    df = pd.read_csv(train_csv_path)
    id_to_name = pd.Series(df.class_name.values, index=df.class_id).to_dict()
    return id_to_name


def load_model(model_path: Path, device: torch.device):
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 196)

    logger.info(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


@app.command()
def main(
        image_path: Path,
        model_path: Path = MODELS_DIR / "model.pkl",
        train_csv: Path = PROCESSED_DATA_DIR / "train.csv"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    classes = load_class_names(train_csv)
    model = load_model(model_path, device)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    logger.info(f"Predicting for: {image_path.name}")
    img = Image.open(image_path).convert("RGB")

    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)

        logger.success("Prediction Results:")
        for i in range(3):
            prob = top_probs[0][i].item() * 100
            class_idx = top_indices[0][i].item()

            class_id = class_idx + 1
            car_name = classes.get(class_id, "Unknown")

            print(f"{i + 1}. {car_name} ({prob:.2f}%)")


if __name__ == "__main__":
    app()