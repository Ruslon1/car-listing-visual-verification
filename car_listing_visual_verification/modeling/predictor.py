from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from car_listing_visual_verification.config import (
    HF_RELEASE_CLASS_MAPPING_FILE,
    MODELS_DIR,
)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


class Predictor:
    def __init__(self, checkpoint_path: Path = MODELS_DIR / "model.pkl",
                 class_mapping_path: Path = HF_RELEASE_CLASS_MAPPING_FILE,
                 image_transforms: transforms.Compose = inference_transforms, device=None):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.class_mapping_path = class_mapping_path
        self.image_transforms = image_transforms

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        class_mapping_df = pd.read_parquet(class_mapping_path).sort_values("class_id")
        self.class_mapping = dict(
            zip(
                class_mapping_df["class_id"].astype(int).tolist(),
                class_mapping_df["class_name"].astype(str).tolist(),
            )
        )

        self.load_model()

    def load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 99)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: Path,):
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        class_id = int(pred_class.item())
        class_name = self.class_mapping.get(class_id, str(class_id))
        score = float(confidence.item())

        return class_id, class_name, score