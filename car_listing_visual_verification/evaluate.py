from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from car_listing_visual_verification import data_loader
from car_listing_visual_verification.config import (
    HF_RELEASE_TEST_FILE,
    HF_RELEASE_CLASS_MAPPING_FILE,
    HF_RELEASE_DIR,
    MODELS_DIR,
)
from car_listing_visual_verification.train import device, val_transforms


def test(
    test_path: Path = HF_RELEASE_TEST_FILE,
    root_dir: Path = HF_RELEASE_DIR,
    checkpoint_path: Path = MODELS_DIR / "model.pkl",
):
    test_dataset = data_loader.CarDataset(test_path, root_dir, val_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 99)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(99)))

    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

    class_map = pd.read_parquet(HF_RELEASE_CLASS_MAPPING_FILE).sort_values("class_id")
    class_names = class_map["class_name"].tolist()

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Test Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "test_confusion_matrix.png", dpi=200)
    plt.close()

    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    logger.success(f"Test Acc: {test_acc:.4f}")
    logger.success(f"Macro Precision: {precision_macro:.4f}")
    logger.success(f"Macro Recall:    {recall_macro:.4f}")
    logger.success(f"Macro F1:        {f1_macro:.4f}")

if __name__ == '__main__':
    test()