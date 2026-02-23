from pathlib import Path

import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from car_listing_visual_verification import data_loader

from torchvision.transforms import transforms
from torchvision import models

from car_listing_visual_verification.config import HF_RELEASE_TRAIN_FILE, HF_RELEASE_VALIDATION_FILE, HF_RELEASE_DIR, \
    MODELS_DIR

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def train(train_path: Path = HF_RELEASE_TRAIN_FILE,
          validation_path: Path = HF_RELEASE_VALIDATION_FILE,
          root_dir: Path = HF_RELEASE_DIR,
          epoches=25,
          ):
    train_dataset = data_loader.CarDataset(train_path, root_dir, train_transforms)
    val_dataset = data_loader.CarDataset(validation_path, root_dir, val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 99)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    adam = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='max', factor=0.1, patience=2)
    unfreeze_epoch = 4

    for epoch in range(epoches):
        if epoch == unfreeze_epoch:
            logger.warning("unfreezing")

            for param in model.parameters():
                param.requires_grad = True

            adam = optim.Adam(model.parameters(), lr=1e-4)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='max', factor=0.1, patience=2)

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epoches}")

        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            adam.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            adam.step()

            current_batch_size = inputs.size(0)
            running_loss += loss.item() * current_batch_size
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += current_batch_size

            progress.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        logger.info(f"Train - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        logger.success(f"Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        torch.save(model.state_dict(), MODELS_DIR / "model.pkl")


if __name__ == '__main__':
    train()