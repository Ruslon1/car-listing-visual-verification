from pathlib import Path
import torch
from loguru import logger
from tqdm import tqdm
import typer
import torch.nn as nn
import torch.optim as optim
from car_listing_visual_verification.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
import car_listing_visual_verification.data_loader as data_loader
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import models

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

app = typer.Typer()


@app.command()
def main(
        train_path: Path = PROCESSED_DATA_DIR / "train.csv",
        images_dir: Path = RAW_DATA_DIR / "cars_train/cars_train",
        validation_path: Path = PROCESSED_DATA_DIR / "val.csv",
):
    train_dataset = data_loader.StanfordCarsDataset(train_path, images_dir, train_transforms)
    val_dataset = data_loader.StanfordCarsDataset(validation_path, images_dir, val_transforms)

    logger.info("Init dataloaders...")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('mps')

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 196)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    num_epochs = 25
    unfreeze_epoch = 5

    logger.info("Starting training pipeline...")

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            logger.warning("unfreezing")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

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

    logger.success("Training Complete")


if __name__ == "__main__":
    app()