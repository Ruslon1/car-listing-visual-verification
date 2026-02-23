import data_loader

from torchvision.transforms import transforms
from torchvision import models

from car_listing_visual_verification.config import PROCESSED_DATA_DIR


def train(train_path: Path = PROCESSED_DATA_DIR / "train.csv",
        images_dir: Path = RAW_DATA_DIR / "cars_train/cars_train",
        validation_path: Path = PROCESSED_DATA_DIR / "val.csv",
epoches=10,
):
    return None