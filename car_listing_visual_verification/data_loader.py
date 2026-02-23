from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, parquet_file: Path, root_dir: Path, transform=None):
        df = pd.read_parquet(parquet_file)

        df = df[df["image_file_name"].notna()].reset_index(drop=True)

        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[int(index)]

        img_path = self.root_dir / row["image_file_name"]
        image = Image.open(img_path).convert("RGB")

        target = row["class_id"]

        target = torch.tensor(int(target), dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, target