import math
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

        x1, y1, x2, y2 = (
            row["car_bbox_x1"],
            row["car_bbox_y1"],
            row["car_bbox_x2"],
            row["car_bbox_y2"],
        )

        if all(math.isfinite(float(v)) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
            w, h = image.size
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))
            image = image.crop((x1, y1, x2, y2))

        target = row["class_id"]

        target = torch.tensor(int(target), dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, target