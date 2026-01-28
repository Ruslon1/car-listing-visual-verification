from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class StanfordCarsDataset(Dataset):
    def __init__(self, csv_file: Path, root_dir: Path, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        image_name = self.root_dir / str(row["image_path"])

        image = Image.open(image_name).convert('RGB')

        bboxes = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
        image = image.crop(bboxes)

        if self.transform:
            image = self.transform(image)

        label = int(row["class_id"]) - 1

        return image, label