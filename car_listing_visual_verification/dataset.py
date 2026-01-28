from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from car_listing_visual_verification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import scipy.io
import pandas as pd


app = typer.Typer()

@app.command()
def main(
    train_mat: Path = RAW_DATA_DIR / "car_devkit/devkit/cars_train_annos.mat",
    meta_mat: Path = RAW_DATA_DIR / "car_devkit/devkit/cars_meta.mat",
    output_path: Path = PROCESSED_DATA_DIR / "cars_train.csv",
):
    logger.info("Loading")
    train_annotations = scipy.io.loadmat(train_mat)["annotations"]
    meta = scipy.io.loadmat(meta_mat)["class_names"]

    rows = []
    for annotation in train_annotations[0]:
        fname = str(annotation["fname"][0])
        class_id = int(annotation["class"][0][0])
        x1 = int(annotation["bbox_x1"][0][0])
        y1 = int(annotation["bbox_y1"][0][0])
        x2 = int(annotation["bbox_x2"][0][0])
        y2 = int(annotation["bbox_y2"][0][0])

        class_index = class_id - 1
        class_name = meta[0][class_index][0]
        rows.append({
            "image_path": fname,
            "class_id": class_id,
            "class_name": class_name,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Exported to csv")

if __name__ == "__main__":
    app()