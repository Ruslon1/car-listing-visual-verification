from pathlib import Path
import scipy.io
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from car_listing_visual_verification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
        train_mat: Path = RAW_DATA_DIR / "car_devkit/devkit/cars_train_annos.mat",
        meta_mat: Path = RAW_DATA_DIR / "car_devkit/devkit/cars_meta.mat",
        train_output: Path = PROCESSED_DATA_DIR / "train.csv",
        val_output: Path = PROCESSED_DATA_DIR / "val.csv",
        split_ratio: float = 0.1,
        seed: int = 42
):
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

    logger.info(f"Splitting data")

    train_df, val_df = train_test_split(
        df,
        test_size=split_ratio,
        random_state=seed,
        stratify=df["class_id"]
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

if __name__ == "__main__":
    app()