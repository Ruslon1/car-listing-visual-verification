from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from car_listing_visual_verification.config import (
    HF_RELEASE_CLASS_MAPPING_FILE,
    MODELS_DIR,
)
from car_listing_visual_verification.modeling.predictor import Predictor

def infer(
        image_path: Path,
        checkpoint_path: Path = MODELS_DIR / "model.pkl",
        class_mapping_path: Path = HF_RELEASE_CLASS_MAPPING_FILE,
):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    predictor = Predictor(checkpoint_path, class_mapping_path)
    class_id, class_name, score = predictor.predict(image_path)

    logger.success(f"Prediction: {class_name} (class_id={class_id}, prob={score:.4f})")


def parse_args():
    parser = ArgumentParser(description="Single image inference")
    parser.add_argument("image_path", type=Path, help="Path to image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=HF_RELEASE_CLASS_MAPPING_FILE,
        help="Path to class_mapping.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        class_mapping_path=args.class_mapping,
    )
