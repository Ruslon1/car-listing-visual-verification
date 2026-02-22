from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from car_listing_visual_verification.config import PROJ_ROOT
from car_listing_visual_verification.data.drom_utils import read_table, write_table

SUPPORTED_FILE_MODES = {"hardlink", "copy", "symlink"}
HF_SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def prepare_hf_release(
    manifest_path: Path,
    class_mapping_path: Path,
    output_dir: Path,
    dataset_name: str,
    dataset_id: str | None = None,
    license_id: str = "other",
    agreement_note: str = (
        "Publication and usage are allowed only under an explicit agreement with drom.ru."
    ),
    file_mode: str = "hardlink",
    force: bool = False,
) -> dict[str, Any]:
    mode = file_mode.strip().lower()
    if mode not in SUPPORTED_FILE_MODES:
        allowed = ", ".join(sorted(SUPPORTED_FILE_MODES))
        raise ValueError(f"Unsupported file_mode={file_mode!r}; expected one of: {allowed}")

    manifest = read_table(manifest_path, required=True)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    required_columns = {
        "listing_id",
        "class_id",
        "class_name",
        "split",
        "image_1_path",
        "image_2_path",
    }
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")

    output_dir = output_dir.resolve()
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_records: list[dict[str, Any]] = []
    listing_records: list[dict[str, Any]] = []

    images_created = 0
    images_reused = 0
    missing_source_images = 0

    for row in manifest.to_dict(orient="records"):
        listing_id = str(row.get("listing_id"))
        class_id = _to_int(row.get("class_id"))
        class_name = _normalize_class_name(row.get("class_name"), class_id=class_id)
        split_original = _to_str(row.get("split")) or "train"
        split_hf = _normalize_split(split_original)

        listing_row = dict(row)
        listing_row["split_original"] = split_original
        listing_row["split_hf"] = split_hf
        listing_row["image_1_file_name"] = None
        listing_row["image_2_file_name"] = None

        for image_slot in (1, 2):
            source_value = row.get(f"image_{image_slot}_path")
            source_path = _resolve_source_path(source_value)
            if source_path is None or not source_path.exists():
                missing_source_images += 1
                continue

            destination_relative = (
                Path("images") / split_hf / class_name / f"{listing_id}_{image_slot}.jpg"
            )
            destination_path = output_dir / destination_relative
            status = _materialize_image(
                source_path=source_path,
                destination_path=destination_path,
                mode=mode,
            )
            if status == "created":
                images_created += 1
            else:
                images_reused += 1

            destination_value = destination_relative.as_posix()
            listing_row[f"image_{image_slot}_file_name"] = destination_value

            image_records.append(
                {
                    "listing_id": listing_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "make": row.get("make"),
                    "model": row.get("model"),
                    "generation": row.get("generation"),
                    "body_type": row.get("body_type"),
                    "year": row.get("year"),
                    "source": row.get("source"),
                    "url": row.get("url"),
                    "scraped_at": row.get("scraped_at"),
                    "split": split_hf,
                    "split_original": split_original,
                    "image_slot": image_slot,
                    "image_file_name": destination_value,
                    "image_phash": row.get(f"image_{image_slot}_phash"),
                    "image_width": row.get(f"image_{image_slot}_width"),
                    "image_height": row.get(f"image_{image_slot}_height"),
                    "image_bytes": row.get(f"image_{image_slot}_bytes"),
                    "image_format": row.get(f"image_{image_slot}_format"),
                    "car_detected": row.get(f"image_{image_slot}_car_detected"),
                    "car_conf": row.get(f"image_{image_slot}_car_conf"),
                    "car_bbox_x1": row.get(f"image_{image_slot}_car_bbox_x1"),
                    "car_bbox_y1": row.get(f"image_{image_slot}_car_bbox_y1"),
                    "car_bbox_x2": row.get(f"image_{image_slot}_car_bbox_x2"),
                    "car_bbox_y2": row.get(f"image_{image_slot}_car_bbox_y2"),
                    "car_bbox_area_ratio": row.get(f"image_{image_slot}_car_bbox_area_ratio"),
                    "exterior_score": row.get(f"image_{image_slot}_exterior_score"),
                    "interior_score": row.get(f"image_{image_slot}_interior_score"),
                    "content_label": row.get(f"image_{image_slot}_content_label"),
                    "content_keep": row.get(f"image_{image_slot}_content_keep"),
                    "content_reason": row.get(f"image_{image_slot}_content_reason"),
                    "content_error": row.get(f"image_{image_slot}_content_error"),
                    "http_status": row.get("http_status"),
                    "parse_version": row.get("parse_version"),
                }
            )

        listing_records.append(listing_row)

    images_df = pd.DataFrame(image_records)
    if images_df.empty:
        raise ValueError("HF export produced zero image rows; check manifest image paths")

    images_df = images_df.sort_values(
        by=["split", "class_id", "listing_id", "image_slot"],
        kind="stable",
    ).reset_index(drop=True)

    listings_df = pd.DataFrame(listing_records)
    listings_df = listings_df.sort_values(
        by=["class_id", "listing_id"],
        kind="stable",
    ).reset_index(drop=True)

    write_table(images_df, output_dir / "metadata.parquet")
    write_table(images_df[images_df["split"] == "train"], output_dir / "train.parquet")
    write_table(images_df[images_df["split"] == "validation"], output_dir / "validation.parquet")
    write_table(images_df[images_df["split"] == "test"], output_dir / "test.parquet")
    write_table(listings_df, output_dir / "listings.parquet")

    if class_mapping_path.exists():
        class_mapping = read_table(class_mapping_path, required=False)
        if not class_mapping.empty:
            write_table(class_mapping, output_dir / "class_mapping.parquet")

    split_counts = images_df["split"].value_counts(dropna=False).sort_index().astype(int).to_dict()
    class_counts = (
        images_df["class_name"].value_counts(dropna=False).sort_values(ascending=False).to_dict()
    )
    listing_count = int(listings_df["listing_id"].nunique())
    class_count = int(images_df["class_name"].nunique())
    image_count = int(len(images_df))

    card_content = render_dataset_card(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        license_id=license_id,
        agreement_note=agreement_note,
        image_count=image_count,
        listing_count=listing_count,
        class_count=class_count,
        split_counts=split_counts,
        class_counts=class_counts,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "README.md").write_text(card_content, encoding="utf-8")

    stats_payload = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "manifest_path": manifest_path.resolve().as_posix(),
        "class_mapping_path": class_mapping_path.resolve().as_posix(),
        "output_dir": output_dir.as_posix(),
        "image_count": image_count,
        "listing_count": listing_count,
        "class_count": class_count,
        "split_counts": split_counts,
        "images_created": images_created,
        "images_reused": images_reused,
        "missing_source_images": missing_source_images,
        "file_mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "release_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return stats_payload


def render_dataset_card(
    dataset_name: str,
    dataset_id: str | None,
    license_id: str,
    agreement_note: str,
    image_count: int,
    listing_count: int,
    class_count: int,
    split_counts: dict[str, int],
    class_counts: dict[str, int],
    generated_at: str,
) -> str:
    top_classes = list(class_counts.items())[:20]
    size_category = _size_category(image_count)

    split_train = split_counts.get("train", 0)
    split_val = split_counts.get("validation", 0)
    split_test = split_counts.get("test", 0)

    table_lines = ["| class_name | images |", "|---|---:|"]
    for class_name, count in top_classes:
        table_lines.append(f"| {class_name} | {count} |")
    top_classes_table = "\n".join(table_lines)

    repo_line = f"- **Dataset repo**: `{dataset_id}`" if dataset_id else ""

    return f"""---
pretty_name: "{dataset_name}"
license: {license_id}
language:
- ru
task_categories:
- image-classification
tags:
- cars
- computer-vision
- drom
size_categories:
- {size_category}
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# {dataset_name}

ML-ready dataset of authorized car listing images collected from drom.ru under an explicit agreement.

## Compliance

- {agreement_note}
- No bypass/evasion collection methods were used.

## Summary

- **Generated at**: `{generated_at}`
- **Listings**: `{listing_count}`
- **Images**: `{image_count}`
- **Classes**: `{class_count}`
{repo_line}

## Splits

- `train`: {split_train}
- `validation`: {split_val}
- `test`: {split_test}

## Files

- `train.parquet`, `validation.parquet`, `test.parquet`: image-level rows (one image per row).
- `metadata.parquet`: full image-level table across all splits.
- `listings.parquet`: listing-level table (two image references per listing).
- `class_mapping.parquet`: class mapping (`class_id` -> `class_name`) if available.
- `images/<split>/<class_name>/<listing_id>_<slot>.jpg`: image files.

## Image-Level Columns

- `image_file_name` relative path to image file inside this dataset repo
- `class_id`, `class_name`, `make`, `model`, `generation`, `body_type`, `year`
- `listing_id`, `source`, `url`, `scraped_at`
- `split`, `split_original`
- `image_slot` (`1` or `2`)
- `image_phash`, `image_width`, `image_height`, `image_bytes`, `image_format`
- `car_detected`, `car_conf`, `car_bbox_x1`, `car_bbox_y1`, `car_bbox_x2`, `car_bbox_y2`, `car_bbox_area_ratio`
- `exterior_score`, `interior_score`, `content_label`, `content_keep`, `content_reason`, `content_error`

## Class Distribution (Top 20 by image count)

{top_classes_table}

## Notes

- This dataset is intended for visual verification/classification experiments.
- Ensure your usage respects the same source agreement constraints.
"""


def _materialize_image(source_path: Path, destination_path: Path, mode: str) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination_path):
        return "reused"

    if mode == "copy":
        shutil.copy2(source_path, destination_path)
        return "created"

    if mode == "symlink":
        os.symlink(source_path.resolve(), destination_path)
        return "created"

    try:
        os.link(source_path, destination_path)
        return "created"
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "created"


def _resolve_source_path(value: Any) -> Path | None:
    raw = _to_str(value)
    if raw is None:
        return None

    source_path = Path(raw)
    if not source_path.is_absolute():
        source_path = PROJ_ROOT / source_path
    return source_path


def _normalize_split(value: str) -> str:
    split = value.strip().lower()
    return HF_SPLIT_MAP.get(split, "train")


def _normalize_class_name(value: Any, class_id: int | None) -> str:
    base = _to_str(value)
    if not base:
        return f"class_{class_id or 0}"
    return base.replace("/", "_").strip()


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _size_category(count: int) -> str:
    if count < 1_000:
        return "n<1K"
    if count < 10_000:
        return "1K<n<10K"
    if count < 100_000:
        return "10K<n<100K"
    if count < 1_000_000:
        return "100K<n<1M"
    return "n>1M"
