from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from car_listing_visual_verification.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

DEFAULT_USER_AGENT = (
    "car-listing-visual-verification/1.0 "
    "(authorized drom.ru data collection; contact: data-team@example.com)"
)


@dataclass(slots=True)
class SourceConfig:
    name: str = "drom"
    base_url: str = "https://www.drom.ru"
    discover_path: str = "/"
    page_param: str = "page"
    page_mode: str = "path"
    page_path_template: str = "page{page}/"
    start_page: int = 1
    max_pages_per_class: int = 50
    stop_on_empty_page: bool = True
    listing_id_regex: str = r"(\d+)(?:\.html)?(?:\?.*)?$"
    timeout_seconds: float = 20.0
    retries: int = 5
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 20.0
    qps: float = 1.0
    burst: float = 3.0
    concurrency: int = 16
    cache_ttl_seconds: int = 86_400
    circuit_breaker_failures: int = 8
    circuit_breaker_cooldown_seconds: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT
    headers: dict[str, str] = field(default_factory=dict)
    listing_url_patterns: list[str] = field(
        default_factory=lambda: [
            r"/[^\s\"'<>]+\.html(?:\?.*)?$",
            r"/auto/[^\s\"'<>]+",
            r"/([a-z0-9\-]+/)+[0-9]{5,}",
        ]
    )


@dataclass(slots=True)
class ClassSpec:
    class_id: int
    make: str
    model: str
    body_type: str
    class_name: str
    generation: str
    search_path: str | None = None
    search_params: dict[str, Any] = field(default_factory=dict)
    max_pages: int | None = None


@dataclass(slots=True)
class DromConfig:
    source: SourceConfig
    classes: list[ClassSpec]


@dataclass(slots=True)
class PipelinePaths:
    raw_pages_dir: Path = RAW_DATA_DIR / "drom" / "pages"
    raw_images_dir: Path = RAW_DATA_DIR / "drom" / "images"
    interim_dir: Path = INTERIM_DATA_DIR / "drom"
    processed_dir: Path = PROCESSED_DATA_DIR
    discovery_path: Path = INTERIM_DATA_DIR / "drom" / "discovery.parquet"
    meta_path: Path = INTERIM_DATA_DIR / "drom" / "meta.parquet"
    images_path: Path = INTERIM_DATA_DIR / "drom" / "images.parquet"
    validated_path: Path = INTERIM_DATA_DIR / "drom" / "validated.parquet"
    dedup_path: Path = INTERIM_DATA_DIR / "drom" / "dedup.parquet"
    manifest_path: Path = PROCESSED_DATA_DIR / "manifest.parquet"
    class_mapping_path: Path = PROCESSED_DATA_DIR / "class_mapping.parquet"

    def ensure_dirs(self) -> None:
        self.raw_pages_dir.mkdir(parents=True, exist_ok=True)
        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def canonical_class_name(
    make: str,
    model: str,
    generation: str | None = None,
    body_type: str | None = None,
) -> str:
    base = f"{make}_{model}"
    if generation:
        base = f"{base}_{generation}"
    if body_type:
        base = f"{base}_{body_type}"
    return slugify(base)


def _to_class_spec(item: dict[str, Any]) -> ClassSpec:
    class_id = int(item["class_id"])
    make = str(item["make"]).strip()
    model = str(item["model"]).strip()
    body_type = item.get("body_type")
    if body_type is None:
        body_type = item.get("search_params", {}).get("body_type")
    body_type = str(body_type).strip() if body_type else None
    if not body_type:
        raise ValueError(f"body_type is required for class_id={class_id}")

    generation = item.get("generation")
    generation = str(generation).strip() if generation else None
    if not generation:
        raise ValueError(f"generation is required for class_id={class_id}")
    class_name = item.get("class_name")
    if class_name:
        class_name = slugify(str(class_name))
    else:
        class_name = canonical_class_name(
            make=make,
            model=model,
            generation=generation,
            body_type=body_type,
        )

    search_path = item.get("search_path")
    if search_path is not None:
        search_path = str(search_path)

    search_params = item.get("search_params", {})
    if not isinstance(search_params, dict):
        raise ValueError(f"search_params must be a mapping for class_id={class_id}")

    max_pages = item.get("max_pages")
    if max_pages is not None:
        max_pages = int(max_pages)

    return ClassSpec(
        class_id=class_id,
        make=make,
        model=model,
        body_type=body_type,
        generation=generation,
        class_name=class_name,
        search_path=search_path,
        search_params=search_params,
        max_pages=max_pages,
    )


def load_classes_config(path: Path) -> DromConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in {path}")

    source_payload = payload.get("source", {})
    if not isinstance(source_payload, dict):
        raise ValueError("source must be a mapping")

    source = SourceConfig(**source_payload)

    classes_payload = payload.get("classes", [])
    if not isinstance(classes_payload, list) or not classes_payload:
        raise ValueError("classes must be a non-empty list")

    classes = [_to_class_spec(item) for item in classes_payload]

    class_ids = [spec.class_id for spec in classes]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("Duplicate class_id detected in classes config")

    class_names = [spec.class_name for spec in classes]
    if len(class_names) != len(set(class_names)):
        raise ValueError("Duplicate class_name detected in classes config")

    signatures = [
        (
            spec.make.lower(),
            spec.model.lower(),
            (spec.generation or "").lower(),
            spec.body_type.lower(),
        )
        for spec in classes
    ]
    if len(signatures) != len(set(signatures)):
        raise ValueError(
            "Duplicate (make, model, generation, body_type) combination detected in classes config"
        )

    return DromConfig(source=source, classes=classes)
