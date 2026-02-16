from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from car_listing_visual_verification.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from car_listing_visual_verification.data.drom_metrics import DromMetrics
from car_listing_visual_verification.data.drom_pipeline import (
    dedup,
    describe_effective_config,
    discover,
    fetch_images,
    fetch_meta,
    prepare_manifest,
    split_manifest,
    validate,
)
from car_listing_visual_verification.data.drom_types import (
    DromConfig,
    PipelinePaths,
    load_classes_config,
)
from car_listing_visual_verification.data.drom_utils import configure_json_logging

app = typer.Typer(help="Drom data collection pipeline")


def _build_paths(
    raw_pages_dir: Path | None,
    raw_images_dir: Path | None,
    interim_dir: Path | None,
    processed_dir: Path | None,
    manifest_path: Path | None,
    class_mapping_path: Path | None,
) -> PipelinePaths:
    raw_pages = raw_pages_dir or (RAW_DATA_DIR / "drom" / "pages")
    raw_images = raw_images_dir or (RAW_DATA_DIR / "drom" / "images")
    interim = interim_dir or (INTERIM_DATA_DIR / "drom")
    processed = processed_dir or PROCESSED_DATA_DIR

    manifest = manifest_path or (processed / "manifest.parquet")
    class_mapping = class_mapping_path or (processed / "class_mapping.parquet")

    return PipelinePaths(
        raw_pages_dir=raw_pages,
        raw_images_dir=raw_images,
        interim_dir=interim,
        processed_dir=processed,
        discovery_path=interim / "discovery.parquet",
        meta_path=interim / "meta.parquet",
        images_path=interim / "images.parquet",
        validated_path=interim / "validated.parquet",
        dedup_path=interim / "dedup.parquet",
        manifest_path=manifest,
        class_mapping_path=class_mapping,
    )


def _load_runtime_config(
    classes: Path,
    qps: float | None,
    concurrency: int | None,
    retries: int | None,
    timeout_seconds: float | None,
) -> DromConfig:
    config = load_classes_config(classes)

    if qps is not None:
        config.source.qps = qps
    if concurrency is not None:
        config.source.concurrency = concurrency
    if retries is not None:
        config.source.retries = retries
    if timeout_seconds is not None:
        config.source.timeout_seconds = timeout_seconds

    return config


def _build_logger(log_level: str) -> logging.Logger:
    configure_json_logging(level=log_level)
    return logging.getLogger("car_listing_visual_verification.data.drom")


def _build_metrics(metrics_port: int, logger: logging.Logger) -> DromMetrics:
    metrics = DromMetrics()
    metrics.maybe_start_server(port=metrics_port, logger=logger)
    return metrics


def _log_start(
    logger: logging.Logger,
    stage: str,
    classes_path: Path,
    config: DromConfig,
    paths: PipelinePaths,
) -> None:
    logger.info(
        "Stage started",
        extra={
            "stage": stage,
            "classes_path": classes_path.as_posix(),
            "paths": {
                "raw_pages_dir": paths.raw_pages_dir.as_posix(),
                "raw_images_dir": paths.raw_images_dir.as_posix(),
                "interim_dir": paths.interim_dir.as_posix(),
                "processed_dir": paths.processed_dir.as_posix(),
                "manifest_path": paths.manifest_path.as_posix(),
            },
            "source": describe_effective_config(config)["source"],
            "class_count": len(config.classes),
        },
    )


@app.command("discover")
def discover_cmd(
    classes: Path = typer.Option(Path("configs/classes.yaml"), help="Path to classes.yaml"),
    force: bool = typer.Option(False, help="Rebuild output and ignore existing rows"),
    no_cache: bool = typer.Option(False, help="Disable HTTP response cache"),
    max_pages: int | None = typer.Option(None, help="Override max pages per class"),
    qps: float | None = typer.Option(None, help="Override host QPS"),
    concurrency: int | None = typer.Option(None, help="Override async concurrency"),
    retries: int | None = typer.Option(None, help="Override max retries"),
    timeout_seconds: float | None = typer.Option(None, help="Override request timeout in seconds"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    config = _load_runtime_config(classes, qps, concurrency, retries, timeout_seconds)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)
    _log_start(logger=logger, stage="discover", classes_path=classes, config=config, paths=paths)

    asyncio.run(
        discover(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
            use_cache=not no_cache,
            max_pages_override=max_pages,
        )
    )


@app.command("fetch-meta")
def fetch_meta_cmd(
    classes: Path = typer.Option(Path("configs/classes.yaml"), help="Path to classes.yaml"),
    force: bool = typer.Option(False, help="Rebuild output and ignore existing rows"),
    no_cache: bool = typer.Option(False, help="Disable HTTP response cache"),
    qps: float | None = typer.Option(None, help="Override host QPS"),
    concurrency: int | None = typer.Option(None, help="Override async concurrency"),
    retries: int | None = typer.Option(None, help="Override max retries"),
    timeout_seconds: float | None = typer.Option(None, help="Override request timeout in seconds"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    config = _load_runtime_config(classes, qps, concurrency, retries, timeout_seconds)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)
    _log_start(logger=logger, stage="fetch_meta", classes_path=classes, config=config, paths=paths)

    asyncio.run(
        fetch_meta(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
            use_cache=not no_cache,
        )
    )


@app.command("fetch-images")
def fetch_images_cmd(
    classes: Path = typer.Option(Path("configs/classes.yaml"), help="Path to classes.yaml"),
    force: bool = typer.Option(False, help="Rebuild output and ignore existing rows"),
    qps: float | None = typer.Option(None, help="Override host QPS"),
    concurrency: int | None = typer.Option(None, help="Override async concurrency"),
    retries: int | None = typer.Option(None, help="Override max retries"),
    timeout_seconds: float | None = typer.Option(None, help="Override request timeout in seconds"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    config = _load_runtime_config(classes, qps, concurrency, retries, timeout_seconds)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)
    _log_start(
        logger=logger, stage="fetch_images", classes_path=classes, config=config, paths=paths
    )

    asyncio.run(
        fetch_images(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
        )
    )


@app.command("validate")
def validate_cmd(
    min_width: int = typer.Option(224, help="Minimum image width"),
    min_height: int = typer.Option(224, help="Minimum image height"),
    min_bytes: int = typer.Option(10000, help="Minimum image size in bytes"),
    keep_invalid_rows: bool = typer.Option(False, help="Keep invalid rows in output"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)

    validate(
        paths=paths,
        logger=logger,
        metrics=metrics,
        min_width=min_width,
        min_height=min_height,
        min_bytes=min_bytes,
        drop_invalid_rows=not keep_invalid_rows,
    )


@app.command("dedup")
def dedup_cmd(
    drop_duplicates: bool = typer.Option(False, help="Drop duplicate listings from output"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)

    dedup(paths=paths, logger=logger, metrics=metrics, drop_duplicates=drop_duplicates)


@app.command("prepare-manifest")
@app.command("build-manifest")
def prepare_manifest_cmd(
    include_duplicates: bool = typer.Option(
        False, help="Include duplicate listings in final manifest"
    ),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)

    prepare_manifest(
        paths=paths,
        logger=logger,
        metrics=metrics,
        include_duplicates=include_duplicates,
    )


@app.command("split")
def split_cmd(
    val_ratio: float = typer.Option(0.1, help="Validation split ratio"),
    test_ratio: float = typer.Option(0.1, help="Test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)

    split_manifest(
        paths=paths,
        logger=logger,
        metrics=metrics,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


@app.command("run-all")
def run_all_cmd(
    classes: Path = typer.Option(Path("configs/classes.yaml"), help="Path to classes.yaml"),
    force: bool = typer.Option(False, help="Rebuild outputs and ignore existing rows"),
    no_cache: bool = typer.Option(False, help="Disable HTTP response cache"),
    max_pages: int | None = typer.Option(None, help="Override max pages per class"),
    qps: float | None = typer.Option(None, help="Override host QPS"),
    concurrency: int | None = typer.Option(None, help="Override async concurrency"),
    retries: int | None = typer.Option(None, help="Override max retries"),
    timeout_seconds: float | None = typer.Option(None, help="Override request timeout in seconds"),
    min_width: int = typer.Option(224, help="Minimum image width"),
    min_height: int = typer.Option(224, help="Minimum image height"),
    min_bytes: int = typer.Option(10000, help="Minimum image size in bytes"),
    val_ratio: float = typer.Option(0.1, help="Validation split ratio"),
    test_ratio: float = typer.Option(0.1, help="Test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    raw_pages_dir: Path | None = typer.Option(None, help="Override raw pages cache directory"),
    raw_images_dir: Path | None = typer.Option(None, help="Override raw images directory"),
    interim_dir: Path | None = typer.Option(None, help="Override interim directory"),
    processed_dir: Path | None = typer.Option(None, help="Override processed directory"),
    manifest_path: Path | None = typer.Option(None, help="Override final manifest path"),
    class_mapping_path: Path | None = typer.Option(
        None, help="Override class mapping output path"
    ),
    metrics_port: int = typer.Option(0, help="Prometheus port (0 disables exporter)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    logger = _build_logger(log_level)
    config = _load_runtime_config(classes, qps, concurrency, retries, timeout_seconds)
    paths = _build_paths(
        raw_pages_dir=raw_pages_dir,
        raw_images_dir=raw_images_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        class_mapping_path=class_mapping_path,
    )
    metrics = _build_metrics(metrics_port, logger)

    _log_start(logger=logger, stage="run_all", classes_path=classes, config=config, paths=paths)

    asyncio.run(
        discover(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
            use_cache=not no_cache,
            max_pages_override=max_pages,
        )
    )
    asyncio.run(
        fetch_meta(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
            use_cache=not no_cache,
        )
    )
    asyncio.run(
        fetch_images(
            config=config,
            paths=paths,
            logger=logger,
            metrics=metrics,
            force=force,
        )
    )
    validate(
        paths=paths,
        logger=logger,
        metrics=metrics,
        min_width=min_width,
        min_height=min_height,
        min_bytes=min_bytes,
        drop_invalid_rows=True,
    )
    dedup(paths=paths, logger=logger, metrics=metrics, drop_duplicates=False)
    prepare_manifest(paths=paths, logger=logger, metrics=metrics, include_duplicates=False)
    split_manifest(
        paths=paths,
        logger=logger,
        metrics=metrics,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    app()
