from __future__ import annotations

import asyncio
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from car_listing_visual_verification.config import PROJ_ROOT
from car_listing_visual_verification.data.drom_client import DromHttpClient
from car_listing_visual_verification.data.drom_metrics import DromMetrics
from car_listing_visual_verification.data.drom_parsers import (
    extract_listing_id,
    extract_listing_urls,
    parse_listing_metadata,
)
from car_listing_visual_verification.data.drom_types import ClassSpec, DromConfig, PipelinePaths
from car_listing_visual_verification.data.drom_utils import (
    ensure_relative,
    merge_incremental,
    normalize_optional_str,
    read_table,
    utc_now_iso,
    write_table,
)

PARSE_VERSION_DISCOVERY = "drom_discovery_v1"
PARSE_VERSION_META = "drom_meta_v1"
PARSE_VERSION_VALIDATE = "drom_validate_v1"
PARSE_VERSION_DEDUP = "drom_dedup_v1"
PARSE_VERSION_MANIFEST = "drom_manifest_v1"


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _clean_optional(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return normalize_optional_str(value)


def _to_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_url(base_url: str, path: str | None, fallback: str) -> str:
    target = path or fallback
    return urljoin(base_url, target)


async def discover(
    config: DromConfig,
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    force: bool = False,
    use_cache: bool = True,
    max_pages_override: int | None = None,
) -> pd.DataFrame:
    paths.ensure_dirs()

    existing_df = pd.DataFrame() if force else read_table(paths.discovery_path)
    seen_keys: set[tuple[str, int]] = set()
    if not existing_df.empty:
        for row in existing_df[["listing_id", "class_id"]].itertuples(index=False):
            seen_keys.add((str(row.listing_id), int(row.class_id)))

    semaphore = asyncio.Semaphore(max(1, min(config.source.concurrency, len(config.classes))))

    async with DromHttpClient(
        source=config.source,
        cache_dir=paths.raw_pages_dir,
        logger=logger,
        metrics=metrics,
    ) as client:

        async def run_for_class(spec: ClassSpec) -> list[dict[str, Any]]:
            async with semaphore:
                return await _discover_one_class(
                    client=client,
                    spec=spec,
                    config=config,
                    logger=logger,
                    known_keys=seen_keys,
                    use_cache=use_cache,
                    max_pages_override=max_pages_override,
                )

        tasks = [asyncio.create_task(run_for_class(spec)) for spec in config.classes]
        class_results = await asyncio.gather(*tasks)

    rows = [row for chunk in class_results for row in chunk]
    new_df = pd.DataFrame(rows)

    if force:
        out_df = new_df.drop_duplicates(subset=["listing_id", "class_id"], keep="last")
    else:
        out_df = merge_incremental(
            existing=existing_df, incoming=new_df, subset=["listing_id", "class_id"]
        )

    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "listing_id",
                "source",
                "url",
                "class_id",
                "class_name",
                "make",
                "model",
                "body_type",
                "generation",
                "discovered_at",
                "discover_page",
                "http_status",
                "parse_version",
                "discover_error",
            ]
        )

    out_df = out_df.sort_values(by=["class_id", "listing_id"], kind="stable")
    write_table(out_df, paths.discovery_path)
    metrics.record_stage_rows(stage="discover", count=len(out_df))

    logger.info(
        "Discovery stage completed",
        extra={
            "rows_total": int(len(out_df)),
            "rows_new": int(len(new_df)),
            "output": paths.discovery_path.as_posix(),
        },
    )
    return out_df


async def _discover_one_class(
    client: DromHttpClient,
    spec: ClassSpec,
    config: DromConfig,
    logger: logging.Logger,
    known_keys: set[tuple[str, int]],
    use_cache: bool,
    max_pages_override: int | None,
) -> list[dict[str, Any]]:
    source = config.source
    rows: list[dict[str, Any]] = []
    local_seen: set[tuple[str, int]] = set()

    max_pages = max_pages_override or spec.max_pages or source.max_pages_per_class
    search_url = _as_url(
        base_url=source.base_url,
        path=spec.search_path,
        fallback=source.discover_path,
    )

    for page in range(source.start_page, source.start_page + max_pages):
        params = dict(spec.search_params)
        params[source.page_param] = page

        try:
            response = await client.get_text(
                url=search_url,
                params=params,
                use_cache=use_cache,
                force_refresh=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Discovery request failed",
                extra={
                    "class_id": spec.class_id,
                    "class_name": spec.class_name,
                    "page": page,
                    "url": search_url,
                    "error": str(exc),
                },
            )
            continue

        if response.status_code >= 400:
            logger.warning(
                "Discovery got non-success status",
                extra={
                    "class_id": spec.class_id,
                    "class_name": spec.class_name,
                    "page": page,
                    "url": search_url,
                    "http_status": response.status_code,
                },
            )
            if response.status_code in {404, 410} and source.stop_on_empty_page:
                break
            continue

        listing_urls = extract_listing_urls(
            content=response.text,
            base_url=source.base_url,
            listing_url_patterns=source.listing_url_patterns,
        )

        page_new_rows = 0
        for listing_url in listing_urls:
            listing_id = extract_listing_id(
                url=listing_url, listing_id_regex=source.listing_id_regex
            )
            key = (listing_id, spec.class_id)
            if key in known_keys or key in local_seen:
                continue

            rows.append(
                {
                    "listing_id": listing_id,
                    "source": source.name,
                    "url": listing_url,
                    "class_id": spec.class_id,
                    "class_name": spec.class_name,
                    "make": spec.make,
                    "model": spec.model,
                    "body_type": spec.body_type,
                    "generation": spec.generation,
                    "discovered_at": utc_now_iso(),
                    "discover_page": page,
                    "http_status": response.status_code,
                    "parse_version": PARSE_VERSION_DISCOVERY,
                    "discover_error": None,
                }
            )
            local_seen.add(key)
            page_new_rows += 1

        if not listing_urls and source.stop_on_empty_page:
            break

        if page_new_rows == 0 and source.stop_on_empty_page:
            break

    logger.info(
        "Class discovery done",
        extra={
            "class_id": spec.class_id,
            "class_name": spec.class_name,
            "rows_new": len(rows),
            "search_url": search_url,
        },
    )
    return rows


async def fetch_meta(
    config: DromConfig,
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    force: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    discovery_df = read_table(paths.discovery_path, required=True)
    existing_df = pd.DataFrame() if force else read_table(paths.meta_path)

    if discovery_df.empty:
        logger.warning("Discovery input is empty", extra={"path": paths.discovery_path.as_posix()})
        write_table(existing_df, paths.meta_path)
        return existing_df

    done_keys: set[tuple[str, int]] = set()
    if not existing_df.empty:
        for row in existing_df[["listing_id", "class_id"]].itertuples(index=False):
            done_keys.add((str(row.listing_id), int(row.class_id)))

    pending_rows = []
    for row in discovery_df.to_dict(orient="records"):
        key = (str(row["listing_id"]), int(row["class_id"]))
        if key in done_keys and not force:
            continue
        pending_rows.append(row)

    if not pending_rows:
        logger.info("No new listings for fetch_meta", extra={"rows_pending": 0})
        write_table(existing_df, paths.meta_path)
        return existing_df

    semaphore = asyncio.Semaphore(config.source.concurrency)

    async with DromHttpClient(
        source=config.source,
        cache_dir=paths.raw_pages_dir,
        logger=logger,
        metrics=metrics,
    ) as client:

        async def fetch_one(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                listing_id = str(row["listing_id"])
                class_id = int(row["class_id"])
                listing_url = str(row["url"])
                scraped_at = utc_now_iso()

                try:
                    response = await client.get_text(
                        url=listing_url,
                        use_cache=use_cache,
                        force_refresh=False,
                    )

                    if response.status_code >= 400:
                        return {
                            "listing_id": listing_id,
                            "source": row.get("source") or config.source.name,
                            "url": listing_url,
                            "scraped_at": response.fetched_at,
                            "make": _clean_optional(row.get("make")),
                            "model": _clean_optional(row.get("model")),
                            "body_type": _clean_optional(row.get("body_type")),
                            "generation": _clean_optional(row.get("generation")),
                            "year": None,
                            "class_name": row.get("class_name"),
                            "class_id": class_id,
                            "image_url_1": None,
                            "image_url_2": None,
                            "http_status": response.status_code,
                            "parse_version": PARSE_VERSION_META,
                            "meta_error": f"http_status={response.status_code}",
                            "title": None,
                            "parser_blob_count": 0,
                        }

                    parsed = parse_listing_metadata(
                        content=response.text,
                        base_url=config.source.base_url,
                    )
                    image_urls = parsed["image_urls"]
                    image_1 = image_urls[0] if len(image_urls) > 0 else None
                    image_2 = image_urls[1] if len(image_urls) > 1 else None

                    return {
                        "listing_id": listing_id,
                        "source": row.get("source") or config.source.name,
                        "url": listing_url,
                        "scraped_at": response.fetched_at,
                        "make": parsed.get("make") or _clean_optional(row.get("make")),
                        "model": parsed.get("model") or _clean_optional(row.get("model")),
                        "body_type": _clean_optional(row.get("body_type")),
                        "generation": parsed.get("generation")
                        or _clean_optional(row.get("generation")),
                        "year": parsed.get("year"),
                        "class_name": row.get("class_name"),
                        "class_id": class_id,
                        "image_url_1": image_1,
                        "image_url_2": image_2,
                        "http_status": response.status_code,
                        "parse_version": PARSE_VERSION_META,
                        "meta_error": None,
                        "title": parsed.get("title"),
                        "parser_blob_count": parsed.get("parser_blob_count", 0),
                    }
                except Exception as exc:  # noqa: BLE001
                    metrics.record_stage_error(stage="fetch_meta")
                    logger.error(
                        "fetch_meta failed",
                        extra={
                            "listing_id": listing_id,
                            "class_id": class_id,
                            "url": listing_url,
                            "error": str(exc),
                        },
                    )
                    return {
                        "listing_id": listing_id,
                        "source": row.get("source") or config.source.name,
                        "url": listing_url,
                        "scraped_at": scraped_at,
                        "make": _clean_optional(row.get("make")),
                        "model": _clean_optional(row.get("model")),
                        "body_type": _clean_optional(row.get("body_type")),
                        "generation": _clean_optional(row.get("generation")),
                        "year": None,
                        "class_name": row.get("class_name"),
                        "class_id": class_id,
                        "image_url_1": None,
                        "image_url_2": None,
                        "http_status": None,
                        "parse_version": PARSE_VERSION_META,
                        "meta_error": str(exc),
                        "title": None,
                        "parser_blob_count": 0,
                    }

        tasks = [asyncio.create_task(fetch_one(row)) for row in pending_rows]
        rows = await asyncio.gather(*tasks)

    new_df = pd.DataFrame(rows)
    if force:
        out_df = new_df.drop_duplicates(subset=["listing_id", "class_id"], keep="last")
    else:
        out_df = merge_incremental(
            existing=existing_df, incoming=new_df, subset=["listing_id", "class_id"]
        )

    out_df = out_df.sort_values(by=["class_id", "listing_id"], kind="stable")
    write_table(out_df, paths.meta_path)
    metrics.record_stage_rows(stage="fetch_meta", count=len(out_df))

    logger.info(
        "fetch_meta stage completed",
        extra={
            "rows_total": int(len(out_df)),
            "rows_new": int(len(new_df)),
            "output": paths.meta_path.as_posix(),
        },
    )
    return out_df


async def fetch_images(
    config: DromConfig,
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    force: bool = False,
) -> pd.DataFrame:
    meta_df = read_table(paths.meta_path, required=True)
    existing_df = pd.DataFrame() if force else read_table(paths.images_path)

    if meta_df.empty:
        logger.warning("Meta input is empty", extra={"path": paths.meta_path.as_posix()})
        write_table(existing_df, paths.images_path)
        return existing_df

    done_keys: set[tuple[str, int]] = set()
    if not existing_df.empty:
        for row in existing_df[["listing_id", "class_id"]].itertuples(index=False):
            done_keys.add((str(row.listing_id), int(row.class_id)))

    pending_rows = []
    for row in meta_df.to_dict(orient="records"):
        key = (str(row["listing_id"]), int(row["class_id"]))
        if key in done_keys and not force:
            continue
        pending_rows.append(row)

    if not pending_rows:
        logger.info("No new listings for fetch_images", extra={"rows_pending": 0})
        write_table(existing_df, paths.images_path)
        return existing_df

    semaphore = asyncio.Semaphore(config.source.concurrency)

    async with DromHttpClient(
        source=config.source,
        cache_dir=paths.raw_pages_dir,
        logger=logger,
        metrics=metrics,
    ) as client:

        async def fetch_listing_images(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                class_name = str(row.get("class_name") or f"class_{row['class_id']}")
                listing_id = str(row["listing_id"])
                class_dir = paths.raw_images_dir / class_name

                first = await _download_and_prepare_image(
                    client=client,
                    url=_clean_optional(row.get("image_url_1")),
                    target_path=class_dir / f"{listing_id}_1.jpg",
                    force=force,
                )
                second = await _download_and_prepare_image(
                    client=client,
                    url=_clean_optional(row.get("image_url_2")),
                    target_path=class_dir / f"{listing_id}_2.jpg",
                    force=force,
                )

                if first["ok"]:
                    metrics.record_image_download()
                if second["ok"]:
                    metrics.record_image_download()

                errors = [err for err in [first.get("error"), second.get("error")] if err]
                stage_error = "; ".join(errors) if errors else None

                return {
                    "listing_id": listing_id,
                    "source": row.get("source") or config.source.name,
                    "url": row.get("url"),
                    "scraped_at": row.get("scraped_at") or utc_now_iso(),
                    "make": _clean_optional(row.get("make")),
                    "model": _clean_optional(row.get("model")),
                    "body_type": _clean_optional(row.get("body_type")),
                    "generation": _clean_optional(row.get("generation")),
                    "year": _to_int(row.get("year")),
                    "class_name": class_name,
                    "class_id": int(row["class_id"]),
                    "image_1_url": _clean_optional(row.get("image_url_1")),
                    "image_2_url": _clean_optional(row.get("image_url_2")),
                    "image_1_path": first.get("path"),
                    "image_2_path": second.get("path"),
                    "image_1_http_status": first.get("http_status"),
                    "image_2_http_status": second.get("http_status"),
                    "image_1_bytes": first.get("bytes"),
                    "image_2_bytes": second.get("bytes"),
                    "image_1_width": first.get("width"),
                    "image_2_width": second.get("width"),
                    "image_1_height": first.get("height"),
                    "image_2_height": second.get("height"),
                    "image_1_format": first.get("format"),
                    "image_2_format": second.get("format"),
                    "fetch_images_error": stage_error,
                }

        tasks = [asyncio.create_task(fetch_listing_images(row)) for row in pending_rows]
        rows = await asyncio.gather(*tasks)

    new_df = pd.DataFrame(rows)
    if force:
        out_df = new_df.drop_duplicates(subset=["listing_id", "class_id"], keep="last")
    else:
        out_df = merge_incremental(
            existing=existing_df, incoming=new_df, subset=["listing_id", "class_id"]
        )

    out_df = out_df.sort_values(by=["class_id", "listing_id"], kind="stable")
    write_table(out_df, paths.images_path)
    metrics.record_stage_rows(stage="fetch_images", count=len(out_df))

    logger.info(
        "fetch_images stage completed",
        extra={
            "rows_total": int(len(out_df)),
            "rows_new": int(len(new_df)),
            "output": paths.images_path.as_posix(),
        },
    )
    return out_df


async def _download_and_prepare_image(
    client: DromHttpClient,
    url: str | None,
    target_path: Path,
    force: bool,
) -> dict[str, Any]:
    if not url:
        return {
            "ok": False,
            "path": None,
            "bytes": None,
            "width": None,
            "height": None,
            "format": None,
            "http_status": None,
            "error": "missing_image_url",
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        info = _inspect_image(target_path)
        return {
            "ok": info["ok"],
            "path": ensure_relative(target_path, PROJ_ROOT),
            "bytes": info.get("bytes"),
            "width": info.get("width"),
            "height": info.get("height"),
            "format": info.get("format"),
            "http_status": 200,
            "error": info.get("error"),
        }

    raw_path = target_path.with_suffix(".orig")
    download = await client.download_file(url=url, target=raw_path, overwrite=True)
    if download.error or download.status_code >= 400:
        raw_path.unlink(missing_ok=True)
        return {
            "ok": False,
            "path": None,
            "bytes": None,
            "width": None,
            "height": None,
            "format": None,
            "http_status": download.status_code,
            "error": download.error or f"http_status={download.status_code}",
        }

    try:
        with Image.open(raw_path) as image:
            rgb = image.convert("RGB")
            rgb.save(target_path, format="JPEG", quality=92)
    except (OSError, UnidentifiedImageError) as exc:
        raw_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        return {
            "ok": False,
            "path": None,
            "bytes": None,
            "width": None,
            "height": None,
            "format": None,
            "http_status": download.status_code,
            "error": f"image_decode_error: {exc}",
        }

    raw_path.unlink(missing_ok=True)
    info = _inspect_image(target_path)
    return {
        "ok": info["ok"],
        "path": ensure_relative(target_path, PROJ_ROOT),
        "bytes": info.get("bytes"),
        "width": info.get("width"),
        "height": info.get("height"),
        "format": info.get("format"),
        "http_status": download.status_code,
        "error": info.get("error"),
    }


def _inspect_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            fmt = (image.format or "UNKNOWN").upper()
        return {
            "ok": True,
            "width": int(width),
            "height": int(height),
            "bytes": path.stat().st_size,
            "format": fmt,
        }
    except (OSError, UnidentifiedImageError) as exc:
        return {
            "ok": False,
            "width": None,
            "height": None,
            "bytes": None,
            "format": None,
            "error": f"image_inspect_error: {exc}",
        }


def validate(
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    min_width: int = 224,
    min_height: int = 224,
    min_bytes: int = 10_000,
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    images_df = read_table(paths.images_path, required=True)
    if images_df.empty:
        logger.warning("Images input is empty", extra={"path": paths.images_path.as_posix()})
        write_table(images_df, paths.validated_path)
        return images_df

    rows: list[dict[str, Any]] = []

    for row in images_df.to_dict(orient="records"):
        first = _validate_image(
            image_path=row.get("image_1_path"),
            min_width=min_width,
            min_height=min_height,
            min_bytes=min_bytes,
        )
        second = _validate_image(
            image_path=row.get("image_2_path"),
            min_width=min_width,
            min_height=min_height,
            min_bytes=min_bytes,
        )

        is_valid = bool(first["ok"] and second["ok"])
        errors = [err for err in [first.get("error"), second.get("error")] if err]
        row_out = {
            **row,
            "image_1_phash": first.get("phash"),
            "image_2_phash": second.get("phash"),
            "image_1_width": first.get("width") or row.get("image_1_width"),
            "image_2_width": second.get("width") or row.get("image_2_width"),
            "image_1_height": first.get("height") or row.get("image_1_height"),
            "image_2_height": second.get("height") or row.get("image_2_height"),
            "image_1_bytes": first.get("bytes") or row.get("image_1_bytes"),
            "image_2_bytes": second.get("bytes") or row.get("image_2_bytes"),
            "image_1_format": first.get("format") or row.get("image_1_format"),
            "image_2_format": second.get("format") or row.get("image_2_format"),
            "is_valid": is_valid,
            "validation_error": "; ".join(errors) if errors else None,
            "parse_version": PARSE_VERSION_VALIDATE,
        }
        rows.append(row_out)

    validated_df = pd.DataFrame(rows)
    if drop_invalid_rows:
        validated_df = validated_df[validated_df["is_valid"]].copy()

    validated_df = validated_df.reset_index(drop=True)
    write_table(validated_df, paths.validated_path)
    metrics.record_stage_rows(stage="validate", count=len(validated_df))

    logger.info(
        "validate stage completed",
        extra={
            "rows_total": int(len(validated_df)),
            "drop_invalid_rows": drop_invalid_rows,
            "output": paths.validated_path.as_posix(),
        },
    )
    return validated_df


def _validate_image(
    image_path: str | None,
    min_width: int,
    min_height: int,
    min_bytes: int,
) -> dict[str, Any]:
    if not image_path:
        return {"ok": False, "error": "missing_image_path"}

    full_path = Path(image_path)
    if not full_path.is_absolute():
        full_path = PROJ_ROOT / full_path

    if not full_path.exists():
        return {"ok": False, "error": f"missing_file: {image_path}"}

    size_bytes = full_path.stat().st_size
    if size_bytes < min_bytes:
        return {
            "ok": False,
            "bytes": size_bytes,
            "error": f"file_too_small: {size_bytes}<{min_bytes}",
        }

    try:
        with Image.open(full_path) as image:
            image.load()
            width, height = image.size
            fmt = (image.format or "UNKNOWN").upper()
            if width < min_width or height < min_height:
                return {
                    "ok": False,
                    "width": width,
                    "height": height,
                    "bytes": size_bytes,
                    "format": fmt,
                    "error": f"small_dimensions: {width}x{height}",
                }

            phash = str(imagehash.phash(image.convert("RGB")))
            return {
                "ok": True,
                "phash": phash,
                "width": width,
                "height": height,
                "bytes": size_bytes,
                "format": fmt,
            }
    except (OSError, UnidentifiedImageError) as exc:
        return {"ok": False, "error": f"corrupted_image: {exc}"}


def dedup(
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    validated_df = read_table(paths.validated_path, required=True)
    if validated_df.empty:
        logger.warning("Validated input is empty", extra={"path": paths.validated_path.as_posix()})
        write_table(validated_df, paths.dedup_path)
        return validated_df

    working = validated_df.sort_values(by=["class_id", "listing_id"], kind="stable").copy()
    seen_hashes: dict[str, str] = {}

    duplicate_flags: list[bool] = []
    duplicate_of_list: list[str | None] = []
    duplicate_reason_list: list[str | None] = []

    for row in working.to_dict(orient="records"):
        listing_id = str(row["listing_id"])
        h1 = _clean_optional(row.get("image_1_phash"))
        h2 = _clean_optional(row.get("image_2_phash"))

        duplicate_of: str | None = None
        duplicate_reason: str | None = None

        for image_hash, hash_name in ((h1, "image_1_phash"), (h2, "image_2_phash")):
            if not image_hash:
                continue
            prev_listing = seen_hashes.get(image_hash)
            if prev_listing and prev_listing != listing_id:
                duplicate_of = prev_listing
                duplicate_reason = f"hash_collision:{hash_name}"
                break

        is_duplicate = duplicate_of is not None
        duplicate_flags.append(is_duplicate)
        duplicate_of_list.append(duplicate_of)
        duplicate_reason_list.append(duplicate_reason)

        if not is_duplicate:
            if h1:
                seen_hashes[h1] = listing_id
            if h2:
                seen_hashes[h2] = listing_id

    working["is_duplicate"] = duplicate_flags
    working["duplicate_of_listing_id"] = duplicate_of_list
    working["duplicate_reason"] = duplicate_reason_list
    working["parse_version"] = PARSE_VERSION_DEDUP

    if drop_duplicates:
        output_df = working[~working["is_duplicate"]].reset_index(drop=True)
    else:
        output_df = working.reset_index(drop=True)

    write_table(output_df, paths.dedup_path)
    metrics.record_stage_rows(stage="dedup", count=len(output_df))

    logger.info(
        "dedup stage completed",
        extra={
            "rows_total": int(len(output_df)),
            "duplicates_detected": int(working["is_duplicate"].sum()),
            "drop_duplicates": drop_duplicates,
            "output": paths.dedup_path.as_posix(),
        },
    )
    return output_df


def prepare_manifest(
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    include_duplicates: bool = False,
) -> pd.DataFrame:
    dedup_df = read_table(paths.dedup_path, required=True)
    if dedup_df.empty:
        logger.warning("Dedup input is empty", extra={"path": paths.dedup_path.as_posix()})
        write_table(dedup_df, paths.manifest_path)
        return dedup_df

    working = dedup_df.copy()
    if "is_duplicate" in working.columns and not include_duplicates:
        working = working[~working["is_duplicate"]].copy()

    if "is_valid" in working.columns:
        working = working[working["is_valid"]].copy()

    manifest = pd.DataFrame(
        {
            "listing_id": working["listing_id"].astype(str),
            "source": working.get("source", "drom"),
            "url": working.get("url"),
            "scraped_at": working.get("scraped_at", utc_now_iso()),
            "make": working.get("make"),
            "model": working.get("model"),
            "body_type": working.get("body_type"),
            "generation": working.get("generation"),
            "year": working.get("year"),
            "class_name": working.get("class_name"),
            "class_id": working.get("class_id"),
            "image_1_path": working.get("image_1_path"),
            "image_2_path": working.get("image_2_path"),
            "image_1_phash": working.get("image_1_phash"),
            "image_2_phash": working.get("image_2_phash"),
            "width": working.get("image_1_width"),
            "height": working.get("image_1_height"),
            "bytes": working.get("image_1_bytes"),
            "format": working.get("image_1_format"),
            "split": "unsplit",
            "http_status": working.get("http_status"),
            "parse_version": PARSE_VERSION_MANIFEST,
            "meta_error": working.get("meta_error"),
            "fetch_images_error": working.get("fetch_images_error"),
            "validation_error": working.get("validation_error"),
            "is_duplicate": working.get("is_duplicate", False),
            "duplicate_of_listing_id": working.get("duplicate_of_listing_id"),
            "duplicate_reason": working.get("duplicate_reason"),
            "image_1_width": working.get("image_1_width"),
            "image_1_height": working.get("image_1_height"),
            "image_1_bytes": working.get("image_1_bytes"),
            "image_1_format": working.get("image_1_format"),
            "image_2_width": working.get("image_2_width"),
            "image_2_height": working.get("image_2_height"),
            "image_2_bytes": working.get("image_2_bytes"),
            "image_2_format": working.get("image_2_format"),
        }
    )

    manifest = manifest.drop_duplicates(subset=["listing_id", "class_id"], keep="last")
    manifest = manifest.sort_values(by=["class_id", "listing_id"], kind="stable").reset_index(
        drop=True
    )
    write_table(manifest, paths.manifest_path)

    class_mapping = (
        manifest[["class_id", "class_name", "make", "model", "body_type", "generation"]]
        .drop_duplicates(subset=["class_id"], keep="first")
        .sort_values(by="class_id", kind="stable")
        .reset_index(drop=True)
    )
    write_table(class_mapping, paths.class_mapping_path)

    metrics.record_stage_rows(stage="prepare_manifest", count=len(manifest))

    logger.info(
        "prepare_manifest stage completed",
        extra={
            "rows_total": int(len(manifest)),
            "manifest_output": paths.manifest_path.as_posix(),
            "class_mapping_output": paths.class_mapping_path.as_posix(),
        },
    )
    return manifest


def split_manifest(
    paths: PipelinePaths,
    logger: logging.Logger,
    metrics: DromMetrics,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    manifest = read_table(paths.manifest_path, required=True)
    if manifest.empty:
        logger.warning("Manifest input is empty", extra={"path": paths.manifest_path.as_posix()})
        write_table(manifest, paths.manifest_path)
        return manifest

    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to <1")

    per_listing = manifest[["listing_id", "class_id"]].drop_duplicates(
        subset=["listing_id"]
    )  # one key
    per_listing = per_listing.reset_index(drop=True)

    if len(per_listing) < 3:
        manifest = manifest.copy()
        manifest["split"] = "train"
        write_table(manifest, paths.manifest_path)
        return manifest

    test_val_ratio = val_ratio + test_ratio
    labels = per_listing["class_id"]

    stratify_labels = labels if _can_stratify(labels, min_count=2) else None
    train_keys, temp_keys = train_test_split(
        per_listing,
        test_size=test_val_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    if temp_keys.empty:
        split_map = {key: "train" for key in train_keys["listing_id"]}
    else:
        temp_val_ratio = val_ratio / test_val_ratio if test_val_ratio > 0 else 0.5
        temp_labels = temp_keys["class_id"]
        temp_stratify = temp_labels if _can_stratify(temp_labels, min_count=2) else None

        val_keys, test_keys = train_test_split(
            temp_keys,
            test_size=1 - temp_val_ratio,
            random_state=seed,
            stratify=temp_stratify,
        )

        split_map = {key: "train" for key in train_keys["listing_id"]}
        split_map.update({key: "val" for key in val_keys["listing_id"]})
        split_map.update({key: "test" for key in test_keys["listing_id"]})

    out = manifest.copy()
    out["split"] = out["listing_id"].map(split_map).fillna("train")
    out = out.sort_values(by=["split", "class_id", "listing_id"], kind="stable").reset_index(
        drop=True
    )

    write_table(out, paths.manifest_path)
    metrics.record_stage_rows(stage="split", count=len(out))

    split_counts = out["split"].value_counts().to_dict()
    logger.info(
        "split stage completed",
        extra={
            "output": paths.manifest_path.as_posix(),
            "split_counts": split_counts,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
    )
    return out


def _can_stratify(labels: pd.Series, min_count: int) -> bool:
    value_counts = labels.value_counts()
    if len(value_counts) < 2:
        return False
    return bool((value_counts >= min_count).all())


def describe_effective_config(config: DromConfig) -> dict[str, Any]:
    source_dict = asdict(config.source)
    source_dict["headers"] = dict(config.source.headers)

    return {
        "source": source_dict,
        "classes": [asdict(spec) for spec in config.classes],
    }
