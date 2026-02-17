# car-listing-visual-verification

Production pipeline for collecting authorized Drom car listings and preparing an ML-ready manifest with two photos per listing.

## Compliance and Scope

This repository assumes an explicit agreement with drom.ru for automated data collection.

- No bypass behavior is implemented (no CAPTCHA solving, no stealth/evasion logic).
- The crawler is rate-limited and configurable for agreed traffic limits.
- Endpoints and request patterns are driven by `configs/classes.yaml` and runtime flags.

## Drom Pipeline

CLI entrypoint:

```bash
python -m car_listing_visual_verification.data.drom --help
```

Stages:

1. `discover` - enumerate listing URLs/IDs for each configured class.
2. `fetch-meta` - fetch listing payload/page and parse metadata + first two image URLs.
3. `fetch-images` - download and normalize first two images per listing.
4. `validate` - check readability, dimensions, size thresholds, corruption.
5. `dedup` - perceptual-hash dedup across images/listings.
6. `prepare-manifest` - generate canonical manifest and class mapping.
7. `split` - assign `train/val/test` by `listing_id` with stratification when possible.

Optional all-in-one command:

```bash
python -m car_listing_visual_verification.data.drom run-all --classes configs/classes.yaml
```

For 300 photos per class (2 photos/listing), run with `150` listings per class:

```bash
python -m car_listing_visual_verification.data.drom run-all \
  --classes configs/classes.yaml \
  --max-listings-per-class 150
```

## Quick Start

Install deps:

```bash
make requirements
```

Run full pipeline with Make targets:

```bash
make data
```

`make data` is optimized for final dataset output:

- runs end-to-end collection with `--no-cache`
- keeps labeled images and final processed manifest
- removes page cache and interim artifacts after completion

Or stage-by-stage:

```bash
make drom-discover
make drom-fetch-meta
make drom-fetch-images
make drom-validate
make drom-dedup
make drom-manifest
make drom-split
```

## Configuration

Primary config file: `configs/classes.yaml`
The repository now ships with a generated 250-class config where each class has exactly one `body_type`.
It also enforces one row per `(make, model)` in this generated config (one generation and one body type per model).
Source host is `https://auto.drom.ru` for listing discovery/fetch.

- `source`: networking, retries, rate limit, cache, parser patterns.
- `classes`: target class definitions.

Each class entry contains:

- `class_id`
- `make`
- `model`
- `body_type` (required; exactly one body style per class, e.g. `sedan` or `van`)
- `generation` (required in current config)
- `class_name` (canonical)
- `search_path`
- `search_params` (endpoint-specific query parameters)

To scale from sample classes to more classes, append class items with unique `class_id` and `class_name`, and keep a single `body_type` per class row.

### Runtime overrides

Use CLI flags for production tuning without editing YAML:

```bash
python -m car_listing_visual_verification.data.drom discover \
  --classes configs/classes.yaml \
  --max-listings-per-class 150 \
  --qps 1.2 \
  --concurrency 16 \
  --retries 6 \
  --timeout-seconds 25 \
  --metrics-port 9108
```

Custom output directories:

```bash
python -m car_listing_visual_verification.data.drom run-all \
  --classes configs/classes.yaml \
  --raw-pages-dir data/raw/drom/pages \
  --raw-images-dir data/raw/drom/images \
  --interim-dir data/interim/drom \
  --processed-dir data/processed \
  --manifest-path data/processed/manifest.parquet
```

## Output Artifacts

Expected layout:

- `data/raw/drom/pages/` - diskcache HTTP cache (HTML/JSON responses).
- `data/raw/drom/images/<class_name>/<listing_id>_1.jpg` and `_2.jpg`.
- `data/interim/drom/discovery.parquet`
- `data/interim/drom/meta.parquet`
- `data/interim/drom/images.parquet`
- `data/interim/drom/validated.parquet`
- `data/interim/drom/dedup.parquet`
- `data/processed/manifest.parquet`
- `data/processed/class_mapping.parquet`

After `make data`, cache/interim files are pruned and recreated as empty directories:

- `data/raw/drom/pages/` (empty)
- `data/interim/drom/` (empty)

## Manifest Contract

`data/processed/manifest.parquet` includes core columns:

- `listing_id`, `source`, `url`, `scraped_at`
- `make`, `model`, `body_type`, `generation`, `year`
- `class_name`, `class_id`
- `image_1_path`, `image_2_path`
- `image_1_phash`, `image_2_phash`
- `width`, `height`, `bytes`, `format`
- `split` (`train`/`val`/`test`)

Diagnostics included:

- `http_status`
- `parse_version`
- `meta_error`, `fetch_images_error`, `validation_error`
- `is_duplicate`, `duplicate_of_listing_id`, `duplicate_reason`
- per-image fields (`image_1_width`, `image_2_width`, etc.)

## Reliability Features

- Async HTTP with `httpx`
- Host token-bucket rate limiting
- Retry with exponential backoff + jitter
- Circuit breaker per host
- Disk cache for pages (`diskcache`)
- Resumable/idempotent stages (skip existing unless `--force`)
- Structured JSON logging
- Optional Prometheus metrics exporter (`--metrics-port`)

## Notes

- Keep `qps`, `concurrency`, and endpoint params aligned with your Drom agreement.
- If endpoint HTML/API structure changes, tune `listing_url_patterns` and class search params in `configs/classes.yaml`.
