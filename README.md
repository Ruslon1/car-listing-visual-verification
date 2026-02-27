# car-listing-visual-verification

Application-side repository for car listing visual verification.

## Repository split

Scraping, filtering, and dataset upload pipeline were moved to:

- `/Users/ruslan/Projects/car-listing-data-pipeline`
- [car-drom-listing-data-pipeline](https://github.com/Ruslon1/car-drom-listing-data-pipeline)

This repository now focuses on model training, evaluation, inference, and API serving.

Dataset used in this project was prepared via that pipeline repository.

## Local data layout

Local dataset artifacts are expected under:

- `/Users/ruslan/Projects/car-listing-visual-verification/data/processed/hf_release`

Required files for API inference:

- `class_mapping.parquet`
- `train.parquet` / `validation.parquet` / `test.parquet` (for training/eval workflows)
- `/Users/ruslan/Projects/car-listing-visual-verification/models/model.pkl`
- `/Users/ruslan/Projects/car-listing-visual-verification/models/yolov8n.pt`

## FastAPI endpoints

- `GET /health` — reports model/gate load status
- `POST /predict` — file upload inference (`multipart/form-data`, field: `file`)

Response labels:

- `no_car` — YOLO gate did not detect vehicle confidently
- `unknown_car` — classifier confidence below threshold
- `car_classified` — vehicle class predicted

## Docker

### 1) Build image

```bash
cd /Users/ruslan/Projects/car-listing-visual-verification
docker build -t clvv-api:latest .
```

### 2) Run container

Mount local `models/` and `data/processed/hf_release/` into the container:

```bash
docker run --rm -p 8000:8000 \
  -v /Users/ruslan/Projects/car-listing-visual-verification/models:/app/models \
  -v /Users/ruslan/Projects/car-listing-visual-verification/data/processed/hf_release:/app/data/processed/hf_release \
  -e CLVV_HF_RELEASE_DIR=/app/data/processed/hf_release \
  clvv-api:latest
```

### 3) Health check

```bash
curl http://127.0.0.1:8000/health
```

### 4) Prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/Users/ruslan/Downloads/imagess.jpg"
```

## Runtime tuning via env vars

- `CLVV_CAR_GATE_MODEL` (default: `/app/models/yolov8n.pt`)
- `CLVV_CAR_GATE_CONF_THR` (default: `0.20`)
- `CLVV_CAR_GATE_MIN_AREA_RATIO` (default: `0.02`)
- `CLVV_CLASSIFIER_UNKNOWN_THRESHOLD` (default: `0.45`)

Example with stricter/softer thresholds:

```bash
docker run --rm -p 8000:8000 \
  -v /Users/ruslan/Projects/car-listing-visual-verification/models:/app/models \
  -v /Users/ruslan/Projects/car-listing-visual-verification/data/processed/hf_release:/app/data/processed/hf_release \
  -e CLVV_HF_RELEASE_DIR=/app/data/processed/hf_release \
  -e CLVV_CAR_GATE_CONF_THR=0.15 \
  -e CLVV_CAR_GATE_MIN_AREA_RATIO=0.01 \
  clvv-api:latest
```

## Non-Docker local run

```bash
cd /Users/ruslan/Projects/car-listing-visual-verification
source .venv/bin/activate
python -m uvicorn car_listing_visual_verification.api.main:app --host 0.0.0.0 --port 8000
```
