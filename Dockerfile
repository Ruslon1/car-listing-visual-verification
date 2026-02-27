FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY car_listing_visual_verification /app/car_listing_visual_verification

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      fastapi~=0.133.1 \
      uvicorn[standard] \
      python-multipart \
      torch~=2.10.0 \
      torchvision~=0.25.0 \
      ultralytics~=8.3.218 \
      pillow~=12.1.0 \
      pandas~=3.0.0 \
      pyarrow~=19.0.1 \
      loguru~=0.7.3 \
      python-dotenv~=1.2.1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "car_listing_visual_verification.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
