from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any

import pandas as pd

_RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_json_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    has_json_handler = any(
        isinstance(getattr(handler, "formatter", None), JsonFormatter)
        for handler in logger.handlers
    )
    if has_json_handler:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]


def read_table(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Table not found: {path}")
        return pd.DataFrame()

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table extension: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported table extension: {path.suffix}")


def merge_incremental(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()

    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=subset, keep="last")
    merged = merged.reset_index(drop=True)
    return merged


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jitter_sleep_seconds(base_seconds: float, cap_seconds: float, attempt: int) -> float:
    exp = min(cap_seconds, base_seconds * (2 ** (attempt - 1)))
    return exp * (0.5 + random.random() * 0.5)


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if not result:
        return None
    return result


def ensure_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
