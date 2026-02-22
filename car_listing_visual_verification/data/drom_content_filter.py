from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContentFilterConfig:
    yolo_model: str = "yolov8n.pt"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    min_car_conf: float = 0.25
    min_car_area_ratio: float = 0.2
    min_exterior_score: float = 0.55
    min_exterior_margin: float = 0.05


class ContentFilter:
    def __init__(self, config: ContentFilterConfig) -> None:
        self.config = config
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Missing dependencies for content filtering. "
                "Install with: pip install ultralytics transformers"
            ) from exc

        self._torch = torch
        self._clip_prompts = [
            "a photo of the exterior of a car",
            "a photo of the interior of a car",
            "a close-up photo of a car dashboard and steering wheel",
            "a close-up photo of a car interior detail",
        ]

        device_name = self._resolve_device(self.config.device, torch)
        self._device_name = device_name
        self._device = torch.device(device_name)

        self._yolo_cache_dir = (
            Path.home() / ".cache" / "car_listing_visual_verification" / "ultralytics"
        )
        self._yolo_cache_dir.mkdir(parents=True, exist_ok=True)
        self._yolo = self._load_yolo_model(YOLO)

        self._clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        self._clip_model = CLIPModel.from_pretrained(self.config.clip_model)
        self._clip_model.to(self._device)
        self._clip_model.eval()

    def _load_yolo_model(self, yolo_cls: Any) -> Any:
        raw_model = self.config.yolo_model.strip()
        explicit_path = Path(raw_model).expanduser()
        has_path_parts = explicit_path.is_absolute() or len(explicit_path.parts) > 1

        if has_path_parts:
            return yolo_cls(explicit_path.as_posix())

        cached_model_path = (self._yolo_cache_dir / raw_model).resolve()
        if not cached_model_path.exists():
            current_dir = Path.cwd()
            try:
                os.chdir(self._yolo_cache_dir)
                yolo_cls(raw_model)
            finally:
                os.chdir(current_dir)

        model = yolo_cls(cached_model_path.as_posix())
        self._cleanup_cwd_model_artifact(raw_model=raw_model, canonical_path=cached_model_path)
        return model

    @staticmethod
    def _cleanup_cwd_model_artifact(raw_model: str, canonical_path: Path) -> None:
        cwd_model_path = Path.cwd() / raw_model
        if not cwd_model_path.exists():
            return
        try:
            if cwd_model_path.resolve() == canonical_path.resolve():
                return
        except OSError:
            pass

        try:
            cwd_model_path.unlink()
        except OSError:
            return

    @staticmethod
    def _resolve_device(preferred: str, torch_module: Any) -> str:
        preferred_norm = preferred.strip().lower()
        if preferred_norm != "auto":
            return preferred_norm

        if (
            getattr(torch_module.backends, "mps", None)
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        if torch_module.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        if not image_path.exists():
            return {
                "car_detected": False,
                "car_conf": None,
                "car_bbox_x1": None,
                "car_bbox_y1": None,
                "car_bbox_x2": None,
                "car_bbox_y2": None,
                "car_bbox_area_ratio": None,
                "exterior_score": None,
                "interior_score": None,
                "content_label": "missing",
                "content_keep": False,
                "content_reason": "missing_file",
                "content_error": f"missing_file: {image_path}",
            }

        detection = self._detect_car(image_path)
        classification = self._classify_exterior(image_path)

        car_detected = bool(detection["car_detected"])
        car_conf = detection.get("car_conf")
        area_ratio = detection.get("car_bbox_area_ratio")
        exterior_score = classification["exterior_score"]
        interior_score = classification["interior_score"]
        content_label = classification["content_label"]

        keep_reasons: list[str] = []
        if not car_detected:
            keep_reasons.append("no_car_detected")
        elif car_conf is not None and car_conf < self.config.min_car_conf:
            keep_reasons.append("low_car_conf")
        elif area_ratio is not None and area_ratio < self.config.min_car_area_ratio:
            keep_reasons.append("small_car_bbox")

        if exterior_score is None:
            keep_reasons.append("missing_clip_score")
        elif exterior_score < self.config.min_exterior_score:
            keep_reasons.append("low_exterior_score")

        if content_label != "exterior":
            keep_reasons.append(f"content_label={content_label}")

        content_keep = not keep_reasons
        content_reason = "ok" if content_keep else "; ".join(keep_reasons)

        errors = [detection.get("content_error"), classification.get("content_error")]
        content_error = "; ".join([item for item in errors if item]) or None

        return {
            "car_detected": car_detected,
            "car_conf": car_conf,
            "car_bbox_x1": detection.get("car_bbox_x1"),
            "car_bbox_y1": detection.get("car_bbox_y1"),
            "car_bbox_x2": detection.get("car_bbox_x2"),
            "car_bbox_y2": detection.get("car_bbox_y2"),
            "car_bbox_area_ratio": area_ratio,
            "exterior_score": exterior_score,
            "interior_score": interior_score,
            "content_label": content_label,
            "content_keep": content_keep,
            "content_reason": content_reason,
            "content_error": content_error,
        }

    def _detect_car(self, image_path: Path) -> dict[str, Any]:
        try:
            yolo_conf = max(0.01, self.config.min_car_conf * 0.5)
            results = self._yolo.predict(
                source=str(image_path),
                conf=yolo_conf,
                verbose=False,
                device=self._device_name,
            )
            if not results:
                return {"car_detected": False, "content_error": "empty_yolo_results"}

            self._cleanup_cwd_model_artifact(
                raw_model=self.config.yolo_model.strip(),
                canonical_path=self._yolo_cache_dir / self.config.yolo_model.strip(),
            )

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return {"car_detected": False}

            names = result.names or {}
            orig_h, orig_w = result.orig_shape
            image_area = float(max(1, orig_h * orig_w))

            best: dict[str, Any] | None = None
            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx].item())
                label = self._class_name(names, cls_id)
                if label != "car":
                    continue

                conf = float(boxes.conf[idx].item())
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].tolist()]
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / image_area)

                candidate = {
                    "car_detected": True,
                    "car_conf": conf,
                    "car_bbox_x1": x1,
                    "car_bbox_y1": y1,
                    "car_bbox_x2": x2,
                    "car_bbox_y2": y2,
                    "car_bbox_area_ratio": area_ratio,
                }
                if best is None:
                    best = candidate
                    continue

                if area_ratio > best["car_bbox_area_ratio"]:
                    best = candidate
                elif area_ratio == best["car_bbox_area_ratio"] and conf > best["car_conf"]:
                    best = candidate

            return best or {"car_detected": False}
        except Exception as exc:  # noqa: BLE001
            return {"car_detected": False, "content_error": f"yolo_error: {exc}"}

    @staticmethod
    def _class_name(names: Any, cls_id: int) -> str:
        label: Any = None
        if isinstance(names, dict):
            label = names.get(cls_id)
        elif isinstance(names, list):
            if 0 <= cls_id < len(names):
                label = names[cls_id]

        if label is None:
            return str(cls_id).lower()
        return str(label).lower().strip()

    def _classify_exterior(self, image_path: Path) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - import guard
            return {
                "exterior_score": None,
                "interior_score": None,
                "content_label": "unknown",
                "content_error": f"pillow_missing: {exc}",
            }

        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")

            inputs = self._clip_processor(
                text=self._clip_prompts,
                images=rgb,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with self._torch.no_grad():
                outputs = self._clip_model(**inputs)
                probs = outputs.logits_per_image[0].softmax(dim=0).detach().cpu().tolist()

            exterior = float(probs[0])
            interior = float(max(probs[1], probs[2], probs[3]))
            margin = exterior - interior

            if margin >= self.config.min_exterior_margin:
                label = "exterior"
            elif -margin >= self.config.min_exterior_margin:
                label = "interior"
            else:
                label = "uncertain"

            return {
                "exterior_score": exterior,
                "interior_score": interior,
                "content_label": label,
                "content_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "exterior_score": None,
                "interior_score": None,
                "content_label": "unknown",
                "content_error": f"clip_error: {exc}",
            }
