from pathlib import Path

from ultralytics import YOLO


class CarGate:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.20,
        min_area_ratio: float = 0.02,
        vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7),
    ) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.min_area_ratio = min_area_ratio
        self.vehicle_class_ids = set(vehicle_class_ids)

    def has_car(self, image_path: Path) -> tuple[bool, float]:
        result = self.model(str(image_path), verbose=False)[0]
        h, w = result.orig_shape
        image_area = float(h * w)

        if result.boxes is None:
            return False, 0.0

        best_conf = 0.0
        for xyxy, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            cls_id = int(cls.item())
            current_conf = float(conf.item())

            if cls_id not in self.vehicle_class_ids:
                continue

            x1, y1, x2, y2 = [float(v.item()) for v in xyxy]
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area

            if current_conf >= self.conf_threshold and area_ratio >= self.min_area_ratio:
                best_conf = max(best_conf, current_conf)

        return best_conf > 0.0, best_conf
