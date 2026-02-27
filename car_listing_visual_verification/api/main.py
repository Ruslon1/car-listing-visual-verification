import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from contextlib import asynccontextmanager

from car_listing_visual_verification.config import MODELS_DIR
from car_listing_visual_verification.modeling.car_gate import CarGate
from car_listing_visual_verification.modeling.predictor import Predictor

CAR_GATE_MODEL = os.getenv("CLVV_CAR_GATE_MODEL", str(MODELS_DIR / "yolov8n.pt"))
CAR_GATE_CONF_THRESHOLD = float(os.getenv("CLVV_CAR_GATE_CONF_THR", "0.20"))
CAR_GATE_MIN_AREA_RATIO = float(os.getenv("CLVV_CAR_GATE_MIN_AREA_RATIO", "0.02"))
CLASSIFIER_UNKNOWN_THRESHOLD = float(os.getenv("CLVV_CLASSIFIER_UNKNOWN_THRESHOLD", "0.45"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    errors = []
    try:
        app.state.predictor = Predictor()
    except Exception as e:
        app.state.predictor = None
        errors.append(f"predictor: {e}")

    try:
        app.state.car_gate = CarGate(
            model_name=CAR_GATE_MODEL,
            conf_threshold=CAR_GATE_CONF_THRESHOLD,
            min_area_ratio=CAR_GATE_MIN_AREA_RATIO,
        )
    except Exception as e:
        app.state.car_gate = None
        errors.append(f"car_gate: {e}")

    app.state.startup_error = " | ".join(errors) if errors else None

    yield


app = FastAPI(
    title="Car listing visual verification API",
    lifespan=lifespan,
)

class PredictResponse(BaseModel):
    label: Literal["no_car", "unknown_car", "car_classified"]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None

@app.get("/health")
def health():
    predictor = getattr(app.state, "predictor", None)
    car_gate = getattr(app.state, "car_gate", None)
    if predictor is None or car_gate is None:
        return {
            "status": "error",
            "model_loaded": predictor is not None,
            "car_gate_loaded": car_gate is not None,
            "error": getattr(app.state, "startup_error", "unknown"),
        }
    return {
        "status": "ok",
        "model_loaded": True,
        "car_gate_loaded": True,
        "device": str(predictor.device),
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    predictor = getattr(app.state, "predictor", None)
    car_gate = getattr(app.state, "car_gate", None)

    if predictor is None or car_gate is None:
        raise HTTPException(status_code=503, detail="Model stack is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    try:
        with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            has_car, gate_conf = car_gate.has_car(Path(tmp.name))
            if not has_car:
                return PredictResponse(label="no_car", confidence=1.0 - gate_conf)

            class_id, class_name, prob = predictor.predict(Path(tmp.name))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}") from e

    if prob < CLASSIFIER_UNKNOWN_THRESHOLD:
        return PredictResponse(label="unknown_car", confidence=prob)

    return PredictResponse(
        label="car_classified",
        confidence=prob,
        class_id=class_id,
        class_name=class_name,
    )
