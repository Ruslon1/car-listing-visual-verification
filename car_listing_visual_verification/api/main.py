from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from car_listing_visual_verification.modeling.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = Predictor()
        app.state.startup_error = None
    except Exception as e:
        app.state.predictor = None
        app.state.startup_error = str(e)

    yield


app = FastAPI(
    title="Car listing visual verification API",
    lifespan=lifespan,
)

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float

@app.get("/health")
def health():
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        return {"status": "error", "model_loaded": False, "error": getattr(app.state, "startup_error", "unknown")}
    return {"status": "ok", "model_loaded": True, "device": str(predictor.device)}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    predictor = getattr(app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    try:
        with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            class_id, class_name, prob = predictor.predict(Path(tmp.name))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}") from e

    return PredictResponse(class_id=class_id, class_name=class_name, probability=prob)