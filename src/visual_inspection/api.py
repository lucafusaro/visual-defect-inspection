from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from visual_inspection.artifacts import load_model_artifact
from visual_inspection.features import (
    create_feature_extractor,
    extract_pil_image_features,
)
from visual_inspection.inference import predict_features

ARTIFACT_PATH = Path("artifacts/bottle_knn_model.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = load_model_artifact(ARTIFACT_PATH)
    model, preprocess = create_feature_extractor()

    app.state.artifact = artifact
    app.state.model = model
    app.state.preprocess = preprocess

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
):
    content = await image.read()

    try:
        with Image.open(BytesIO(content)) as pil_image:
            features = extract_pil_image_features(
                pil_image,
                request.app.state.model,
                request.app.state.preprocess,
            )
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from error

    return predict_features(features, request.app.state.artifact)
