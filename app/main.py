# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.model import YOLOInference
import io
from pathlib import Path
from PIL import Image

app = FastAPI(
    title="YOLO Inference API",
    description="Real-time object detection via REST API",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")

# Chargement du modèle au démarrage (une seule fois)
model = YOLOInference("models/yolo11n_GK.onnx")

@app.get("/health")
def health_check():
    """Endpoint de santé — utile pour Kubernetes liveness probe"""
    return {"status": "healthy", "model_loaded": model.is_loaded()}

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Reçoit une image, retourne les détections YOLO.
    
    - **file**: image (JPEG, PNG)
    - **confidence**: seuil de confiance (0.0 - 1.0)
    """
    # Lire l'image uploadée
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Inférence
    detections = model.predict(image, conf_threshold=confidence)
    
    return JSONResponse(content={
        "filename": file.filename,
        "detections": detections,
        "count": len(detections)
    })

@app.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...)
):
    """Inférence sur plusieurs images en une requête"""
    results = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        detections = model.predict(image)
        results.append({
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        })
    return JSONResponse(content={"results": results})