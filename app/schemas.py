# app/schemas.py
from pydantic import BaseModel

class Detection(BaseModel):
    bbox: list[float]
    confidence: float
    class_id: int

class DetectionResponse(BaseModel):
    filename: str
    detections: list[Detection]
    count: int