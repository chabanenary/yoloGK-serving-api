# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_detect_with_image():
    # Crée une image de test
    img = Image.new("RGB", (640, 640), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"confidence": "0.3"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "count" in data

def test_detect_no_file():
    response = client.post("/detect")
    assert response.status_code == 422  # Validation error

def test_health_model_loaded():
    response = client.get("/health")
    assert response.json()["model_loaded"] is True