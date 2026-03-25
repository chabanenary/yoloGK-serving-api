# app/model.py
import onnxruntime as ort
import numpy as np
from PIL import Image

CLASS_NAMES = ["adult", "child", "creditcard", "id", "nude"]


class YOLOInference:
    def __init__(self, model_path: str):
        # --- Mode CPU (natif) ---
        self.session = ort.InferenceSession(model_path)
        # --- Mode GPU NVIDIA (décommenter pour production avec CUDA) ---
        # self.session = ort.InferenceSession(
        #     model_path,
        #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        # )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # ex: [1, 3, 640, 640]
        
    def is_loaded(self) -> bool:
        return self.session is not None
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Redimensionne et normalise l'image pour YOLO"""
        img = image.resize((self.input_shape[3], self.input_shape[2]))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)     # Ajoute batch dimension
        return img
    
    def postprocess(self, outputs, conf_threshold: float) -> list:
        """Parse les sorties YOLO11/YOLOv8 : shape [1, 9, 8400] (4 bbox + 5 classes)"""
        detections = []
        raw = outputs[0]  # [1, 9, 8400]

        # Transpose → [8400, 9]
        raw = np.transpose(raw[0], (1, 0))

        for det in raw:
            bbox = det[:4]           # cx, cy, w, h
            class_scores = det[4:]   # 5 scores
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if conf >= conf_threshold:
                cx, cy, w, h = bbox
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                detections.append({
                    "bbox": [round(float(x1), 1), round(float(y1), 1),
                             round(float(x2), 1), round(float(y2), 1)],
                    "confidence": round(conf, 3),
                    "class_id": class_id,
                    "class": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                })

        # Tri par confiance décroissante
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
    
    def predict(self, image: Image.Image, conf_threshold: float = 0.5) -> list:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, conf_threshold)