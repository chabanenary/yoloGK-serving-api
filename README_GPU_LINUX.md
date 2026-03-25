# GPU Support — Linux/NVIDIA

Résumé des modifications apportées pour activer l'inférence GPU avec ONNX Runtime.

## Ce qui a été modifié

### `Dockerfile`
- Base image changée de `python:3.11-slim` vers `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- Python 3.11 installé manuellement par-dessus l'image CUDA
- `--platform=linux/amd64` forcé (onnxruntime-gpu n'a pas de wheel ARM64)

### `requirements.txt`
- `onnxruntime==1.19.0` → `onnxruntime-gpu==1.19.0`

### `app/model.py`
- Session ONNX configurée avec deux providers :
  ```python
  providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
  ```
  → GPU si disponible, sinon fallback CPU automatique

---

## Build et run

```bash
# Build (depuis macOS ARM64, forcer la plateforme)
podman build --platform linux/amd64 -t yolo-serving-api .

# Run sur serveur Linux avec GPU NVIDIA
podman run --gpus all -p 8000:8000 yolo-serving-api

# Run sans GPU (fallback CPU)
podman run -p 8000:8000 yolo-serving-api
```

---

## Prérequis sur le serveur hôte

Pour que `--gpus all` fonctionne, le serveur doit avoir :

- Driver NVIDIA installé
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/) installé
- CUDA 12.x + cuDNN 9.x disponibles

---

## Situation sur macOS Apple Silicon (M1/M2/M3)

| Problème | Cause |
|----------|-------|
| CUDA non détecté | Pas de driver NVIDIA sur Mac |
| `libcudnn.so.9` introuvable | cuDNN absent |
| Inférence lente | CPU + émulation QEMU (amd64 sur arm64) |

**Fallback automatique** : l'API démarre quand même en CPU grâce au `CPUExecutionProvider`.

Pour développer en local sans pénalité d'émulation, lancer nativement :

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Options cloud recommandées

| Provider | Instance | GPU |
|----------|----------|-----|
| AWS | `g4dn.xlarge` | T4 |
| Google Cloud | `n1-standard` + T4 | T4 |
| RunPod | GPU pod | T4 / A100 |
