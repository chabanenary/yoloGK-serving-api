# yoloGK-serving-api

API REST de détection d'objets basée sur un modèle YOLO11 custom, servie avec FastAPI et packagée via Docker.

## Classes détectées

Le modèle `yolo11n_GK.onnx` est entraîné sur 5 classes :

| ID | Classe |
|----|--------|
| 0 | adult |
| 1 | child |
| 2 | creditcard |
| 3 | id |
| 4 | nude |

## Stack

- **FastAPI** — API REST
- **ONNX Runtime** — inférence YOLO11 sans dépendance PyTorch
- **Pillow** — prétraitement des images
- **Uvicorn** — serveur ASGI
- **Docker / Podman** — containerisation

## Structure

```
yolo-serving-api/
├── app/
│   ├── main.py          # Routes FastAPI + UI statique
│   ├── model.py         # Chargement ONNX + pré/post-traitement
│   └── static/
│       └── index.html   # Interface graphique de test
├── models/
│   └── yolo11n_GK.onnx  # Modèle custom (non versionné)
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Lancer l'API

### Avec Docker / Podman

```bash
# Build
podman build -t yolo-serving-api .

# Run
podman run -p 8000:8000 yolo-serving-api
```

### Avec docker-compose

```bash
docker-compose up --build
```

### En local (sans Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Interface graphique

Ouvre [http://localhost:8000](http://localhost:8000) dans ton navigateur.

- Glisse-dépose une image ou clique pour en choisir une
- Ajuste le seuil de confiance
- Clique sur **Détecter**

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Interface graphique |
| `GET` | `/health` | Santé de l'API |
| `POST` | `/detect` | Détection sur une image |
| `POST` | `/detect/batch` | Détection sur plusieurs images |

### Exemple `/detect`

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```
 ```bash
 curl -X POST "127.0.0.1:8000/detect" \ 
  -F "file=@image.jpeg" \
  -F "confidence=0.5"
```

Réponse :

```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "bbox": [120.0, 45.0, 380.0, 620.0],
      "confidence": 0.874,
      "class_id": 0,
      "class": "adult"
    }
  ],
  "count": 1
}
```

### Exemple `/detect/batch`

```bash
curl -X POST "http://localhost:8000/detect/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

## Documentation interactive

Swagger UI disponible sur [http://localhost:8000/docs](http://localhost:8000/docs)

## Tests

Le fichier `tests/test_api.py` contient 4 tests automatisés :

| Test | Description |
|------|-------------|
| `test_health` | Vérifie que l'endpoint `/health` répond 200 |
| `test_health_model_loaded` | Vérifie que le modèle est bien chargé au démarrage |
| `test_detect_with_image` | Envoie une image synthétique (640×640 rouge) à `/detect` et vérifie la structure de la réponse |
| `test_detect_no_file` | Vérifie que `/detect` sans fichier retourne une erreur 422 |

### Lancer les tests

**Dans le container (après build) :**

```bash
podman run --rm yolo-serving-api python -m pytest tests/test_api.py -v
```

**Sans rebuild, en montant le dossier local :**

```bash
podman run --rm -v $(pwd)/tests:/app/tests yolo-serving-api python -m pytest tests/test_api.py -v
```
