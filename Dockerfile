# Dockerfile

# --- Mode CPU (natif) ---
FROM python:3.11-slim

# --- Mode GPU NVIDIA (décommenter pour production avec CUDA) ---
# FROM --platform=linux/amd64 nvidia/cuda:12.4.1-runtime-ubuntu22.04
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.11 python3.11-dev python3-pip \
#     libgl1 libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*
# RUN ln -s /usr/bin/python3.11 /usr/local/bin/python && \
#     ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
