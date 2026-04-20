FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY inference_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Project source (ML core + inference service)
COPY src/           ./src/
COPY configs/       ./configs/
COPY inference_service/app/     ./inference_service/app/
COPY inference_service/src/     ./inference_service/src/

# Runtime data directories (model + storage)
RUN mkdir -p /app/models /app/data/features /app/data/metadata

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "inference_service.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]