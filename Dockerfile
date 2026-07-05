# --------------------------------------------------------------------------
# Lightweight CPU-only image for serving a ResNet50 food classifier
# --------------------------------------------------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files / buffering stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install OS packages needed by Pillow for common image formats (jpeg/png/webp).
# Combined into one RUN + cleanup to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency layer -------------------------------------------------------
# Copy ONLY requirements.txt first so Docker can cache this layer.
# As long as requirements.txt doesn't change, `pip install` won't re-run
# even if application code changes below.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application layer -------------------------------------------------------
# Now copy the rest of the app. Code changes only invalidate this layer,
# not the (slow) pip install layer above.
COPY app.py .
COPY custom_checkpoint.py .
COPY labels.json .
COPY nutrition_db.json .
COPY templates/ templates/
COPY models/ models/

# Default weight file location inside the image; override at `docker run`
# time with -e MODEL_PATH=/app/models/your_file.pth if you mount a different one.
ENV MODEL_PATH=/app/models/resnet50_food.pth
ENV LABELS_PATH=/app/labels.json
ENV NUTRITION_PATH=/app/nutrition_db.json

EXPOSE 5000

# Basic container-level health check, hits the /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run with gunicorn (production WSGI server) instead of Flask's dev server.
# 1 worker is usually enough for CPU-bound single-model inference; increase
# --workers if you have more CPU cores and want concurrent requests, but note
# each worker loads its own full copy of the model into memory.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "60", "app:app"]
