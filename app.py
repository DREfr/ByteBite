"""
app.py
------
Flask + PyTorch (CPU) inference service for a fine-tuned ResNet50
food-image classifier.

Endpoints
---------
GET  /            -> HTML upload UI
POST /predict      -> multipart/form-data image upload, returns JSON
GET  /health        -> simple liveness probe (useful for Docker HEALTHCHECK)

Configuration is done via environment variables so the same image can be
reused with different weight files / label sets without rebuilding:

    MODEL_PATH   path to the .pth/.ckpt weight file  (default: models/resnet50_food.pth)
    LABELS_PATH  path to a JSON list of class names   (default: labels.json)
"""

import io
import json
import logging
import os

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from custom_checkpoint import is_custom_checkpoint, parse_custom_checkpoint

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/resnet50_food.pth")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")
NUTRITION_PATH = os.environ.get("NUTRITION_PATH", "nutrition_db.json")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("food-classifier")

# Force CPU-only execution regardless of what's compiled into the torch build.
DEVICE = torch.device("cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# --------------------------------------------------------------------------
# Label loading
# --------------------------------------------------------------------------
def load_labels(path: str) -> list:
    """Load an ordered list of class names. Index position == model output index."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labels file not found at '{path}'. Provide a JSON array of class "
            f"names in the same order used during training, e.g. "
            f'["french_fries", "hamburger", "pancakes", "pizza", "sushi", "tiramisu"]'
        )
    with open(path, "r") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Labels file '{path}' must contain a non-empty JSON array.")
    return labels


CLASS_NAMES = load_labels(LABELS_PATH)
NUM_CLASSES = len(CLASS_NAMES)
log.info("Loaded %d classes from %s: %s", NUM_CLASSES, LABELS_PATH, CLASS_NAMES)


# --------------------------------------------------------------------------
# Nutrition lookup
# --------------------------------------------------------------------------
def load_nutrition_db(path: str) -> dict:
    """
    Load a {class_name: {serving, calories, protein_g, carbs_g, fat_g}} map.
    Returns an empty dict (rather than raising) if the file is missing, so
    classification still works even without nutrition data configured.
    """
    if not os.path.exists(path):
        log.warning("Nutrition DB not found at '%s'; nutrition info will be omitted.", path)
        return {}
    with open(path, "r") as f:
        return json.load(f)


NUTRITION_DB = load_nutrition_db(NUTRITION_PATH)
log.info("Loaded nutrition data for %d classes from %s", len(NUTRITION_DB), NUTRITION_PATH)


# --------------------------------------------------------------------------
# Model definition & weight loading
# --------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    """Recreate the exact ResNet50 architecture used at training time."""
    model = models.resnet50(weights=None)  # no ImageNet download; we load our own weights
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _strip_prefix(state_dict: dict, prefixes=("model.", "backbone.", "module.")) -> dict:
    """
    Some training frameworks (PyTorch Lightning, DataParallel, custom
    trainers) save state_dict keys with a wrapper prefix such as
    'model.conv1.weight' instead of 'conv1.weight'. This strips the first
    matching prefix so the keys line up with a plain torchvision ResNet50.
    """
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_weights(path: str) -> dict:
    """
    Load a checkpoint file and normalize it down to a plain state_dict
    matching torchvision's ResNet50 key names, regardless of whether the
    file is:
      - this team's custom binary format (resnet50_food*.ckpt) -- a
        hand-rolled protobuf-style serialization, NOT a standard PyTorch
        checkpoint. See custom_checkpoint.py for the full format writeup.
      - a standard PyTorch state_dict (torch.save(model.state_dict(), path))
      - a full checkpoint dict with a 'state_dict' / 'model_state_dict' key
        (common with PyTorch Lightning .ckpt files)
      - a pickled full model object (torch.save(model, path))
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. Set the MODEL_PATH environment "
            f"variable or place your weight file at that location."
        )

    if is_custom_checkpoint(path):
        log.info("Detected custom binary checkpoint format for '%s'.", path)
        return parse_custom_checkpoint(path)

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # Someone saved the entire model object rather than a state_dict.
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
        return _strip_prefix(checkpoint)

    raise TypeError(f"Unrecognized checkpoint format in '{path}': {type(checkpoint)}")


def load_model() -> nn.Module:
    model = build_model(NUM_CLASSES)
    state_dict = load_weights(MODEL_PATH)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys when loading weights (using random init for these): %s", missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint that were ignored: %s", unexpected)

    model.to(DEVICE)
    model.eval()  # disable dropout/batchnorm updates for inference
    return model


log.info("Loading model from %s ...", MODEL_PATH)
MODEL = load_model()
log.info("Model loaded and ready for inference.")


# --------------------------------------------------------------------------
# Preprocessing (must match training-time transforms)
# --------------------------------------------------------------------------
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet channel means
            std=[0.229, 0.224, 0.225],   # ImageNet channel std devs
        ),
    ]
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_bytes: bytes, top_k: int = 5) -> list:
    """Run preprocessing + inference and return the top_k (label, confidence) pairs."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():  # no need to track gradients at inference time
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, NUM_CLASSES)
    top_probs, top_idxs = torch.topk(probabilities, k=top_k)

    return [
        {"label": CLASS_NAMES[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_idxs)
    ]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "num_classes": NUM_CLASSES}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Expected form field 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}
        ), 400

    try:
        image_bytes = file.read()
        predictions = predict_image(image_bytes)
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid or readable image."}), 400
    except Exception as exc:  # noqa: BLE001 - surface any inference error as a 500
        log.exception("Inference failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    return jsonify(
        {
            "top_prediction": predictions[0],
            "top_5": predictions,
            "nutrition": NUTRITION_DB.get(predictions[0]["label"]),
        }
    ), 200


if __name__ == "__main__":
    # Dev-server entry point. In the Docker image, gunicorn is used instead (see Dockerfile).
    app.run(host="0.0.0.0", port=5000, debug=False)
