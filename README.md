# Food Image Classifier

A containerized food image classifier with nutrition lookup, built with **Flask + PyTorch (CPU-only) + Docker**. Fine-tuned ResNet50 identifies a photo of a meal as one of six food classes and returns a nutritional breakdown for it.

Built as a course project. The course specification (`ByteBite`) describes a related but not identical system — several deliberate deviations from that spec are documented below.

---

## Classes

```
french_fries · hamburger · pancakes · pizza · sushi · tiramisu
```

## Architecture

| Layer | Choice |
|---|---|
| Model | ResNet50 (torchvision), fine-tuned, 6-class linear head |
| Inference | PyTorch, CPU-only |
| Web framework | Flask + Gunicorn |
| Frontend | Static HTML/CSS/JS (no build step, no external CDN) |
| Containerization | Docker (`python:3.10-slim` base) |

### Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Upload UI |
| `/predict` | POST | `multipart/form-data` image upload → JSON prediction + nutrition |
| `/health` | GET | Liveness probe (used by the Docker `HEALTHCHECK`) |

---

## Deviations from the ByteBite course spec

The spec describes Streamlit + FastAPI + MobileNetV3. This project intentionally diverges in the following ways:

1. **Flask instead of Streamlit/FastAPI.** Flask was already a working, tested stack; running two web frameworks in parallel added complexity without benefit for a single-developer course project. All required user-facing behavior (upload, prediction, nutrition lookup, charts, error handling) is implemented natively in Flask + vanilla JS rather than Streamlit widgets.
2. **ResNet50 instead of MobileNetV3.** The provided fine-tuned weights are ResNet50; the architecture in `app.py` (`build_model()`) matches the checkpoint exactly.
3. **Custom checkpoint format.** The provided `.pth`/`.ckpt` weight files are **not** standard PyTorch checkpoints — they use a hand-rolled, protobuf-style binary format (name + shape + dtype + raw float32 bytes per tensor). This was fully reverse-engineered and is handled transparently by `custom_checkpoint.py`, which detects the format via a magic-byte sniff test and remaps its key names (`gamma`→`weight`, `beta`→`bias`, `moving_mean`→`running_mean`, `moving_variance`→`running_var`, `down_sample`→`downsample`, `classifier`→`fc`) onto a stock `torchvision.models.resnet50()` state dict. Verified: 267/267 renamed tensors match the target architecture's learnable parameters and running buffers exactly, with zero missing or unexpected keys (the remaining 53 keys in a full 320-entry ResNet50 state dict are `num_batches_tracked` counters, which the checkpoint doesn't store and which `load_state_dict(strict=False)` leaves at their default value — harmless for inference).
4. **Offline operation (REQ-012).** All fonts are system font stacks; there is no dependency on Google Fonts, a chart library CDN, or any other external network resource at runtime. Nutrition charts are rendered as inline SVG.

## ⚠️ Known limitation: placeholder nutrition data

The values in `nutrition_db.json` (calories, protein, carbs, fat per class) are **placeholder estimates**, not sourced from a verified nutrition database or lab data. This should be flagged explicitly if this project is submitted or evaluated — swap in a real data source (e.g., USDA FoodData Central) before treating the numbers as accurate.

---

## Running it

### Project layout expected by the Dockerfile

```
.
├── Dockerfile
├── requirements.txt
├── app.py
├── custom_checkpoint.py
├── labels.json
├── nutrition_db.json
├── templates/
│   └── index.html
└── models/
    └── resnet50_food.pth
```

### Build & run

```bash
docker build -t food-classifier .
docker run --rm -p 8080:5000 --name food-classifier food-classifier
```

> Port 8080 is used on the host instead of 5000 because port 5000 conflicts with the macOS AirPlay Receiver on Apple Silicon Macs running Docker Desktop. The container's internal port is still 5000.

Then open **http://localhost:8080**.

### Useful commands

```bash
# Run in the background
docker run -d --rm -p 8080:5000 --name food-classifier food-classifier

# Tail logs
docker logs -f food-classifier

# Stop
docker stop food-classifier

# Liveness check
curl http://localhost:8080/health

# Prediction via curl
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8080/predict
```

### Configuration

Set at container runtime via environment variables (see `app.py`):

| Variable | Default |
|---|---|
| `MODEL_PATH` | `/app/models/resnet50_food.pth` |
| `LABELS_PATH` | `/app/labels.json` |
| `NUTRITION_PATH` | `/app/nutrition_db.json` |

---

## Features

- **Classification**: top-5 predictions with confidence scores
- **Nutrition lookup**: server-side join against `nutrition_db.json` by predicted label
- **Nutritional Analysis Dashboard**: calorie/protein/carb/fat callouts plus an inline SVG donut chart of the macro-nutrient split
- **Fallback alert**: if a predicted class has no nutrition entry, the UI shows an explicit "data unavailable" notice rather than failing silently
- **Client + server-side error handling**: unsupported file types, oversized uploads, and corrupted/unreadable images are caught client-side before upload where possible, with the server (`ALLOWED_EXTENSIONS` check, `UnidentifiedImageError` handling in `app.py`) as a backstop. Errors surface as dismissible toast notifications rather than crashing the page.
- **Responsive layout**: two-column desktop layout (upload panel + results panel), collapsing to a single column on narrower screens

---

## Testing

| Class | Tested? | Notes |
|---|---|---|
| french_fries | ✅ | *(fill in observed accuracy / example results)* |
| hamburger | ✅ | |
| pancakes | ✅ | |
| pizza | ✅ | |
| sushi | ✅ | |
| tiramisu | ✅ | |

> All six classes have been exercised end-to-end through the running container. Add specific accuracy figures, sample image counts, or misclassification notes here if the course submission requires quantified results.

---

## Possible next steps (not yet implemented)

- `docker-compose` setup for simplified local orchestration
- "Model Management" admin use case from the course spec (out of scope; stretch goal only)
- Replacing placeholder nutrition data with a verified source
