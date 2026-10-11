"""Single-image inference with the frozen, validation-selected E2 checkpoint."""
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from threading import Lock
import time

from PIL import Image, ImageOps

from app import ROOT

LOGGER = logging.getLogger(__name__)
# Preserve Pillow's decoder before the lazy Ultralytics import installs its HEIF hook.
OPEN_IMAGE = Image.open
CLASSES = {0: "RBC", 1: "WBC", 2: "Platelets"}
DEFAULT_WEIGHTS = ROOT / "outputs/experiments/e2_class_weight_640/weights/best.pt"


class DemoError(Exception):
    """A concise message suitable for the demo UI."""


def resolve_weights(weights=None):
    """CLI > BLOOD_CELL_WEIGHTS > original E2 run; relative paths use repo root."""
    value = weights if weights is not None else os.environ.get("BLOOD_CELL_WEIGHTS")
    path = Path(value).expanduser() if value else DEFAULT_WEIGHTS
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_image(image):
    """Decode only a local JPEG/PNG or PIL image; never fetch a URL."""
    if image is None:
        raise DemoError("Upload a JPG, JPEG or PNG image first.")
    try:
        if isinstance(image, Image.Image):
            image.load()
            if image.format not in (None, "JPEG", "PNG"):
                raise DemoError("Please use a JPG, JPEG or PNG image.")
            return ImageOps.exif_transpose(image).convert("RGB")
        if not isinstance(image, (str, Path)):
            raise DemoError("The image could not be read. Upload a valid JPG, JPEG or PNG.")
        path = Path(image)
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise DemoError("Please use a JPG, JPEG or PNG image.")
        with OPEN_IMAGE(path, formats=("JPEG", "PNG")) as decoded:
            if decoded.format not in ("JPEG", "PNG"):
                raise DemoError("Please use a JPG, JPEG or PNG image.")
            decoded.load()
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DemoError("The image could not be read. Upload a valid JPG, JPEG or PNG.") from exc


@dataclass
class DetectionResult:
    image: Image.Image
    counts: dict[str, int]
    total: int
    latency_ms: float
    device: str


class BloodCellDetector:
    def __init__(self, weights=None):
        self.weights = resolve_weights(weights)
        self._model = None
        self._torch = None
        self._device = None
        self._lock = Lock()

    def check_weights(self):
        """Existence check only: startup does not load a model or download weights."""
        if not self.weights.is_file():
            raise DemoError(
                "E2 best.pt was not found. Provide its local path with --weights "
                "or BLOOD_CELL_WEIGHTS. No weights are downloaded or trained automatically."
            )
        if self.weights.suffix.lower() != ".pt":
            raise DemoError("Provide the original E2 best.pt checkpoint.")

    def _load_model(self):
        if self._model is not None:
            return
        self.check_weights()
        try:
            evidence = json.loads((ROOT / "docs/evidence/frozen_results.json").read_text(encoding="utf-8"))
            with self.weights.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if evidence["selected_experiment"] != "E2" or digest != evidence["test"]["checkpoint_sha256"]:
                raise DemoError("This file is not the frozen E2 checkpoint. Provide the selected E2 best.pt.")
            import torch
            from ultralytics import YOLO

            model = YOLO(str(self.weights))
            if model.task != "detect" or model.names != CLASSES:
                raise DemoError("The checkpoint must detect 0: RBC, 1: WBC and 2: Platelets.")
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._torch = torch
            self._model = model
        except DemoError:
            raise
        except Exception as exc:
            LOGGER.exception("E2 model loading failed")
            raise DemoError("E2 could not be loaded. Check the checkpoint and installed dependencies.") from exc

    def predict(self, image, confidence=0.25):
        # Serialize model loading and prediction, including calls outside Gradio.
        with self._lock:
            started = time.perf_counter()
            source = read_image(image)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise DemoError("Confidence must be between 0.05 and 0.95.") from exc
            if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
                raise DemoError("Confidence must be between 0.05 and 0.95.")
            self._load_model()
            try:
                if self._device.startswith("cuda"):
                    self._torch.cuda.synchronize()
                result = self._model.predict(
                    source=source, imgsz=640, conf=confidence, device=self._device,
                    verbose=False, save=False, save_txt=False, save_crop=False,
                )[0]
                counts = {name: 0 for name in CLASSES.values()}
                if result.boxes is not None:
                    for class_id in result.boxes.cls.cpu().tolist():
                        if class_id not in CLASSES:
                            raise DemoError("Unexpected detection class. Check the E2 checkpoint.")
                        counts[CLASSES[class_id]] += 1
                # Ultralytics plot() returns BGR; Gradio/PIL expect RGB.
                annotated = Image.fromarray(result.plot()[..., ::-1].copy())
                if self._device.startswith("cuda"):
                    self._torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                return DetectionResult(annotated, counts, sum(counts.values()), elapsed,
                                       "CUDA" if self._device.startswith("cuda") else "CPU")
            except DemoError:
                raise
            except Exception as exc:
                LOGGER.exception("Single-image inference failed")
                raise DemoError("Detection failed. Try another image or restart the local demo.") from exc
