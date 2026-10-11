"""Local interactive demo; disposable framework state stays in this repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for variable, folder in (
    ("YOLO_CONFIG_DIR", "ultralytics"),
    ("MPLCONFIGDIR", "matplotlib"),
    ("GRADIO_TEMP_DIR", "gradio"),
):
    cache = ROOT / ".cache" / folder
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(cache)
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
