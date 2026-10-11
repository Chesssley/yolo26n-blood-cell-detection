"""Run the local Gradio blood-cell demo: python app/app.py --weights best.pt."""
import argparse
import logging
from pathlib import Path
import sys

# Support both python app/app.py and python -m app.app from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import BloodCellDetector, DemoError, read_image
import gradio as gr

LOGGER = logging.getLogger(__name__)
READY = "Upload an image, adjust confidence, then select Detect."
DISCLAIMER = (
    "Research / educational demonstration only. "
    "Not for clinical diagnosis, medical decision-making, or treatment."
)


def detect_image(detector, image, confidence):
    try:
        result = detector.predict(image, confidence)
        return (result.image, result.counts["RBC"], result.counts["WBC"],
                result.counts["Platelets"], result.total, round(result.latency_ms, 2),
                result.device, "Detection complete.")
    except DemoError as exc:
        return (None, None, None, None, None, None, "", str(exc))
    except Exception:
        LOGGER.exception("Unexpected demo error")
        return (None, None, None, None, None, None, "", "Detection failed. Restart the local demo and try again.")


def preview_image(image):
    if image is None:
        return (None, None, None, None, None, None, None, "", READY)
    try:
        return (read_image(image), None, None, None, None, None, None, "", READY)
    except DemoError as exc:
        return (None, None, None, None, None, None, None, "", str(exc))


def clear_inputs():
    return (None, None, 0.25, None, None, None, None, None, None, "", READY)


def build_demo(detector):
    with gr.Blocks(title="YOLO26n Blood Cell Detection", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# YOLO26n Blood Cell Detection\nInteractive object detection demo for RBC, WBC and Platelets.")
        gr.Markdown("**" + DISCLAIMER + "**")
        with gr.Row():
            with gr.Column():
                upload = gr.File(label="Upload JPG / JPEG / PNG", file_types=[".jpg", ".jpeg", ".png"],
                                 file_count="single", type="filepath")
                original = gr.Image(label="Original image", interactive=False, format="png", buttons=[])
                confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                with gr.Row():
                    detect = gr.Button("Detect", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column():
                annotated = gr.Image(label="Detection result", interactive=False, format="png", buttons=[])
                with gr.Row():
                    rbc = gr.Number(label="RBC", interactive=False, precision=0)
                    wbc = gr.Number(label="WBC", interactive=False, precision=0)
                    platelets = gr.Number(label="Platelets", interactive=False, precision=0)
                total = gr.Number(label="Total detections", interactive=False, precision=0)
                latency = gr.Number(label="Local end-to-end inference time (ms)", interactive=False, precision=2)
                device = gr.Textbox(label="Device", interactive=False)
                gr.Markdown("Time includes image decoding, first-use model loading, prediction and drawing. "
                            "Browser upload/display and queue waiting are excluded; this is not an experiment benchmark.")
        status = gr.Markdown(READY)
        outputs = [annotated, rbc, wbc, platelets, total, latency, device, status]
        event = detect.click(lambda image, conf: detect_image(detector, image, conf),
                             [upload, confidence], outputs, concurrency_limit=1, api_name="detect")
        # Cancel delivery of old results when the displayed input changes.
        # An in-flight synchronous prediction may finish, but cannot update the UI.
        upload.change(preview_image, upload, [original, *outputs],
                      cancels=[event], api_name=False)
        clear.click(clear_inputs, outputs=[upload, original, confidence, *outputs],
                    cancels=[event], api_name=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Local educational YOLO26n E2 blood-cell detection demo.")
    parser.add_argument("--weights", help="Path to the frozen E2 best.pt; overrides BLOOD_CELL_WEIGHTS.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    detector = BloodCellDetector(args.weights)
    try:
        detector.check_weights()
    except DemoError as exc:
        parser.exit(2, f"Error: {exc}\n")
    build_demo(detector).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, show_error=False,
    )


if __name__ == "__main__":
    main()
