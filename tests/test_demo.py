"""Demo contract checks using synthetic images and mocked predictions, never evaluation."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app import ROOT
from app.app import READY, build_demo, clear_inputs, detect_image, preview_image
from app.inference import (BloodCellDetector, CLASSES, DEFAULT_WEIGHTS, DemoError,
                           read_image, resolve_weights)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.weights = self.folder / "best.pt"
        self.weights.write_bytes(b"synthetic checkpoint placeholder; not a model")
        self.image = Image.new("RGB", (4, 3), (11, 22, 33))

    def fake_result(self, classes):
        cls = Mock()
        cls.cpu.return_value.tolist.return_value = classes
        result = Mock()
        result.boxes = SimpleNamespace(cls=cls)
        result.plot.return_value = np.full((3, 4, 3), [33, 22, 11], dtype=np.uint8)
        return result

    def test_weight_precedence_and_repo_relative_paths(self):
        with patch.dict("os.environ", {"BLOOD_CELL_WEIGHTS": "weights/env.pt"}):
            self.assertEqual(resolve_weights("weights/cli.pt"), ROOT / "weights/cli.pt")
            self.assertEqual(resolve_weights(), ROOT / "weights/env.pt")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_weights(), DEFAULT_WEIGHTS)

    def test_missing_weights_never_construct_model(self):
        with patch("ultralytics.YOLO") as yolo:
            detector = BloodCellDetector(self.folder / "absent.pt")
            with self.assertRaisesRegex(DemoError, "E2 best.pt was not found"):
                detector.predict(self.image)
            yolo.assert_not_called()

    def test_empty_corrupt_and_unsupported_inputs(self):
        invalid = self.folder / "broken.png"
        invalid.write_bytes(b"not an image")
        gif = self.folder / "renamed.png"
        self.image.save(gif, format="GIF")
        for value in (None, invalid, gif, 42, self.folder / "image.txt"):
            with self.subTest(value=value), self.assertRaises(DemoError):
                read_image(value)
        for extension in ("jpg", "jpeg", "png"):
            valid = self.folder / ("valid." + extension)
            self.image.save(valid)
            self.assertEqual(read_image(valid).mode, "RGB")

    def test_invalid_confidence_precedes_model_loading(self):
        detector = BloodCellDetector(self.weights)
        with patch.object(detector, "_load_model") as load:
            for conf in (None, "invalid", float("nan"), 0.04, 0.96):
                with self.subTest(conf=conf), self.assertRaises(DemoError):
                    detector.predict(self.image, conf)
            load.assert_not_called()

    def test_frozen_checkpoint_hash_is_required(self):
        detector = BloodCellDetector(self.weights)
        with patch("ultralytics.YOLO") as yolo:
            with self.assertRaisesRegex(DemoError, "not the frozen E2"):
                detector.predict(self.image)
            yolo.assert_not_called()

    def matching_evidence(self):
        return {"selected_experiment": "E2", "test": {
            "checkpoint_sha256": hashlib.sha256(self.weights.read_bytes()).hexdigest()}}

    def test_lazy_model_reuse_cpu_counts_color_and_latency(self):
        model = Mock(task="detect", names=CLASSES)
        model.predict.return_value = [self.fake_result([0, 0, 1, 2])]
        with patch("app.inference.json.loads", return_value=self.matching_evidence()), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("ultralytics.YOLO", return_value=model) as yolo:
            detector = BloodCellDetector(self.weights)
            demo = build_demo(detector)
            self.assertEqual(demo.config["title"], "YOLO26n Blood Cell Detection")
            yolo.assert_not_called()
            for _ in range(2):
                result = detector.predict(self.image, 0.35)
                self.assertEqual(result.counts, {"RBC": 2, "WBC": 1, "Platelets": 1})
                self.assertEqual(result.total, 4)
                self.assertEqual(result.device, "CPU")
                self.assertGreater(result.latency_ms, 0)
                self.assertEqual(result.image.getpixel((0, 0)), (11, 22, 33))
            yolo.assert_called_once_with(str(self.weights))
            args = model.predict.call_args.kwargs
            self.assertEqual((args["imgsz"], args["conf"], args["device"]), (640, 0.35, "cpu"))
            self.assertFalse(args["save"])
            demo.close()

    def test_class_mapping_and_load_failure(self):
        for model, text in ((Mock(task="detect", names={0: "WBC"}), "checkpoint must detect"),
                            (RuntimeError("internal details"), "could not be loaded")):
            with self.subTest(text=text), \
                    patch("app.inference.json.loads", return_value=self.matching_evidence()), \
                    patch("ultralytics.YOLO", side_effect=model if isinstance(model, Exception) else None,
                          return_value=model), \
                    patch("app.inference.LOGGER"):
                with self.assertRaisesRegex(DemoError, text):
                    BloodCellDetector(self.weights).predict(self.image)

    def ready_detector(self, result):
        detector = BloodCellDetector(self.weights)
        detector._model = Mock()
        detector._model.predict.return_value = [result]
        detector._device = "cpu"
        return detector

    def test_zero_detections(self):
        empty = self.fake_result([])
        absent = self.fake_result([])
        absent.boxes = None
        for result in (empty, absent):
            output = self.ready_detector(result).predict(self.image)
            self.assertEqual(output.total, 0)
            self.assertEqual(output.counts, {name: 0 for name in CLASSES.values()})

    def test_cuda_device_and_synchronized_time(self):
        detector = self.ready_detector(self.fake_result([2]))
        detector._device = "cuda:0"
        detector._torch = Mock()
        result = detector.predict(self.image)
        self.assertEqual(result.device, "CUDA")
        self.assertEqual(detector._torch.cuda.synchronize.call_count, 2)

    def test_inference_exception_is_concise_and_clears_outputs(self):
        detector = self.ready_detector(self.fake_result([]))
        detector._model.predict.side_effect = RuntimeError("sensitive internal details")
        with self.assertLogs("app.inference", level="ERROR"):
            output = detect_image(detector, self.image, 0.25)
        self.assertEqual(output[:6], (None,) * 6)
        self.assertNotIn("sensitive", output[-1])
        self.assertIn("Detection failed", output[-1])

    def test_ui_empty_input_and_reset(self):
        output = detect_image(BloodCellDetector(self.weights), None, 0.25)
        self.assertIn("Upload", output[-1])
        self.assertEqual(preview_image(None)[-1], READY)
        reset = clear_inputs()
        self.assertEqual(len(reset), 11)
        self.assertEqual(reset[2], 0.25)
        self.assertEqual(reset[-1], READY)
        self.assertTrue(all(value is None for value in reset[3:9]))


if __name__ == "__main__":
    unittest.main()
