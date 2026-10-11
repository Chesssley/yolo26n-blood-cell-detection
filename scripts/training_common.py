"""Shared project paths, artifact helpers and strict train/val model-loader guards."""
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = ROOT / "weights/pretrained/yolo26n.pt"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_info(path):
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha(path)}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def image_paths(split):
    ids = (ROOT / f"configs/splits/{split}.txt").read_text().split()
    return {ROOT / f"data/processed/bccd/images/{split}/{image_id}.jpg" for image_id in ids}


def guard_files(dataset, split):
    actual = {Path(p).resolve() for p in dataset.im_files}
    check(actual == image_paths(split), f"Model loader membership differs from corrected {split}")
    return {"split": split, "count": len(actual), "images": sorted(p.relative_to(ROOT).as_posix() for p in actual)}


class ValOnlyValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        check(self.args.split == "val", "Forbidden validation split")
        check(Path(img_path).resolve() == ROOT / "data/processed/bccd/images/val", "Forbidden validation path")
        dataset = super().build_dataset(img_path, mode, batch)
        guard_files(dataset, "val")
        return dataset


class TrainValOnlyTrainer(DetectionTrainer):
    def get_dataset(self):
        check(self.args.split == "val", "Forbidden trainer validation split")
        data = super().get_dataset()
        data.pop("test", None)  # test path never reaches model dataloaders
        return data

    def build_dataset(self, img_path, mode="train", batch=None):
        check(mode in ("train", "val"), "Forbidden trainer dataset mode")
        check(Path(img_path).resolve() == ROOT / f"data/processed/bccd/images/{mode}", "Forbidden trainer path")
        dataset = super().build_dataset(img_path, mode, batch)
        guard_files(dataset, mode)
        return dataset

    def get_validator(self):
        return ValOnlyValidator(self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args), _callbacks=self.callbacks)


def metrics(result):
    values = result.results_dict
    output = {k: float(v) for k, v in values.items()}
    check(all(math.isfinite(v) for v in output.values()), "Nonfinite validation metric")
    output["per_class"] = {result.names[int(c)]: dict(zip(("precision", "recall", "map50", "map50_95"),
                            map(float, result.box.class_result(i)))) for i, c in enumerate(result.box.ap_class_index)}
    return output
