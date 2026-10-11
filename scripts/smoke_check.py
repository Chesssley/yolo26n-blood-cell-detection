"""Opt-in engineering check: train/val loading and synthetic CUDA forward only."""
import argparse
import os
import sys
from pathlib import Path

import _bootstrap
from blood_cell_detection.data import ROOT, CLASSES, check_processed, load_json, require, save_json


def run():
    require(Path(sys.prefix).resolve() == ROOT / ".venv", "Use this repository own .venv")
    os.chdir(ROOT)
    os.environ.update(YOLO_CONFIG_DIR=str(ROOT / ".cache/ultralytics"),
                      MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), YOLO_OFFLINE="true")
    dataset = check_processed()
    import torch
    import ultralytics
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.nn.tasks import DetectionModel
    require(ultralytics.__version__ == "8.4.175" and torch.cuda.is_available(), "Pinned CUDA environment required")
    data = {"names":dict(enumerate(CLASSES)), "nc":3}
    counts = load_json(ROOT / "docs/evidence/frozen_results.json")["dataset"]["split_classes"]
    for split in ("train", "val"):
        ds = YOLODataset(img_path=str(ROOT / f"data/processed/bccd/images/{split}"), data=data,
                         task="detect", imgsz=640, augment=False, cache=False, batch_size=1, rect=False)
        expected = set((ROOT / f"configs/splits/{split}.txt").read_text().split())
        require({Path(p).stem for p in ds.im_files} == expected, "Loader membership drift")
        actual = {c:0 for c in CLASSES}
        for index, label in enumerate(ds.labels):
            item = ds[index]
            require(item["img"].ndim == 3 and len(item["cls"]) == len(label["cls"]), "Loader transform failure")
            for c in label["cls"].flatten():
                actual[CLASSES[int(c)]] += 1
        require(actual == counts[split], "Loader label count drift")
    # Random architecture, no pretrained download or metric evaluation.
    cfg = Path(ultralytics.__file__).parent / "cfg/models/26/yolo26.yaml"
    import yaml
    architecture = yaml.safe_load(cfg.read_text())
    architecture["scale"] = "n"
    torch.manual_seed(42)
    model = DetectionModel(architecture, nc=3, verbose=False).cuda().eval()
    with torch.inference_mode():
        output = model(torch.zeros((1,3,640,640), device="cuda"))
    require(output is not None, "Synthetic forward failed")
    require(check_processed() == dataset, "Data mutated by smoke check")
    save_json(ROOT / "outputs/reproduction/smoke_check.json", {"status":"PASS", "dataset":dataset,
        "scope":"train/val loader transforms plus one synthetic CUDA forward; no performance metrics", "test_calls":0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-smoke", action="store_true")
    args = parser.parse_args()
    require(args.run_smoke, "Smoke check disabled; explicitly enable for a new reproduction")
    run()
