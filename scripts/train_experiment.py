"""Explicit fresh reproduction: independent pretrained initialization, train/val only."""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import _bootstrap
from blood_cell_detection.data import ROOT, check_processed, load_json, read_yaml, require, save_json, sha

CONFIGS = {"E1": "e1_baseline_640", "E2": "e2_class_weight_640", "E3": "e3_class_weight_512"}


def run(experiment):
    require(Path(sys.prefix).resolve() == ROOT / ".venv", "Use this repository's own .venv")
    os.chdir(ROOT)
    os.environ.update(YOLO_CONFIG_DIR=str(ROOT / ".cache/ultralytics"),
                      MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), YOLO_OFFLINE="true",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    config = read_yaml(ROOT / f"configs/experiments/{CONFIGS[experiment]}.yaml")
    require(config.pop("experiment_id") == experiment, "Wrong experiment ID")
    published = load_json(ROOT / "docs/evidence/frozen_results.json")
    require(config == published["experiments"][experiment]["config"], "Frozen experiment configuration changed")
    dataset_audit = check_processed()
    smoke = load_json(ROOT / "outputs/reproduction/smoke_check.json")
    require(smoke["status"] == "PASS" and smoke["dataset"] == dataset_audit, "Complete the smoke check on current data first")
    pretrained = ROOT / config["model"]
    require(pretrained.exists() and sha(pretrained) == published["pretrained_sha"], "Official pretrained hash mismatch")
    run_dir = ROOT / "outputs/reproduction" / experiment
    require(not run_dir.exists(), "Existing attempt; refusing overwrite or resume")
    run_dir.mkdir(parents=True)
    dataset_file = run_dir / "train_val_only.yaml"
    import yaml
    dataset_file.write_text(yaml.safe_dump({"path": str(ROOT / "data/processed/bccd"),
        "train": "images/train", "val": "images/val", "names": {0: "RBC", 1: "WBC", 2: "Platelets"}},
        sort_keys=False), encoding="utf-8")
    # No test path is supplied to either trainer or validator.
    from training_common import TrainValOnlyTrainer, ValOnlyValidator, guard_files, metrics
    import torch
    import ultralytics
    import torchvision
    from ultralytics import YOLO
    require((torch.__version__, torchvision.__version__, ultralytics.__version__) ==
            ("2.10.0+cu128", "0.25.0+cu128", "8.4.175"), "Pinned framework version drift")
    require(torch.cuda.is_available(), "CUDA required by the recorded protocol")
    args = dict(config, model=str(pretrained), data=str(dataset_file), project=str(run_dir), name="fit",
                split="val", val=True, resume=False, exist_ok=False)
    attempts = {"status": "PREPARED", "train_calls": 0, "independent_val_calls": 0, "test_calls": 0}
    save_json(run_dir / "attempt.json", attempts)
    save_json(run_dir / "input_hashes.json", {"dataset": dataset_audit, "pretrained_sha256": sha(pretrained),
                                              "config_sha256": sha(ROOT / f"configs/experiments/{CONFIGS[experiment]}.yaml")})
    save_json(run_dir / "environment.json", {"Python": platform.python_version(), "OS": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0)})
    model = YOLO(str(pretrained))
    epoch_start, epochs = [0.0], []

    def ready(trainer):
        require(trainer.device.type == "cuda" and bool(trainer.amp), "Device/AMP drift")
        require(trainer.args.cls_pw == config["cls_pw"] and not trainer.args.resume, "Training protocol drift")
        expected = read_yaml(ROOT / f"configs/resolved/{experiment.lower()}.yaml")
        actual = vars(trainer.args)
        allowed = {"model", "data", "project", "name", "save_dir"}
        for key, value in expected.items():
            if key not in allowed:
                require(str(actual.get(key)) == str(value), "Uncontrolled resolved parameter: " + key)
        save_json(run_dir / "resolved_args.json", vars(trainer.args))
        optimizer = {"class": type(trainer.optimizer).__name__,
                     "groups": [{k: v for k, v in g.items() if k != "params"} for g in trainer.optimizer.param_groups]}
        save_json(run_dir / "optimizer.json", optimizer)
        save_json(run_dir / "loaders.json", {"train": guard_files(trainer.train_loader.dataset, "train"),
                                              "val": guard_files(trainer.test_loader.dataset, "val")})
        weights = getattr(trainer.model, "class_weights", None)
        save_json(run_dir / "class_weights.json", {"cls_pw": config["cls_pw"],
            "train_counts": trainer.get_class_counts().tolist(),
            "weights": weights.detach().cpu().tolist() if weights is not None else [1.0]*3})
        torch.cuda.reset_peak_memory_stats(0)

    def verify_loss(trainer):
        if config["cls_pw"] == 0:
            return
        criterion = trainer.model.criterion
        criteria = [criterion.one2many, criterion.one2one] if hasattr(criterion, "one2many") else [criterion]
        for loss in criteria:
            require(torch.equal(loss.class_weights.flatten(), trainer.model.class_weights.flatten()),
                    "Loss does not consume measured class weights")

    def begin(trainer):
        epoch_start[0] = time.perf_counter()

    def end(trainer):
        epochs.append({"epoch": trainer.epoch + 1, "train_val_seconds": time.perf_counter() - epoch_start[0]})

    model.add_callback("on_pretrain_routine_end", ready)
    model.add_callback("on_train_epoch_start", begin)
    model.add_callback("on_fit_epoch_end", end)
    model.add_callback("on_train_batch_end", verify_loss)
    attempts.update(status="TRAINING", train_calls=1)
    save_json(run_dir / "attempt.json", attempts)
    before = time.perf_counter()
    model.train(trainer=TrainValOnlyTrainer, **args)
    resources = {"wall_seconds": time.perf_counter() - before,
        "epochs": epochs, "allocated_bytes": torch.cuda.max_memory_allocated(0),
        "reserved_bytes": torch.cuda.max_memory_reserved(0), "scope": "single historical training; callbacks unsynchronized"}
    save_json(run_dir / "training_resources.json", resources)
    best = run_dir / "fit/weights/best.pt"
    save_json(run_dir / "selected_checkpoint.json", {"path": best.relative_to(ROOT).as_posix(), "sha256": sha(best)})
    restored = YOLO(str(best))
    attempts.update(status="VALIDATING", independent_val_calls=1)
    save_json(run_dir / "attempt.json", attempts)
    result = restored.val(validator=ValOnlyValidator, data=str(dataset_file), split="val", imgsz=config["imgsz"],
        batch=16, device=0, workers=0, quantize=16, rect=True, conf=0.001, iou=0.7, max_det=300, nms=None,
        project=str(run_dir), name="independent_validation", exist_ok=False)
    save_json(run_dir / "validation_metrics.json", metrics(result))
    save_json(ROOT / f"outputs/reproduction/{experiment.lower()}_metrics.json", {
        "config": config, "checkpoints": {"best": {"path": best.relative_to(ROOT).as_posix(), "sha256": sha(best)}},
        "training": resources, "validation": metrics(result)})
    attempts["status"] = "COMPLETED"
    save_json(run_dir / "attempt.json", attempts)
    print("Fresh reproduction complete. Published evidence is unchanged; no test evaluation entry is provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=CONFIGS, required=True)
    parser.add_argument("--run-training", action="store_true", help="Explicitly enable a new local reproduction")
    args = parser.parse_args()
    require(args.run_training, "Training disabled; supply --run-training only for a new reproduction")
    run(args.experiment)
