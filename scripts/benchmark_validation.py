"""Synchronized, alternating, fresh-process E2/E3 benchmark on corrected VAL only.

Five measured repeats each; each worker warms up two full passes before one timed pass.
No metrics/AP evaluation, images or prediction payload are saved. Existing report refuses rerun.
"""
import sys
if "--worker" not in sys.argv and "--run-benchmark" not in sys.argv:
    raise SystemExit("Benchmark disabled; explicitly enable a new reproduction")
import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from training_common import ROOT, check, sha, save, timestamp, image_paths, file_info
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor

REPORTS = ROOT / "outputs/reproduction"
REPEATS = 5
WARMUPS = 2
PARAMS = dict(batch=16, device=0, quantize=16, workers=0, conf=0.25, iou=0.7,
              max_det=300, rect=False, nms=False, augment=False, agnostic_nms=False,
              classes=None, verbose=False, save=False, save_txt=False, save_crop=False,
              save_json=False, plots=False, show=False, visualize=False)


class BenchmarkPredictor(DetectionPredictor):
    def preprocess(self, images):
        tensor = super().preprocess(images)
        if not hasattr(self, "benchmark_shapes"):
            self.benchmark_shapes = []
        self.benchmark_shapes.append(list(tensor.shape))
        return tensor


def stats(values):
    return dict(median=statistics.median(values), mean=statistics.mean(values),
                std=statistics.pstdev(values), min=min(values), max=max(values), n=len(values))


def worker(experiment, repeat):
    check(torch.cuda.is_available(),
          "Benchmark CUDA/hardware drift")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    summary = json.loads((REPORTS / f"{experiment.lower()}_metrics.json").read_text())
    checkpoint = ROOT / summary["checkpoints"]["best"]["path"]
    check(sha(checkpoint) == summary["checkpoints"]["best"]["sha256"], "Benchmark checkpoint drift")
    imgsz = 640 if experiment == "E2" else 512
    check(summary["config"]["imgsz"] == imgsz, "Checkpoint experiment/resolution mismatch")
    sources = sorted(image_paths("val"))
    manifest = ROOT / ".cache/e2_e3_benchmark_val.txt"
    check(manifest.read_text().splitlines() == [str(p) for p in sources] and len(sources) == 86,
          "Benchmark VAL manifest drift")
    model = YOLO(str(checkpoint))
    check(Path(model.ckpt_path).resolve() == checkpoint.resolve(), "Wrong benchmark checkpoint")
    batches = []

    def capture_batch(predictor):
        paths = [Path(p).resolve() for p in predictor.batch[0]]
        check(all(p in image_paths("val") for p in paths), "Non-VAL benchmark batch")
        check(0 < len(paths) <= 16, "Actual benchmark batch exceeds 16")
        batches.append([p.relative_to(ROOT).as_posix() for p in paths])

    model.add_callback("on_predict_batch_start", capture_batch)

    def consume():
        count = 0
        speed = dict(preprocess=0.0, inference=0.0, postprocess=0.0)
        for result in model.predict(source=str(manifest), imgsz=imgsz, stream=True, predictor=BenchmarkPredictor, **PARAMS):
            check(Path(result.path).resolve() == sources[count], "Benchmark order/membership drift")
            count += 1
            for key in speed:
                speed[key] += float(result.speed[key])
        check(count == 86, "Incomplete benchmark pass")
        return {k: v / count for k,v in speed.items()}

    for _ in range(WARMUPS):
        consume()
        torch.cuda.synchronize(0)
    check(model.predictor.model.fp16 and model.predictor.device.type == "cuda", "Benchmark FP16/CUDA drift")
    resolved = vars(model.predictor.args)
    for key, value in PARAMS.items():
        check(str(resolved[key]) == str(value), f"Prediction parameter drift: {key}")
    batches.clear()
    model.predictor.benchmark_shapes.clear()
    gc.collect()
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    speed = consume()
    torch.cuda.synchronize(0)
    seconds = time.perf_counter() - start
    check(math.isfinite(seconds) and seconds > 0, "Invalid timing")
    check([p for batch in batches for p in batch] == [p.relative_to(ROOT).as_posix() for p in sources],
          "Benchmark actual batched order mismatch")
    check([len(b) for b in batches] == [16,16,16,16,16,6], "Actual loader batching drift")
    check(model.predictor.benchmark_shapes == [[n,3,imgsz,imgsz] for n in [16]*5+[6]], "Actual input tensor shape drift")
    record = dict(experiment=experiment, repeat=repeat, timestamp=timestamp(),
                  checkpoint=file_info(checkpoint), imgsz=imgsz, device=str(model.predictor.device),
                  fp16=bool(model.predictor.model.fp16), prediction_parameters=PARAMS,
                  actual_batch_sizes=[len(b) for b in batches], manifest_sha256=sha(manifest),
                  actual_tensor_shapes=model.predictor.benchmark_shapes,
                  total_wall_seconds=seconds, ms_per_image=seconds*1000/86, images_per_second=86/seconds,
                  ultralytics_speed_ms_per_image=speed,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                  warmup_passes_excluded=WARMUPS, synchronize_before_and_after=True,
                  test_model_evaluation=False, test_prediction=False)
    check(sha(checkpoint) == summary["checkpoints"]["best"]["sha256"], "Benchmark mutated checkpoint")
    save(ROOT / f".cache/benchmark_{experiment.lower()}_{repeat}.json", record)


def run():
    check(Path(sys.prefix).resolve() == ROOT / ".venv", "Use this repository own .venv")
    (ROOT / ".cache").mkdir(parents=True, exist_ok=True)
    output = REPORTS / "e2_vs_e3_efficiency.json"
    check(not output.exists(), "Existing benchmark report; refusing automatic repetition")
    check(not (ROOT / ".cache/benchmark_attempt.json").exists(), "Existing benchmark attempt; inspect before retry")
    sources = sorted(image_paths("val"))
    check(len(sources) == 86, "Corrected VAL count drift")
    manifest = ROOT / ".cache/e2_e3_benchmark_val.txt"
    manifest.write_text("\n".join(map(str, sources))+"\n", encoding="utf-8")
    images = [{"path":p.relative_to(ROOT).as_posix(), "sha256":sha(p)} for p in sources]
    save(ROOT / ".cache/benchmark_attempt.json", dict(status="RUNNING", start=timestamp(), repeats=REPEATS))
    records, order = {"E2": [], "E3": []}, []
    # Alternate order across rounds. Only one model process owns CUDA at any time.
    for repeat in range(1, REPEATS+1):
        for experiment in (("E2","E3") if repeat % 2 else ("E3","E2")):
            order.append(dict(repeat=repeat, experiment=experiment))
            log = ROOT / f".cache/benchmark_{experiment.lower()}_{repeat}.log"
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", experiment,
                                "--repeat", str(repeat)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            record=json.loads((ROOT/f".cache/benchmark_{experiment.lower()}_{repeat}.json").read_text())
            records[experiment].append(record)
            print(f"{experiment} repeat {repeat}: {record['ms_per_image']:.6f} ms/image", flush=True)
    summary = {}
    for experiment, rows in records.items():
        measured = {key:stats([r[key] for r in rows]) for key in
                    ("total_wall_seconds", "ms_per_image", "images_per_second",
                     "peak_allocated_bytes", "peak_reserved_bytes")}
        measured["ultralytics_speed_ms_per_image"] = {key:stats([r["ultralytics_speed_ms_per_image"][key] for r in rows])
                                                       for key in ("preprocess","inference","postprocess")}
        training=json.loads((REPORTS/f"{experiment.lower()}_metrics.json").read_text())["training"]
        summary[experiment] = dict(imgsz=640 if experiment=="E2" else 512, checkpoint=rows[0]["checkpoint"],
                                   measured=measured, repeats=rows, training=training)
    check(all(sha(ROOT/r['path'])==r['sha256'] for r in images), "VAL images changed during benchmark")
    output_record = dict(status="PASS", timestamp=timestamp(), hardware=torch.cuda.get_device_name(0),
         repeats_per_model=REPEATS, warmup_passes_per_worker=WARMUPS,
         seed=42, deterministic=True, alternating_order=order, same_val_images=images,
         val_count=86, val_order="lexicographic absolute path", manifest_sha256=sha(manifest),
         prediction_parameters=PARAMS, gpu_synchronization=True, results=summary,
         wall_scope="warm-cache disk loading + preprocessing + GPU inference + postprocessing + Results iteration; excludes model load and warmup",
         speed_scope="Ultralytics synchronized stage profiles; excludes loading/iteration/callback overhead; reported separately from wall",
         memory_scope="fresh worker per model/repeat, after two warmups; allocator peak during measured pass includes resident model and warm allocator reserve",
         training_scope="Recorded original E2/E3 mean train+val callback time and ready-to-train-completion allocator peaks; sequential same-hardware runs, not repeated controlled training",
         test_model_evaluation=False, test_prediction=False, saved_prediction_payload=False,
         references=["https://docs.ultralytics.com/modes/predict/", "https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html"])
    save(output, output_record)
    save(ROOT/".cache/benchmark_attempt.json",dict(status="COMPLETED",end=timestamp(), repeats=REPEATS))
    print("VAL-ONLY CONTROLLED INFERENCE BENCHMARK: PASS")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--worker", choices=("E2","E3"))
    parser.add_argument("--repeat", type=int)
    args=parser.parse_args()
    if args.worker:
        check(args.repeat in range(1,REPEATS+1), "Invalid benchmark repeat")
        worker(args.worker,args.repeat)
    else:
        check(args.run_benchmark, "Benchmark disabled; explicitly enable a new reproduction")
        run()
