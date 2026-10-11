# Controlled experiments

| Experiment | Input | cls_pw | Purpose | Validation mAP50-95 |
| --- | --- | --- | --- | --- |
| E1 | 640 | 0.0 | Baseline | 0.636218 |
| E2 | 640 | 0.5 | Class weighting | 0.637143 |
| E3 | 512 | 0.5 | Resolution trade-off | 0.635402 |

Each experiment starts independently from the official YOLO26n pretrained asset, with no checkpoint chaining or resume.
E1→E2 changes only `cls_pw`; E2→E3 changes only input size. Data membership, seed, augmentation, optimizer policy, batch size and other semantic training arguments are controlled.

## Fixed protocol

- Seed 42; deterministic mode enabled; batch 16; AMP enabled; CUDA device 0.
- Maximum 100 epochs; early-stopping patience 20; workers 0; no image cache.
- `optimizer=auto` resolved to AdamW at initial learning rate 0.001429 in all three historical runs.
- Full portable resolved configurations are in `configs/resolved/`; original training arguments and actual optimizer states were captured before the first batch.
- Primary selection rule: highest independent validation mAP50-95 from the restored `best.pt` of E1/E2/E3. No test-driven tuning.
- Historical actual/best epochs: E1 49/29, E2 69/49, E3 60/40. Built-in early-stopping/checkpoint fitness is an Ultralytics behavior; cross-experiment selection uses independent validation AP50-95.

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| E1 | 0.825267 | 0.914447 | 0.919868 | 0.636218 |
| E2 | 0.863390 | 0.898984 | 0.912741 | 0.637143 |
| E3 | 0.863857 | 0.873104 | 0.918488 | 0.635402 |

E2 won by a small margin. One seed and one split provide no statistical-significance evidence.

## Class weighting

Ultralytics 8.4.175 applies mean-normalized inverse-frequency weights proportional to `count ** (-cls_pw)`.
Training counts are RBC 2408, WBC 215, Platelets 211. At power 0.5, measured trainer weights are approximately 0.388357, 1.299691 and 1.311952.
Weights were verified in both detection-loss branches. Overall AP50-95 increased slightly; RBC improved, WBC decreased and Platelets were almost unchanged/slightly lower.
The result is a limited class-dependent trade-off, with no consistent minority-class AP benefit.

## Resolution and timing

640→512 changes validation AP50-95 by -0.174 percentage points; Platelets AP50-95 falls more than the overall average.

| Metric | 640 (E2) | 512 (E3) | Change |
| --- | --- | --- | --- |
| avg train+val time (s/epoch) | 3.0852 | 2.6518 | -14.05% |
| training peak allocated VRAM (GiB) | 2.4760 | 1.6007 | -35.35% |
| training peak reserved VRAM (GiB) | 2.7422 | 1.7422 | -36.47% |
| median pipeline latency (ms/image) | 2.6759 | 2.3673 | -11.53% |
| median throughput (images/s) | 373.7086 | 422.4213 | +13.03% |

The controlled benchmark uses the same 86 validation images in lexicographic order. It alternates E2/E3 order across five rounds and launches a fresh worker for each model/repeat.
Each worker excludes model loading and two complete warmup passes, checks actual FP16/batch/input shapes and synchronizes CUDA before/after external wall timing.
Batch sizes are 16/16/16/16/16/6; square inputs, `conf=0.25`, `iou=0.7`, `nms=False` and no augmentation or saved predictions.
Reported medians include warm-cache disk loading, preprocessing, inference, postprocessing and result iteration, rather than GPU kernel time alone.
Training VRAM and callback epoch times are historical single-run measurements. They are not repeated training benchmarks, batch-1 latency or edge deployment measurements.

## Explicit new reproduction

Acquire the official pretrained asset listed in `docs/evidence/provenance.json`, save it as `weights/pretrained/yolo26n.pt`, and verify its SHA-256 before starting.
Complete the data audit, full near-duplicate scan and preparation first. Inspect the conversion/audit outputs and run the data-loader smoke check described in `docs/methodology.md`.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_check.py --run-smoke
.\.venv\Scripts\python.exe scripts/train_experiment.py --experiment E1 --run-training
.\.venv\Scripts\python.exe scripts/train_experiment.py --experiment E2 --run-training
.\.venv\Scripts\python.exe scripts/train_experiment.py --experiment E3 --run-training
.\.venv\Scripts\python.exe scripts/benchmark_validation.py --run-benchmark
```

Training refuses existing run directories and uses a train/validation-only YAML plus explicit loader guards.
New outputs go to `outputs/reproduction/`, never to `docs/evidence/`.
The benchmark operates only on those new E2/E3 weights and validation images. It reports measured hardware and cannot be expected to match published numbers on a different system.
Training/benchmarking is opt-in and has not been rerun for documentation publication.
The published final test is closed; no test execution command is included.
