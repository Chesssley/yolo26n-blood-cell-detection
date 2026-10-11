# STAGE 11 REPORT

Date: 2026-10-11 (Asia/Shanghai). Scope: local educational inference demo only.
Outcome: required checks PASS; dependency ResourceWarnings are disclosed below.

## 1. Preflight

- The initially named course-source repository was on `main` at `d773cc5c6cb36fb5436ec71bf24bf4415c0bae9d`, with no remote and two pre-existing untracked teacher DOCX files. It has neither the generated public README template nor `scripts/render_results.py`. These files and the source repository were left unchanged.
- The user explicitly approved implementation in the companion `yolo26n-blood-cell-detection-public` repository and read-only use of the course-source E2 checkpoint/sample.
- Public repository: `main`, initial HEAD `963e5ea155c760e3f2c633144d457aa05393ba1e`; clean status; origin `Chesssley/yolo26n-blood-cell-detection`; remote main matched HEAD.
- Public `.venv` was absent. A separate environment was created using CPython 3.12.10. No installed packages/environment were copied from another project.
- Dependencies were resolved against `requirements.txt` and the existing environment constraints, then installed from the exact resolver artifact URLs with SHA-256 checks. Shared pip cache remained on D:.
- Final preflight: own interpreter/pip, Python 3.12.10, pip check, Gradio import and README renderer verification PASS. PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Ultralytics 8.4.175; CUDA available.
- Initial README verification in the empty environment failed for missing NumPy; after installing requirements it passed before App/documentation changes.

## 2. Implementation

- `app/__init__.py`: project-local disposable caches; offline Ultralytics and disabled automatic dependency installation/Gradio analytics.
- `app/inference.py`: image decoding, weight resolution, frozen-checkpoint identity check, lazy detector, serial model reuse, class counts and timed RGB visualization.
- `app/app.py`: localhost Gradio UI and direct/module CLI entry points.
- `requirements-app.txt`: original requirements/constraints plus actually resolved and installed `gradio==6.30.0`. Existing ML dependency files unchanged.
- `tests/test_demo.py`: 11 synthetic/mocked contract checks. Tests do not evaluate any dataset or load a real checkpoint.

## 3. Model loading

Priority: CLI `--weights`, environment `BLOOD_CELL_WEIGHTS`, then repository-relative `outputs/experiments/e2_class_weight_640/weights/best.pt`.
This default matches the original frozen run; the source run has no `fit/` layer. The public repository contains no checkpoint.
The smoke test supplied the existing source checkpoint with `--weights`, without copying it.

The SHA-256 matched frozen evidence: `a732185a984047ddf5f5ede934441821442a3765c0916ba08bc434e2bd131b02`.
The app also requires the detection task and exact mapping `0: RBC`, `1: WBC`, `2: Platelets`.
No automatic weight download or retraining exists. Startup checks existence; the first valid Detect loads the model, and later requests reuse it. Prediction uses `imgsz=640` and the supplied confidence.

## 4. UI

Upload/drop a JPG/JPEG/PNG, preview original image, choose confidence (0.05–0.95, default 0.25, step 0.05), Detect, then view boxes, three counts, total, local time and device. Clear restores the initial form.
Errors use concise status messages and clear stale results; internal inference/load exceptions are logged to the terminal.
The medical-use restriction is prominent in both UI and README. Localhost only; no public share link.

## 5. README

Added one Highlight and Interactive Demo instructions to `docs/templates/README.md.in`, then ran the renderer to generate README.
Documented installation, both launch entry points, overrides, original E2 identity, CPU/CUDA selection, usage, temporary image cache, timing scope and non-clinical restriction.
Renderer and `--verify`: PASS. Existing metrics and rendered experiment/result documents unchanged.

## 6. Validation

| Check | Result |
| --- | --- |
| Python syntax / compileall | PASS |
| Gradio 6.30.0 and App module imports | PASS |
| Both CLI entry points `--help` | PASS |
| Actual missing-weight CLI: exit 2, concise error, no traceback | PASS |
| `pip install -r requirements-app.txt`, `pip check` | PASS |
| 11 unit checks: precedence, empty/corrupt/unsupported images, confidence, checkpoint SHA, class mapping, load/inference errors, model reuse, CPU selection, zero detections, BGR/RGB conversion, CUDA timing, reset | PASS |
| Browser upload, original preview, one real Detect, result image/counts/time/device | PASS |
| Browser empty input and corrupt PNG messages; final Clear state | PASS |
| README render and `--verify` | PASS |
| Frozen public evidence/configuration hashes via renderer | PASS |
| 1,571 source data/split/final-test/checkpoint file hashes unchanged | PASS |
| Git diff review and whitespace checks | PASS |

Exactly one real prediction on train sample `BloodImage_00001.jpg`, confidence 0.25: RBC 34, WBC 1, Platelets 0, total 35, CUDA, 3113.11 ms. Bounding boxes rendered and were visually inspected in the browser. A real screenshot was retained only in ignored local cache.
This is an engineering smoke check, not an accuracy evaluation. Time includes decoding, first-use model loading, prediction and plotting; excludes browser upload/display and queue waiting.
CPU selection/behavior and zero detections were checked with mocks; no real CPU performance run or extra real-image prediction was performed.

Warnings: Gradio construction in synchronous Windows unit tests emitted unclosed event-loop ResourceWarnings; all assertions and live browser checks passed. No warning was suppressed.
Ultralytics' optional HEIF fallback initially broke the corrupt-PNG test after model imports. JPEG/PNG decoding now retains the original Pillow decoder; automatic installation is disabled. The repaired test and browser corrupt-input check passed.

## 7. Experiment integrity

- Retraining E1/E2/E3: **NO**.
- Validation dataset evaluation: **NO**.
- Final-test rerun: **NO**.
- Frozen evidence, metrics, model selection or conclusions modified: **NO**.
- Raw data, annotations and train/val/test split modified: **NO**.
- Models, datasets, outputs, environments, caches, temporary images or screenshots included in the commit: **NO**.

## 8. Git status

Feature branch: `feat/stage11-interactive-demo`. Only App code, App requirements, tests, generated README/template and Stage 11 reports are included.
Pre-existing course-source DOCX files remain untouched/untracked. Public repository commit/status and PR identity are reported in the final delivery message after creation.

## 9. Pull Request

Target: `main`. Title/commit message: `feat: add interactive blood cell detection demo`.
Create through Codex's connected GitHub Create Pull Request tool after validation. Leave open for human review; do not merge.

Implementation references: [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/), [Gradio official image component](https://github.com/gradio-app/gradio/blob/main/gradio/components/image.py).
