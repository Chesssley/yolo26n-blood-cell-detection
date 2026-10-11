# Methodology

## Provenance and data quality

BCCD is pinned to upstream commit `d272fb14cdff6e473fafeeeba32aba5f560e9e43`.
The archive, license and every raw file have SHA-256 records. Raw images/XML remain byte-identical to upstream.
Audit checks image/XML pairing, dimensions, class vocabulary, bounding-box geometry, exact bytes/decoded-pixel duplicates and split coverage.
All 66,066 unordered image pairs were screened with pHash, dHash and aligned grayscale correlation. Thresholds produce candidates; visual review confirms acquisition identity.

## Minimal corrected split

`BloodImage_00037` and `BloodImage_00038` depict the same acquisition field and originally crossed validation/training.
Only 00037 moves from validation to training. Both images remain; original ordering is retained with the moved image appended.
Train/validation/test is 206/86/72. Test membership is byte-identical to upstream. Hash/group checks reject future drift.
No patient/slide metadata is available, so group-aware correction does not prove patient independence.

## Deterministic preparation

Two exact registered zero-area RBC objects are removed only from processed labels: 00338 `(504,337,504,337)` and 00343 `(181,329,181,329)`.
Unknown defects stop preparation. No coordinate guessing, clamping, ±1 origin changes or raw repair occurs.
VOC numeric endpoints convert to normalized YOLO `xywh`:

```text
xc = (xmin+xmax)/(2W)    yc = (ymin+ymax)/(2H)
bw = (xmax-xmin)/W      bh = (ymax-ymin)/H
```

Class IDs: 0 RBC, 1 WBC, 2 Platelets. Twelve decimal places preserve endpoint round trips; the measured maximum error is 2.4004975784919225e-10 pixels across 4,886 valid objects.
Prepared images are byte copies. Hashes, labels, all split members and coordinate round trips are verified after construction.
The original implementation also passed delete/rebuild byte determinism checks. Publication does not rerun that destructive check.

## Experiments and selection

E1/E2/E3 were specified before formal results. Every run starts from the same official pretrained bytes with seed 42, controlled settings and independent output directories.
The original data-loading and three-epoch engineering smoke checks passed before the formal experiments. Smoke results are not presented as model performance.
Train/validation membership guards block test data from training and benchmark loaders.
Models are ranked by independent validation mAP50-95 from restored best checkpoints. The winner, checkpoint hash and evaluation protocol are frozen before test evaluation.
The selected E2 checkpoint SHA-256 is `a732185a984047ddf5f5ede934441821442a3765c0916ba08bc434e2bd131b02`.
One test evaluation supplied metrics and predictions from the same pass. Offline error analysis/figures reuse those predictions; no second test inference was needed.

## New reproduction boundaries

Public data scripts verify pinned raw and split hashes, screen all pairs, preserve raw data and refuse overwrite.
`scripts/smoke_check.py` validates train/validation data-loader membership and checks the model architecture with one synthetic CUDA forward. It never loads test data or computes performance metrics.
A new independent reproduction should complete this engineering check before explicit training.
Historical frozen scores and the closed test record are immutable documentation. New train/validation/benchmark outputs are separate, and there is no automatic test rerun entry.

For timing scope, optimizer settings and controlled variables, see [experiments](experiments.md).
