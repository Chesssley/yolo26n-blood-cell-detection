# Dataset audit

| Item | Measured result |
| --- | ---: |
| RGB images / VOC XML files | 364 / 364 |
| Image size | 640 × 480 |
| Raw objects | 4,888 |
| Invalid zero-area RBC objects | 2 |
| Modeling objects | 4,886 |
| Exhaustively screened image pairs | 66,066 |
| Confirmed acquisition group | 1 |
| Corrected train / validation / test | 206 / 86 / 72 |

Raw class counts are RBC 4,155, WBC 372, Platelets 361. Modeling RBC count is 4,153; other class counts are unchanged.
The two invalid boxes are registered by image ID, class and exact original coordinates. Only modeling labels omit them; raw data remains unchanged.
The confirmed same-field group 00037/00038 is kept together in training by moving 00037 from validation.
No images are removed, no random re-split occurs and test membership/order/bytes remain unchanged.

The near-duplicate scan uses grayscale pHash (32×32 DCT, 64 bits), dHash (9×8, 64 bits) and Pearson correlation after 128×96 resizing.
Candidate rule: `(pHash≤12 and (dHash≤24 or corr≥0.60)) or (dHash≤12 and corr≥0.65) or corr≥0.80`.
Every pair is checked, with no random sampling. Only the reviewed acquisition pair was found; no additional test-related candidate was found under these thresholds.
Threshold screening cannot exclude all shifted/cropped/rotated correlations or establish patient/slide independence.

Raw hashes, corrected split hashes, candidate measurements and reviewed groups are in `docs/evidence/` and `configs/`.
These decisions protect data quality, evaluation integrity and reproducibility. A fresh scan producing new candidates stops preparation pending explicit review.
