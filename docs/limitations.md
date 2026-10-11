# Limitations

- BCCD contains only 364 images and one corrected split. The 72-image test set supports a limited estimate.
- Only seed 42 was used. Small validation differences are not significance or stability evidence.
- WBC and Platelets have limited sample sizes (372 and 361 modeling objects).
- Patient/slide metadata is unavailable. Acquisition screening does not establish patient independence or exclude all related fields.
- Label coverage can be incomplete. Errors are defined against existing annotations, which remain unchanged.
- Only one nonzero class-weighting power and two resolutions were studied; observed minority-class and size effects are experiment specific.
- AP50 is higher than AP50-95; stricter localization remains a challenge. mAP is not classification accuracy.
- The original E1 independent validation runtime snapshot was not captured; its protocol was reconstructed from executed code. It was not rerun to fill that record.
- Training resource values come from single runs with different stopping epochs; epoch callbacks were not independently synchronized.
- The efficiency benchmark depends on RTX 4070 SUPER, batch 16, warmed FP16 pipeline timing. It is not batch-1 latency, an edge benchmark or production deployment evidence.
- No clinical validation, external independent cohort or diagnostic-system claim is made.

Future work could use multiple seeds, patient-aware splits, external datasets, improved annotation coverage and separate deployment benchmarks. These are proposals, not completed results.
