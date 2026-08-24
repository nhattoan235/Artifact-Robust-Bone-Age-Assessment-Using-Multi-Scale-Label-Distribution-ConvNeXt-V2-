# Audit mask reuse for Plan C

## Source inspected

`D:\Learning\DoAn Tot nghiep\gpt-image-bone-age-synthesis`

## Findings

The project contains useful segmentation code and partial masks, but it does not contain a complete, directly reusable hand-mask cache for the current C protocol.

| Artifact | Finding | Decision for C |
|---|---|---|
| `models/cau_hinh_C/mask_generator.py` | Contains deterministic hand-segmentation logic and label-mask logic | Reuse as a candidate implementation only after test-blind audit |
| `models/cau_hinh_C/output/*_mask_label_only.png` | 600 partial masks; IDs overlap current official train only | Use as training-only geometry/QC samples, not as complete cache |
| `data/experiment_v2/synthetic_train/*_mask_label_only.png` | 200 masks for synthetic parents | Do not use as the main C cache |
| `models/cau_hinh_C/Ket_qua_200_SAM_final` | 200 masks corresponding to the locked test IDs | Do not use to design, tune or validate C; test-derived artifact |
| `data/experiment_v2/manifests/real_train.csv` + `real_validation.csv` | 11,350 + 1,261 = 12,611 images | Different split from current C development pool of 14,036 |
| `mask_eligibility.csv` | 11,350 rows, 1,939 eligible | Incomplete for the current 14,036-image C pool |

## Important distinction

Files named `mask_label_only` are intended for label/inpainting protection. They are not automatically hand masks suitable for ROI geometry. C needs a hand component mask or a verified hand bounding geometry.

The external code comments also document that some geometric thresholds were measured from the 200-image test set. Therefore, the code cannot be accepted unchanged as a clean C preprocessing protocol.

## Safe reuse decision

1. Reuse the segmentation code structure and the 600 train-only masks for implementation/QC review.
2. Exclude the 200 test masks from all C design, threshold selection and validation.
3. Re-audit or re-derive geometry thresholds using development data only.
4. Generate a fresh cache covering all 14,036 development images, with image SHA, mask SHA, crop coordinates and fallback reason.
5. Generate test masks only after C OOF gates pass and the test protocol is locked.

## Effect on schedule

The source saves implementation time but does not remove the cache-generation stage:

- code reuse and audit: 30–90 minutes;
- development mask/ROI cache for 14,036 images: approximately 20–90 minutes after the segmentation pipeline is ready;
- full C training estimate remains approximately 16–24 GPU-hours.

No C training should start until the fresh development-only cache passes the 14,036-ID and fallback-rate gates.

## Pilot result

A 32-image development-only pilot was run with the reused `segment_hand` implementation. It produced 16 valid masks and 16 `segment_hand_failed` fallbacks (50% fallback). This fails C's maximum 1% fallback gate, so the full 14,036-image cache was intentionally not generated from this implementation.

The pilot failure is concentrated in difficult cassette/bright-background images; therefore, writing zero masks for all failures would create a formally complete but scientifically unusable cache. The segmentation method must be revised or a development-only verified hand-mask source must be supplied before full cache generation.
