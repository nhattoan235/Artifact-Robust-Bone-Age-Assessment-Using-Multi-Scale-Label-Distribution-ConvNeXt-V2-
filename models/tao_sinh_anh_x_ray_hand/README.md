# Artifact-only cleaning for RSNA hand X-rays

This project provides three deliberately separate outputs:

1. `run_artifact_only.py` edits only detected label boxes and is the preferred
   full-frame method.
2. `run_hand_roi.py` makes lossless crops from original pixels.
3. `run_masked_hand.py` is an experimental silhouette composite and should
   only be used after its mask is approved.

The full-frame pipeline does **not** cut the hand out or replace the complete
background.

Safety invariants:

- the protected anatomy mask is intentionally oversized;
- artifact masks are clipped to the outside of protected anatomy;
- every pixel outside the artifact mask remains byte-identical;
- every protected-anatomy pixel remains byte-identical;
- every input receives an output; uncertain cases are queued for review.

## Run

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  run_artifact_only.py `
  --input-dir data\original_200 `
  --output-dir outputs\artifact_only_200_v2 `
  --seed-protection-dir annotations\protection_seeds_v2
```

Optional approved masks use the filename `<Case_ID>.png`:

```powershell
... --seed-protection-dir annotations\protected `
    --manual-artifact-dir annotations\artifacts
```

White pixels in an artifact mask are edited. White pixels in a protection
mask can never be edited.

## Outputs

- `cleaned/`: exactly one PNG per input;
- `protected_masks/`: conservative no-edit masks;
- `artifact_masks/`: pixels eligible for modification;
- `review/`: four-panel visual QA;
- `artifact_only_qc.csv`: status, review reasons, and pixel invariants.

`AUTO_CLEANED_REVIEW_REQUIRED` is still emitted as an image so the 200-case
cohort stays complete. It must not be treated as approved until its review
panel or manual mask has been checked.

## Current 200-case run

The current conservative run is `outputs/artifact_only_200_v2`:

- 200/200 cleaned PNGs and 200/200 review panels were written;
- all 200 cases pass both byte-level preservation checks;
- automatic cleaning is intentionally conservative and does not remove every
  edge label;
- use `artifact_only_qc.csv` as the review queue, then paint a white manual
  artifact mask for any missed label and rerun with
  `--manual-artifact-dir`.

The output is therefore a reproducible first pass plus a deterministic manual
review workflow, not a claim of 200 clinically approved images.

## Paired bone-age benchmark

`run_open_boneage_benchmark.py` evaluates every available image variant with
the same frozen `ianpan/bone-age` three-fold ensemble. The required ground
truth is `external/aimi_bonn_deeplasia/data/rsna_test.csv`.

Run a small smoke test first:

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  run_open_boneage_benchmark.py `
  --metadata-csv external\aimi_bonn_deeplasia\data\rsna_test.csv `
  --group original=data\original_200 `
  --group ours=outputs\artifact_only_200_manual_v3\cleaned `
  --protocol fullframe `
  --output-dir outputs\boneage_benchmark_smoke `
  --limit 10
```

Run the complete paired analysis:

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  run_open_boneage_benchmark.py `
  --metadata-csv external\aimi_bonn_deeplasia\data\rsna_test.csv `
  --group original=data\original_200 `
  --group ours=outputs\artifact_only_200_manual_v3\cleaned `
  --protocol fullframe `
  --protocol histmatch `
  --protocol fixed_crop `
  --protocol fixed_crop_histmatch `
  --crop-mask-dir outputs\artifact_only_200_manual_v3\protected_masks `
  --output-dir outputs\boneage_benchmark_200
```

When the authors' generated radiographs become available, add
`--group author=<directory>`. The directory must contain one image named
`<Case_ID>.png` (or another supported image extension) for every evaluated
case. Aggregate MAE values quoted from the paper are not mixed into the paired
test because they were produced by a different, unavailable checkpoint.

Benchmark outputs include:

- `predictions_long.csv`: one prediction per case, group, and protocol;
- `summary_metrics.csv`: MAE, RMSE, bias, confidence intervals, and accuracy
  within 6/12/24 months;
- `paired_vs_original.csv`: prediction drift and paired change in absolute
  error versus the original image;
- `case_level_vs_original.csv`: case-level drift and error change for outlier
  review;
- `pairwise_group_comparisons.csv`: direct paired tests for every group pair,
  including the authors' images versus ours when that group is supplied;
- accuracy and Bland-Altman plots;
- `run_manifest.json`: pinned model revision, paths, hashes, seed, device, and
  preprocessing configuration.

## Lossless ROI crops

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  run_hand_roi.py `
  --input-dir data\original_200 `
  --protection-dir outputs\artifact_only_200_v2\protected_masks `
  --output-dir outputs\hand_roi_200_v1
```

ROI crops contain original pixels only. They are useful as a second
experimental arm, but rectangular cropping alone cannot remove a marker that
lies inside the hand bounding box.

## Manual review tool

Start the local mask editor:

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  annotation_server.py
```

Open `http://127.0.0.1:8765`. The editor starts with the v2 review queue and
loads the automatic mask as a draft. Paint only labels/marker plates, erase
false-positive regions, then select **Lưu mask**. Approved masks are written
to `annotations/manual_artifacts/<Case_ID>.png`.

Keyboard shortcuts:

- `B`: brush;
- `E`: eraser;
- left/right arrow: previous/next case;
- `Ctrl+S`: save mask.

After annotation, rerun:

```powershell
& "$PWD\.boneage_env\Scripts\python.exe" `
  run_artifact_only.py `
  --input-dir data\original_200 `
  --output-dir outputs\artifact_only_200_manual_v1 `
  --seed-protection-dir annotations\protection_seeds_v2 `
  --manual-artifact-dir annotations\manual_artifacts
```
