# Worst 5 Error Analysis

Sorted by Dice ascending, then HD95 descending.

## Rank 1
- case_id: BraTS-GLI-00734-001
- slice_idx: 24
- dice: 0.000000
- hd95: 75.01799885644469
- pred_area: 132
- gt_area: 108
- lesion_ratio: 0.00270433
- failure_reason: small lesion size
- improvement_suggestion: increase lesion-aware sampling, patch-based training, and small-lesion slice weighting

## Rank 2
- case_id: BraTS-GLI-00559-000
- slice_idx: 136
- dice: 0.000000
- hd95: nan
- pred_area: 0
- gt_area: 241
- lesion_ratio: 0.00603466
- failure_reason: missed lesion / under-segmentation; small lesion size
- improvement_suggestion: increase lesion-aware sampling, patch-based training, and small-lesion slice weighting; increase foreground weighting or try Tversky/Focal Tversky loss

## Rank 3
- case_id: BraTS-GLI-00370-000
- slice_idx: 40
- dice: 0.000000
- hd95: nan
- pred_area: 0
- gt_area: 108
- lesion_ratio: 0.00270433
- failure_reason: missed lesion / under-segmentation; small lesion size
- improvement_suggestion: increase lesion-aware sampling, patch-based training, and small-lesion slice weighting; increase foreground weighting or try Tversky/Focal Tversky loss

## Rank 4
- case_id: BraTS-GLI-00608-000
- slice_idx: 58
- dice: 0.000000
- hd95: 28.020016631624348
- pred_area: 71
- gt_area: 74
- lesion_ratio: 0.00185296
- failure_reason: small lesion size; low contrast (score=0.437)
- improvement_suggestion: increase lesion-aware sampling, patch-based training, and small-lesion slice weighting; use contrast augmentation, multi-modal balancing, attention gates, and boundary loss

## Rank 5
- case_id: BraTS-GLI-00237-000
- slice_idx: 30
- dice: 0.000000
- hd95: nan
- pred_area: 0
- gt_area: 88
- lesion_ratio: 0.00220353
- failure_reason: missed lesion / under-segmentation; small lesion size
- improvement_suggestion: increase lesion-aware sampling, patch-based training, and small-lesion slice weighting; increase foreground weighting or try Tversky/Focal Tversky loss
