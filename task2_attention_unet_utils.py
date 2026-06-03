"""Attention U-Net and Task 2 test-evaluation helpers."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from scipy.ndimage import (
        binary_erosion as scipy_binary_erosion,
        distance_transform_edt,
        label as scipy_connected_components,
    )

    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    scipy_binary_erosion = None
    distance_transform_edt = None
    scipy_connected_components = None
    SCIPY_AVAILABLE = False


def split_cases_patient_level(
    case_ids,
    train_ratio=0.65,
    val_ratio=0.20,
    test_ratio=0.15,
    seed=42,
):
    """Split case IDs before expanding them to slice files."""
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios.sum():.4f}")

    if len(case_ids) < 3:
        raise ValueError("At least 3 cases are required for train/val/test split.")

    shuffled_cases = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled_cases)

    num_cases = len(shuffled_cases)
    num_test = max(1, int(round(num_cases * test_ratio)))
    num_val = max(1, int(round(num_cases * val_ratio)))

    if num_val + num_test >= num_cases:
        num_test = max(1, min(num_test, num_cases - 2))
        num_val = max(1, min(num_val, num_cases - num_test - 1))

    test_cases = sorted(shuffled_cases[:num_test])
    val_cases = sorted(shuffled_cases[num_test : num_test + num_val])
    train_cases = sorted(shuffled_cases[num_test + num_val :])

    return train_cases, val_cases, test_cases


def collect_files_for_cases(case_to_files, selected_cases):
    files = []
    for case_id in selected_cases:
        files.extend(case_to_files[case_id])
    return sorted(files)


def write_case_list(path, cases):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for case_id in cases:
            f.write(str(case_id) + "\n")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def match_spatial_shape(x, reference):
    """Pad or resize decoder features to match the skip feature map."""
    diff_y = reference.size(2) - x.size(2)
    diff_x = reference.size(3) - x.size(3)

    if diff_y == 0 and diff_x == 0:
        return x

    if diff_y < 0 or diff_x < 0:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return F.pad(
        x,
        [
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
        ],
    )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = Attention_block(
            F_g=decoder_channels,
            F_l=skip_channels,
            F_int=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, decoder_feature, skip_feature):
        decoder_feature = self.up(decoder_feature)
        decoder_feature = match_spatial_shape(decoder_feature, skip_feature)
        skip_feature = self.attention(g=decoder_feature, x=skip_feature)
        x = torch.cat([skip_feature, decoder_feature], dim=1)
        return self.conv(x)


class AttUNet(nn.Module):
    """U-Net with attention gates on encoder skip connections."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


def to_python_value(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy().tolist()

    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        return x.tolist()

    return x


def binary_dice(pred_mask, gt_mask, eps=1e-7):
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())

    if pred_sum + gt_sum == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps))


def extract_surface(mask):
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    return np.logical_xor(mask, binary_erosion_2d(mask))


def binary_erosion_2d(mask):
    if SCIPY_AVAILABLE:
        return scipy_binary_erosion(mask, border_value=0)

    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("NumPy fallback erosion expects a 2D mask.")

    h, w = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    return eroded


def surface_distances_direct(source_surface, target_surface, chunk_size=1024):
    source_coords = np.argwhere(source_surface)
    target_coords = np.argwhere(target_surface)

    if source_coords.size == 0 or target_coords.size == 0:
        return np.array([], dtype=np.float64)

    distances = []
    target_coords = target_coords.astype(np.float64)

    for start in range(0, len(source_coords), chunk_size):
        source_chunk = source_coords[start : start + chunk_size].astype(np.float64)
        diff = source_chunk[:, None, :] - target_coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        distances.append(np.min(dist, axis=1))

    return np.concatenate(distances)


def count_connected_components(mask):
    mask = np.asarray(mask).astype(bool)

    if SCIPY_AVAILABLE:
        _, component_count = scipy_connected_components(mask.astype(np.uint8))
        return int(component_count)

    visited = np.zeros_like(mask, dtype=bool)
    component_count = 0
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            component_count += 1
            stack = [(y, x)]
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

    return component_count


def hd95_binary(pred_mask, gt_mask):
    """Compute robust 2D HD95 from binary masks using distance transforms."""
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    pred_empty = not pred_mask.any()
    gt_empty = not gt_mask.any()

    if pred_empty and gt_empty:
        return 0.0

    if pred_empty or gt_empty:
        return np.nan

    pred_surface = extract_surface(pred_mask)
    gt_surface = extract_surface(gt_mask)

    if SCIPY_AVAILABLE:
        distance_to_gt = distance_transform_edt(~gt_surface)
        distance_to_pred = distance_transform_edt(~pred_surface)

        distances = np.concatenate(
            [
                distance_to_gt[pred_surface],
                distance_to_pred[gt_surface],
            ]
        )
    else:
        distances = np.concatenate(
            [
                surface_distances_direct(pred_surface, gt_surface),
                surface_distances_direct(gt_surface, pred_surface),
            ]
        )

    if distances.size == 0:
        return np.nan

    return float(np.percentile(distances, 95))


def region_metrics_from_arrays(pred_np, label_np, region_names):
    region_metrics = []

    for c, name in enumerate(region_names):
        pred_c = pred_np[c].astype(bool)
        gt_c = label_np[c].astype(bool)
        region_metrics.append(
            {
                "region": str(name),
                "dice": binary_dice(pred_c, gt_c),
                "hd95": hd95_binary(pred_c, gt_c),
                "pred_area": int(pred_c.sum()),
                "gt_area": int(gt_c.sum()),
                "pred_empty": bool(not pred_c.any()),
                "gt_empty": bool(not gt_c.any()),
            }
        )

    return region_metrics


def summarize_sample_metrics(pred_np, label_np, region_names):
    region_metrics = region_metrics_from_arrays(pred_np, label_np, region_names)

    dice_values = np.array([m["dice"] for m in region_metrics], dtype=np.float64)
    hd95_values = np.array([m["hd95"] for m in region_metrics], dtype=np.float64)

    pred_union = np.any(pred_np > 0, axis=0)
    gt_union = np.any(label_np > 0, axis=0)

    image_area = int(gt_union.size)
    pred_area = int(pred_union.sum())
    gt_area = int(gt_union.sum())

    mean_hd95 = (
        np.nan if np.all(np.isnan(hd95_values)) else float(np.nanmean(hd95_values))
    )

    row = {
        "dice": float(np.mean(dice_values)),
        "hd95": mean_hd95,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "lesion_ratio": float(gt_area / max(image_area, 1)),
        "pred_empty": bool(pred_area == 0),
        "gt_empty": bool(gt_area == 0),
    }

    for metric in region_metrics:
        region_key = metric["region"].lower()
        row[f"dice_{region_key}"] = metric["dice"]
        row[f"hd95_{region_key}"] = metric["hd95"]
        row[f"pred_area_{region_key}"] = metric["pred_area"]
        row[f"gt_area_{region_key}"] = metric["gt_area"]
        row[f"pred_empty_{region_key}"] = metric["pred_empty"]
        row[f"gt_empty_{region_key}"] = metric["gt_empty"]

    return row


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []

    # Preserve requested column order, then append any extra metric columns
    # present in rows. This keeps per-region Dice/HD95 fields without failing.
    fieldnames = list(fieldnames)
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    return float(np.mean(values)), float(np.std(values)), float(np.median(values))


def build_summary_rows(metric_rows, region_names):
    dice_values = [row["dice"] for row in metric_rows]
    hd95_values = [row["hd95"] for row in metric_rows]

    mean_dice, std_dice, median_dice = mean_std_median(dice_values)
    mean_hd95, std_hd95, median_hd95 = mean_std_median(hd95_values)

    summary = {
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_hd95": mean_hd95,
        "std_hd95": std_hd95,
        "median_dice": median_dice,
        "median_hd95": median_hd95,
        "num_test_cases": len({row["case_id"] for row in metric_rows}),
        "num_test_samples": len(metric_rows),
    }

    for region_name in region_names:
        key = str(region_name).lower()
        region_dice = [row[f"dice_{key}"] for row in metric_rows]
        region_hd95 = [row[f"hd95_{key}"] for row in metric_rows]
        summary[f"mean_dice_{key}"] = mean_std_median(region_dice)[0]
        summary[f"mean_hd95_{key}"] = mean_std_median(region_hd95)[0]

    return [summary]


@torch.no_grad()
def evaluate_model_on_test_loader(model, loader, device, region_names, threshold=0.5):
    model.eval()
    metric_rows = []
    dataset_offset = 0

    pbar = tqdm(loader, desc="Test", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > threshold).astype(np.uint8)
        labels_np = labels.detach().cpu().numpy().astype(np.uint8)

        batch_size = images.size(0)
        for i in range(batch_size):
            case_id = to_python_value(batch["case_id"][i])
            slice_idx = int(to_python_value(batch["slice_idx"][i]))
            path = to_python_value(batch["path"][i])
            dataset_idx = dataset_offset + i

            row = {
                "case_id": str(case_id),
                "slice_idx": slice_idx,
                "dataset_idx": dataset_idx,
                "path": str(path),
            }
            row.update(summarize_sample_metrics(preds[i], labels_np[i], region_names))
            metric_rows.append(row)

        dataset_offset += batch_size

    return metric_rows


def get_sample_arrays(sample):
    image = sample["image"]
    label = sample["label"]

    image_np = (
        image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    )
    label_np = (
        label.detach().cpu().numpy() if torch.is_tensor(label) else np.asarray(label)
    )

    case_id = to_python_value(sample["case_id"])
    slice_idx = int(to_python_value(sample["slice_idx"]))
    return image_np, label_np, str(case_id), slice_idx


@torch.no_grad()
def predict_dataset_sample(model, dataset, dataset_idx, device, threshold=0.5):
    model.eval()
    sample = dataset[int(dataset_idx)]
    image_np, label_np, case_id, slice_idx = get_sample_arrays(sample)

    image_tensor = sample["image"].unsqueeze(0).to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred_np = (probs > threshold).astype(np.uint8)

    return image_np, label_np.astype(np.uint8), pred_np, case_id, slice_idx


def estimate_background_value_from_corners(img):
    h, w = img.shape
    corner_values = np.array(
        [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]],
        dtype=np.float32,
    )
    return float(np.median(corner_values))


def normalize_mri_for_display_with_background(img, eps=1e-6):
    img = img.astype(np.float32)
    background_value = estimate_background_value_from_corners(img)
    foreground_mask = np.abs(img - background_value) > eps

    out = np.zeros_like(img, dtype=np.float32)
    if foreground_mask.sum() == 0:
        return out

    foreground_values = img[foreground_mask]
    vmin = np.percentile(foreground_values, 1)
    vmax = np.percentile(foreground_values, 99)

    if vmax <= vmin:
        return out

    out[foreground_mask] = (img[foreground_mask] - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def estimate_contrast_score(image_np, gt_union, image_channel=3, eps=1e-6):
    if gt_union.sum() == 0:
        return np.nan

    channel = min(image_channel, image_np.shape[0] - 1)
    img = image_np[channel].astype(np.float32)
    background_value = estimate_background_value_from_corners(img)
    brain_mask = np.abs(img - background_value) > eps
    non_lesion_mask = np.logical_and(brain_mask, ~gt_union.astype(bool))

    if non_lesion_mask.sum() == 0:
        return np.nan

    lesion_mean = float(np.mean(img[gt_union.astype(bool)]))
    background_mean = float(np.mean(img[non_lesion_mask]))
    background_std = float(np.std(img[non_lesion_mask]))
    return abs(lesion_mean - background_mean) / (background_std + eps)


def build_improvement_suggestion(reasons):
    reason_text = " | ".join(reasons).lower()
    suggestions = []

    if "small lesion" in reason_text:
        suggestions.append(
            "increase lesion-aware sampling, patch-based training, and small-lesion slice weighting"
        )
    if "low contrast" in reason_text:
        suggestions.append(
            "use contrast augmentation, multi-modal balancing, attention gates, and boundary loss"
        )
    if "artifact" in reason_text or "fragmented" in reason_text:
        suggestions.append(
            "strengthen artifact augmentation, data cleaning, and connected-component post-processing"
        )
    if "over-segmentation" in reason_text or "false positive" in reason_text:
        suggestions.append("add false-positive penalty and hard negative mining")
    if "under-segmentation" in reason_text or "missed lesion" in reason_text:
        suggestions.append(
            "increase foreground weighting or try Tversky/Focal Tversky loss"
        )

    if not suggestions:
        suggestions.append(
            "inspect visually and consider boundary-aware loss plus targeted augmentation"
        )

    return "; ".join(dict.fromkeys(suggestions))


def analyze_failure_case(image_np, label_np, pred_np, image_channel=3):
    gt_union = np.any(label_np > 0, axis=0)
    pred_union = np.any(pred_np > 0, axis=0)

    gt_area = int(gt_union.sum())
    pred_area = int(pred_union.sum())
    image_area = int(gt_union.size)
    lesion_ratio = float(gt_area / max(image_area, 1))

    reasons = []
    if gt_area == 0 and pred_area > 0:
        reasons.append("false positive on empty-label slice")
    elif gt_area > 0 and pred_area == 0:
        reasons.append("missed lesion / under-segmentation")
    elif gt_area > 0:
        if pred_area > 1.5 * gt_area:
            reasons.append("over-segmentation")
        elif pred_area < 0.5 * gt_area:
            reasons.append("under-segmentation")

    if lesion_ratio < 0.01 and gt_area > 0:
        reasons.append("small lesion size")

    contrast_score = estimate_contrast_score(
        image_np, gt_union, image_channel=image_channel
    )
    if np.isnan(contrast_score):
        if gt_area > 0:
            reasons.append("possible low contrast, needs visual confirmation")
    elif contrast_score < 0.5:
        reasons.append(f"low contrast (score={contrast_score:.3f})")

    component_count = count_connected_components(pred_union)
    if component_count > 10:
        reasons.append(
            f"fragmented prediction / possible artifact ({component_count} components)"
        )

    if not reasons:
        reasons.append("mixed boundary error, needs visual confirmation")

    return {
        "failure_reason": "; ".join(reasons),
        "improvement_suggestion": build_improvement_suggestion(reasons),
        "contrast_score": contrast_score,
        "pred_components": int(component_count),
        "lesion_ratio": lesion_ratio,
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def sort_worst_cases(metric_rows, top_k=5):
    def sort_key(row):
        hd95_value = row["hd95"]
        hd95_for_sort = float("inf") if np.isnan(hd95_value) else hd95_value
        return (row["dice"], -hd95_for_sort)

    return sorted(metric_rows, key=sort_key)[:top_k]


def save_worst5_markdown(path, worst_rows):
    lines = [
        "# Worst 5 Error Analysis",
        "",
        "Sorted by Dice ascending, then HD95 descending.",
        "",
    ]

    for rank, row in enumerate(worst_rows, start=1):
        lines.extend(
            [
                f"## Rank {rank}",
                f"- case_id: {row['case_id']}",
                f"- slice_idx: {row['slice_idx']}",
                f"- dice: {row['dice']:.6f}",
                f"- hd95: {row['hd95']}",
                f"- pred_area: {row['pred_area']}",
                f"- gt_area: {row['gt_area']}",
                f"- lesion_ratio: {row['lesion_ratio']:.8f}",
                f"- failure_reason: {row['failure_reason']}",
                f"- improvement_suggestion: {row['improvement_suggestion']}",
                "",
            ]
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def sanitize_filename(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def visualize_worst_case(image_np, label_np, pred_np, row, save_path, image_channel=3):
    channel = min(image_channel, image_np.shape[0] - 1)
    bg_display = normalize_mri_for_display_with_background(image_np[channel])

    gt_union = np.any(label_np > 0, axis=0)
    pred_union = np.any(pred_np > 0, axis=0)
    false_positive = np.logical_and(pred_union, ~gt_union)
    false_negative = np.logical_and(gt_union, ~pred_union)

    error_rgb = np.zeros((*gt_union.shape, 3), dtype=np.float32)
    error_rgb[false_positive] = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    error_rgb[false_negative] = np.array([0.1, 0.4, 1.0], dtype=np.float32)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(bg_display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Image ch{channel}\n{row['case_id']}\nslice {row['slice_idx']}")
    axes[0].axis("off")

    axes[1].imshow(bg_display, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(~gt_union, gt_union), cmap="Reds", alpha=0.55)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(bg_display, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        np.ma.masked_where(~pred_union, pred_union), cmap="Blues", alpha=0.55
    )
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(bg_display, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(~gt_union, gt_union), cmap="Reds", alpha=0.45)
    axes[3].imshow(
        np.ma.masked_where(~pred_union, pred_union), cmap="Blues", alpha=0.45
    )
    axes[3].set_title(f"Overlay\nDice={row['dice']:.4f}")
    axes[3].axis("off")

    axes[4].imshow(bg_display, cmap="gray", vmin=0, vmax=1)
    error_mask = np.repeat((~np.any(error_rgb > 0, axis=-1))[..., None], 3, axis=-1)
    axes[4].imshow(np.ma.array(error_rgb, mask=error_mask), alpha=0.75)
    axes[4].set_title("Error Map\nFP red / FN blue")
    axes[4].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def enrich_and_save_worst5(
    model,
    dataset,
    metric_rows,
    device,
    output_dir,
    threshold=0.5,
    top_k=5,
):
    output_dir = Path(output_dir)
    worst5_cases_path = output_dir / "worst5_cases.csv"
    worst5_md_path = output_dir / "worst5_error_analysis.md"
    worst5_vis_dir = output_dir / "worst5_visualizations"
    worst5_vis_dir.mkdir(parents=True, exist_ok=True)

    worst_rows = sort_worst_cases(metric_rows, top_k=top_k)
    enriched_rows = []

    for rank, row in enumerate(worst_rows, start=1):
        image_np, label_np, pred_np, case_id, slice_idx = predict_dataset_sample(
            model=model,
            dataset=dataset,
            dataset_idx=row["dataset_idx"],
            device=device,
            threshold=threshold,
        )

        analysis = analyze_failure_case(
            image_np=image_np, label_np=label_np, pred_np=pred_np
        )
        enriched = dict(row)
        enriched.update(analysis)
        enriched["rank"] = rank
        enriched_rows.append(enriched)

        filename = (
            f"worst_{rank:02d}_case_{sanitize_filename(case_id)}_"
            f"slice_{slice_idx}.png"
        )
        visualize_worst_case(
            image_np=image_np,
            label_np=label_np,
            pred_np=pred_np,
            row=enriched,
            save_path=worst5_vis_dir / filename,
            image_channel=3,
        )

    worst_fieldnames = [
        "rank",
        "case_id",
        "slice_idx",
        "dataset_idx",
        "dice",
        "hd95",
        "pred_area",
        "gt_area",
        "lesion_ratio",
        "failure_reason",
        "improvement_suggestion",
        "contrast_score",
        "pred_components",
        "path",
    ]
    write_csv_rows(worst5_cases_path, enriched_rows, fieldnames=worst_fieldnames)
    save_worst5_markdown(worst5_md_path, enriched_rows)

    return enriched_rows


def run_test_evaluation(
    model,
    test_loader,
    test_dataset,
    device,
    output_dir,
    best_model_path,
    region_names,
    threshold=0.5,
    save_worst5=True,
):
    """Load the best checkpoint, evaluate test data, and optionally save worst-5 analysis."""
    output_dir = Path(output_dir)
    best_model_path = Path(best_model_path)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found: {best_model_path}. Run training before test evaluation."
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded best model for test evaluation:", best_model_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Best validation mean Dice:", checkpoint.get("best_mean_dice"))

    metric_rows = evaluate_model_on_test_loader(
        model=model,
        loader=test_loader,
        device=device,
        region_names=region_names,
        threshold=threshold,
    )

    if len(metric_rows) == 0:
        raise RuntimeError("No test samples were evaluated.")

    per_case_path = output_dir / "test_metrics_per_case.csv"
    summary_path = output_dir / "test_metrics_summary.csv"
    write_csv_rows(per_case_path, metric_rows, fieldnames=list(metric_rows[0].keys()))

    summary_rows = build_summary_rows(metric_rows, region_names=region_names)
    write_csv_rows(summary_path, summary_rows)

    if save_worst5:
        enrich_and_save_worst5(
            model=model,
            dataset=test_dataset,
            metric_rows=metric_rows,
            device=device,
            output_dir=output_dir,
            threshold=threshold,
            top_k=5,
        )

    summary = summary_rows[0]
    print(f"Test Dice: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"Test HD95: {summary['mean_hd95']:.4f} ± {summary['std_hd95']:.4f}")
    print("Per-case/per-slice metrics:", per_case_path)
    print("Summary metrics:", summary_path)
    if save_worst5:
        print("Worst 5 CSV:", output_dir / "worst5_cases.csv")
        print("Worst 5 markdown:", output_dir / "worst5_error_analysis.md")
        print("Worst 5 visualizations:", output_dir / "worst5_visualizations")
    else:
        print("Worst-5 error analysis skipped.")

    return metric_rows, summary_rows
