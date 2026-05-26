"""Visualization helpers for BraTS-GLI Task 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from task1_utils import (
    MODALITY_ORDER,
    REGION_ORDER,
    get_channel_min_padding_values,
    get_tumor_slice,
    pad_2d_to_target,
    transform_labels_to_regions,
)


PathLike = Union[str, Path]


def _prepare_save_path(save_path: Optional[PathLike]) -> Optional[Path]:
    if save_path is None:
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _slice_2d(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    if slice_idx < 0 or slice_idx >= volume.shape[2]:
        raise IndexError(f"slice_idx {slice_idx} is outside [0, {volume.shape[2] - 1}].")
    return np.rot90(volume[:, :, slice_idx])


def get_label_region_display_slice(region_masks: np.ndarray) -> int:
    """Pick a slice that makes WT, TC, and ET regions visible when possible."""
    if region_masks.ndim != 4 or region_masks.shape[0] != 3:
        raise ValueError(f"Expected region masks with shape [3, H, W, D], got {region_masks.shape}.")

    depth = region_masks.shape[3]
    slice_sums = region_masks.reshape(3, -1, depth).sum(axis=1)
    wt_sums, tc_sums, et_sums = slice_sums

    all_regions = np.where((wt_sums > 0) & (tc_sums > 0) & (et_sums > 0))[0]
    if all_regions.size:
        scores = et_sums[all_regions] * 1000 + tc_sums[all_regions] * 10 + wt_sums[all_regions]
        return int(all_regions[int(np.argmax(scores))])

    if np.any(et_sums > 0):
        return int(np.argmax(et_sums))
    if np.any(tc_sums > 0):
        return int(np.argmax(tc_sums))
    if np.any(wt_sums > 0):
        return int(np.argmax(wt_sums))
    return depth // 2


def _robust_window(image: np.ndarray) -> Tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0

    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    vmin, vmax = np.percentile(values, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _segmentation_rgba(seg_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgba = np.zeros(seg_slice.shape + (4,), dtype=np.float32)
    colors = {
        1: (0.00, 0.75, 1.00, alpha),
        2: (1.00, 0.82, 0.15, alpha),
        3: (1.00, 0.10, 0.12, alpha),
        4: (1.00, 0.10, 0.12, alpha),
    }
    for label, color in colors.items():
        rgba[seg_slice == label] = color
    return rgba


def _region_rgba(mask_slice: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgba = np.zeros(mask_slice.shape + (4,), dtype=np.float32)
    rgba[mask_slice > 0] = (color[0], color[1], color[2], 0.85)
    return rgba


def _mask_rgba(mask_slice: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.40) -> np.ndarray:
    rgba = np.zeros(mask_slice.shape + (4,), dtype=np.float32)
    rgba[mask_slice > 0] = (color[0], color[1], color[2], alpha)
    return rgba


def _save_or_close(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    path = _prepare_save_path(save_path)
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _add_intensity_colorbar(fig: plt.Figure, ax: plt.Axes, image_artist: Any) -> None:
    """Add a compact MRI intensity colorbar next to one subplot."""
    colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.035, pad=0.02)
    colorbar.ax.tick_params(labelsize=7)


def visualize_overlay(
    modality: np.ndarray,
    seg: np.ndarray,
    slice_idx: int,
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> None:
    """Save a semi-transparent segmentation overlay on one modality slice."""
    image = _slice_2d(modality, slice_idx)
    seg_slice = _slice_2d(seg, slice_idx)
    vmin, vmax = _robust_window(image)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(_segmentation_rgba(seg_slice))
    ax.set_title(title or f"Segmentation overlay | slice {slice_idx}")
    ax.axis("off")
    _save_or_close(fig, save_path)


def visualize_case_modalities(case: Dict[str, Any], save_path: Optional[PathLike] = None) -> None:
    """Save four modalities plus ground-truth and overlay at the tumor-max slice."""
    seg = case["seg"]
    slice_idx = get_tumor_slice(seg)
    case_id = case["case_id"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, modality_name in zip(axes[:4], MODALITY_ORDER):
        image = _slice_2d(case["modalities"][modality_name], slice_idx)
        vmin, vmax = _robust_window(image)
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{case_id} | {modality_name} | slice {slice_idx}")
        ax.axis("off")

    seg_slice = _slice_2d(seg, slice_idx)
    axes[4].imshow(_segmentation_rgba(seg_slice, alpha=0.90))
    axes[4].set_title(f"{case_id} | GT labels | slice {slice_idx}")
    axes[4].axis("off")

    t1ce_slice = _slice_2d(case["modalities"]["T1ce"], slice_idx)
    vmin, vmax = _robust_window(t1ce_slice)
    axes[5].imshow(t1ce_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[5].imshow(_segmentation_rgba(seg_slice))
    axes[5].set_title(f"{case_id} | T1ce + GT | slice {slice_idx}")
    axes[5].axis("off")

    fig.tight_layout()
    _save_or_close(fig, save_path)


def visualize_raw_four_modalities_overlay(
    case: Dict[str, Any],
    slice_idx: Optional[int] = None,
    save_path: Optional[PathLike] = None,
    case_label: str = "case",
) -> None:
    """Save one raw figure with four modalities and four GT overlays."""
    seg = case["seg"]
    if slice_idx is None:
        slice_idx = get_tumor_slice(seg)
    case_id = str(case["case_id"])
    seg_slice = _slice_2d(seg, slice_idx)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for row, modality_name in enumerate(MODALITY_ORDER):
        image = _slice_2d(case["modalities"][modality_name], slice_idx)
        vmin, vmax = _robust_window(image)

        plain_artist = axes[row, 0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        _add_intensity_colorbar(fig, axes[row, 0], plain_artist)
        axes[row, 0].set_title(
            f"{case_id}\nraw | {modality_name} | slice {slice_idx}",
            fontsize=9,
            pad=8,
        )
        axes[row, 0].axis("off")

        overlay_artist = axes[row, 1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(_segmentation_rgba(seg_slice, alpha=0.38))
        _add_intensity_colorbar(fig, axes[row, 1], overlay_artist)
        axes[row, 1].set_title(
            f"{case_id}\nraw | {modality_name} + GT | slice {slice_idx}",
            fontsize=9,
            pad=8,
        )
        axes[row, 1].axis("off")

    fig.suptitle(f"{case_label}: raw four modalities with GT overlay", y=0.992, fontsize=11)
    fig.text(
        0.5,
        0.01,
        "GT labels: 1=NCR/NET (cyan), 2=ED (yellow), 4=ET (red).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965), h_pad=1.8, w_pad=0.8)
    _save_or_close(fig, save_path)


def visualize_processed_four_modalities_overlay(
    case: Dict[str, Any],
    slice_idx: Optional[int] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    save_path: Optional[PathLike] = None,
    case_label: str = "case",
) -> None:
    """Save one processed figure with padded modalities and WT label overlay."""
    regions = case.get("label_regions")
    if regions is None:
        regions = transform_labels_to_regions(case["seg"])
    if slice_idx is None:
        slice_idx = get_tumor_slice(regions[0])

    case_id = str(case["case_id"])
    if target_shape is None:
        target_h, target_w = int(case["seg"].shape[0]), int(case["seg"].shape[1])
    else:
        target_h, target_w = int(target_shape[0]), int(target_shape[1])

    label_slice = regions[:, :, :, slice_idx].astype(np.uint8)
    label_slice = pad_2d_to_target(label_slice, target_h, target_w, value=0)
    wt_slice = np.rot90(label_slice[0])

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for row, modality_name in enumerate(MODALITY_ORDER):
        image = case["modalities"][modality_name][:, :, slice_idx][None, :, :].astype(np.float32)
        image_padding_value = get_channel_min_padding_values(image)
        image = pad_2d_to_target(image, target_h, target_w, value=image_padding_value)[0]
        image = np.rot90(image)
        vmin, vmax = _robust_window(image)

        plain_artist = axes[row, 0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        _add_intensity_colorbar(fig, axes[row, 0], plain_artist)
        axes[row, 0].set_title(
            f"{case_id}\nprocessed | {modality_name} | slice {slice_idx}",
            fontsize=9,
            pad=8,
        )
        axes[row, 0].axis("off")

        overlay_artist = axes[row, 1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(_mask_rgba(wt_slice, (0.10, 0.65, 1.00), alpha=0.38))
        _add_intensity_colorbar(fig, axes[row, 1], overlay_artist)
        axes[row, 1].set_title(
            f"{case_id}\nprocessed | {modality_name} + WT | slice {slice_idx}",
            fontsize=9,
            pad=8,
        )
        axes[row, 1].axis("off")

    fig.suptitle(
        f"{case_label}: processed four modalities with WT overlay "
        f"(padded to {target_h}x{target_w})",
        y=0.992,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Processed = z-score normalized, cropped, center-padded. "
        "Overlay uses WT from label channels [WT, TC, ET].",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965), h_pad=1.8, w_pad=0.8)
    _save_or_close(fig, save_path)


def visualize_label_regions(
    region_masks: np.ndarray,
    case_id: str,
    slice_idx: Optional[int] = None,
    save_path: Optional[PathLike] = None,
) -> None:
    """Save WT, TC, ET region masks. Expected shape is [3, H, W, D]."""
    if region_masks.ndim != 4 or region_masks.shape[0] != 3:
        raise ValueError(f"Expected region masks with shape [3, H, W, D], got {region_masks.shape}.")

    if slice_idx is None:
        slice_idx = get_label_region_display_slice(region_masks)

    colors = {
        "WT": (0.15, 0.70, 1.00),
        "TC": (1.00, 0.70, 0.05),
        "ET": (1.00, 0.10, 0.12),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for channel, (region_name, ax) in enumerate(zip(REGION_ORDER, axes)):
        mask_slice = _slice_2d(region_masks[channel], slice_idx)
        ax.imshow(np.zeros_like(mask_slice), cmap="gray", vmin=0, vmax=1)
        ax.imshow(_region_rgba(mask_slice, colors[region_name]))
        ax.set_title(f"{case_id} | {region_name} | slice {slice_idx}")
        ax.axis("off")

    fig.tight_layout()
    _save_or_close(fig, save_path)


def visualize_case_label_regions(
    case: Dict[str, Any], save_path: Optional[PathLike] = None
) -> None:
    """Save WT, TC, ET region masks for a processed case."""
    regions = case.get("label_regions")
    if regions is None:
        regions = transform_labels_to_regions(case["seg"])
    visualize_label_regions(regions, str(case["case_id"]), save_path=save_path)


def plot_contrast_statistics(
    contrast_df: Any,
    save_path: Optional[PathLike] = None,
    title: str = "ET vs healthy brain contrast",
    ylabel: str = "Mean absolute contrast",
) -> Any:
    """Save a bar plot comparing ET-vs-healthy contrast across modalities."""
    grouped = (
        contrast_df.groupby("modality")["ET_vs_healthy_contrast"]
        .mean()
        .reindex(MODALITY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F2CF5B"]
    ax.bar(grouped.index, grouped.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("MRI modality")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_or_close(fig, save_path)
    return grouped
