"""Utilities for BraTS-GLI Task 1 preprocessing.

Array convention used in this project:
- 3D volumes are stored as [H, W, D].
- Region labels are stored as [3, H, W, D].
- Region channel 0 = WT, channel 1 = TC, channel 2 = ET.
- Saved 2D image slices are [4, H, W].
- Saved 2D label slices are [3, H, W].
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
BBox = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

MODALITY_ORDER = ("T1", "T1ce", "T2", "FLAIR")
REGION_ORDER = ("WT", "TC", "ET")
NCR_NET_LABELS = (1,)
EDEMA_LABELS = (2,)
# Project/PDF canonical label definition: 4 = GD-enhancing tumor (ET).
ET_LABELS = (4,)
LEGACY_ET_LABELS = (3,)
VALID_SEG_LABELS = (0,) + NCR_NET_LABELS + EDEMA_LABELS + ET_LABELS

MODALITY_ALIASES = {
    "T1": {"t1n", "t1"},
    "T1ce": {"t1c", "t1ce"},
    "T2": {"t2w", "t2"},
    "FLAIR": {"t2f", "flair"},
    "seg": {"seg"},
}


def is_nifti_file(path: PathLike) -> bool:
    """Return True for .nii and .nii.gz files."""
    name = Path(path).name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def strip_nifti_suffix(path: PathLike) -> str:
    """Return a filename without the .nii or .nii.gz suffix."""
    name = Path(path).name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(path).stem


def identify_nifti_role(path: PathLike) -> Optional[str]:
    """Identify a BraTS modality or segmentation role from a filename."""
    stem = strip_nifti_suffix(path).lower()
    tokens = [tok for tok in re.split(r"[-_]", stem) if tok]
    if not tokens:
        return None

    last_token = tokens[-1]
    for role, aliases in MODALITY_ALIASES.items():
        if last_token in aliases:
            return role
    return None


def _iter_nifti_files(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and is_nifti_file(path)
    )


def _scan_case_file_map(case_dir: Path) -> Dict[str, Path]:
    file_map: Dict[str, Path] = {}
    for path in _iter_nifti_files(case_dir):
        role = identify_nifti_role(path)
        if role is None:
            continue
        if role in file_map:
            raise ValueError(
                f"Duplicate files for role '{role}' in case '{case_dir.name}': "
                f"{file_map[role].name} and {path.name}"
            )
        file_map[role] = path
    return file_map


def _contains_segmentation_file(case_dir: Path) -> bool:
    try:
        return any(
            identify_nifti_role(path) == "seg" for path in _iter_nifti_files(case_dir)
        )
    except ValueError:
        return True


def find_patient_cases(dataset_dir: PathLike) -> List[Path]:
    """Find patient case folders under a dataset root.

    The function supports a dataset root containing patient folders directly.
    If no direct patient folders are found, it falls back to a recursive search
    for directories containing a segmentation NIfTI file.
    """
    root = Path(dataset_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    if _contains_segmentation_file(root):
        return [root]

    direct_cases = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and _contains_segmentation_file(path)
    )
    if direct_cases:
        return direct_cases

    case_dirs = set()
    for pattern in ("*.nii", "*.nii.gz"):
        for path in root.rglob(pattern):
            if path.is_file() and identify_nifti_role(path) == "seg":
                case_dirs.add(path.parent)

    cases = sorted(case_dirs)
    if not cases:
        raise FileNotFoundError(
            f"No BraTS case folders with segmentation NIfTI files were found under: {root}"
        )
    return cases


def load_nifti(path: PathLike) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Load a NIfTI file and return array, affine, and header."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to read NIfTI files. Install it with "
            "'pip install nibabel' or 'conda install -c conda-forge nibabel'."
        ) from exc

    nifti_path = Path(path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {nifti_path}")
    if not is_nifti_file(nifti_path):
        raise ValueError(f"Expected a .nii or .nii.gz file, got: {nifti_path}")

    image = nib.load(str(nifti_path))
    array = np.asarray(image.get_fdata(dtype=np.float32))
    return array, image.affine.copy(), image.header.copy()


def _format_shape_map(shapes: Dict[str, Tuple[int, ...]]) -> str:
    return ", ".join(f"{name}: {shape}" for name, shape in shapes.items())


def standardize_segmentation_labels(
    seg: np.ndarray, allow_label3_as_et: bool = True
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Map segmentation labels to the project/PDF canonical labels.

    The assignment defines label 4 as ET. Some BraTS-GLI files are distributed
    with ET stored as label 3; when allowed, label 3 is remapped to 4 before
    downstream WT/TC/ET conversion.
    """
    standardized = np.asarray(seg).astype(np.uint8, copy=True)
    unique_labels = set(int(value) for value in np.unique(standardized))

    accepted_labels = set(VALID_SEG_LABELS)
    if allow_label3_as_et:
        accepted_labels.update(LEGACY_ET_LABELS)

    unknown = sorted(unique_labels - accepted_labels)
    if unknown:
        warnings.warn(f"Unexpected segmentation labels found: {unknown}.")

    mapping: Dict[int, int] = {}
    if allow_label3_as_et and any(label in unique_labels for label in LEGACY_ET_LABELS):
        for legacy_label in LEGACY_ET_LABELS:
            standardized[standardized == legacy_label] = ET_LABELS[0]
            mapping[int(legacy_label)] = int(ET_LABELS[0])

    return standardized, mapping


def load_case(case_dir: PathLike) -> Dict[str, Any]:
    """Load four MRI modalities and segmentation for one BraTS case."""
    case_path = Path(case_dir)
    if not case_path.exists():
        raise FileNotFoundError(f"Case directory does not exist: {case_path}")
    if not case_path.is_dir():
        raise NotADirectoryError(f"Case path is not a directory: {case_path}")

    file_map = _scan_case_file_map(case_path)
    required = list(MODALITY_ORDER) + ["seg"]
    missing = [name for name in required if name not in file_map]
    if missing:
        found = (
            ", ".join(sorted(path.name for path in _iter_nifti_files(case_path)))
            or "none"
        )
        raise FileNotFoundError(
            f"Case '{case_path.name}' is missing required files for: {missing}. "
            f"Found NIfTI files: {found}"
        )

    modalities: Dict[str, np.ndarray] = {}
    affine = None
    header = None
    shapes: Dict[str, Tuple[int, ...]] = {}

    for name in MODALITY_ORDER:
        array, this_affine, this_header = load_nifti(file_map[name])
        if array.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume for {name} in case '{case_path.name}', "
                f"got shape {array.shape}."
            )
        modalities[name] = array.astype(np.float32, copy=False)
        shapes[name] = tuple(array.shape)
        if affine is None:
            affine = this_affine
            header = this_header

    seg_array, seg_affine, seg_header = load_nifti(file_map["seg"])
    if seg_array.ndim != 3:
        raise ValueError(
            f"Expected a 3D segmentation in case '{case_path.name}', got shape {seg_array.shape}."
        )
    raw_seg = np.rint(seg_array).astype(np.uint8)
    seg_original_labels = sorted(int(value) for value in np.unique(raw_seg))
    seg, seg_label_mapping = standardize_segmentation_labels(raw_seg)
    shapes["seg"] = tuple(seg.shape)

    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"Shape mismatch in case '{case_path.name}'. "
            f"All modalities and seg must match. Shapes: {_format_shape_map(shapes)}"
        )

    return {
        "case_id": case_path.name,
        "case_dir": case_path,
        "modalities": modalities,
        "modality_paths": {name: file_map[name] for name in MODALITY_ORDER},
        "seg": seg,
        "seg_path": file_map["seg"],
        "seg_original_labels": seg_original_labels,
        "seg_label_mapping": seg_label_mapping,
        "affine": affine,
        "header": header,
        "seg_affine": seg_affine,
        "seg_header": seg_header,
        "original_shape": tuple(seg.shape),
        "cropped_shape": None,
        "bbox": None,
        "normalization": {},
    }


def get_tumor_slice(seg: np.ndarray) -> int:
    """Return the slice index with the largest tumor area."""
    if seg.ndim != 3:
        raise ValueError(f"Expected seg shape [H, W, D], got {seg.shape}.")

    tumor_areas = np.sum(seg > 0, axis=(0, 1))
    if tumor_areas.size == 0:
        raise ValueError("Segmentation has no slice dimension.")
    if int(tumor_areas.max()) == 0:
        warnings.warn("Empty tumor mask: using the middle slice for visualization.")
        return int(seg.shape[2] // 2)
    return int(np.argmax(tumor_areas))


def zscore_normalize(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    apply_to_all: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Z-score normalize a volume using brain-mask statistics.

    Mean and standard deviation are computed only inside ``mask`` (or ``volume > 0`` if
    no mask is provided). By default, the same z-score transform is then applied to the
    whole volume, including background voxels, so background values can become negative.
    Set ``apply_to_all=False`` only when old "background stays zero" behavior is needed.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if mask is None:
        valid_mask = volume > 0
    else:
        if mask.shape != volume.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match volume shape {volume.shape}."
            )
        valid_mask = mask.astype(bool)

    valid_mask = valid_mask & np.isfinite(volume)

    if not np.any(valid_mask):
        warnings.warn(
            "No valid brain voxels found; returning the input volume as float32."
        )
        return volume.astype(np.float32, copy=True), float("nan"), float("nan")

    values = volume[valid_mask].astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        warnings.warn(f"Very small standard deviation ({std}); using eps={eps}.")
        std = float(eps)

    if apply_to_all:
        normalized = ((volume - mean) / std).astype(np.float32)
    else:
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[valid_mask] = ((volume[valid_mask] - mean) / std).astype(np.float32)
    return normalized, mean, std


def get_joint_nonzero_mask(modality_list: Sequence[np.ndarray]) -> np.ndarray:
    """Return the union nonzero mask from multiple aligned modalities."""
    if not modality_list:
        raise ValueError("modality_list cannot be empty.")

    first_shape = modality_list[0].shape
    brain_mask = np.zeros(first_shape, dtype=bool)
    for idx, volume in enumerate(modality_list):
        if volume.shape != first_shape:
            raise ValueError(
                f"Modality at index {idx} has shape {volume.shape}, expected {first_shape}."
            )
        brain_mask |= (np.asarray(volume) != 0) & np.isfinite(volume)
    return brain_mask


def get_nonzero_bbox(modality_list: Sequence[np.ndarray]) -> BBox:
    """Compute a 3D bounding box from the joint nonzero area of modalities."""
    brain_mask = get_joint_nonzero_mask(modality_list)
    if not np.any(brain_mask):
        warnings.warn(
            "Joint nonzero mask is empty; using the full image extent as bbox."
        )
        return tuple((0, int(size)) for size in brain_mask.shape)  # type: ignore[return-value]

    coords = np.where(brain_mask)
    bbox = tuple(
        (int(axis_coords.min()), int(axis_coords.max()) + 1) for axis_coords in coords
    )
    return bbox  # type: ignore[return-value]


def crop_to_bbox(volume: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a 3D volume with a bbox of ((h0, h1), (w0, w1), (d0, d1))."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    if len(bbox) != 3:
        raise ValueError(f"Expected a 3-axis bbox, got: {bbox}")

    slices = []
    for axis, (start, end) in enumerate(bbox):
        if start < 0 or end > volume.shape[axis] or start >= end:
            raise ValueError(
                f"Invalid bbox for axis {axis}: {(start, end)} with shape {volume.shape}."
            )
        slices.append(slice(start, end))
    return volume[tuple(slices)]


def crop_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Crop modalities and segmentation with one shared joint-nonzero bbox."""
    modalities = case["modalities"]
    if "brain_mask" in case and case["brain_mask"] is not None:
        bbox = get_nonzero_bbox([case["brain_mask"].astype(np.uint8)])
    else:
        modality_volumes = [modalities[name] for name in MODALITY_ORDER]
        bbox = get_nonzero_bbox(modality_volumes)

    cropped = dict(case)
    cropped["modalities"] = {
        name: crop_to_bbox(volume, bbox) for name, volume in modalities.items()
    }
    cropped["seg"] = crop_to_bbox(case["seg"], bbox)
    if "brain_mask" in case and case["brain_mask"] is not None:
        cropped["brain_mask"] = crop_to_bbox(
            case["brain_mask"].astype(np.uint8), bbox
        ).astype(bool)

    cropped["bbox"] = bbox
    cropped["cropped_shape"] = tuple(cropped["seg"].shape)
    return cropped


def ceil_to_multiple(value: int, multiple: int = 16) -> int:
    """Round an integer up to the nearest multiple."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}.")
    value = int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def _cropped_shape_from_case_or_dir(
    item: Union[Dict[str, Any], PathLike],
) -> Tuple[int, int, int]:
    if isinstance(item, dict):
        if item.get("cropped_shape") is not None:
            return tuple(int(x) for x in item["cropped_shape"])  # type: ignore[return-value]
        modalities = item["modalities"]
        bbox = get_nonzero_bbox([modalities[name] for name in MODALITY_ORDER])
    else:
        case = load_case(item)
        brain_mask = get_joint_nonzero_mask(
            [case["modalities"][name] for name in MODALITY_ORDER]
        )
        bbox = get_nonzero_bbox([brain_mask.astype(np.uint8)])

    return tuple(int(end - start) for start, end in bbox)  # type: ignore[return-value]


def compute_target_padding_shape(
    cases_or_case_dirs: Iterable[Union[Dict[str, Any], PathLike]], multiple: int = 16
) -> Tuple[int, int]:
    """Compute padded [H, W] target from max cropped H/W rounded to a multiple."""
    max_h = 0
    max_w = 0
    count = 0
    for item in cases_or_case_dirs:
        count += 1
        cropped_h, cropped_w, _ = _cropped_shape_from_case_or_dir(item)
        max_h = max(max_h, int(cropped_h))
        max_w = max(max_w, int(cropped_w))

    if count == 0:
        raise ValueError("cases_or_case_dirs cannot be empty.")

    return ceil_to_multiple(max_h, multiple), ceil_to_multiple(max_w, multiple)


def pad_2d_to_target(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    value: Union[int, float, Sequence[Union[int, float]], np.ndarray] = 0,
) -> np.ndarray:
    """Center-pad a channel-first 2D array [C, H, W] to [C, target_h, target_w].

    ``value`` can be a scalar for all channels or a length-C sequence for per-channel
    padding values.
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got shape {arr.shape}.")

    channels, height, width = arr.shape
    if height > target_h or width > target_w:
        raise ValueError(
            f"Cannot pad shape {arr.shape} to target ({target_h}, {target_w}); "
            "input is larger than target."
        )

    pad_h = int(target_h - height)
    pad_w = int(target_w - width)
    top = pad_h // 2
    left = pad_w // 2

    values = np.asarray(value)
    if values.ndim == 0:
        channel_values = np.full(channels, values.item(), dtype=arr.dtype)
    else:
        channel_values = values.reshape(-1)
        if channel_values.size != channels:
            raise ValueError(
                f"Expected {channels} padding values for shape {arr.shape}, "
                f"got {channel_values.size}."
            )
        channel_values = channel_values.astype(arr.dtype, copy=False)

    padded = np.empty((channels, int(target_h), int(target_w)), dtype=arr.dtype)
    for channel_idx, pad_value in enumerate(channel_values):
        padded[channel_idx].fill(pad_value)
    padded[:, top : top + height, left : left + width] = arr
    return padded


def get_channel_min_padding_values(arr: np.ndarray) -> np.ndarray:
    """Return one finite minimum value per channel for [C, H, W] image padding."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got shape {arr.shape}.")

    pad_values = []
    for channel_idx in range(arr.shape[0]):
        channel = arr[channel_idx]
        finite = channel[np.isfinite(channel)]
        pad_values.append(float(finite.min()) if finite.size else 0.0)
    return np.asarray(pad_values, dtype=np.float32)


def transform_labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """Convert BraTS labels to [3, H, W, D] masks: WT, TC, ET.

    This follows the project/PDF canonical labels:
    1 = NCR/NET, 2 = ED, 4 = ET. If a dataset stores ET as label 3,
    call standardize_segmentation_labels first to remap 3 to 4.
    """
    if seg.ndim != 3:
        raise ValueError(f"Expected seg shape [H, W, D], got {seg.shape}.")

    seg = np.asarray(seg)
    allowed = set(VALID_SEG_LABELS)
    unique_labels = set(int(value) for value in np.unique(seg))
    unknown = sorted(unique_labels - allowed)
    if unknown:
        warnings.warn(
            f"Unexpected segmentation labels found after standardization: {unknown}."
        )

    wt = np.isin(seg, NCR_NET_LABELS + EDEMA_LABELS + ET_LABELS)
    tc = np.isin(seg, NCR_NET_LABELS + ET_LABELS)
    et = np.isin(seg, ET_LABELS)

    if np.any(et & ~tc) or np.any(tc & ~wt):
        warnings.warn("Region nesting check failed: expected ET subset TC subset WT.")

    return np.stack([wt, tc, et], axis=0).astype(np.uint8)


def _masked_mean(volume: np.ndarray, mask: np.ndarray, label: str) -> float:
    if not np.any(mask):
        warnings.warn(f"Empty mask for {label}; returning NaN.")
        return float("nan")
    return float(np.asarray(volume)[mask].mean())


def compute_contrast_statistics(
    case: Dict[str, Any], data_stage: str = "processed"
) -> Any:
    """Compute tumor-vs-healthy intensity statistics for each modality.

    ``data_stage`` is recorded in the output so raw and processed contrasts can be
    compared without mixing their intensity scales.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to save contrast statistics. Install it with "
            "'pip install pandas' or 'conda install pandas'."
        ) from exc

    seg = case["seg"]
    regions = case.get("label_regions")
    if regions is None:
        regions = transform_labels_to_regions(seg)

    if "brain_mask" in case and case["brain_mask"] is not None:
        brain_mask = case["brain_mask"].astype(bool)
        brain_mask_source = "case_brain_mask"
        if brain_mask.shape != seg.shape:
            raise ValueError(
                f"Brain mask shape {brain_mask.shape} does not match seg shape {seg.shape}."
            )
    else:
        if str(data_stage).startswith("processed"):
            raise ValueError(
                "Processed contrast requires case['brain_mask']. "
                "Do not recompute a nonzero mask from z-score volumes because background "
                "is no longer zero."
            )
        brain_mask = get_joint_nonzero_mask(
            [case["modalities"][name] for name in MODALITY_ORDER]
        )
        brain_mask_source = "joint_nonzero_modalities"

    healthy_mask = brain_mask & (seg == 0)
    wt_mask = regions[0].astype(bool)
    et_mask = regions[2].astype(bool)

    rows = []
    for modality_name in MODALITY_ORDER:
        volume = case["modalities"][modality_name]
        mean_et = _masked_mean(volume, et_mask, f"{case['case_id']} {modality_name} ET")
        mean_wt = _masked_mean(volume, wt_mask, f"{case['case_id']} {modality_name} WT")
        mean_healthy = _masked_mean(
            volume, healthy_mask, f"{case['case_id']} {modality_name} healthy brain"
        )

        rows.append(
            {
                "case_id": case["case_id"],
                "data_stage": data_stage,
                "brain_mask_source": brain_mask_source,
                "modality": modality_name,
                "mean_ET": mean_et,
                "mean_WT": mean_wt,
                "mean_healthy": mean_healthy,
                "ET_vs_healthy_contrast": abs(mean_et - mean_healthy),
                "WT_vs_healthy_contrast": abs(mean_wt - mean_healthy),
                "ET_voxels": int(et_mask.sum()),
                "WT_voxels": int(wt_mask.sum()),
                "healthy_voxels": int(healthy_mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def save_processed_2d_slices(
    case: Dict[str, Any],
    output_dir: PathLike,
    include_empty: bool = False,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Save processed 2D slices as compressed .npz files."""
    output_root = Path(output_dir)
    case_id = str(case["case_id"])
    case_output_dir = output_root / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    seg = case["seg"]
    regions = case.get("label_regions")
    if regions is None:
        regions = transform_labels_to_regions(seg)

    if regions.shape[0] != 3 or regions.shape[1:] != seg.shape:
        raise ValueError(
            f"Expected region shape [3, H, W, D] matching seg {seg.shape}, got {regions.shape}."
        )

    total_slices = int(seg.shape[2])
    saved_slices = 0
    tumor_slices = 0
    original_shape = np.asarray(case.get("original_shape", seg.shape), dtype=np.int32)
    cropped_shape = np.asarray(case.get("cropped_shape", seg.shape), dtype=np.int32)
    if target_shape is None:
        target_h, target_w = int(seg.shape[0]), int(seg.shape[1])
    else:
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
    padded_shape = np.asarray([target_h, target_w, total_slices], dtype=np.int32)

    for slice_idx in range(total_slices):
        image = np.stack(
            [case["modalities"][name][:, :, slice_idx] for name in MODALITY_ORDER],
            axis=0,
        ).astype(np.float32)
        label = regions[:, :, :, slice_idx].astype(np.uint8)

        image_padding_values = get_channel_min_padding_values(image)
        image = pad_2d_to_target(
            image, target_h, target_w, value=image_padding_values
        ).astype(np.float32)
        label = pad_2d_to_target(label, target_h, target_w, value=0).astype(np.uint8)

        has_tumor = bool(label.sum() > 0)
        if has_tumor:
            tumor_slices += 1
        if not include_empty and not has_tumor:
            continue

        save_path = case_output_dir / f"{case_id}_slice_{slice_idx:03d}.npz"
        np.savez_compressed(
            save_path,
            image=image,
            label=label,
            case_id=np.asarray(case_id),
            slice_idx=np.asarray(slice_idx, dtype=np.int32),
            original_shape=original_shape,
            cropped_shape=cropped_shape,
            padded_shape=padded_shape,
            target_h=np.asarray(target_h, dtype=np.int32),
            target_w=np.asarray(target_w, dtype=np.int32),
            image_padding_values=image_padding_values.astype(np.float32),
            modality_order=np.asarray(MODALITY_ORDER),
            label_order=np.asarray(REGION_ORDER),
        )
        saved_slices += 1

    return {
        "saved_slices": saved_slices,
        "total_slices": total_slices,
        "tumor_slices": tumor_slices,
        "target_h": target_h,
        "target_w": target_w,
        "padded_shape": tuple(int(x) for x in padded_shape),
        "output_dir": case_output_dir,
    }


def shape_to_string(shape: Iterable[int]) -> str:
    """Serialize a shape for CSV output."""
    return "x".join(str(int(value)) for value in shape)


def bbox_to_string(bbox: Optional[BBox]) -> str:
    """Serialize a bbox for CSV output."""
    if bbox is None:
        return ""
    return ";".join(f"{start}:{end}" for start, end in bbox)
