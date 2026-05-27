"""Main pipeline for BraTS-GLI Task 1 preprocessing and visualization.

Edit DATASET_DIR below, or pass --dataset_dir from the command line.
No deep learning model is trained in this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from task1_utils import (
    MODALITY_ORDER,
    bbox_to_string,
    compute_target_padding_shape,
    compute_contrast_statistics,
    crop_case,
    find_patient_cases,
    get_joint_nonzero_mask,
    get_tumor_slice,
    load_case,
    save_processed_2d_slices,
    shape_to_string,
    transform_labels_to_regions,
    zscore_normalize,
)
from task1_visualization import (
    plot_contrast_statistics,
    visualize_processed_four_modalities_overlay,
    visualize_label_regions,
    visualize_raw_four_modalities_overlay,
    visualize_top_t1ce_et_contrast_cases,
)

DATASET_DIR = Path("projectdataset/datasets/aiocta/brats2023-part-1/versions/1")
OUTPUT_DIR = Path("outputs_task1")
INCLUDE_EMPTY = False
MAX_CASES = None
REQUIRED_SEG_LABELS = (1, 2, 4)


def check_runtime_dependencies() -> None:
    """Fail early with a clear message when Task 1 dependencies are missing."""
    required = {
        "numpy": "numpy",
        "nibabel": "nibabel",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "tqdm": "tqdm",
    }
    missing = [
        package
        for package, module_name in required.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them with: pip install numpy nibabel matplotlib pandas tqdm"
        )


def normalize_case_modalities(case: Dict[str, Any]) -> Dict[str, Any]:
    """Z-score normalize each modality with joint-brain statistics applied to all voxels."""
    normalized_case = dict(case)
    raw_modalities = case["modalities"]
    normalized_modalities = {}
    normalization_stats = {}

    brain_mask = get_joint_nonzero_mask(
        [raw_modalities[name] for name in MODALITY_ORDER]
    )
    for modality_name in MODALITY_ORDER:
        normalized, mean, std = zscore_normalize(
            raw_modalities[modality_name],
            mask=brain_mask,
            apply_to_all=True,
        )
        normalized_modalities[modality_name] = normalized
        bg_zscore = float("nan")
        if np.isfinite(mean) and np.isfinite(std) and std != 0:
            bg_zscore = float((0.0 - mean) / std)
        normalization_stats[modality_name] = {
            "mean": mean,
            "std": std,
            "background_value_after_zscore": bg_zscore,
        }

    normalized_case["modalities"] = normalized_modalities
    normalized_case["normalization"] = normalization_stats
    normalized_case["brain_mask"] = brain_mask
    return normalized_case


def prepare_case_for_task1(case_dir: Path) -> Dict[str, Any]:
    """Load, normalize, crop, and convert labels for one case."""
    raw_case = load_case(case_dir)
    normalized_case = normalize_case_modalities(raw_case)
    cropped_case = crop_case(normalized_case)
    cropped_case["label_regions"] = transform_labels_to_regions(cropped_case["seg"])
    return {"raw_case": raw_case, "processed_case": cropped_case}


def _ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed_slices"
    summary_dir = output_dir / "summary_csv"
    for path in (figures_dir, processed_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "figures": figures_dir,
        "processed_slices": processed_dir,
        "summary_csv": summary_dir,
    }


def _clear_generated_outputs(output_dirs: Dict[str, Path]) -> None:
    """Remove previous generated Task 1 outputs that can make reruns misleading."""
    for png_path in output_dirs["figures"].glob("*.png"):
        png_path.unlink()

    for child in output_dirs["processed_slices"].iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for csv_path in output_dirs["summary_csv"].glob("*.csv"):
        csv_path.unlink()


def _mapping_to_string(mapping: Dict[int, int]) -> str:
    if not mapping:
        return ""
    return ";".join(f"{source}->{target}" for source, target in sorted(mapping.items()))


def _missing_required_seg_labels(seg: np.ndarray) -> List[int]:
    """Return required BraTS tumor labels that are absent after label standardization."""
    present = set(int(value) for value in np.unique(seg))
    return [label for label in REQUIRED_SEG_LABELS if label not in present]


def _case_summary_row(
    case: Dict[str, Any], slice_info: Dict[str, Any]
) -> Dict[str, Any]:
    regions = case["label_regions"]
    row = {
        "case_id": case["case_id"],
        "original_shape": shape_to_string(case["original_shape"]),
        "cropped_shape": shape_to_string(case["cropped_shape"]),
        "padded_shape": shape_to_string(slice_info["padded_shape"]),
        "target_h": int(slice_info["target_h"]),
        "target_w": int(slice_info["target_w"]),
        "bbox": bbox_to_string(case.get("bbox")),
        "seg_original_labels": ",".join(
            str(x) for x in case.get("seg_original_labels", [])
        ),
        "seg_label_mapping": _mapping_to_string(case.get("seg_label_mapping", {})),
        "number_of_total_slices": int(case["cropped_shape"][2]),
        "number_of_tumor_slices": int(slice_info["tumor_slices"]),
        "number_of_saved_slices": int(slice_info["saved_slices"]),
        "WT_voxels": int(regions[0].sum()),
        "TC_voxels": int(regions[1].sum()),
        "ET_voxels": int(regions[2].sum()),
    }
    normalization = case.get("normalization", {})
    for modality_name in MODALITY_ORDER:
        bg_value = normalization.get(modality_name, {}).get(
            "background_value_after_zscore", ""
        )
        row[f"{modality_name}_bg_zscore"] = bg_value
    return row


def _best_et_contrast_modality(
    contrast_df: Any,
    value_column: str = "ET_vs_healthy_contrast",
) -> str:
    if value_column not in contrast_df.columns:
        return "N/A"
    grouped = (
        contrast_df.groupby("modality")[value_column].mean().reindex(MODALITY_ORDER)
    )
    valid = grouped.dropna()
    if valid.empty:
        return "N/A"
    return str(valid.idxmax())


def process_one_case(
    case_dir: Path,
    output_dirs: Dict[str, Path],
    include_empty: bool,
    target_shape: tuple[int, int],
) -> Dict[str, Any]:
    """Run Task 1 preprocessing for one case and return summary objects."""
    raw_case = load_case(case_dir)
    case_id = raw_case["case_id"]
    print(f"[INFO] Loaded {case_id}: shape={raw_case['original_shape']}")
    if raw_case.get("seg_label_mapping"):
        print(
            f"[INFO] Standardized labels for {case_id}: "
            f"{_mapping_to_string(raw_case['seg_label_mapping'])}"
        )

    missing_labels = _missing_required_seg_labels(raw_case["seg"])
    if missing_labels:
        reason = (
            "missing required segmentation label(s) after standardization: "
            + ",".join(str(label) for label in missing_labels)
        )
        print(f"[WARN] Skipping {case_id}: {reason}.")
        return {
            "skipped": True,
            "case_id": case_id,
            "skip_reason": reason,
            "saved_slices": 0,
        }

    raw_case_for_contrast = dict(raw_case)
    raw_case_for_contrast["label_regions"] = transform_labels_to_regions(
        raw_case["seg"]
    )
    raw_case_for_contrast["brain_mask"] = get_joint_nonzero_mask(
        [raw_case["modalities"][name] for name in MODALITY_ORDER]
    )
    raw_contrast_df = compute_contrast_statistics(
        raw_case_for_contrast, data_stage="raw"
    )

    normalized_case = normalize_case_modalities(raw_case)
    cropped_case = crop_case(normalized_case)
    cropped_case["label_regions"] = transform_labels_to_regions(cropped_case["seg"])

    print(
        f"[INFO] Cropped {case_id}: "
        f"{raw_case['original_shape']} -> {cropped_case['cropped_shape']}"
    )
    bg_values = [
        cropped_case.get("normalization", {})
        .get(modality_name, {})
        .get("background_value_after_zscore", float("nan"))
        for modality_name in MODALITY_ORDER
    ]
    print(
        "[INFO] Background z-score values "
        + ", ".join(
            (
                f"{modality_name}={bg_value:.4f}"
                if np.isfinite(bg_value)
                else f"{modality_name}=nan"
            )
            for modality_name, bg_value in zip(MODALITY_ORDER, bg_values)
        )
    )

    processed_contrast_df = compute_contrast_statistics(
        cropped_case, data_stage="processed_zscore"
    )
    slice_info = save_processed_2d_slices(
        cropped_case,
        output_dirs["processed_slices"],
        include_empty=include_empty,
        target_shape=target_shape,
    )
    print(
        f"[INFO] Saved {slice_info['saved_slices']} 2D slices for {case_id} "
        f"(tumor slices={slice_info['tumor_slices']}, total={slice_info['total_slices']})."
    )

    return {
        "skipped": False,
        "case": cropped_case,
        "raw_contrast_df": raw_contrast_df,
        "processed_contrast_df": processed_contrast_df,
        "summary_row": _case_summary_row(cropped_case, slice_info),
        "saved_slices": int(slice_info["saved_slices"]),
    }


def select_representative_case_ids(
    case_rows: List[Dict[str, Any]], n: int = 3
) -> List[str]:
    """Select representative cases by low/median/high WT tumor burden."""
    if not case_rows:
        return []
    if len(case_rows) <= n:
        return [str(row["case_id"]) for row in case_rows]

    sorted_rows = sorted(
        case_rows, key=lambda row: (int(row["WT_voxels"]), str(row["case_id"]))
    )
    positions = [round(i * (len(sorted_rows) - 1) / (n - 1)) for i in range(n)]

    selected = []
    for position in positions:
        case_id = str(sorted_rows[int(position)]["case_id"])
        if case_id not in selected:
            selected.append(case_id)
    return selected


def _processed_slice_from_raw_slice(
    processed_case: Dict[str, Any], raw_slice_idx: int
) -> int:
    bbox = processed_case.get("bbox")
    if bbox is None:
        return get_tumor_slice(processed_case["seg"])

    processed_idx = int(raw_slice_idx) - int(bbox[2][0])
    if 0 <= processed_idx < processed_case["seg"].shape[2]:
        return processed_idx
    return get_tumor_slice(processed_case["seg"])


def generate_representative_visualizations(
    case_rows: List[Dict[str, Any]],
    case_id_to_dir: Dict[str, Path],
    output_dirs: Dict[str, Path],
    target_shape: tuple[int, int],
    n: int = 3,
) -> List[str]:
    """Generate compact raw/processed visualizations for representative cases."""
    selected_ids = select_representative_case_ids(case_rows, n=n)

    for index, case_id in enumerate(selected_ids, start=1):
        prepared = prepare_case_for_task1(case_id_to_dir[case_id])
        raw_case = prepared["raw_case"]
        processed_case = prepared["processed_case"]

        raw_slice_idx = get_tumor_slice(raw_case["seg"])
        processed_slice_idx = _processed_slice_from_raw_slice(
            processed_case, raw_slice_idx
        )
        case_label = f"case {index:03d}"

        visualize_raw_four_modalities_overlay(
            raw_case,
            slice_idx=raw_slice_idx,
            save_path=output_dirs["figures"]
            / f"raw_case_{index:03d}_four_modalities_overlay.png",
            case_label=case_label,
        )
        visualize_processed_four_modalities_overlay(
            processed_case,
            slice_idx=processed_slice_idx,
            target_shape=target_shape,
            save_path=output_dirs["figures"]
            / f"processed_case_{index:03d}_four_modalities_overlay.png",
            case_label=case_label,
        )

        if index == 1:
            visualize_label_regions(
                processed_case["label_regions"],
                case_id,
                save_path=output_dirs["figures"] / "label_regions_WT_TC_ET.png",
            )

    return selected_ids


def generate_top_t1ce_et_contrast_visualization(
    raw_contrast_df: Any,
    case_id_to_dir: Dict[str, Path],
    output_dirs: Dict[str, Path],
    top_n: int = 5,
) -> List[str]:
    """Save a compact figure of raw T1ce cases with largest ET contrast."""
    contrast_column = "ET_vs_healthy_abs_contrast"
    if contrast_column not in raw_contrast_df.columns:
        contrast_column = "ET_vs_healthy_contrast"
    t1ce_rows = raw_contrast_df[
        raw_contrast_df["modality"].astype(str).str.lower() == "t1ce"
    ].copy()
    t1ce_rows = t1ce_rows.dropna(subset=[contrast_column])
    t1ce_rows = t1ce_rows.sort_values(contrast_column, ascending=False).head(top_n)

    examples = []
    for rank, (_, row) in enumerate(t1ce_rows.iterrows(), start=1):
        case_id = str(row["case_id"])
        case_dir = case_id_to_dir.get(case_id)
        if case_dir is None:
            continue
        raw_case = load_case(case_dir)
        et_areas = np.sum(raw_case["seg"] == 4, axis=(0, 1))
        if et_areas.size == 0 or int(et_areas.max()) == 0:
            continue
        examples.append(
            {
                "rank": rank,
                "case": raw_case,
                "contrast": float(row[contrast_column]),
                "slice_idx": int(np.argmax(et_areas)),
            }
        )

    if examples:
        visualize_top_t1ce_et_contrast_cases(
            examples,
            save_path=output_dirs["figures"] / "top5_t1ce_et_contrast_examples.png",
        )

    return [str(example["case"]["case_id"]) for example in examples]


def inspect_random_saved_npz(
    processed_dir: Path, target_shape: tuple[int, int]
) -> None:
    """Print shape and label sanity checks for one saved .npz slice."""
    npz_files = sorted(processed_dir.rglob("*.npz"))
    if not npz_files:
        print("[WARN] No saved .npz files found for random inspection.")
        return

    sample_path = random.Random(1312).choice(npz_files)
    with np.load(sample_path) as sample:
        image = sample["image"]
        label = sample["label"]
        label_sums = [int(label[channel].sum()) for channel in range(label.shape[0])]
        image_channel_mins = [
            float(np.nanmin(image[channel])) for channel in range(image.shape[0])
        ]
        label_unique = np.unique(label)
        image_padding_values = (
            sample["image_padding_values"].astype(np.float32)
            if "image_padding_values" in sample.files
            else np.asarray(image_channel_mins, dtype=np.float32)
        )

    expected_image_shape = (4, int(target_shape[0]), int(target_shape[1]))
    expected_label_shape = (3, int(target_shape[0]), int(target_shape[1]))
    if (
        tuple(image.shape) != expected_image_shape
        or tuple(label.shape) != expected_label_shape
    ):
        raise ValueError(
            f"Saved npz shape mismatch in {sample_path}: "
            f"image {image.shape}, label {label.shape}, "
            f"expected {expected_image_shape} and {expected_label_shape}."
        )
    if not set(label_unique.tolist()).issubset({0, 1}):
        raise ValueError(
            f"Saved label should be binary 0/1, got values {label_unique.tolist()}."
        )

    print(f"Random npz check: {sample_path}")
    print(f"image shape: {image.shape}")
    print(f"label shape: {label.shape}")
    print(
        f"image channel min values: {[round(value, 4) for value in image_channel_mins]}"
    )
    print(
        "image padding values: "
        f"{[round(float(value), 4) for value in image_padding_values.tolist()]}"
    )
    print(f"label unique values: {label_unique.tolist()}")
    print(f"label voxel sums WT/TC/ET: {label_sums}")


def main(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    include_empty: bool = INCLUDE_EMPTY,
    max_cases: Optional[int] = MAX_CASES,
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    """Run the complete Task 1 pipeline over all discovered cases."""
    check_runtime_dependencies()

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for summary CSV output.") from exc

    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir)
    output_dirs = _ensure_output_dirs(output_dir)
    _clear_generated_outputs(output_dirs)

    print(f"[INFO] Dataset directory: {dataset_dir}")
    print(f"[INFO] Output directory: {output_dir}")

    case_dirs = find_patient_cases(dataset_dir)
    if max_cases is not None:
        case_dirs = case_dirs[: int(max_cases)]
    print(f"[INFO] Found {len(case_dirs)} case(s).")

    print("[INFO] Scanning cropped shapes for unified padding target...")
    target_h, target_w = compute_target_padding_shape(
        tqdm(case_dirs, desc="Padding shape scan"),
        multiple=16,
    )
    target_shape = (target_h, target_w)
    print(f"[INFO] Target padded 2D shape: {target_h} x {target_w}")

    case_rows: List[Dict[str, Any]] = []
    raw_contrast_frames = []
    processed_contrast_frames = []
    skipped_cases: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    total_saved_slices = 0
    case_id_to_dir = {case_dir.name: case_dir for case_dir in case_dirs}

    for case_dir in tqdm(case_dirs, desc="Task 1 preprocessing"):
        try:
            result = process_one_case(
                case_dir=case_dir,
                output_dirs=output_dirs,
                include_empty=include_empty,
                target_shape=target_shape,
            )
            if result.get("skipped"):
                skipped_cases.append(
                    {
                        "case_id": str(result["case_id"]),
                        "reason": str(result["skip_reason"]),
                    }
                )
                continue
            case_rows.append(result["summary_row"])
            raw_contrast_frames.append(result["raw_contrast_df"])
            processed_contrast_frames.append(result["processed_contrast_df"])
            total_saved_slices += result["saved_slices"]
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] Failed case {case_dir.name}: {message}")
            errors.append({"case_id": case_dir.name, "error": message})
            if stop_on_error:
                raise

    if not case_rows:
        raise RuntimeError("No cases were processed successfully.")

    case_summary_df = pd.DataFrame(case_rows)
    raw_contrast_df = pd.concat(raw_contrast_frames, ignore_index=True)
    processed_contrast_df = pd.concat(processed_contrast_frames, ignore_index=True)

    case_summary_path = output_dirs["summary_csv"] / "case_summary.csv"
    contrast_path = output_dirs["summary_csv"] / "contrast_statistics.csv"
    raw_contrast_path = output_dirs["summary_csv"] / "raw_contrast_statistics.csv"
    case_summary_df.to_csv(case_summary_path, index=False)
    processed_contrast_df.to_csv(contrast_path, index=False)
    raw_contrast_df.to_csv(raw_contrast_path, index=False)

    if errors:
        errors_path = output_dirs["summary_csv"] / "processing_errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        print(f"[WARN] {len(errors)} case(s) failed. See: {errors_path}")
    if skipped_cases:
        skipped_path = output_dirs["summary_csv"] / "skipped_cases.csv"
        pd.DataFrame(skipped_cases).to_csv(skipped_path, index=False)
        print(
            f"[WARN] {len(skipped_cases)} case(s) skipped for incomplete labels. See: {skipped_path}"
        )

    visualization_case_ids = generate_representative_visualizations(
        case_rows,
        case_id_to_dir,
        output_dirs,
        target_shape=target_shape,
        n=3,
    )

    raw_contrast_plot_path = (
        output_dirs["figures"] / "raw_intensity_contrast_barplot.png"
    )
    processed_contrast_plot_path = (
        output_dirs["figures"] / "processed_intensity_contrast_barplot.png"
    )
    contrast_plot_path = output_dirs["figures"] / "intensity_contrast_barplot.png"
    plot_contrast_statistics(
        raw_contrast_df,
        save_path=raw_contrast_plot_path,
        title="Raw ET vs healthy brain absolute contrast",
        ylabel="Mean |ET - healthy|",
        value_column="ET_vs_healthy_abs_contrast",
    )
    plot_contrast_statistics(
        processed_contrast_df,
        save_path=processed_contrast_plot_path,
        title="Processed ET vs healthy brain relative contrast",
        ylabel="Mean |ET - healthy| / |healthy|",
        value_column="ET_vs_healthy_relative_contrast",
    )
    plot_contrast_statistics(
        raw_contrast_df,
        save_path=contrast_plot_path,
        title="Raw ET vs healthy brain absolute contrast",
        ylabel="Mean |ET - healthy|",
        value_column="ET_vs_healthy_abs_contrast",
    )
    top_t1ce_case_ids = generate_top_t1ce_et_contrast_visualization(
        raw_contrast_df,
        case_id_to_dir,
        output_dirs,
        top_n=5,
    )

    best_raw_modality = _best_et_contrast_modality(
        raw_contrast_df,
        value_column="ET_vs_healthy_abs_contrast",
    )
    best_processed_modality = _best_et_contrast_modality(
        processed_contrast_df,
        value_column="ET_vs_healthy_relative_contrast",
    )

    inspect_random_saved_npz(output_dirs["processed_slices"], target_shape=target_shape)

    print("Task 1 finished.")
    print(f"Number of cases processed: {len(case_rows)}")
    print(f"Number of cases skipped for incomplete labels: {len(skipped_cases)}")
    print(f"Number of 2D slices saved: {total_saved_slices}")
    print(f"Target padded 2D shape: {target_h} x {target_w}")
    print(f"Visualization cases: {', '.join(visualization_case_ids)}")
    print(f"Top T1ce ET contrast cases: {', '.join(top_t1ce_case_ids) or 'N/A'}")
    print(f"Best raw modality for ET absolute contrast: {best_raw_modality}")
    print(
        f"Best processed modality for ET relative contrast: {best_processed_modality}"
    )
    print(f"Outputs saved to: {output_dir}/")

    return {
        "cases_processed": len(case_rows),
        "cases_skipped_incomplete_labels": len(skipped_cases),
        "slices_saved": total_saved_slices,
        "target_h": target_h,
        "target_w": target_w,
        "visualization_cases": visualization_case_ids,
        "top_t1ce_et_contrast_cases": top_t1ce_case_ids,
        "best_raw_modality_for_ET_absolute_contrast": best_raw_modality,
        "best_processed_modality_for_ET_relative_contrast": best_processed_modality,
        "case_summary_csv": case_summary_path,
        "contrast_statistics_csv": contrast_path,
        "raw_contrast_statistics_csv": raw_contrast_path,
        "skipped_cases": skipped_cases,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Task 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="BraTS-GLI Task 1 preprocessing pipeline"
    )
    parser.add_argument("--dataset_dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--include_empty",
        action="store_true",
        default=INCLUDE_EMPTY,
        help="Save all slices, including empty labels.",
    )
    parser.add_argument(
        "--max_cases", type=int, default=MAX_CASES, help="Optional debug limit."
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one case fails.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        include_empty=args.include_empty,
        max_cases=args.max_cases,
        stop_on_error=args.stop_on_error,
    )
