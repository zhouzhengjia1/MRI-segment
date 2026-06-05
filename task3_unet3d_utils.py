"""Slab-based 3D U-Net components for Task 3 BraTS segmentation."""

from __future__ import annotations

import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm


SLICE_INDEX_PATTERN = re.compile(r"(?:^|[_-])slice[_-]?(\d+)$")


def read_npz_value(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        return x.tolist()

    return x


def get_slice_idx_from_npz_or_filename(path):
    path = Path(path)

    try:
        with np.load(path, allow_pickle=True) as data:
            if "slice_idx" in data:
                return int(read_npz_value(data["slice_idx"]))
    except Exception as exc:
        raise ValueError(f"Could not read slice_idx from npz file: {path}") from exc

    match = SLICE_INDEX_PATTERN.search(path.stem)
    if match is not None:
        return int(match.group(1))

    raise ValueError(
        f"Could not determine slice_idx for {path}. Expected a 'slice_idx' "
        "field inside the npz file or a filename ending with '_slice_045.npz'."
    )


def build_case_to_sorted_files(slice_files):
    case_to_items = defaultdict(list)

    for path in slice_files:
        path = Path(path)
        case_id = path.parent.name
        slice_idx = get_slice_idx_from_npz_or_filename(path)
        case_to_items[case_id].append((slice_idx, path))

    case_to_files = {}
    for case_id, items in sorted(case_to_items.items()):
        seen = set()
        for slice_idx, path in items:
            if slice_idx in seen:
                raise ValueError(
                    f"Duplicate slice_idx={slice_idx} found for case_id={case_id}. "
                    f"One duplicate path is {path}."
                )
            seen.add(slice_idx)

        case_to_files[case_id] = [path for _, path in sorted(items, key=lambda x: x[0])]

    return case_to_files


def scan_dataset_3d(data_root):
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    all_npz_files = sorted(data_root.glob("*/*.npz"))
    if len(all_npz_files) == 0:
        raise FileNotFoundError(f"No .npz slice files found under: {data_root}")

    case_to_files = build_case_to_sorted_files(all_npz_files)
    case_ids = sorted(case_to_files.keys())

    print("Number of cases:", len(case_ids))
    print("Number of slices:", len(all_npz_files))

    counts = [len(case_to_files[c]) for c in case_ids]
    print("Slices per case:")
    print("  min   :", min(counts))
    print("  max   :", max(counts))
    print("  mean  :", float(np.mean(counts)))
    print("  median:", float(np.median(counts)))

    first_case = case_ids[0]
    first_indices = [get_slice_idx_from_npz_or_filename(p) for p in case_to_files[first_case][:5]]
    print("Example sorted case:", first_case)
    print("First sorted slice indices:", first_indices)

    return case_to_files, case_ids


def inspect_npz_3d(path, image_key="image", label_key="label"):
    with np.load(path, allow_pickle=True) as data:
        print("File:", path)
        print("Keys:", list(data.files))

        image = data[image_key]
        label = data[label_key]

        print("image shape:", image.shape, image.dtype)
        print("label shape:", label.shape, label.dtype)
        print("label unique:", np.unique(label))

        if "case_id" in data:
            print("case_id:", read_npz_value(data["case_id"]))

        if "slice_idx" in data:
            print("slice_idx:", read_npz_value(data["slice_idx"]))
        else:
            print("slice_idx from filename:", get_slice_idx_from_npz_or_filename(path))

        if "modality_order" in data:
            print("modality_order:", read_npz_value(data["modality_order"]))

        if "label_order" in data:
            print("label_order:", read_npz_value(data["label_order"]))


def apply_augmentation_3d(image, label):
    # image: [4, D, H, W], label: [3, D, H, W].
    # Use spatial flips only; depth order is preserved.
    if random.random() < 0.5:
        image = torch.flip(image, dims=[3])
        label = torch.flip(label, dims=[3])

    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        label = torch.flip(label, dims=[2])

    if random.random() < 0.3:
        noise_std = random.uniform(0.0, 0.03)
        image = image + noise_std * torch.randn_like(image)

    return image, label


def slab_collate_fn(samples):
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "label": torch.stack([sample["label"] for sample in samples], dim=0),
        "case_id": [sample["case_id"] for sample in samples],
        "center_slice_idx": torch.tensor(
            [sample["center_slice_idx"] for sample in samples], dtype=torch.long
        ),
        "center_depth_idx": torch.tensor(
            [sample["center_depth_idx"] for sample in samples], dtype=torch.long
        ),
        "slice_indices": [sample["slice_indices"] for sample in samples],
        "paths": [sample["paths"] for sample in samples],
    }


class BraTSSlab3DDataset(Dataset):
    def __init__(
        self,
        files,
        case_to_files,
        slab_depth=16,
        slab_mode="center",
        augment=False,
        image_key="image",
        label_key="label",
        num_modalities=4,
        out_channels=3,
        image_height=192,
        image_width=208,
    ):
        self.files = [Path(path) for path in files]
        self.case_to_files = {
            str(case_id): [Path(path) for path in paths]
            for case_id, paths in case_to_files.items()
        }
        self.slab_depth = int(slab_depth)
        self.slab_mode = str(slab_mode)
        self.augment = augment
        self.image_key = image_key
        self.label_key = label_key
        self.num_modalities = int(num_modalities)
        self.out_channels = int(out_channels)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        if self.slab_depth <= 0:
            raise ValueError(f"slab_depth must be positive, got {self.slab_depth}")
        if self.slab_mode not in {"center", "start"}:
            raise ValueError(f"slab_mode must be 'center' or 'start', got {self.slab_mode}")

        # For even slab depths, the center slice sits slightly left of the middle.
        self.center_depth_idx = (self.slab_depth - 1) // 2

        self.file_to_case_id = {}
        self.file_to_position = {}
        self.file_to_slice_idx = {}
        for case_id, case_files in self.case_to_files.items():
            for position, path in enumerate(case_files):
                path = Path(path)
                self.file_to_case_id[path] = str(case_id)
                self.file_to_position[path] = position
                self.file_to_slice_idx[path] = get_slice_idx_from_npz_or_filename(path)

        missing = [path for path in self.files if path not in self.file_to_position]
        if missing:
            raise ValueError(
                "Some dataset files are missing from case_to_files. "
                f"First missing file: {missing[0]}"
            )

        self.samples = []
        for path in self.files:
            self.samples.append(
                {
                    "case_id": self.file_to_case_id[path],
                    "position": self.file_to_position[path],
                    "path": path,
                }
            )

    def __len__(self):
        return len(self.samples)

    def _load_slice_arrays(self, path):
        with np.load(path, allow_pickle=True) as data:
            image = data[self.image_key].astype(np.float32)
            label = data[self.label_key].astype(np.float32)

        expected_image_shape = (
            self.num_modalities,
            self.image_height,
            self.image_width,
        )
        expected_label_shape = (
            self.out_channels,
            self.image_height,
            self.image_width,
        )

        if image.shape != expected_image_shape:
            raise ValueError(
                f"Unexpected image shape {image.shape} in {path}; "
                f"expected {expected_image_shape}"
            )

        if label.shape != expected_label_shape:
            raise ValueError(
                f"Unexpected label shape {label.shape} in {path}; "
                f"expected {expected_label_shape}"
            )

        return image, label

    def _slab_positions(self, case_files, base_position):
        last_position = len(case_files) - 1

        if self.slab_mode == "center":
            left = self.center_depth_idx
            offsets = range(-left, self.slab_depth - left)
            raw_positions = [base_position + offset for offset in offsets]
        else:
            raw_positions = [base_position + depth_idx for depth_idx in range(self.slab_depth)]

        return [min(max(pos, 0), last_position) for pos in raw_positions]

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        case_id = sample_info["case_id"]
        base_position = sample_info["position"]
        case_files = self.case_to_files[case_id]

        slab_positions = self._slab_positions(case_files, base_position)
        slab_paths = [case_files[pos] for pos in slab_positions]
        slice_indices = [int(self.file_to_slice_idx[path]) for path in slab_paths]

        center_path = slab_paths[self.center_depth_idx]
        center_slice_idx = int(self.file_to_slice_idx[center_path])

        image_slices = []
        label_slices = []
        for path in slab_paths:
            image, label = self._load_slice_arrays(path)
            image_slices.append(image)
            label_slices.append(label)

        image = np.stack(image_slices, axis=1)
        label = np.stack(label_slices, axis=1)

        expected_image_shape = (
            self.num_modalities,
            self.slab_depth,
            self.image_height,
            self.image_width,
        )
        expected_label_shape = (
            self.out_channels,
            self.slab_depth,
            self.image_height,
            self.image_width,
        )

        if image.shape != expected_image_shape:
            raise ValueError(
                f"Unexpected slab image shape {image.shape}; expected {expected_image_shape}"
            )
        if label.shape != expected_label_shape:
            raise ValueError(
                f"Unexpected slab label shape {label.shape}; expected {expected_label_shape}"
            )

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.augment:
            image, label = apply_augmentation_3d(image, label)

        return {
            "image": image,
            "label": label,
            "case_id": str(case_id),
            "center_slice_idx": center_slice_idx,
            "center_depth_idx": int(self.center_depth_idx),
            "slice_indices": slice_indices,
            "paths": [str(path) for path in slab_paths],
        }


def norm3d(num_channels, norm_layer="instance"):
    if norm_layer == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm_layer == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    raise ValueError(f"Unsupported norm_layer: {norm_layer}")


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm3d(out_channels, norm_layer=norm_layer),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm3d(out_channels, norm_layer=norm_layer),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def match_shape_3d(x, reference):
    diff_d = reference.size(2) - x.size(2)
    diff_h = reference.size(3) - x.size(3)
    diff_w = reference.size(4) - x.size(4)

    if diff_d == 0 and diff_h == 0 and diff_w == 0:
        return x

    if diff_d < 0 or diff_h < 0 or diff_w < 0:
        return F.interpolate(
            x,
            size=reference.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    return F.pad(
        x,
        [
            diff_w // 2,
            diff_w - diff_w // 2,
            diff_h // 2,
            diff_h - diff_h // 2,
            diff_d // 2,
            diff_d - diff_d // 2,
        ],
    )


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(1, 2, 2), norm_layer="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=pool_kernel),
            DoubleConv3D(in_channels, out_channels, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.block(x)


class Up3D(nn.Module):
    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        upsample_scale=(1, 2, 2),
        norm_layer="instance",
    ):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=upsample_scale,
            mode="trilinear",
            align_corners=True,
        )
        self.conv = DoubleConv3D(
            decoder_channels + skip_channels,
            out_channels,
            norm_layer=norm_layer,
        )

    def forward(self, decoder_feature, skip_feature):
        decoder_feature = self.up(decoder_feature)
        decoder_feature = match_shape_3d(decoder_feature, skip_feature)
        x = torch.cat([skip_feature, decoder_feature], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_channels=16,
        pool_kernel=(1, 2, 2),
        upsample_scale=(1, 2, 2),
        norm_layer="instance",
    ):
        super().__init__()
        self.in_conv = DoubleConv3D(in_channels, base_channels, norm_layer=norm_layer)

        self.down1 = Down3D(base_channels, base_channels * 2, pool_kernel, norm_layer)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, pool_kernel, norm_layer)
        self.down3 = Down3D(base_channels * 4, base_channels * 8, pool_kernel, norm_layer)
        self.down4 = Down3D(base_channels * 8, base_channels * 16, pool_kernel, norm_layer)

        self.up1 = Up3D(base_channels * 16, base_channels * 8, base_channels * 8, upsample_scale, norm_layer)
        self.up2 = Up3D(base_channels * 8, base_channels * 4, base_channels * 4, upsample_scale, norm_layer)
        self.up3 = Up3D(base_channels * 4, base_channels * 2, base_channels * 2, upsample_scale, norm_layer)
        self.up4 = Up3D(base_channels * 2, base_channels, base_channels, upsample_scale, norm_layer)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

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


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int, norm_layer="instance"):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm3d(F_int, norm_layer=norm_layer),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm3d(F_int, norm_layer=norm_layer),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[-3:] != x1.shape[-3:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp3D(nn.Module):
    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        upsample_scale=(1, 2, 2),
        norm_layer="instance",
    ):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=upsample_scale,
            mode="trilinear",
            align_corners=True,
        )
        self.attention = AttentionBlock3D(
            F_g=decoder_channels,
            F_l=skip_channels,
            F_int=max(skip_channels // 2, 1),
            norm_layer=norm_layer,
        )
        self.conv = DoubleConv3D(
            decoder_channels + skip_channels,
            out_channels,
            norm_layer=norm_layer,
        )

    def forward(self, decoder_feature, skip_feature):
        decoder_feature = self.up(decoder_feature)
        decoder_feature = match_shape_3d(decoder_feature, skip_feature)
        skip_feature = self.attention(g=decoder_feature, x=skip_feature)
        x = torch.cat([skip_feature, decoder_feature], dim=1)
        return self.conv(x)


class AttUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_channels=16,
        pool_kernel=(1, 2, 2),
        upsample_scale=(1, 2, 2),
        norm_layer="instance",
    ):
        super().__init__()
        self.in_conv = DoubleConv3D(in_channels, base_channels, norm_layer=norm_layer)

        self.down1 = Down3D(base_channels, base_channels * 2, pool_kernel, norm_layer)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, pool_kernel, norm_layer)
        self.down3 = Down3D(base_channels * 4, base_channels * 8, pool_kernel, norm_layer)
        self.down4 = Down3D(base_channels * 8, base_channels * 16, pool_kernel, norm_layer)

        self.up1 = AttentionUp3D(base_channels * 16, base_channels * 8, base_channels * 8, upsample_scale, norm_layer)
        self.up2 = AttentionUp3D(base_channels * 8, base_channels * 4, base_channels * 4, upsample_scale, norm_layer)
        self.up3 = AttentionUp3D(base_channels * 4, base_channels * 2, base_channels * 2, upsample_scale, norm_layer)
        self.up4 = AttentionUp3D(base_channels * 2, base_channels, base_channels, upsample_scale, norm_layer)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

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


class DiceLoss3D(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class DiceBCELoss3D(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss3D()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        total = self.dice_weight * dice + self.bce_weight * bce

        loss_parts = {
            "dice_loss": float(dice.detach().cpu()),
            "bce_loss": float(bce.detach().cpu()),
        }

        return total, loss_parts


class DiceAccumulator3D:
    def __init__(self, num_channels=3, threshold=0.5, smooth=1e-5):
        self.num_channels = num_channels
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = torch.zeros(num_channels, dtype=torch.float64)
        self.pred_sum = torch.zeros(num_channels, dtype=torch.float64)
        self.target_sum = torch.zeros(num_channels, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = targets.float()
        dims = (0, 2, 3, 4)

        self.intersection += torch.sum(preds * targets, dim=dims).detach().cpu().double()
        self.pred_sum += torch.sum(preds, dim=dims).detach().cpu().double()
        self.target_sum += torch.sum(targets, dim=dims).detach().cpu().double()

    def compute(self):
        dice = (2.0 * self.intersection + self.smooth) / (
            self.pred_sum + self.target_sum + self.smooth
        )
        return dice.numpy()


@torch.no_grad()
def dice_per_sample_channel_3d(logits, targets, threshold=0.5, smooth=1e-5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    dims = (2, 3, 4)

    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    return (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)


def sample_loss_values_3d(logits, targets, threshold=0.5, dice_weight=1.0, bce_weight=1.0):
    dice_values = dice_per_sample_channel_3d(logits, targets, threshold=threshold)
    dice_loss = 1.0 - dice_values.mean(dim=1)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    ).mean(dim=(1, 2, 3, 4))
    return dice_weight * dice_loss + bce_weight * bce


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch_3d(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_bce_loss = 0.0
    num_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and scaler.is_enabled():
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, loss_parts = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, loss_parts = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_size
        running_dice_loss += loss_parts["dice_loss"] * batch_size
        running_bce_loss += loss_parts["bce_loss"] * batch_size
        num_samples += batch_size

        pbar.set_postfix(
            {"loss": running_loss / max(num_samples, 1), "lr": get_current_lr(optimizer)}
        )

    return {
        "loss": running_loss / num_samples,
        "dice_loss": running_dice_loss / num_samples,
        "bce_loss": running_bce_loss / num_samples,
    }


@torch.no_grad()
def validate_one_epoch_3d(model, loader, criterion, device, out_channels=3, threshold=0.5):
    model.eval()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_bce_loss = 0.0
    num_samples = 0

    dice_acc = DiceAccumulator3D(num_channels=out_channels, threshold=threshold)
    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss, loss_parts = criterion(logits, labels)

        running_loss += loss.item() * batch_size
        running_dice_loss += loss_parts["dice_loss"] * batch_size
        running_bce_loss += loss_parts["bce_loss"] * batch_size
        num_samples += batch_size

        dice_acc.update(logits, labels)
        pbar.set_postfix({"loss": running_loss / max(num_samples, 1)})

    dice_values = dice_acc.compute()

    return {
        "loss": running_loss / num_samples,
        "dice_loss": running_dice_loss / num_samples,
        "bce_loss": running_bce_loss / num_samples,
        "dice_wt": float(dice_values[0]),
        "dice_tc": float(dice_values[1]),
        "dice_et": float(dice_values[2]),
        "mean_dice": float(np.mean(dice_values)),
    }


def normalize_mri_for_display(img, eps=1e-6):
    img = img.astype(np.float32)
    foreground_mask = np.abs(img) > eps
    out = np.zeros_like(img, dtype=np.float32)

    if foreground_mask.sum() == 0:
        return out

    values = img[foreground_mask]
    vmin = np.percentile(values, 1)
    vmax = np.percentile(values, 99)

    if vmax <= vmin:
        return out

    out[foreground_mask] = (img[foreground_mask] - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def show_3d_slab_sample(sample, region_names, modality_index=3):
    image = sample["image"].numpy()
    label = sample["label"].numpy()
    center_depth_idx = int(sample["center_depth_idx"])
    bg = normalize_mri_for_display(image[modality_index, center_depth_idx])
    gt_union = np.any(label[:, center_depth_idx] > 0, axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Image center slice\n{sample['case_id']}\nslice {sample['center_slice_idx']}"
    )
    axes[0].axis("off")

    for i, name in enumerate(region_names):
        axes[i + 1].imshow(bg, cmap="gray", vmin=0, vmax=1)
        axes[i + 1].imshow(
            np.ma.masked_where(label[i, center_depth_idx] <= 0.5, label[i, center_depth_idx]),
            cmap="Reds",
            alpha=0.5,
        )
        axes[i + 1].set_title(f"GT {name}")
        axes[i + 1].axis("off")

    axes[4].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(np.ma.masked_where(~gt_union, gt_union), cmap="Reds", alpha=0.45)
    axes[4].set_title("GT union")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_3d_slab_sample(model, dataset, dataset_idx, device, threshold=0.5):
    model.eval()
    sample = dataset[int(dataset_idx)]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred = (probs > threshold).astype(np.uint8)
    return sample, pred


def visualize_3d_slab_prediction(
    model,
    dataset,
    dataset_idx,
    device,
    save_path,
    modality_index=3,
    threshold=0.5,
    show=True,
):
    sample, pred = predict_3d_slab_sample(
        model=model,
        dataset=dataset,
        dataset_idx=dataset_idx,
        device=device,
        threshold=threshold,
    )

    image_np = sample["image"].numpy()
    label_np = sample["label"].numpy().astype(np.uint8)
    center_depth_idx = int(sample["center_depth_idx"])

    bg = normalize_mri_for_display(image_np[modality_index, center_depth_idx])
    gt_center = label_np[:, center_depth_idx]
    pred_center = pred[:, center_depth_idx]

    gt_union = np.any(gt_center > 0, axis=0)
    pred_union = np.any(pred_center > 0, axis=0)
    false_positive = np.logical_and(pred_union, ~gt_union)
    false_negative = np.logical_and(gt_union, ~pred_union)

    error_rgb = np.zeros((*gt_union.shape, 3), dtype=np.float32)
    error_rgb[false_positive] = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    error_rgb[false_negative] = np.array([0.1, 0.4, 1.0], dtype=np.float32)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Center image\n{sample['case_id']}\nslice {sample['center_slice_idx']}"
    )
    axes[0].axis("off")

    axes[1].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(~gt_union, gt_union), cmap="Reds", alpha=0.55)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(~pred_union, pred_union), cmap="Blues", alpha=0.55)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(bg, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(~gt_union, gt_union), cmap="Reds", alpha=0.45)
    axes[3].imshow(np.ma.masked_where(~pred_union, pred_union), cmap="Blues", alpha=0.45)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    axes[4].imshow(bg, cmap="gray", vmin=0, vmax=1)
    error_mask = np.repeat((~np.any(error_rgb > 0, axis=-1))[..., None], 3, axis=-1)
    axes[4].imshow(np.ma.array(error_rgb, mask=error_mask), alpha=0.75)
    axes[4].set_title("Error Map\nFP red / FN blue")
    axes[4].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []

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


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[~np.isnan(values)]


def summarize_values(values):
    values = finite_values(values)
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


@torch.no_grad()
def evaluate_model_on_slab_loader_3d(
    model,
    loader,
    device,
    region_names,
    threshold=0.5,
    dice_weight=1.0,
    bce_weight=1.0,
):
    model.eval()
    rows = []
    dataset_offset = 0

    pbar = tqdm(loader, desc="Test 3D", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        dice_values = dice_per_sample_channel_3d(
            logits=logits,
            targets=labels,
            threshold=threshold,
        ).detach().cpu().numpy()
        loss_values = sample_loss_values_3d(
            logits=logits,
            targets=labels,
            threshold=threshold,
            dice_weight=dice_weight,
            bce_weight=bce_weight,
        ).detach().cpu().numpy()

        batch_size = images.size(0)
        for i in range(batch_size):
            row = {
                "case_id": str(batch["case_id"][i]),
                "center_slice_idx": int(batch["center_slice_idx"][i].item()),
                "slice_indices": " ".join(str(x) for x in batch["slice_indices"][i]),
                "dataset_idx": dataset_offset + i,
                "loss": float(loss_values[i]),
                "dice": float(np.mean(dice_values[i])),
                "paths": "|".join(str(x) for x in batch["paths"][i]),
            }
            for channel, name in enumerate(region_names):
                row[f"dice_{str(name).lower()}"] = float(dice_values[i, channel])
            rows.append(row)

        dataset_offset += batch_size

    return rows


def build_test_summary_rows(metric_rows, region_names):
    summary = {}
    mean_dice, std_dice, min_dice, max_dice = summarize_values([r["dice"] for r in metric_rows])
    mean_loss, std_loss, min_loss, max_loss = summarize_values([r["loss"] for r in metric_rows])

    summary.update(
        {
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "min_dice": min_dice,
            "max_dice": max_dice,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "num_test_cases": len({row["case_id"] for row in metric_rows}),
            "num_test_slabs": len(metric_rows),
        }
    )

    for name in region_names:
        key = f"dice_{str(name).lower()}"
        region_mean, region_std, _, _ = summarize_values([r[key] for r in metric_rows])
        summary[f"mean_{key}"] = region_mean
        summary[f"std_{key}"] = region_std

    return [summary]


def sort_worst_slab_cases(metric_rows, top_k=5):
    return sorted(metric_rows, key=lambda row: (row["dice"], -row["loss"]))[:top_k]


def sanitize_filename(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def save_worst_slab_visualizations(model, dataset, worst_rows, device, output_dir, threshold=0.5):
    vis_dir = Path(output_dir) / "worst5_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(worst_rows, start=1):
        case_id = sanitize_filename(row["case_id"])
        save_path = vis_dir / (
            f"worst_{rank:02d}_case_{case_id}_center_slice_{row['center_slice_idx']}.png"
        )
        visualize_3d_slab_prediction(
            model=model,
            dataset=dataset,
            dataset_idx=int(row["dataset_idx"]),
            device=device,
            save_path=save_path,
            modality_index=3,
            threshold=threshold,
            show=False,
        )


def run_test_evaluation_3d(
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

    print("Loaded best model for 3D test evaluation:", best_model_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Best validation mean Dice:", checkpoint.get("best_mean_dice"))

    metric_rows = evaluate_model_on_slab_loader_3d(
        model=model,
        loader=test_loader,
        device=device,
        region_names=region_names,
        threshold=threshold,
    )

    if len(metric_rows) == 0:
        raise RuntimeError("No test slabs were evaluated.")

    summary_rows = build_test_summary_rows(metric_rows, region_names=region_names)

    metrics_path = output_dir / "test_metrics.csv"
    summary_path = output_dir / "test_metrics_summary.csv"
    write_csv_rows(metrics_path, metric_rows, fieldnames=list(metric_rows[0].keys()))
    write_csv_rows(summary_path, summary_rows, fieldnames=list(summary_rows[0].keys()))

    worst_rows = []
    if save_worst5:
        worst_rows = sort_worst_slab_cases(metric_rows, top_k=5)
        worst_path = output_dir / "worst5_cases.csv"
        write_csv_rows(worst_path, worst_rows, fieldnames=list(worst_rows[0].keys()))
        save_worst_slab_visualizations(
            model=model,
            dataset=test_dataset,
            worst_rows=worst_rows,
            device=device,
            output_dir=output_dir,
            threshold=threshold,
        )

    summary = summary_rows[0]
    print(f"Test slab Dice: {summary['mean_dice']:.4f} +/- {summary['std_dice']:.4f}")
    print(f"Test slab Loss: {summary['mean_loss']:.4f} +/- {summary['std_loss']:.4f}")
    print("Slab metrics:", metrics_path)
    print("Summary metrics:", summary_path)
    if save_worst5:
        print("Worst 5 CSV:", output_dir / "worst5_cases.csv")
        print("Worst 5 visualizations:", output_dir / "worst5_visualizations")

    return metric_rows, summary_rows, worst_rows
