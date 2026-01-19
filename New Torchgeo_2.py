"""
TorchGeo-first minimal script.

Training (TorchGeo official pattern):
  - LoveDADataModule(...)
  - SemanticSegmentationTask(...)
  - Trainer.fit(task, datamodule)

Inference:
  - Trainer.predict(task, dataloaders=...)
  - EXTRA feature kept: force pure white pixels (RGB=255,255,255) to background (class 0)
"""

from __future__ import annotations
import os
import csv
from pathlib import Path
import numpy as np
import torch
import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import LoveDADataModule
from torchgeo.datasets import stack_samples
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models.unet import Unet_Weights


# ------------------- Config
ROOT_DIR = r"D:\LoveDA"
SCENE = ["urban"]          # Only train on Urban
# Use very small batch size and workers on CPU to reduce memory usage
BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_EPOCHS = 50
DEFAULT_ROOT_DIR = "checkpoints"

# Toggle training; set to True to fine-tune on D:\LoveDA.
# For now we only use TorchGeo's pretrained U-Net weights (no local training) to avoid CPU OOM.
RUN_TRAIN = False

# Inference on your own PNGs (set to the provided folder)
PREDICT_IMAGE_DIR = r"C:\Users\Linlin\Desktop\CV 11.22\Project\更新\3.边界裁剪\边界裁剪图片_已处理"
PREDICT_OUT_DIR = "segmentation_results"
CKPT_PATH = os.path.join(DEFAULT_ROOT_DIR, "last.ckpt")  # Adjust if your ckpt name differs

# Statistics output
STATS_CSV_PATH = os.path.join(PREDICT_OUT_DIR, "segmentation_statistics.csv")
EXCLUDE_BACKGROUND_IN_RATIO = True  # background (class 0) not counted in the denominator

CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}
# Color palette for visualization (H,W,3)
PALETTE = np.array(
    [
        [255, 255, 255],   # 0 background - white
        [220, 20, 60],     # 1 building   - red
        [255, 255, 0],     # 2 road       - yellow
        [0, 0, 255],       # 3 water      - blue
        [210, 180, 140],   # 4 barren     - tan
        [0, 128, 0],       # 5 forest     - green
        [144, 238, 144],   # 6 agriculture- light green
    ],
    dtype=np.uint8,
)


# ------------------- Inference dataset (minimal)
class PngFolderDataset(Dataset):
    def __init__(self, image_dir: str) -> None:
        self.image_paths = sorted(
            p for p in Path(image_dir).glob("*.png") if p.is_file()
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found in: {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img)  # (H, W, 3) uint8
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3, H, W) float
        return {"image": x, "path": str(path)}


def apply_white_to_background(mask: np.ndarray, rgb01_chw: torch.Tensor) -> np.ndarray:
    """Set pure white pixels in the INPUT RGB image to background (class 0) in the mask."""
    rgb = (rgb01_chw.detach().cpu().numpy() * 255.0).round().astype(np.uint8)  # (3,H,W)
    rgb_hwc = np.transpose(rgb[:3], (1, 2, 0))  # (H,W,3)
    is_white = np.all(rgb_hwc == 255, axis=2)
    mask = mask.copy()
    mask[is_white] = 0
    return mask


def main() -> None:
    # ------------------- Train (optional)
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        # No pretrained weights here to avoid version/state_dict mismatch issues
        weights=None,
        in_channels=3,
        num_classes=7,
        loss="ce",
        ignore_index=0,      # background
    )

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        default_root_dir=DEFAULT_ROOT_DIR,
    )

    if RUN_TRAIN:
        datamodule = LoveDADataModule(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            root=ROOT_DIR,
            scene=SCENE,
        )
        trainer.fit(model=task, datamodule=datamodule)

    # ------------------- Predict on your PNG folder
    os.makedirs(PREDICT_OUT_DIR, exist_ok=True)

    predict_ds = PngFolderDataset(PREDICT_IMAGE_DIR)
    predict_loader = DataLoader(
        predict_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=stack_samples,
    )

    # Load checkpoint; if missing and RUN_TRAIN=True, fall back to in-memory task
    if os.path.exists(CKPT_PATH):
        predict_task = SemanticSegmentationTask.load_from_checkpoint(CKPT_PATH)
    else:
        if not RUN_TRAIN:
            print(
                f"⚠ Checkpoint not found: {CKPT_PATH}. "
                "Using randomly initialized weights for inference. "
                "Set CKPT_PATH to a trained checkpoint or enable RUN_TRAIN=True to train first."
            )
        predict_task = task
    predict_task.eval()

    idx_global = 0
    stats_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in predict_loader:
            x = batch["image"]  # (B,3,H,W) float in [0,1]
            # TorchGeo SemanticSegmentationTask forward expects tensors, not dicts.
            # Use the image tensor as input.
            out = predict_task(x)
            # During inference, forward usually returns logits tensor (B,C,H,W).
            logits = out if isinstance(out, torch.Tensor) else out["mask"]
            # First detect white pixels in the INPUT; white pixels are forced to background (class 0)
            # so they do not participate in recognition.
            rgb_uint8 = (x.detach().cpu().numpy() * 255.0).round().astype(np.uint8)  # (B,3,H,W)
            white_masks = np.all(np.transpose(rgb_uint8[:, :3], (0, 2, 3, 1)) == 255, axis=3)  # (B,H,W)

            # Argmax prediction
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            # Force white pixels to background before any other postprocess
            pred[white_masks] = 0
            paths = batch["path"]

            for b in range(pred.shape[0]):
                mask = pred[b]
                stem = Path(paths[b]).stem
                # Save colorized visualization (no raw mask output)
                color_mask = PALETTE[mask]
                out_path = os.path.join(PREDICT_OUT_DIR, f"{stem}_vis.png")
                Image.fromarray(color_mask).save(out_path)
                idx_global += 1

                # ---- Per-image class ratio statistics
                counts = np.bincount(mask.flatten(), minlength=7).astype(np.int64)
                if EXCLUDE_BACKGROUND_IN_RATIO:
                    denom = int(counts[1:].sum())
                else:
                    denom = int(counts.sum())

                row: dict[str, object] = {
                    "image": stem,
                    "mask_path": out_path,
                    "denominator_pixels": denom,
                }
                for cls_id, cls_name in CLASS_NAMES.items():
                    cls_count = int(counts[cls_id])
                    if denom > 0:
                        if EXCLUDE_BACKGROUND_IN_RATIO and cls_id == 0:
                            ratio = 0.0
                        else:
                            ratio = cls_count / denom
                    else:
                        ratio = 0.0
                    row[f"{cls_name}_ratio"] = ratio
                    row[f"{cls_name}_pixels"] = cls_count

                stats_rows.append(row)

    print(f"Saved {idx_global} masks to: {PREDICT_OUT_DIR}")

    # Write CSV statistics
    if stats_rows:
        # stable column order
        fieldnames: list[str] = ["image", "mask_path", "denominator_pixels"]
        for cls_id in range(7):
            name = CLASS_NAMES[cls_id]
            fieldnames.append(f"{name}_ratio")
        for cls_id in range(7):
            name = CLASS_NAMES[cls_id]
            fieldnames.append(f"{name}_pixels")

        with open(STATS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)

        print(f"Saved per-image statistics CSV to: {STATS_CSV_PATH}")


if __name__ == "__main__":
    main()