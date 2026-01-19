import matplotlib
matplotlib.use('Agg')

from torchgeo.datamodules import LoveDADataModule
import torch
import lightning as L
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datasets import LoveDA

if __name__ == '__main__':
    root_dir = "./datasets/LoveDA"

    print("\n" + "="*60)
    print("Downloading LoveDA dataset (if not already present)...")
    print("="*60)

    print("\nDownloading training dataset...")
    train_dataset_download = LoveDA(root=root_dir, split="train", scene=["urban"], download=True)
    print(f"✓ Training dataset ready: {len(train_dataset_download)} samples")
    print(f"  Sample image shape: {train_dataset_download[0]['image'].shape}")
    print(f"  Sample mask shape: {train_dataset_download[0]['mask'].shape}")

    print("\nDownloading validation dataset...")
    val_dataset_download = LoveDA(root=root_dir, split="val", scene=["urban"], download=True)
    print(f"✓ Validation dataset ready: {len(val_dataset_download)} samples")

    print("\n" + "="*60)
    print("Loading LoveDA dataset using torchgeo LoveDADataModule...")
    print("="*60)

    datamodule = LoveDADataModule(
        root=root_dir,
        scene=["urban"],
        batch_size=4,
        num_workers=4,
        download=False,
    )

    print("\nSetting up datamodule for training stage...")
    datamodule.setup(stage="fit")

    print("✓ LoveDADataModule setup completed successfully")

    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    
    sample = train_dataset[0]
    if 'mask' in sample:
        mask = sample['mask']
        print(f"✓ Mask shape: {mask.shape}")
        print(f"✓ Mask values: {mask}")
        if isinstance(mask, torch.Tensor):
            unique_vals = torch.unique(mask)
            print(f"✓ Mask unique values: {unique_vals.tolist()}")
            print(f"✓ Mask value range: [{mask.min().item()}, {mask.max().item()}]")

    task = SemanticSegmentationTask(
        model="unetplusplus",
        weights=True,
        num_classes=7,
        lr=0.001
    )

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir="checkpoints/",
        # num_sanity_val_steps=0
    )

    trainer.fit(model=task, datamodule=datamodule)
    trainer.save_checkpoint("final_model.ckpt")
