# Set matplotlib backend to non-interactive to avoid warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from torchgeo.datamodules import LoveDADataModule
import torch
import lightning as L  
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datasets import LoveDA
from collections import Counter
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.datasets import RasterDataset
import rasterio
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from PIL import Image
import glob

# Load LoveDA dataset using torchgeo API
root_dir = "./datasets/LoveDA"

print("\n" + "="*60)
print("Downloading LoveDA dataset (if not already present)...")
print("="*60)

# Download the dataset using LoveDA class
# This will download the data if it doesn't exist
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

# Initialize LoveDADataModule with proper parameters
# download=False since we already downloaded above
datamodule = LoveDADataModule(
    root=root_dir,
    scene=["urban"],  # Only Urban scene as per requirement
    batch_size=16,
    num_workers=4,
    download=False,  # Data already downloaded above
)

# Setup the datamodule (this will load the datasets)
# Stage "fit" is used for training and validation datasets
print("\nSetting up datamodule for training stage...")
datamodule.setup(stage="fit")

print("✓ LoveDADataModule setup completed successfully")

# Access the train dataset for class weight calculation
train_dataset = datamodule.train_dataset
val_dataset = datamodule.val_dataset

print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Val dataset: {len(val_dataset)} samples")

# ------------------- 1. Calculate class distribution and weights
# Use torchgeo's dataloader to properly sample from the dataset
print("\n" + "="*60)
print("Calculating class distribution from training dataset...")
print("="*60)

class_counts = Counter()

# Get a sample of training data using the datamodule's train dataloader
# This ensures we're using the same sampling strategy as training
train_dataloader = datamodule.train_dataloader()
sample_count = 0
max_samples = 1000  # Limit samples for faster computation

for batch in train_dataloader:
    # torchgeo batches contain 'image' and 'mask' keys
    masks = batch['mask']  # Shape: [batch_size, H, W]
    
    for mask in masks:
        mask_flat = mask.flatten()
        # Convert to list for Counter (handles both tensor and numpy)
        if isinstance(mask_flat, torch.Tensor):
            mask_list = mask_flat.cpu().numpy().tolist()
        else:
            mask_list = mask_flat.tolist()
        class_counts.update(mask_list)
        sample_count += 1
        
        if sample_count >= max_samples:
            break
    
    if sample_count >= max_samples:
        break

print(f"Processed {sample_count} samples for class distribution")

# Calculate class weights (num_classes=7, need 7 weight values for classes 0-6)
# LoveDA dataset has 7 classes: 0-6
# Exclude agriculture (class 6) from training and recognition
total_pixels = sum([class_counts[i] for i in range(1, 6)])  # Count pixels for classes 1-5 (excluding agriculture)
num_valid_classes = 5  # Classes 1-5 are valid classes (excluding agriculture)

weights_list = []
for i in range(7):  # Classes 0-6, total 7 classes
    if i == 0 or i == 6:
        # Class 0 (background) and class 6 (agriculture) will be ignored, weight set to 1.0
        weights_list.append(1.0)
    elif class_counts[i] > 0:
        # Classes 1-5: calculate inverse frequency weights (excluding agriculture)
        weights_list.append(total_pixels / (class_counts[i] * num_valid_classes))
    else:
        weights_list.append(1.0)  # If a class has no samples, weight set to 1.0

weights = torch.tensor(weights_list)
print(f"Class weights (classes 0-6, classes 0 and 6 (agriculture) will be ignored): {weights}")

# ------------------- 2. Create Semantic Segmentation Task
task = SemanticSegmentationTask(
    segmentation_model="unet",
    encoder_weights="imagenet",
    learning_rate=0.001
)

# ------------------- 3. Create Trainer
trainer = L.Trainer(
    accelerator="cpu",
    devices=1,
    default_root_dir="checkpoints/"
)

# ------------------- 4. Train the model
trainer.fit(model=task, datamodule=datamodule)


# Custom dataset class for inference (if needed)
# For LoveDA inference, you can use the LoveDA dataset directly or create a custom one
class MySatelliteDataset(RasterDataset):
    filename_glob = "*.png"          # Image filename pattern
    is_image = True                  # Image only, no mask
    separate_files = False           # If multiple bands in single file
    all_bands = ['R', 'G', 'B']      # Adjust according to your bands (RGB example)
    rgb_bands = ['R', 'G', 'B']



# ==================== Inference Section (Run after training) ====================
# Note: The following code is for model inference and should be run after training

# Class names mapping (according to LoveDA dataset)
# LoveDA has 7 classes: 0-6
CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture"
}

# Configuration: input image directory and output paths
# Use the same root directory as training, or specify a different one
INPUT_IMAGE_DIR = root_dir  # Use the same root directory as training
OUTPUT_DIR = "segmentation_results"  # Output directory
MODEL_PATH = "checkpoints/last.ckpt"  # Model checkpoint path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model
print("Loading model...")
task = SemanticSegmentationTask.load_from_checkpoint(MODEL_PATH)
task.eval()
task.freeze()  # Freeze model for inference

# Create inference dataset using LoveDA dataset
# For inference, you can use the test split or create a custom dataset
print("Loading dataset...")

# Download test dataset if needed
print("Downloading test dataset (if not already present)...")
test_dataset_download = LoveDA(root=INPUT_IMAGE_DIR, split="test", scene=["urban"], download=True)
print(f"✓ Test dataset ready: {len(test_dataset_download)} samples")

# Option 1: Use LoveDA test dataset
inference_datamodule = LoveDADataModule(
    root=INPUT_IMAGE_DIR,
    scene=["urban"],
    batch_size=4,
    num_workers=4,
    download=False,  # Data already downloaded above
)
inference_datamodule.setup(stage="test")
test_dataset = inference_datamodule.test_dataset

# Option 2: Use custom dataset with GridGeoSampler (if you have custom images)
# dataset = MySatelliteDataset(root=INPUT_IMAGE_DIR)
# sampler = GridGeoSampler(dataset, size=512, stride=256)
# dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=stack_samples)

# Use test dataloader from datamodule
dataloader = inference_datamodule.test_dataloader()

# Store statistics results
results = []

print("Starting semantic segmentation prediction...")
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        # torchgeo batches contain 'image' and 'mask' keys
        # Get input images (torchgeo uses 'image' key)
        input_images = batch['image'] if isinstance(batch, dict) else batch[0]
        
        # Predict using torchgeo's task
        # The task expects a dict with 'image' key, or we can pass the batch directly
        pred_output = task(batch) if isinstance(batch, dict) else task({'image': input_images})
        
        # Extract prediction from output
        if isinstance(pred_output, dict):
            pred_output = pred_output.get('pred', pred_output.get('mask', pred_output.get('prediction')))
        elif isinstance(pred_output, torch.Tensor):
            pass  # Already a tensor
        else:
            pred_output = pred_output
        
        # Get prediction mask
        if isinstance(pred_output, torch.Tensor):
            pred_mask = torch.argmax(pred_output, dim=1).cpu().numpy()
        else:
            pred_mask = np.argmax(pred_output, axis=1)
            
        # Process each sample in the batch
        for b in range(pred_mask.shape[0]):
            mask = pred_mask[b].copy()
            
            # Get RGB image and convert to 0-255 range
            if isinstance(input_images, torch.Tensor):
                rgb = input_images[b].cpu().numpy()
            else:
                rgb = input_images[b]
            
            # Normalize if needed (torchgeo typically normalizes images)
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            
            # Detect pure white pixels (RGB all 255) and set as background
            if rgb.shape[0] >= 3:
                rgb_hwc = np.transpose(rgb[:3], (1, 2, 0))
                is_white = np.all(rgb_hwc == 255, axis=2)
                mask[is_white] = 0
            
            # Set agriculture (class 6) to background (class 0) - exclude from recognition
            mask[mask == 6] = 0
        
            # Try to get original filename from dataset
            # For LoveDA dataset, we can extract info from the dataset
            park_name = f"image_{batch_idx}_{b}"  # Default name
            
            # If using LoveDA dataset, try to get scene/image info
            if hasattr(test_dataset, 'files') and len(test_dataset.files) > 0:
                try:
                    # Get file index corresponding to current sample
                    file_idx = batch_idx * dataloader.batch_size + b
                    if file_idx < len(test_dataset.files):
                        file_path = test_dataset.files[file_idx]
                        filename = os.path.basename(file_path)
                        # Extract name from filename
                        name_without_ext = os.path.splitext(filename)[0]
                        park_name = re.sub(r'[_\-].*$', '', name_without_ext)
                        if not park_name or len(park_name) < 2:
                            park_name = name_without_ext
                except:
                    pass
            
            # Generate output filename
            output_filename = f"{park_name}_segmentation_{batch_idx}_{b}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save segmented image
            Image.fromarray(mask.astype(np.uint8)).save(output_path)
            
            # Count pixels for each class (excluding background class 0 and agriculture class 6)
            unique_classes, counts = np.unique(mask, return_counts=True)
            class_counts_dict = dict(zip(unique_classes, counts))
            
            # Calculate total pixels (excluding background class 0 and agriculture class 6)
            total_valid = sum([count for cls, count in class_counts_dict.items() if cls != 0 and cls != 6])
            
            # Calculate class percentages (excluding background class 0 and agriculture class 6)
            result_row = {"Park Name": park_name, "Image File": output_filename}
            
            if total_valid > 0:
                for cls_id in range(1, 6):  # Classes 1-5 (excluding agriculture class 6)
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    pixel_count = class_counts_dict.get(cls_id, 0)
                    percentage = (pixel_count / total_valid) * 100
                    result_row[cls_name] = f"{percentage:.2f}%"
            else:
                # If no valid classes detected
                for cls_id in range(1, 6):  # Classes 1-5 (excluding agriculture)
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    result_row[cls_name] = "0.00%"
            
            results.append(result_row)
            
            if (batch_idx * dataloader.batch_size + b + 1) % 10 == 0:
                print(f"  Processed {batch_idx * dataloader.batch_size + b + 1} images...")

# Generate CSV statistics table
print("\nGenerating CSV statistics table...")
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "segmentation_statistics.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"\nSegmentation completed!")
print(f"Segmented images saved to: {OUTPUT_DIR}")
print(f"Statistics table saved to: {csv_path}")
print(f"Total processed: {len(results)} images")