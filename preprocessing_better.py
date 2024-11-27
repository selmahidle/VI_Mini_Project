import os
from glob import glob
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, 
    LoadImaged, 
    ToTensord, 
    EnsureTyped,
    EnsureChannelFirstd, 
    Spacingd, 
    NormalizeIntensityd, 
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Orientationd,
    ResizeWithPadOrCropd,
    MapTransform,
    RandSpatialCropd,
    LoadImage
)
from monai.data import CacheDataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm

train_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
test_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"

compute_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image")
])


""" 
Function to get info about the datasets intensity levels
"""
def compute_mean_std(image_paths):
    pixel_values = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        transformed = compute_transforms({"image": img_path})
        image = transformed["image"].numpy()  
        pixel_values.append(image.flatten())  
    
    pixel_values = np.concatenate(pixel_values, axis=0)
    
    mean = np.mean(pixel_values)
    std = np.std(pixel_values)
    return mean, std


train_images = sorted(glob(os.path.join(train_dir, "*", "midRT", "*_T2.nii.gz")))
test_images = sorted(glob(os.path.join(test_dir, "*", "midRT", "*_T2.nii.gz")))
all_images = train_images + test_images
dataset_mean, dataset_std = compute_mean_std(all_images)

train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")),  # keep original sampling
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, 64)), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2), 
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.3), 
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3), 
    NormalizeIntensityd(keys="image", subtrahend=dataset_mean, divisor=dataset_std, channel_wise=False),
    ToTensord(keys=["image", "mask"])
])

val_test_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")), 
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, 64)), 
    NormalizeIntensityd(keys="image", subtrahend=dataset_mean, divisor=dataset_std, channel_wise=False),
    ToTensord(keys=["image", "mask"])
])


""" 
Function to get dataloaders
"""

def get_dataloaders(time, val_size=0.1, batch_size=4):
    if time not in {"mid", "pre"}:
        raise ValueError("Invalid time. Expected 'mid' or 'pre'.")
    
    foldername = f"{time}RT"
    train_images = sorted(glob(os.path.join(train_dir, "*", foldername, "*_T2.nii.gz")))
    train_masks = sorted(glob(os.path.join(train_dir, "*", foldername, "*_mask.nii.gz")))
    test_images = sorted(glob(os.path.join(test_dir, "*", foldername, "*_T2.nii.gz")))
    test_masks = sorted(glob(os.path.join(test_dir, "*", foldername, "*_mask.nii.gz")))

    # ensure that images and masks are correctly paired
    assert len(train_images) == len(train_masks), "Mismatch in training images and masks"
    assert len(test_images) == len(test_masks), "Mismatch in test images and masks"

    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_masks, test_size=val_size, random_state=42
    )

    train_files = [{"image": img, "mask": msk} for img, msk in zip(X_train, y_train)]
    val_files = [{"image": img, "mask": msk} for img, msk in zip(X_val, y_val)]
    test_files = [{"image": img, "mask": msk} for img, msk in zip(test_images, test_masks)]

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=val_test_transforms, cache_rate=1.0)
    test_ds = CacheDataset(data=test_files, transform=val_test_transforms, cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders(time="mid", batch_size=4)

"""

def print_unique_values(loader, loader_name):

    Print unique values for masks in a DataLoader, along with their filenames.

    print(f"--- {loader_name} ---")
    count = 0
    for batch in loader:
        masks = batch["mask"]
        file_paths = batch["mask_meta_dict"]["filename_or_obj"]  # Get filenames from metadata
        for i in range(masks.size(0)):  # Loop through the batch
            unique_values = torch.unique(masks[i])  # Get unique values for each mask
            print(f"Filename: {file_paths[i]}")
            print(f"Unique values: {unique_values}")
            count += 1
            if count == 10:  # Stop after 10 masks
                return


def print_raw_mask_unique_values(mask_paths, loader_name):

    Print unique values for raw, untransformed masks using MONAI's LoadImage,
    along with their filenames.

    print(f"--- Raw {loader_name} Masks ---")
    load_mask = LoadImage(image_only=True)  # Load only the mask, no metadata
    for idx, mask_path in enumerate(mask_paths[:10]):  # Limit to first 10 masks
        mask = load_mask(mask_path)  # Load the raw mask
        unique_values = torch.unique(torch.tensor(mask)).tolist()  # Get unique values
        print(f"Filename: {mask_path}")
        print(f"Unique values: {sorted(unique_values)}")
    print()

# Fetch data loaders
train_loader, val_loader, test_loader = get_dataloaders(time="mid", batch_size=1)

train_masks = sorted(glob(os.path.join(train_dir, "*", "midRT", "*_mask.nii.gz")))
val_masks, test_masks = train_test_split(
    train_masks, test_size=0.1, random_state=42
)

# Print unique values for raw masks
print_raw_mask_unique_values(train_masks, "Train Raw")
print_raw_mask_unique_values(val_masks, "Validation Raw")
print_raw_mask_unique_values(test_masks, "Test Raw")

# Print unique values for masks in each loader
print_unique_values(train_loader, "Train Loader")
print_unique_values(val_loader, "Validation Loader")
print_unique_values(test_loader, "Test Loader")

"""
