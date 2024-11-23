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
    RandShiftIntensityd,
    RandScaleIntensityd,
    Orientationd,
    ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader
import torch

train_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
test_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"

""" 
Define transformations
"""

train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(
        keys=["image", "mask"], 
        pixdim=(0.5, 0.5, 2.0), 
        mode=("bilinear", "nearest")
    ),
    ResizeWithPadOrCropd(
        keys=["image", "mask"], 
        spatial_size=(512, 512, 64)
    ), 
    RandFlipd(
        keys=["image", "mask"], 
        prob=0.5, 
        spatial_axis=[0, 1, 2]
    ), 
    RandScaleIntensityd(
        keys="image", 
        factors=0.1, 
        prob=0.5
    ),
    RandShiftIntensityd(
        keys="image", 
        offsets=0.1, 
        prob=0.5
    ),
    NormalizeIntensityd(
        keys="image", 
        nonzero=True, 
        channel_wise=True
    ),
    ToTensord(keys=["image", "mask"])
])

val_test_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(
        keys=["image", "mask"], 
        pixdim=(0.5, 0.5, 2.0), 
        mode=("bilinear", "nearest")
    ),
    ResizeWithPadOrCropd(
        keys=["image", "mask"], 
        spatial_size=(512, 512, 64)
    ), 
    NormalizeIntensityd(
        keys="image", 
        nonzero=True, 
        channel_wise=True
    ),
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

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_test_transforms)
    test_ds = Dataset(data=test_files, transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
