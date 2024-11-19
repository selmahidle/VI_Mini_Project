"""
Preprocessing for 3D Images for the Visual Intelligence Mini-Project

Based on theese tutorials: https://www.youtube.com/watch?v=83FLt4fPNGs and https://www.youtube.com/watch?v=hqgZuatm8eE
And https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb 
"""

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
    MapTransform,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    Orientationd,
    ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import nibabel as nib

train_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
test_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"


"""
Define a transform to convert the multi-classes labels into multi-labels segmentation task in One-Hot format
"""
class ConvertToMultiChannelHNTSd(MapTransform):
    """
    Convert HNTS-MRG labels to multi-channel format:
    - Channel 0: Background (label 0)
    - Channel 1: GTVp (label 1, primary tumor)
    - Channel 2: GTVn (label 2, lymph nodes)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 0)  # Channel 0: Background
            result.append(d[key] == 1)  # Channel 1: GTVp
            result.append(d[key] == 2)  # Channel 2: GTVn
            d[key] = torch.stack(result, axis=0).float()
        return d


"""
Define tranformations
"""
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image"]),
    EnsureTyped(keys=["image", "mask"]),
    ConvertToMultiChannelHNTSd(keys=["mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")),  # keep original sampling
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, 64)), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2), 
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), 
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),  
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ToTensord(keys=["image", "mask"])
])

val_test_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image"]),
    EnsureTyped(keys=["image", "mask"]),
    ConvertToMultiChannelHNTSd(keys=["mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")), 
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, 64)), 
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
    ToTensord(keys=["image", "mask"])
])


"""
Get the appropriate dataloaders based on the specified time.
The time param can only be either "mid" or "pre".
"""

def get_dataloaders(time, val_size=0.1, batch_size=4):

    if time not in {"mid", "pre"}:
        raise ValueError("Invalid time. Expected 'mid' or 'pre'.")

    foldername = str(time) + "RT"
    train_images = sorted(glob(os.path.join(train_dir, "*", foldername, "*_T2.nii.gz")))
    train_masks = sorted(glob(os.path.join(train_dir, "*", foldername, "*_mask.nii.gz")))

    test_images = sorted(glob(os.path.join(test_dir, "*", foldername, "*_T2.nii.gz")))
    test_masks = sorted(glob(os.path.join(test_dir, "*", foldername, "*_mask.nii.gz")))

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=val_size, random_state=42)

    train_files = [{"image": image_filename, "mask": mask_filename} for image_filename, mask_filename in zip(X_train, y_train)]
    val_files = [{"image": image_filename, "mask": mask_filename} for image_filename, mask_filename in zip(X_val, y_val)]
    test_files = [{"image": image_filename, "mask": mask_filename} for image_filename, mask_filename in zip(test_images, test_masks)]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_test_transforms)
    test_ds = Dataset(data=test_files, transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)

    return train_loader, val_loader, test_loader


"""
Plot one slice of the first patient
Can be useful for understanding but not needed for preprocessing


test_patient = first(preRT_train_loader)
plt.figure("Test", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Slice 40 of a patient")
plt.imshow(test_patient["image"][0, 0, :, :, 40], cmap="grey")

plt.subplot(1, 2, 2)
plt.title("Mask of slice 40")
plt.imshow(test_patient["mask"][0, 0, :, :, 40])
plt.show()
"""
