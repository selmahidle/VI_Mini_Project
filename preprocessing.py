"""
Preprocessing for 3D Images for the Visual Intelligence Mini-Project

Based on theese tutorials: https://www.youtube.com/watch?v=83FLt4fPNGs and https://www.youtube.com/watch?v=hqgZuatm8eE
"""

import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized
from monai.data import Dataset, DataLoader
from monai.utils import first
import matplotlib.pyplot as plt


# TODO: there is also a test folder in IDUN

# For IDUN - set data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
data_dir = "/Users/selmahidle/Documents/skole/visuell_intelligens/mini_project/data"


"""
preRT image data loading
"""

preRT_images = sorted(glob(os.path.join(data_dir, '*', 'preRT', '*_T2.nii.gz')))
preRT_masks = sorted(glob(os.path.join(data_dir, "*", "preRT", "*_mask.nii.gz")))

# TODO: This is a 60/20/20 split because I only have 5 images downloaded, change to 80/10/10 when running on IDUN by setting the first test_size to 0.2 and not 0.4
X_preRT_train, X_preRT_test_and_val, y_preRT_train, y_preRT_test_and_val = train_test_split(preRT_images, preRT_masks, test_size=0.4, random_state=42)
X_preRT_test, X_preRT_val, y_preRT_test, y_preRT_val = train_test_split(X_preRT_test_and_val, y_preRT_test_and_val, test_size=0.5, random_state=42)

preRT_train_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(X_preRT_train, y_preRT_train)]
preRT_val_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(X_preRT_val, y_preRT_val)]
preRT_test_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(X_preRT_test, y_preRT_test)]


"""
midRT image data loading
"""

midRT_images = sorted(glob(os.path.join(data_dir, "*", "midRT", "*_T2.nii.gz")))
midRT_masks = sorted(glob(os.path.join(data_dir, "*", "midRT", "*_mask.nii.gz")))

# TODO: This is a 60/20/20 split because I only have 5 images downloaded, change to 80/10/10 when running on IDUN by setting the first test_size to 0.2 and not 0.4
X_midRT_train, X_midRT_test_and_val, y_midRT_train, y_midRT_test_and_val = train_test_split(midRT_images, midRT_masks, test_size=0.4, random_state=42)
X_midRT_test, X_midRT_val, y_midRT_test, y_midRT_val = train_test_split(X_midRT_test_and_val, y_midRT_test_and_val, test_size=0.5, random_state=42)

midRT_train_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_train, y_midRT_train)]
midRT_val_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_val, y_midRT_val)]
midRT_test_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_test, y_midRT_test)]


"""
Define tranformations
TODO: should val also have the same transformations as train, or should it be the same as test?
"""

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]), # AddChanneld
        Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2)), # usikker på om dette burde være med
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "mask"], source_key=["image"]),
        Resized(keys=["image", "mask"], spatial_size=[128, 128, 128]), # usikker på om 128 er riktig for våre bilder hvertfall høyde, bredde og antall slices
        ToTensord(keys=["image", "mask"])
    ]
)

val_test_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ToTensord(keys=["image", "mask"])
    ]
)


"""
Create datasets and dataloaders
"""

preRT_train_ds = Dataset(data=preRT_train_files, transform=train_transforms)
midRT_train_ds = Dataset(data=midRT_train_files, transform=train_transforms)
preRT_val_ds = Dataset(data=preRT_val_files, transform=val_test_transforms)
midRT_val_ds = Dataset(data=midRT_val_files, transform=val_test_transforms)
preRT_test_ds = Dataset(data=preRT_test_files, transform=val_test_transforms)
midRT_test_ds = Dataset(data=midRT_test_files, transform=val_test_transforms)

# TODO: maybe change the batch size
preRT_train_loader = DataLoader(preRT_train_ds, batch_size = 1)
midRT_train_loader = DataLoader(midRT_train_ds, batch_size = 1)
preRT_val_loader = DataLoader(preRT_val_ds, batch_size = 1)
midRT_val_loader = DataLoader(midRT_val_ds, batch_size = 1)
preRT_test_loader = DataLoader(preRT_test_ds, batch_size = 1)
midRT_test_loader = DataLoader(midRT_test_ds, batch_size = 1)

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


