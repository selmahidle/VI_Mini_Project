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
preRT image data loading
"""

preRT_train_images = sorted(glob(os.path.join(train_dir, '*', 'preRT', '*_T2.nii.gz')))
preRT_train_masks = sorted(glob(os.path.join(train_dir, "*", "preRT", "*_mask.nii.gz")))

preRT_test_images = sorted(glob(os.path.join(test_dir, '*', 'preRT', '*_T2.nii.gz')))
preRT_test_masks = sorted(glob(os.path.join(test_dir, "*", "preRT", "*_mask.nii.gz")))

X_preRT_train, X_preRT_val, y_preRT_train, y_preRT_val = train_test_split(preRT_train_images, preRT_train_masks, test_size=0.1, random_state=42)

preRT_train_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(X_preRT_train, y_preRT_train)]
preRT_val_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(X_preRT_val, y_preRT_val)]
preRT_test_files = [{"image": preRT_image_filename, "mask": preRT_mask_filename} for preRT_image_filename, preRT_mask_filename in zip(preRT_test_images, preRT_test_masks)]


"""
midRT image data loading
TODO: kopier fra det over
"""

"""
midRT_images = sorted(glob(os.path.join(data_dir, "*", "midRT", "*_T2.nii.gz")))
midRT_masks = sorted(glob(os.path.join(data_dir, "*", "midRT", "*_mask.nii.gz")))

X_midRT_train, X_midRT_test_and_val, y_midRT_train, y_midRT_test_and_val = train_test_split(midRT_images, midRT_masks, test_size=0.4, random_state=42)
X_midRT_test, X_midRT_val, y_midRT_test, y_midRT_val = train_test_split(X_midRT_test_and_val, y_midRT_test_and_val, test_size=0.5, random_state=42)

midRT_train_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_train, y_midRT_train)]
midRT_val_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_val, y_midRT_val)]
midRT_test_files = [{"image": midRT_image_filename, "mask": midRT_mask_filename} for midRT_image_filename, midRT_mask_filename in zip(X_midRT_test, y_midRT_test)]
"""

"""
Define a new transform to convert labels
Here we convert the multi-classes labels into multi-labels segmentation task in One-Hot format.
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
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")), # TODO: maybe test pixdim=(1.0, 1.0, 1.0) to resample, pixdim=(0.5, 0.5, 2.0) keeps original spacing
    ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, 128)),
    RandSpatialCropd(keys=["image", "mask"], roi_size=[128, 128, 64], random_size=False),
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
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "mask"])
])



"""
Create datasets and dataloaders
"""

preRT_train_ds = Dataset(data=preRT_train_files, transform=train_transforms)
#midRT_train_ds = Dataset(data=midRT_train_files, transform=train_transforms)
preRT_val_ds = Dataset(data=preRT_val_files, transform=val_test_transforms)
#midRT_val_ds = Dataset(data=midRT_val_files, transform=val_test_transforms)
preRT_test_ds = Dataset(data=preRT_test_files, transform=val_test_transforms)
#midRT_test_ds = Dataset(data=midRT_test_files, transform=val_test_transforms)

preRT_train_loader = DataLoader(preRT_train_ds, batch_size=4, collate_fn=pad_list_data_collate)
preRT_val_loader = DataLoader(preRT_val_ds, batch_size=4, collate_fn=pad_list_data_collate)
preRT_test_loader = DataLoader(preRT_test_ds, batch_size=4, collate_fn=pad_list_data_collate)


#midRT_train_loader = DataLoader(midRT_train_ds, batch_size = 32)
#midRT_val_loader = DataLoader(midRT_val_ds, batch_size = 32)
#midRT_test_loader = DataLoader(midRT_test_ds, batch_size = 32)



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