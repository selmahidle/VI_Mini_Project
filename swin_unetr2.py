import os
from glob import glob
from sklearn.model_selection import train_test_split
from monai.transforms import (
    RandFlipd, RandRotate90d, ToTensord,
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, Resized
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import torch
import torch.optim as optim
import psutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

#attempted to get a higher dice, but does not work well

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# Define data directory
data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"

# Function to log memory usage
def log_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    print(f"{message} Memory Usage: {mem:.2f} GB")

log_memory_usage("Before data loading")

# Load preRT images and masks
preRT_images = sorted(glob(os.path.join(data_dir, '*', 'preRT', '*_T2.nii.gz')))
preRT_masks = sorted(glob(os.path.join(data_dir, '*', 'preRT', '*_mask.nii.gz')))

log_memory_usage("After data loading")

# Split into train, validation, and test sets
X_preRT_train, X_preRT_test_and_val, y_preRT_train, y_preRT_test_and_val = train_test_split(
    preRT_images, preRT_masks, test_size=0.2, random_state=42)
X_preRT_test, X_preRT_val, y_preRT_test, y_preRT_val = train_test_split(
    X_preRT_test_and_val, y_preRT_test_and_val, test_size=0.5, random_state=42)

# Create file dictionaries
preRT_train_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_train, y_preRT_train)]
preRT_val_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_val, y_preRT_val)]
preRT_test_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_test, y_preRT_test)]

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2), mode=["bilinear", "nearest"]),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64), mode=["trilinear", "nearest"]),
    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "mask"])
])

val_test_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2), mode=["bilinear", "nearest"]),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64), mode=["trilinear", "nearest"]),
    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "mask"])
])

# Create datasets and dataloaders
preRT_train_ds = Dataset(data=preRT_train_files, transform=train_transforms)
preRT_val_ds = Dataset(data=preRT_val_files, transform=val_test_transforms)
preRT_test_ds = Dataset(data=preRT_test_files, transform=val_test_transforms)

# Set batch size
batch_size = 2
preRT_train_loader = DataLoader(preRT_train_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)
preRT_val_loader = DataLoader(preRT_val_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)
preRT_test_loader = DataLoader(preRT_test_ds, batch_size=batch_size, collate_fn=pad_list_data_collate)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

# Define the Swin UNETR model
model = SwinUNETR(
    img_size=(128, 128, 64),
    in_channels=1,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True
).to(device)

sample_mask_path = preRT_masks[0]
sample_mask = nib.load(sample_mask_path).get_fdata()

# Print unique values
print("Unique values in raw mask:", np.unique(sample_mask))

# Define the loss function and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Define the metric
dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)

# TensorBoard writer
writer = SummaryWriter()

# Training and validation loop
epochs = 20
best_metric = -1
best_metric_epoch = -1

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in preRT_train_loader:
        step += 1
        inputs, masks = batch_data["image"].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if step % 10 == 0:
            print(f"Step {step}, Train Loss: {loss.item():.4f}")

    epoch_avg_loss = epoch_loss / step
    print(f"Epoch {epoch + 1} Average Loss: {epoch_avg_loss:.4f}")
    writer.add_scalar("Loss/train", epoch_avg_loss, epoch)
    scheduler.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for val_data in preRT_val_loader:
            val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
            val_outputs = model(val_inputs)
            val_outputs = torch.softmax(val_outputs, dim=1)

            val_masks_onehot = one_hot(val_masks, num_classes=3)
            val_outputs_foreground = val_outputs[:, 1:, ...]
            val_masks_onehot_foreground = val_masks_onehot[:, 1:, ...]

            dice_metric(y_pred=val_outputs_foreground, y=val_masks_onehot_foreground)

        dice_scores = dice_metric.aggregate()
        dice_metric.reset()
        if isinstance(dice_scores, tuple):
            dice_scores = dice_scores[0]

        valid_scores = dice_scores[~torch.isnan(dice_scores)]
        mean_dice = valid_scores.mean().item() if valid_scores.numel() > 0 else float('nan')
        print(f"Validation Mean Dice: {mean_dice:.4f}")
        writer.add_scalar("Dice/validation", mean_dice, epoch)

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

# Visualize predictions
sample_data = next(iter(preRT_val_loader))
inputs, masks = sample_data["image"].to(device), sample_data["mask"].to(device)
outputs = model(inputs)
predicted_masks = torch.argmax(outputs, dim=1).cpu()

plt.figure(figsize=(12, 6))
for i in range(min(4, batch_size)):
    plt.subplot(2, 4, i + 1)
    plt.title("Input Image")
    plt.imshow(inputs[i, 0].cpu(), cmap="gray")

    plt.subplot(2, 4, i + 5)
    plt.title("Predicted Mask")
    plt.imshow(predicted_masks[i], cmap="jet")
plt.show()
