import os
from glob import glob
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, Resized
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import torch
import torch.optim as optim

# Define data directory
data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"

# Load preRT images and masks
preRT_images = sorted(glob(os.path.join(data_dir, '*', 'preRT', '*_T2.nii.gz')))
preRT_masks = sorted(glob(os.path.join(data_dir, '*', 'preRT', '*_mask.nii.gz')))

# Split into train, validation, and test sets
X_preRT_train, X_preRT_test_and_val, y_preRT_train, y_preRT_test_and_val = train_test_split(preRT_images, preRT_masks, test_size=0.2, random_state=42)
X_preRT_test, X_preRT_val, y_preRT_test, y_preRT_val = train_test_split(X_preRT_test_and_val, y_preRT_test_and_val, test_size=0.5, random_state=42)

# Create file dictionaries
preRT_train_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_train, y_preRT_train)]
preRT_val_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_val, y_preRT_val)]
preRT_test_files = [{"image": img, "mask": msk} for img, msk in zip(X_preRT_test, y_preRT_test)]

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2)),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64)),
    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "mask"])
])

val_test_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64)),
    ToTensord(keys=["image", "mask"])
])

# Create datasets and dataloaders
preRT_train_ds = Dataset(data=preRT_train_files, transform=train_transforms)
preRT_val_ds = Dataset(data=preRT_val_files, transform=val_test_transforms)
preRT_test_ds = Dataset(data=preRT_test_files, transform=val_test_transforms)

preRT_train_loader = DataLoader(preRT_train_ds, batch_size=8, collate_fn=pad_list_data_collate)
preRT_val_loader = DataLoader(preRT_val_ds, batch_size=8, collate_fn=pad_list_data_collate)
preRT_test_loader = DataLoader(preRT_test_ds, batch_size=8, collate_fn=pad_list_data_collate)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Swin UNETR model
model = SwinUNETR(
    img_size=(128, 128, 64),  # Input image dimensions
    in_channels=1,  # Input channels (grayscale images)
    out_channels=3,  # Number of segmentation classes
    feature_size=48,  # Size of the initial feature maps
    use_checkpoint=True  # Enable gradient checkpointing to save memory
).to(device)

# Define the loss function and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the metric
dice_metric = DiceMetric(include_background=True, reduction="mean")

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
        outputs = model(inputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if step % 10 == 0:
            print(f"Step {step}, Train Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / step:.4f}")

    model.eval()
    with torch.no_grad():
        val_dice = []
        for val_data in preRT_val_loader:
            val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
            val_outputs = model(val_inputs)
            val_outputs = torch.softmax(val_outputs, dim=1)
            val_masks_onehot = one_hot(val_masks, num_classes=3)
            dice_metric(y_pred=val_outputs, y=val_masks_onehot)

        dice_scores = dice_metric.aggregate()
        dice_metric.reset()
        mean_dice = dice_scores.mean().item()
        print(f"Validation Mean Dice: {mean_dice:.4f}")

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

print(f"Training Complete. Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")

# Testing the model
print("Testing the model...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    test_dice = []
    for test_data in preRT_test_loader:
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        test_outputs = model(test_inputs)
        test_outputs = torch.softmax(test_outputs, dim=1)
        test_masks_onehot = one_hot(test_masks, num_classes=3)
        dice_metric(y_pred=test_outputs, y=test_masks_onehot)

    test_scores = dice_metric.aggregate()
    dice_metric.reset()
    mean_test_dice = test_scores.mean().item()
    print(f"Test Mean Dice: {mean_test_dice:.4f}")
