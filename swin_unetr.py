import os
from glob import glob
from sklearn.model_selection import train_test_split
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

from preprocessing import get_dataloaders

from monai.transforms import Resize
from torch.utils.data import DataLoader

resize = Resize(spatial_size=(256, 256, 64))


class ResizeDatasetWrapper:
    def __init__(self, dataloader, resize_transform):
        self.dataloader = dataloader
        self.resize_transform = resize_transform

    def __iter__(self):
        for batch in self.dataloader:
            batch["image"] = self.resize_transform(batch["image"])
            batch["mask"] = self.resize_transform(batch["mask"])
            yield batch

    def __len__(self):
        return len(self.dataloader)


train_loader, val_loader, test_loader = get_dataloaders(time="mid", batch_size=1)

train_loader = ResizeDatasetWrapper(train_loader, resize)
val_loader = ResizeDatasetWrapper(val_loader, resize)
test_loader = ResizeDatasetWrapper(test_loader, resize)

train_loader, val_loader, test_loader = get_dataloaders(time="mid", batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

model = SwinUNETR(
    img_size=(256, 256, 64),
    in_channels=1,
    out_channels=3,
    feature_size=48
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)

epochs = 2
best_metric = -1
best_metric_epoch = -1

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, masks = batch_data["image"].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_avg_loss = epoch_loss / step
    print(f"Epoch {epoch + 1} Average Loss: {epoch_avg_loss:.4f}")
    scheduler.step()

    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for val_data in val_loader:
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

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_swin_model.pth")
            print("Saved Best Model")
print(f"Training Complete. Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")

# Testing the model
print("Testing the model...")
model.load_state_dict(torch.load("best_swin_model.pth"))
model.eval()
with torch.no_grad():
    test_dice = []
    for test_data in test_loader:
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        test_outputs = model(test_inputs)
        test_outputs = torch.softmax(test_outputs, dim=1)
        test_masks_onehot = one_hot(test_masks, num_classes=3)
        dice_metric(y_pred=test_outputs, y=test_masks_onehot)

    test_scores = dice_metric.aggregate()
    dice_metric.reset()
    mean_test_dice = test_scores.mean().item()
    print(f"Test Mean Dice: {mean_test_dice:.4f}")
