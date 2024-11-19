# Import necessary components from preprocessing.py
from preprocessing import preRT_train_loader, preRT_val_loader, preRT_test_loader

import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

train_loader = preRT_train_loader
val_loader = preRT_val_loader
test_loader = preRT_test_loader

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,  # background and two tumor labels
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3
).to(device)

# Define the loss function and optimizer
loss_function = loss_function = DiceLoss(to_onehot_y=False, softmax=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Metrics
dice_metric = DiceMetric(include_background=True, reduction="mean")

# Training and validation loop
epochs = 20
best_metric = -1
best_metric_epoch = -1
val_interval = 1

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

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_dice = []
            for val_data in preRT_val_loader:
                val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
                val_outputs = model(val_inputs)
                val_outputs = torch.sigmoid(val_outputs)  # Sigmoid activation for binary segmentation
                dice = dice_metric(y_pred=val_outputs, y=val_masks)
                val_dice.append(dice.item())

            mean_dice = torch.tensor(val_dice).mean().item()
            print(f"Validation Dice: {mean_dice:.4f}")

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
    for test_data in test_loader:
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        test_outputs = model(test_inputs)
        test_outputs = torch.sigmoid(test_outputs)  # Sigmoid activation for binary segmentation
        dice = dice_metric(y_pred=test_outputs, y=test_masks)
        test_dice.append(dice.item())

    mean_test_dice = torch.tensor(test_dice).mean().item()
    print(f"Test Dice: {mean_test_dice:.4f}")