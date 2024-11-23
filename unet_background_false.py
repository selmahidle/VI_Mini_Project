"""
U-Net trained for head and neck tumor segmentation on the data from the HNTSMRG Challenge
"""


from preprocessing import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
import matplotlib.pyplot as plt

train_loader, val_loader, test_loader = get_dataloaders(time="mid", batch_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Define the model, loss function and optimizer
"""
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3
).to(device)


max_epochs = 2
best_metric = -1
best_metric_epoch = -1
val_interval = 1

loss_function = DiceLoss(include_background=False,to_onehot_y=True, softmax=True) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
dice_metric = DiceMetric(include_background=False, reduction="none")
dice_metric_batch = DiceMetric(include_background=False, reduction="none")
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

epoch_loss_values = []
metric_values = []
metric_values_class1 = []  # For GTVp
metric_values_class2 = []  # For GTVn


"""
Training and validation loop
"""

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
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

    lr_scheduler.step(loss)
    epoch_loss_values.append(epoch_loss / step)
    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / step:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
                val_outputs = model(val_inputs)
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_outputs, y=val_masks)
                dice_metric_batch(y_pred=val_outputs, y=val_masks)

            mean_dice = dice_metric.aggregate().mean().item()
            metric_batch = dice_metric_batch.aggregate()

            metric_class1 = metric_batch[1].item()  # GTVp
            metric_class2 = metric_batch[2].item()  # GTVn

            metric_values.append(mean_dice)
            metric_values_class1.append(metric_class1)
            metric_values_class2.append(metric_class2)

            dice_metric.reset()
            dice_metric_batch.reset()

            print(
                f"Validation Metrics - Mean Dice: {mean_dice:.4f}, "
                f"GTVp Dice: {metric_class1:.4f}, GTVn Dice: {metric_class2:.4f}"
            )

            if mean_dice > best_metric:
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_unet_wo_background_model.pth")
                print("Saved Best Model")

print(f"Training Complete. Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")


"""
Testing the model
"""
dice_metric = DiceMetric(include_background=True, reduction="none")
dice_metric_batch = DiceMetric(include_background=True, reduction="none")
metric_values = []
metric_values_class0 = []  # For background
metric_values_class1 = []  # For GTVp
metric_values_class2 = []  # For GTVn

print("Testing the model...")
model.load_state_dict(torch.load("best_unet_wo_background_model.pth"))
model.eval()
with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        test_outputs = model(test_inputs)
        test_outputs = torch.argmax(test_outputs, dim=1, keepdim=True)
        dice_metric(y_pred=test_outputs, y=test_masks)
        dice_metric_batch(y_pred=test_outputs, y=test_masks)

    mean_test_dice = dice_metric.aggregate().mean().item()
    metric_batch = dice_metric_batch.aggregate()
    test_metric_class0 = metric_batch[0].item()
    test_metric_class1 = metric_batch[1].item()
    test_metric_class2 = metric_batch[2].item()
    mean_test_dice = mean_metric = (test_metric_class0 + test_metric_class1 + test_metric_class2) / 3
    print(
        f"Test Metrics - Mean Dice: {mean_test_dice:.4f}, "
        f"Background Dice: {test_metric_class0:.4f}, GTVp Dice: {test_metric_class1:.4f}, GTVn Dice: {test_metric_class2:.4f}"
    )
    dice_metric.reset()
    dice_metric_batch.reset()


"""
Plot the loss and dice metric
"""

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, y, color="red", label="Train Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Validation Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Epoch")
plt.ylabel("Mean Dice")
plt.plot(x, y, color="green", label="Val Dice")
plt.legend()
plt.show()
plt.savefig('unet_results_1.png')
print("Plot saved as unet_wo_background_results_1.png")

plt.figure("class_metrics", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Val Mean Dice - Class GTVp (primary tumor)")
x = [val_interval * (i + 1) for i in range(len(metric_values_class1))]
y = metric_values_class1
plt.xlabel("Epoch")
plt.ylabel("Mean Dice")
plt.plot(x, y, color="blue", label="GTVp Dice")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Val Mean Dice - Class GTVn (lymph nodes)")
x = [val_interval * (i + 1) for i in range(len(metric_values_class2))]
y = metric_values_class2
plt.xlabel("Epoch")
plt.ylabel("Mean Dice")
plt.plot(x, y, color="brown", label="GTVn Dice")
plt.legend()

plt.show()
plt.savefig('unet_wo_background_results_2.png')
print("Plot saved as unet_results_2.png")