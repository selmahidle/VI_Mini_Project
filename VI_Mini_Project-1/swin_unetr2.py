"""
U-Net trained for head and neck tumor segmentation on the data from the HNTSMRG Challenge
"""


from preprocessing import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
import matplotlib.pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

train_loader, val_loader, test_loader = get_dataloaders(time="pre", batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

"""
Define the model, loss function and optimizer
"""
model = SwinUNETR(
    img_size=(256, 256, 32),
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
   # channels=(16, 32, 64, 128, 256),
    #strides=(2, 2, 2, 2),
    #kernel_size=3,
    #up_kernel_size=3
).to(device)


max_epochs = 100
best_metric = -1
best_metric_epoch = -1
val_interval = 1

def compute_class_weights(loader):
    class_counts = torch.zeros(3)
    for batch in loader:
        masks = batch['mask']
        for c in range(3):
            class_counts[c] += (masks == c).sum().item()
    total_voxels = class_counts.sum()
    class_frequencies = class_counts / total_voxels
    class_weights = 1.0 / class_frequencies
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return class_weights

class_weights = compute_class_weights(train_loader).to(device)
print(f"Normalized Class Weights: {class_weights}")


loss_function = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True, weight=class_weights) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

epoch_loss_values = []
metric_values = []
metric_values_class0 = []
metric_values_class1 = []  # For GTVp
metric_values_class2 = []  # For GTVn


post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
post_mask = Compose([AsDiscrete(to_onehot=3)])

def check_class_presence(loader, dataset_name):
    class_presence = set()
    for batch in loader:
        masks = batch['mask']
        unique_labels = torch.unique(masks)
        class_presence.update(unique_labels.cpu().numpy())
    print(f"Unique labels in {dataset_name} dataset: {sorted(class_presence)}")

# Assuming you have train_loader, val_loader, and test_loader
check_class_presence(train_loader, "training")
check_class_presence(val_loader, "validation")
check_class_presence(test_loader, "testing")



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
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_masks = [post_mask(i) for i in decollate_batch(val_masks)]


                print(f"Val Outputs Shape: {[i.shape for i in val_outputs]}")
                print(f"Val Masks Shape: {[i.shape for i in val_masks]}")

                # Ensure outputs and masks are not None
                if not val_outputs or not val_masks:
                    raise ValueError("Validation outputs or masks are None or empty.")

                # Calculate metrics
                dice_metric(y_pred=val_outputs, y=val_masks)
                dice_metric_batch(y_pred=val_outputs, y=val_masks)

        # Aggregate metrics
       # mean_dice = dice_metric.aggregate()
        #metric_per_class = dice_metric_batch.aggregate()
        mean_dice = dice_metric.aggregate().item()
        metric_per_class = dice_metric_batch.aggregate().cpu().numpy()

        # Append mean_dice to metric_values
        metric_values.append(mean_dice)

        # Append per-class dice scores
        metric_values_class0.append(metric_per_class[0])
        metric_values_class1.append(metric_per_class[1])
        metric_values_class2.append(metric_per_class[2])

        # Debugging aggregate outputs
        print(f"Mean Dice: {mean_dice}")
        print(f"Dice per class: {metric_per_class}")

        if mean_dice is None or metric_per_class is None:
            raise ValueError("Aggregation returned None. Check inputs to metric computation.")

        # Reset metrics
        dice_metric.reset()
        dice_metric_batch.reset()

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Saved Best Model")

print(f"Training Complete. Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")


"""
Testing the model
"""

print("Testing the model...")
model.load_state_dict(torch.load("best_unet_model.pth"))
model.eval()
with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
        test_masks = [post_mask(i) for i in decollate_batch(test_masks)]
        dice_metric(y_pred=test_outputs, y=test_masks)
        dice_metric_batch(y_pred=test_outputs, y=test_masks)

    # Aggregate metrics
    mean_dice = dice_metric.aggregate()
    metric_per_class = dice_metric_batch.aggregate()

    print(f"Testing Mean Dice: {mean_dice}")
    print(f"Dice per class: {metric_per_class}")

    # Reset metrics
    dice_metric.reset()
    dice_metric_batch.reset()



"""
Plot the loss and dice metric
"""

# Plot the training loss
plt.figure("Loss", (6, 4))
plt.title("Epoch Average Loss")
epochs = [i + 1 for i in range(len(epoch_loss_values))]
plt.plot(epochs, epoch_loss_values, color="red", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('swin_loss_u12.png')
plt.show()
print("Loss plot saved as swin_loss_u12.png")

# Plot the validation mean dice
plt.figure("Mean Dice", (6, 4))
plt.title("Validation Mean Dice")
epochs = [val_interval * (i + 1) for i in range(len(metric_values))]
plt.plot(epochs, metric_values, color="green", label="Val Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Mean Dice")
plt.legend()
plt.savefig('swin_mean_dice1.png')
plt.show()
print("Mean Dice plot saved as swin_mean_dice1.png")

# Plot the per-class dice scores
plt.figure("Per-Class Dice", (8, 6))
plt.title("Validation Dice Score per Class")
epochs = [val_interval * (i + 1) for i in range(len(metric_values_class1))]
plt.plot(epochs, metric_values_class0, color="blue", marker='o', label="Background")
plt.plot(epochs, metric_values_class1, color="orange", marker='o', label="GTVp")
plt.plot(epochs, metric_values_class2, color="purple", marker='o', label="GTVn")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.savefig('swin_per_class_dice1.png')
plt.show()
print("Per-class Dice plot saved as swin_per_class_dice1.png")

