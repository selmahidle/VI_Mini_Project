import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, ToTensord
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
import time
from monai.networks.nets import UNet
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the preprocessing transforms for a single image
single_image_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="LPS"),
    Spacingd(keys=["image", "mask"], pixdim=(0.5, 0.5, 2.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", subtrahend=0.0, divisor=1.0, channel_wise=False),  # Replace with actual mean and std
    ToTensord(keys=["image", "mask"]),
])

# Define file paths for a single image and mask
single_image_path = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test/8/preRT/8_preRT_T2.nii.gz"
single_mask_path = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test/8/preRT/8_preRT_mask.nii.gz"

# Apply preprocessing to the single image
data = {"image": single_image_path, "mask": single_mask_path}
transformed_data = single_image_transforms(data)
single_image = transformed_data["image"].unsqueeze(0).to(device)  # Add batch dimension
single_mask = transformed_data["mask"].unsqueeze(0).to(device)  # Add batch dimension (for evaluation)

# Define the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=3,
    dropout=0.1,
    norm=Norm.BATCH,
).to(device)

# Load the model weights
print("Loading model weights...")
state_dict = torch.load("/cluster/home/annaost/VI_Mini_Project-1/best_unet_model.pth")

# If the saved file contains additional keys like "model" or "state_dict", extract the correct one
if "state_dict" in state_dict:
    model.load_state_dict(state_dict["state_dict"], strict=False)
else:
    model.load_state_dict(state_dict, strict=False)

print("Model loaded successfully. Running inference on a single image...")
model.eval()

# Perform inference
roi_size = (128, 128, 128)
sw_batch_size = 1

with torch.no_grad():
    start_time = time.time()
    test_output = sliding_window_inference(single_image, roi_size, sw_batch_size, model)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time for single image: {inference_time:.4f} seconds")

# Post-process the output
predicted_mask = torch.argmax(test_output, dim=1).detach().cpu()  # Convert to class indices

# Visualize the input image, ground truth mask, and predicted mask
plt.figure("Single Image Inference", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(single_image[0, 0, :, :, single_image.shape[-1] // 2].cpu(), cmap="gray")
plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(single_mask[0, 0, :, :, single_mask.shape[-1] // 2].cpu())
plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(predicted_mask[0, :, :, predicted_mask.shape[-1] // 2])
plt.savefig("single_image_inference.png")
plt.show()
