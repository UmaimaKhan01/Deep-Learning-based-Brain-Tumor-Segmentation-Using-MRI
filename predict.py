import os
import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
    ToTensord
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

# Fix path issues
base_dir = r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main"
data_dir = os.path.join(base_dir, "Task01_BrainTumour")

# Check if the data_dir exists
if not os.path.exists(data_dir):
    data_dir = r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Task01_BrainTumour"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Could not find data directory at {data_dir}")

# Ensure the directory for test file exists
image_dir = os.path.join(data_dir, "imagesTr")
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Could not find image directory at {image_dir}")

# Get the first available test file
test_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
if not test_files:
    raise FileNotFoundError(f"No .nii.gz files found in {image_dir}")

test_file = test_files[0]
print(f"Using test file: {test_file}")

# Define paths
model_path = os.path.join(base_dir, "best_metric_model_fold1.pth")
if not os.path.exists(model_path):
    model_files = [f for f in os.listdir(base_dir) if f.startswith("best_metric_model_fold") and (f.endswith(".pth") or f.endswith(".txt"))]
    if model_files:
        model_path = os.path.join(base_dir, model_files[0])
    else:
        raise FileNotFoundError(f"No model files found in {base_dir}")

print(f"Using model file: {model_path}")

# Create output directory
output_dir = os.path.join(base_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"prediction_{os.path.basename(test_file)}")

# Define test transforms that match training transforms
# The key here is to use the same spatial processing as during training
test_transform = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS"),  # Same orientation as training
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  # Same spacing as training
    # Use a fixed size crop that matches what the model expects
    CenterSpatialCropd(keys=["image"], roi_size=(128, 128, 64)),  # Critical to match model's expected input size
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image"])
])

# Load test data
test_ds = Dataset(data=[{"image": test_file}], transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model with exact same architecture as during training
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Try to load the model
try:
    if model_path.endswith('.txt'):
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        # Try with weights_only=True
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model loaded with weights_only=True")
    except Exception as e2:
        print(f"Error loading model with weights_only=True: {e2}")
        raise

model.eval()

# Generate prediction
try:
    with torch.no_grad():
        print("Processing test data...")
        batch = next(iter(test_loader))
        test_inputs = batch["image"].to(device)
        print(f"Input shape: {test_inputs.shape}")
        
        print("Running model inference...")
        test_outputs = model(test_inputs)
        print(f"Output shape: {test_outputs.shape}")
        
        print("Applying sigmoid and threshold...")
        test_outputs = torch.sigmoid(test_outputs)
        test_outputs = (test_outputs > 0.5).float()
        
        # Save prediction
        from monai.transforms import SaveImage
        saver = SaveImage(output_dir=output_dir, output_postfix="seg", output_ext=".nii.gz")
        metadata = {"filename_or_obj": [os.path.basename(test_file)]}
        saver(test_outputs[0], metadata)
        
        # Also create a visualization
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert to numpy for visualization
        output_np = test_outputs.detach().cpu().numpy()[0]  # First batch item
        
        # Create a simple visualization
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        # Find a slice with some segmentation
        mid_slice = output_np.shape[3] // 2
        
        for i in range(3):  # For each of the 3 tumor classes
            axes[i].imshow(output_np[i, :, :, mid_slice], cmap='viridis')
            axes[i].set_title(f"Tumor class {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_visualization.png"))
        print(f"Visualization saved to {os.path.join(output_dir, 'prediction_visualization.png')}")
        
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()