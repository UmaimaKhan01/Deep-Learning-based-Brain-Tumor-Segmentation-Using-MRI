import os
import torch
import numpy as np
import nibabel as nib
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.losses import DiceLoss

# Add your dataset path here
data_dir = r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main\Task01_BrainTumour"

# Define training transforms
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 64), random_size=False),  # Smaller ROI to save memory
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])

# Create full dataset
full_dataset = Dataset(
    data=[
        {"image": os.path.join(data_dir, "imagesTr", x), "label": os.path.join(data_dir, "labelsTr", x)}
        for x in os.listdir(os.path.join(data_dir, "imagesTr")) if x.endswith(".nii.gz")
    ],
    transform=train_transform,
)

# Filter out samples with no foreground labels
def filter_empty_labels(dataset):
    filtered_data = []
    for sample in tqdm(dataset, desc="Filtering empty labels"):
        label = sample["label"]  # The label is already loaded as a tensor
        if torch.any(label > 0):  # Check if there's any non-zero label
            filtered_data.append(sample)
    return filtered_data

filtered_data = filter_empty_labels(full_dataset)
print(f"Filtered dataset size: {len(filtered_data)}")

# Use only 10% of the dataset
subset_size = int(len(filtered_data) * 0.1)
full_dataset = torch.utils.data.Subset(filtered_data, range(subset_size))

# Set up 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Training loop for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\nFold {fold + 1}")
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # Reduced batch size and workers
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)  # Reduced batch size and workers

        # Model setup
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        # Loss function and optimizer
        loss_function = DiceLoss(
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            batch=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)  # Use 95th percentile
        val_dice_metric = DiceMetric(include_background=False, reduction="mean")
        val_hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)

        # Training parameters
        max_epochs = 20  # Reduced number of epochs
        best_metric = -1
        best_metric_epoch = -1

        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0.0
            dice_metric.reset()
            hausdorff_metric.reset()

            # Training loop
            for batch_data in tqdm(train_loader):
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Compute metrics
                dice_metric(y_pred=outputs, y=labels)
                hausdorff_metric(y_pred=outputs, y=labels)

            epoch_loss /= len(train_loader)
            print(f"Train Loss: {epoch_loss:.4f} | Train Dice: {dice_metric.aggregate().item():.4f} | Train Hausdorff: {hausdorff_metric.aggregate().item():.4f}")

            # Validation loop
            model.eval()
            val_dice_metric.reset()
            val_hausdorff_metric.reset()
            with torch.no_grad():
                for val_data in tqdm(val_loader):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_inputs)

                    # Post-process predictions to ensure they are binary
                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs = (val_outputs > 0.5).float()

                    val_dice_metric(y_pred=val_outputs, y=val_labels)
                    val_hausdorff_metric(y_pred=val_outputs, y=val_labels)

            val_dice = val_dice_metric.aggregate().item()
            val_hausdorff = val_hausdorff_metric.aggregate().item()
            print(f"Val Dice: {val_dice:.4f} | Val Hausdorff: {val_hausdorff:.4f}")

            # Save best model
            if val_dice > best_metric:
                best_metric = val_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f"best_metric_model_fold{fold + 1}.pth")
                print(f"New best metric: {best_metric:.4f} at epoch {best_metric_epoch}")

        print(f"Training complete for Fold {fold + 1}. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == '__main__':
    train_model()