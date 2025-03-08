import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class BRATSDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        samples = []
        image_files = sorted(os.listdir(self.image_dir))
        print("Loading samples...")
        for image_file in tqdm(image_files):
            if not image_file.endswith(".nii.gz"):
                continue  # Skip non-NIfTI files
            image_path = os.path.join(self.image_dir, image_file)
            label_path = os.path.join(self.label_dir, image_file)

            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Label file missing: {label_path}")
                continue

            try:
                label = nib.load(label_path).get_fdata()
            except Exception as e:
                print(f"Error loading label file: {label_path} ({e})")
                continue

            for i in range(label.shape[2]):
                if np.any(label[:, :, i] > 0):  # Only include slices with tumor regions
                    samples.append((image_path, label_path, i))
        print("Loaded ", len(samples), "samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path, slice_index = self.samples[idx]
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Extract specific slices
        image_slice = image[:, :, slice_index, :3]  # Take first 3 channels
        label_slice = label[:, :, slice_index]

        image_slice = torch.tensor(image_slice, dtype=torch.float32).permute(2, 0, 1)  # Add channel dimension
        label_slice = torch.tensor(label_slice, dtype=torch.long)

        if self.transform:
            image_slice = self.transform(image_slice)
            label_slice = self.transform(label_slice)

        return image_slice, label_slice