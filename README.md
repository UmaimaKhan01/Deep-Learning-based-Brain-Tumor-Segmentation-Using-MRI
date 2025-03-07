# Deep Learning-based Brain Tumor Segmentation Using MRI

This repository contains an implementation of a 3D U-Net model for brain tumor segmentation using the BraTS challenge dataset. The implementation utilizes the MONAI framework, PyTorch, and other libraries for deep learning in medical imaging.

## Overview

Brain tumor segmentation is a critical task in medical image analysis that assists in diagnosis, treatment planning, and monitoring. This project implements a 3D U-Net architecture to segment brain tumors from multi-modal MRI scans, identifying three different tumor regions:
1. Necrotic and non-enhancing tumor (label 1)
2. Peritumoral edema (label 2)
3. GD-enhancing tumor (label 4)

## Dataset

The dataset used is from the Brain Tumor Image Segmentation (BraTS) challenge, available [here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). Download the `Task01_BrainTumour.tar` file and extract it.

The dataset structure should look like:
```
Task01_BrainTumour/
├── dataset.json
├── imagesTr/      # Training images
├── labelsTr/      # Training labels
└── imagesTs/      # Test images (not used)
```

## Requirements

To run this code, you'll need:

```
python >= 3.6
torch >= 1.7.0
monai >= 0.6.0
nibabel
numpy
scikit-learn
tqdm
matplotlib
```

You can install the required packages using pip:
```bash
pip install torch monai nibabel numpy scikit-learn tqdm matplotlib
```

## Code Structure

- `mypython.py`: Main training script that implements 5-fold cross-validation.
- `predict.py`: Script for generating segmentation predictions on a test image.
- `results.txt`: Contains the cross-validation results.

## Running the Code

### Training

To train the model using 5-fold cross-validation:

```bash
python mypython.py
```

This script will:
1. Load and preprocess the BraTS dataset
2. Filter out samples with no foreground labels
3. Use 10% of the filtered dataset for faster training
4. Perform 5-fold cross-validation
5. Save the best model for each fold as `best_metric_model_fold{fold_number}.pth`

Note: You'll need to update the `data_dir` variable in the script to point to your BraTS dataset location.

### Prediction

To generate a prediction on a test image:

```bash
python predict.py
```

This script will:
1. Load a trained model
2. Process a test image
3. Generate a segmentation prediction
4. Save the prediction as a NIfTI file
5. Create a visualization of the segmentation results

Note: You'll need to update the `base_dir` and `data_dir` variables in the script to point to your directory structure.

## Results

The 5-fold cross-validation results are summarized below:

| Fold | Best Val Dice | Epoch | Hausdorff Distance |
|------|--------------|-------|-------------------|
| 1    | 0.4051       | 20    | 55.6956           |
| 2    | 0.3563       | 20    | 68.3177           |
| 3    | 0.4561       | 19    | 46.0424           |
| 4    | 0.5071       | 20    | 35.3593           |
| 5    | 0.4304       | 18    | 46.4126           |

**Average Val Dice: 0.4310**

## Model Architecture

The implemented 3D U-Net has the following configuration:

- Spatial Dimensions: 3
- Input Channels: 4 (corresponding to the four MRI modalities)
- Output Channels: 3 (corresponding to the three tumor classes)
- Feature Channels: (16, 32, 64, 128, 256)
- Strides: (2, 2, 2, 2)
- Residual Units: 2 per layer

## Implementation Details

### Preprocessing Pipeline

The preprocessing pipeline includes:
- Loading NIfTI data (4 MRI modalities and segmentation mask)
- Converting labels from [1,2,4] to consecutive [0,1,2]
- Cropping to non-zero region to remove background
- Normalizing each modality to zero mean and unit variance
- Resampling to isotropic spacing of (1.0, 1.0, 1.0) mm
- Extracting random patches of size 128×128×128 during training
- Applying data augmentation (random flipping, rotation)

### Training Strategy

The training process includes:
- Loss Function: Combination of Dice loss and Cross-Entropy loss
- Optimizer: Adam optimizer with initial learning rate of 1e-4
- Epochs: Maximum of 20 epochs per fold
- Early Stopping: Based on validation Dice coefficient
- Evaluation Metrics: Dice similarity coefficient and Hausdorff distance

## References

[1] Bakas, S., Reyes, M., Int., E. & Menze, B. Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. arXiv preprint arXiv:1811.02629 (2018).

[2] Bakas, S. et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data 4, 1–13 (2017).

[3] Menze, B. H. et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Transactions on Medical Imaging 34, 1993–2024 (2015).

[4] MONAI: Medical Open Network for AI. https://github.com/Project-MONAI/MONAI

## License

This project is provided as-is for educational purposes.

## Author

Umaima Khan - fn653419@ucf.edu
