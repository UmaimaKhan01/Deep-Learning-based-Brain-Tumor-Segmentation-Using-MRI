import cv2
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Define paths
IMG_ROOT = r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main\Task01_BrainTumour\imagesTr"
LABEL_ROOT = r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main\Task01_BrainTumour\labelsTr"
IMG_OUTPUT_ROOT = './train/image_T1'
LABEL_OUTPUT_ROOT = './train/label'

L0 = 0      # Background
L1 = 50     # Necrotic and Non-enhancing Tumor
L2 = 100    # Edema
L3 = 150    # Enhancing Tumor

def nii2jpg_img(img_path, output_root):
    img_name = os.path.basename(img_path).split('.')[0]
    output_path = os.path.join(output_root, img_name)
    os.makedirs(output_path, exist_ok=True)
    
    img = nib.load(img_path).get_fdata()
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    
    for i in range(img.shape[2]):
        cv2.imwrite(os.path.join(output_path, f'{img_name}_{i}.jpg'), img[:, :, i, 0])

def nii2jpg_label(label_path, output_root):
    label_name = os.path.basename(label_path).split('.')[0]
    output_path = os.path.join(output_root, label_name)
    os.makedirs(output_path, exist_ok=True)
    
    label = nib.load(label_path).get_fdata().astype(np.uint8) * 50
    for i in range(label.shape[2]):
        cv2.imwrite(os.path.join(output_path, f'{label_name}_{i}.jpg'), label[:, :, i])

# Process all label files
for path in tqdm(os.listdir(LABEL_ROOT)):
    if not path.endswith('.nii.gz'):
        continue
    label_path = os.path.join(LABEL_ROOT, path)
    nii2jpg_label(label_path, LABEL_OUTPUT_ROOT)