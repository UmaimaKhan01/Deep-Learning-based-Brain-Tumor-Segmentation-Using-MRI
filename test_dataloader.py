from dataloader import BRATSDataset

dataset = BRATSDataset(
    image_dir=r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main\Task01_BrainTumour\imagesTr",
    label_dir=r"C:\Users\umaim\Downloads\Brain-Tumour-Segmentation-main\Brain-Tumour-Segmentation-main\Task01_BrainTumour\labelsTr"
)
print("Total valid slices:", len(dataset))