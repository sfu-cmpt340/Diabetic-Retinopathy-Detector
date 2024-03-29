import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader

class MultiLabelLesionDataset(Dataset):
    def __init__(self, images_dir, ground_truth_dirs, transform=None):
        """
        images_dir: Path to the directory containing images.
        ground_truth_dirs: Dictionary mapping lesion types to their ground truth directories.
        transform: PyTorch transforms to apply to the images.
        """
        self.images_dir = images_dir
        self.ground_truth_dirs = ground_truth_dirs
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # print(img_path)
        img = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            img = self.transform(img)
        
        labels = []
        for lesion_type, (gt_dir, abbreviation) in self.ground_truth_dirs.items():
            # Dynamically construct the gt_path based on the lesion type
            gt_filename = img_name.replace('.jpg', f'_{abbreviation}.tif')
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if os.path.exists(gt_path):
                gt_image = Image.open(gt_path)
                gt_array = np.array(gt_image)
                labels.append(1 if np.any(gt_array > 0) else 0)
            else:
                labels.append(0)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return img, labels
    
transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ground_truth_dirs = {
     'Microaneurysms': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/1. Microaneurysms', 'MA'),
    'Haemorrhages': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/2. Haemorrhages', 'HE'),
    'Hard_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/3. Hard Exudates', 'EX'),
    'Soft_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/4. Soft Exudates', 'SE'),
    'Optic_Disc': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/5. Optic Disc', 'OD'),

}
# dataset = MultiLabelLesionDataset(images_dir='./data_lesion_detection/1. Original Images/train',
#                                   ground_truth_dirs=ground_truth_dirs,
#                                   transform=transform)
                        
# for i in range(len(dataset)):
#     img, labels = dataset[i]
#     print(f"Image Index: {i}")
#     print(f"Labels: {labels.numpy()}")
#     # # Optionally, display the image using matplotlib
#     # import matplotlib.pyplot as plt
#     # plt.imshow(img.permute(1, 2, 0))  # Rearrange the channels for plotting
#     # plt.title(f"Labels: {labels.numpy()}")
#     # plt.show()
    
#     if i == 5:  # Limit the output to first 6 images
#         break

