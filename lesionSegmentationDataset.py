import os
import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes


class MultiLesionSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        """
        images_dir: Path to the directory containing images.
        lesion_dirs: Dictionary mapping lesion types to their respective directories.
        transform: Transformations to apply to the images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        print(img_path)
        
        # Initialize an empty mask of the same size as the image
        mask = np.zeros((np.array(img).shape[0], np.array(img).shape[1]), dtype=np.uint8)


        # Aggregate masks from each lesion type
        for lesion_type, (mask_dir, abbreviation) in self.masks_dir.items():
            mask_path = os.path.join(mask_dir, self.images[idx].replace('.jpg', f'_{abbreviation}.tif'))
            if os.path.exists(mask_path):
                lesion_mask = Image.open(mask_path)
                lesion_mask = np.array(lesion_mask)
                # Update mask
                mask = np.maximum(mask, lesion_mask)

    # Convert the aggregated binary mask to a tensor mask
        if self.mask_transform:
            mask_tensor = self.mask_transform(Image.fromarray(mask))
        else:
            mask_tensor = ToTensor()(Image.fromarray(mask))

    # Convert the image to tensor
        img_tensor = self.image_transform(img) if self.image_transform else ToTensor()(img)

        mask_tensor = ToTensor()(Image.fromarray(mask))
        img_tensor = ToTensor()(img)

        # Calculate bounding boxes from mask_tensor
        pos = np.where(mask)
        boxes = torch.tensor([[np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]], dtype=torch.float32) if pos[0].size > 0 else torch.tensor([], dtype=torch.float32).reshape(0, 4)

        num_objs = len(torch.unique(mask_tensor)) - 1  # Assuming first id is background
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "masks": mask_tensor.unsqueeze(0),  # Add batch dimension if it's missing
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        return img_tensor, targets




# # Example usage
# ground_truth_dirs = {
#     'Microaneurysms': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/1. Microaneurysms', 'MA'),
#     'Haemorrhages': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/2. Haemorrhages', 'HE'),
#     'Hard_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/3. Hard Exudates', 'EX'),
#     'Soft_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/4. Soft Exudates', 'SE'),
#     'Optic_Disc': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/5. Optic Disc', 'OD'),

# }
# image_transforms = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Mask transform only converts mask to tensor without normalization
# mask_transforms = Compose([
#     Resize((224, 224)),
#     ToTensor(),
# ])

# dataset = MultiLesionSegmentationDataset(images_dir='./data_lesion_detection/1. Original Images/train',
#                                          masks_dir=ground_truth_dirs,
#                                          image_transform=image_transforms,
#                                          mask_transform=mask_transforms)

# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# for images, masks in data_loader:
#     print(images.shape, masks.shape)
#     break
    # Proceed with your training loop
