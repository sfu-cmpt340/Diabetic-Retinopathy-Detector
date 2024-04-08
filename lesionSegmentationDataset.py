from email.mime import base
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes
import cv2
from os.path import exists

def visualize_image_mask(image_path, mask_tensor):
    """
    Visualize an image with overlay of a binary mask.

    Args:
        image_path (str): Path to the original image.
        mask_tensor (torch.Tensor): Binary mask tensor.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = mask_tensor.numpy()  # Assuming mask_tensor is a single mask here for simplicity

    # Overlay the mask on the image
    # Assuming the mask is binary, we'll color the mask region in red
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # Coloring mask region in red

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Image with Mask Overlay')
    plt.axis('off')
    
    plt.show()

def find_boxes(image, klass=0):
    """
    Find bounding boxes for segmented regions in a binary mask image.

    Args:
        image (numpy.ndarray): Binary mask image.
        klass (int): Class label.

    Returns:
        dict: Dictionary containing bounding boxes, masks, and class labels.
    """
    if image is None:
        return {}
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (image>0).astype(np.uint8))
#     print(labels.shape, labels.max())
    stats[:,2] += stats[:,0]
    stats[:,3] += stats[:,1]
    stats[:,[0,1,2,3]] = stats[:,[1,0,3,2]]
    stats = stats[:, :-1]
    boxes = stats[1:]
    masks = np.zeros((len(boxes), *image.shape), dtype=np.uint8)
    for i in range(len(boxes)):
        masks[i, boxes[i, 0]:boxes[i, 2], boxes[i, 1]:boxes[i, 3]] = (
                    labels[boxes[i, 0]:boxes[i, 2], boxes[i, 1]:boxes[i, 3]] == (i + 1))
    boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
    klass = np.array([klass] * len(boxes))
    return dict(
        boxes=torch.from_numpy(boxes.astype(np.float32)),
        masks=torch.from_numpy(masks),
        labels=torch.from_numpy(klass.astype(np.int64)))


class LesionSegMask(Dataset):
    def __init__(self,images_path,ground_truth_dir, root=None):
        """
        Custom dataset class for loading images and corresponding ground truth masks.

        Args:
            images_path (str): Path to the directory containing original images.
            ground_truth_dir (dict): Dictionary containing paths to directories containing ground truth masks.
            root (str): Root directory path.
        """
        if root is None:
            raise Exception("data root directory is none")

        original_images_path = os.path.join(root, images_path)

    # Assuming ground_truth_dirs is a parameter similar to ground_truth_dirs_test defined earlier
        self.images = []

        # Get all original image file names
        original_images = [img for img in os.listdir(original_images_path) if img.endswith('.jpg')]

        for img in original_images:
            # Extract the base name without the extension
            base_name = img.split('.')[0]
            image_path = os.path.join(original_images_path, img)

            # Prepare to collect mask paths for this image
            mask_paths = []

            # Iterate over each lesion type and build the path to each corresponding mask
            for lesion_type, (mask_dir, mask_suffix) in ground_truth_dir.items():
                mask_name = f"{base_name}_{mask_suffix}.tif"
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                else:
                    mask_paths.append(None)  # Append None if the mask does not exist

            # Ensure that mask_paths contains all required masks, otherwise skip this image
            if len(mask_paths) == len(ground_truth_dir):
                self.images.append((image_path, *mask_paths))
            self.ratio = 0.5

        if not self.images:
            raise ValueError("No images were found with the corresponding masks.")


    def __getitem__(self, index):
        i = self.images[index]
        # print(self.images)
        image = cv2.imread(i[0], cv2.IMREAD_COLOR)
        dsize = (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio))
        image = cv2.resize(image, dsize=dsize)
        
        label = {}

        for cls, path in enumerate(i[1:]):
                if path is None or not exists(path):
                    continue
                mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(mask_img, dsize=dsize)
                res = find_boxes(mask_img, cls)
                for k, v in res.items():
                    if k in label:
                        label[k] = torch.cat((label[k], v))
                        # label[k].append(v)
                    else:
                        label[k] = v
        image = torch.from_numpy(image)
        image = image.float() / 255.0
        print("this is masks")
        visualize_image_mask(i[0],label["masks"])
        print(label["masks"])
        return image, label

    def __len__(self):
        return len(self.images)


