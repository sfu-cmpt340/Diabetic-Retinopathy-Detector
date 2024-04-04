from __future__ import print_function, division
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset 
import torch.optim as optim
import numpy as np
import main_fine_tuning as resnetModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def BASE_DR():
    path = 'gaussian_filtered_images'
    X = []
    y = []
    desired_size = (128, 128) 
    Classes = {'0':0, '1':1, '2':2, '3':3, '4':4}

    for i in Classes:
        folder_path ='gaussian_filtered_images/' +i
        for j in os.listdir(folder_path):
            img = cv2.imread(folder_path+'/'+j)
            img = cv2.resize(img, desired_size)
            # normalize values
            img = img / 255  #-->Apply normalization because we want pixel values to be scaled to the range 0-1
            X.append(img)
            y.append(Classes[i])
    X = np.array(X)
    y = np.array(y)
    np.save('X_np', X)
    np.save('y_np', y)

    class_names = Classes

    class_counts = [0] * len(class_names)


    for subset_dir in ['gaussian_filtered_images']:
    # Iterate over the subfolders (classes)
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(subset_dir, class_name)
            # Count the number of images in the class directory
            class_counts[class_idx] += len(os.listdir(class_dir))
    print("lol")

    total_images = sum(class_counts)
    class_weights = [total_images/count for count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(class_weights_tensor.shape, class_weights_tensor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32, stratify=y)
    
    X_train = X_train.reshape((-1, 128, 128, 3))
    X_test = X_test.reshape((-1, 128, 128, 3))

    X_train_tensor = torch.tensor(X_train.transpose((0, 3, 1, 2)), dtype=torch.float32)  # Convert and rearrange dimensions, then convert to float32 and move to GPU

# Assuming y_train_tensor is your target tensor
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Convert y_train to tensor of type long and move to GPU


    # Create TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    X_test_tensor = torch.tensor(X_test.transpose((0, 3, 1, 2)), dtype=torch.float32)  # Convert and rearrange dimensions, then convert to float32 and move to GPU

# Assuming y_train_tensor is your target tensor
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # Convert y_train to tensor of type long and move to GPU


    # training = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize images to 224x224
    #     transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    #     transforms.RandomRotation(20),  # Randomly rotate images by 20 degrees
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    # ])
    # testing = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize images to 224x224
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    # ]),

    # Create TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    dsets = {
        'training': train_dataset,
        'testing':test_dataset
    }


    # Define batch size for training
    batch_size = 32  # Adjust batch size as needed

    # Create DataLoader for training data
    sample = train_dataset[0]
    print(len(sample))

    # Extract labels from the TensorDataset
# Assuming your train_dataset is a TensorDataset and the second element of each item is the label
    labels = torch.stack([label for _, label in train_dataset])

    # Calculate sample weights using the labels
    sample_weights = class_weights_tensor[labels].tolist()  # Convert tensor to a list

    # Create the WeightedRandomSampler with your sample weights
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Check the DataLoader
  
   
    print("DataLoaders are ready.")


    dset_loaders = {
    'training': DataLoader(train_dataset, sampler = sampler,batch_size=32, num_workers=0),
    'testing': DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    }
    for i, (x, y) in enumerate(dset_loaders['training']):
        print("batch index {}, 0/1/2/3/4: {}/{}/{}/{}/{}".format(
            i, (y == 0).sum().item(), (y == 1).sum().item(), (y == 2).sum().item(), 
            (y == 3).sum().item(), (y == 4).sum().item()))

    print("Number of batches in train_loader:", len(dset_loaders["training"]))
    print("Sample weights for the first batch:", sample_weights[:32])  # Adjust the slice as needed
    dset_sizes = {x: len(dsets[x]) for x in ['training', 'testing']}

    # Set up the DataLoaders in a dictionary


    return dset_loaders,dsets,dset_sizes

    