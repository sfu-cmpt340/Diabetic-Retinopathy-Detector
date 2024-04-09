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
    """
    Function to prepare and load data for Diabetic Retinopathy classification.

    Returns:
        dset_loaders: Dictionary containing training and testing DataLoader objects.
        dsets: Dictionary containing training and testing dataset objects.
        dset_sizes: Dictionary containing sizes of training and testing datasets.
    """
    # Define the path to the dataset
    path = 'gaussian_filtered_images'

    # Initialize lists to store images and labels
    X = []
    y = []
    
    # Define the desired size of the images
    desired_size = (128, 128) 

    # Define the classes
    Classes = {'0':0, '1':1, '2':2, '3':3, '4':4}

    # Iterate over the classes folder
    for i in Classes:
        folder_path ='gaussian_filtered_images/' +i
        for j in os.listdir(folder_path):
            # Read and resize image
            img = cv2.imread(folder_path+'/'+j)
            img = cv2.resize(img, desired_size)
            # normalize values
            img = img / 255  #-->Apply normalization because we want pixel values to be scaled to the range 0-1
            X.append(img)
            y.append(Classes[i])
    
    # Convert lists to numpy arrays ansd save them
    X = np.array(X)
    y = np.array(y)
    np.save('X_np', X)
    np.save('y_np', y)

    # Define the class names
    class_names = Classes

    class_counts = [0] * len(class_names)


    for subset_dir in ['gaussian_filtered_images']:
    # Iterate over the subfolders (classes)
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(subset_dir, class_name)
            # Count the number of images in the class directory
            class_counts[class_idx] += len(os.listdir(class_dir))

    # Calculate the class weights for weighted sampling
    total_images = sum(class_counts)
    class_weights = [total_images/count for count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
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

    # Create TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    dsets = {
        'training': train_dataset,
        'testing':test_dataset
    }

    # Extract labels from the TensorDataset
# Assuming your train_dataset is a TensorDataset and the second element of each item is the label
    labels = torch.stack([label for _, label in train_dataset])

    # Calculate sample weights using the labels
    sample_weights = class_weights_tensor[labels].tolist()  # Convert tensor to a list

    # Create the WeightedRandomSampler with your sample weights
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print("DataLoaders are ready.")

    # Create DataLoader objects for training and testing datasets
    dset_loaders = {
    'training': DataLoader(train_dataset, sampler = sampler,batch_size=32, num_workers=0),
    'testing': DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    }

    dset_sizes = {x: len(dsets[x]) for x in ['training', 'testing']}

    # Set up the DataLoaders in a dictionary

    print("DataLoaders are ready.")
    
    return dset_loaders,dsets,dset_sizes

    