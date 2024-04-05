import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Define the path to the original input directory of images
ORIG_INPUT_DATASET = "3rd_dataset"

# Define the base path to the new directory that will contain our images after the split
BASE_PATH = "disease_no_disease"

# Define the path to the training directory
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])

# Define the path to the testing directory
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# Define the amount of data that will be used for training (80%)
TRAIN_SPLIT = 0.80

# Create the training and testing directories if they don't exist
for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Grab the list of images in the original input dataset
images = os.listdir(ORIG_INPUT_DATASET)

# Shuffle the list of images
random.shuffle(images)

# Split the images into training and testing sets
split_index = int(len(images) * TRAIN_SPLIT)
train_images = images[:split_index]
test_images = images[split_index:]

# Move training images to the training directory
for image in train_images:
    src = os.path.sep.join([ORIG_INPUT_DATASET, image])
    dst = os.path.sep.join([TRAIN_PATH, image])
    shutil.move(src, dst)

# Move testing images to the testing directory
for image in test_images:
    src = os.path.sep.join([ORIG_INPUT_DATASET, image])
    dst = os.path.sep.join([TEST_PATH, image])
    shutil.move(src, dst)
