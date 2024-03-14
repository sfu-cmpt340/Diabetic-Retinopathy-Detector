import pandas as pd
import shutil
import os

# Load CSV data
data = pd.read_csv('trainLabels.csv')  # Assuming 'data.csv' contains image names

# Set paths
input_folder = "data/val"  # Folder containing all images
output_folder = "data_split"  # Output directory to store split data
os.makedirs(output_folder, exist_ok=True)

# Create directories for each class
unique_classes = data['level'].unique()  # Adjust 'class_column' to your CSV column name for classifications
for class_name in unique_classes:
    class_dir = os.path.join(output_folder, str(class_name))
    os.makedirs(class_dir, exist_ok=True)

# Process images based on classifications
for idx, row in data.iterrows():
    image_name = row['image']  # Adjust 'image_name_column' to your CSV column name for image names
    image_path = os.path.join(input_folder, image_name)
    # print(image_name)
    image_path += ".jpeg"
    if(image_name == "10_left"):
        print("ds")
        print(image_path)

    if os.path.isfile(image_path):
        class_name = row['level']  # Adjust 'class_column' to your CSV column name for classifications
        dest_dir = os.path.join(output_folder, str(class_name))
        shutil.move(image_path, dest_dir)
    else:
        print(f"Image {image_name} not found in {input_folder}")

print("Splitting images based on classifications completed.")
