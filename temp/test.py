import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import Lesion_Detection_Segmentation
from PIL import Image
from torchvision.transforms import functional as F

def test_single_image(image_path, lesion_segmentation_module, device, transform=None):
    # Ensure the model is in evaluation mode
    lesion_segmentation_module.eval()

    # Load the image
    image = Image.open(image_path).convert("RGB")

    
    # Apply the necessary transformations
    if transform is not None:
        image = transform(image)
    else:
        # Default transformations if none are provided
        image = F.to_tensor(image)
        image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Add a batch dimension
  

    # Disable gradient computation
    with torch.no_grad():
        # Perform the forward pass and get predictions
        predictions = lesion_segmentation_module([image])

        # The output is a list with one element per image,
        # so we get the first element since we only have one image
        prediction = predictions[0]

        # Here you can process the prediction dictionary as needed
        # For example, print it or convert it to a certain format
        print(prediction)

        # Return the prediction if you want to use it outside this function
        return prediction

def visualize_segmentation(model, image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Convert image to tensor
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
        
        # Move tensor to device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Perform inference
        prediction = model(image_tensor)[0]  # Only one image, so select the first element
        print(prediction)
        
        # Get the predicted masks
        masks = prediction['masks'].cpu().numpy()  # Convert to numpy array
        
        # Convert the masks to binary
        masks = (masks > 0.5).astype(np.uint8)
        
        image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

# Plot the original image
        plt.figure(figsize=(20, 10))  # Increase figure size for better visibility
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.axis('off')
        plt.title('Original Image')

        # Sum up the masks for all classes
        composite_mask = np.sum(masks, axis=0)

        # Ensure the composite mask is binary
        composite_mask = (composite_mask > 0).astype(np.uint8)
        composite_mask = composite_mask.squeeze()

        # Plot the composite segmentation mask
        plt.subplot(1, 3, 2)
        plt.imshow(composite_mask, cmap='gray', alpha=0.7)  # Use a grayscale colormap
        plt.axis('off')
        plt.title('Composite Segmentation')

        # Optionally, plot each class mask separately
        for i in range(masks.shape[1]):
            class_mask = (masks[0, i] > 0.5).astype(np.uint8)  # Binary mask for the i-th class

            plt.subplot(1, 3, 3)  # Adjust subplot as needed
            plt.imshow(class_mask, alpha=0.7)  # No colormap, just binary
            plt.axis('off')
            plt.title(f'Segmentation Class {i+1}')

        plt.show()

# Example usage:
# Assuming you have a trained model named `trained_model`
# And an image named `sample_image`

# visualize_segmentation(trained_model, sample_image)

if __name__ == '__main__': 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lesion_detection_model = Lesion_Detection_Segmentation.LesionDetectionModel(num_classes=5, learning_rate=1e-3,device=device)

    # Retrieve the feature extractor
    checkpoint = torch.load('lesion_detection_model.pth')

    # Load the model state_dict
    lesion_detection_model.model.load_state_dict(checkpoint)

    # Now you can access the feature extractor
    feature_extractor = lesion_detection_model.get_feature_extractor()
    img = Image.open('./data_lesion_detection/1. Original Images/test/IDRiD_55.jpg')
    img.show()
    model = Lesion_Detection_Segmentation.LesionSegmentationModule(feature_extractor=feature_extractor, num_classes=6)
    model.load_state_dict(torch.load('lesion_segmentation_model_dict.pth'))

    # Assuming you have a sample image named 'sample_image'
    # visualizing segmentation
    # visualize_segmentation(model, img)
    test_single_image('./data_lesion_detection/1. Original Images/test/IDRiD_55.jpg',model,device=device)
