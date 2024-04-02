import torch
import os
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch import nn
from PIL import Image
import torch.nn.functional as F

class DRGradingSubNetwork(nn.Module):
    def __init__(self, resnet18, lesion_segmentation_module, num_classes):
        super(DRGradingSubNetwork, self).__init__()
        self.resnet18 = resnet18
        self.lesion_segmentation_module = lesion_segmentation_module
        self.fc = nn.Linear(512 + 512, num_classes)  # Adjust input size based on your feature sizes
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print("here resnet before ")
        features_resnet18 = self.resnet18(x)

        print(features_resnet18.shape)
        features_resnet18 = features_resnet18.squeeze()
        print("here resnet after ")

        print(features_resnet18.shape)

        features_lesion = self.lesion_segmentation_module.mask_rcnn_model.backbone(x)
        # features_resnet18_tensor = torch.mean(features_resnet18, dim=1, keepdim=True)
        # features_resnet18_tensor = torch.tensor(features_resnet18_tensor)
        # print(features_resnet18_tensor)
        print("here sef before ")

        print(features_lesion.shape)
        features_lesion = features_lesion.squeeze()
        print("here seg after ")

        # print(features_lesion.size())

        # pooled_features_lesion = features_lesion.squeeze()
        print(features_lesion.shape)
        # print(features_lesion.shape)

        print(features_resnet18.shape)




        combined_features = torch.cat((features_resnet18, features_lesion), dim=0)
        output = self.fc(self.relu(combined_features))
        return output

def main():
    # Instantiate DR grading sub-network
    resNet18 = models.resnet18()
    state_dict = torch.load('finetune_net.pth')

    # Remove fully connected layer weights from the state dict

    # Load the modified state dict into the ResNet18 model
    resNet18.fc = nn.Linear(resNet18.fc.in_features, 5)
    resNet18.load_state_dict(state_dict, strict=False)
    resNet18 = torch.nn.Sequential(*(list(resNet18.children())[:-1]))

    lesion_segmentation_module = torch.load('lesion_segmentation_model.pth')
    num_classes = 5  # Number of classes for DR grading

    dr_grading_subnetwork = DRGradingSubNetwork(resNet18, lesion_segmentation_module, num_classes)

    # Freeze weights of ResNet18 and lesion segmentation module
    for param in dr_grading_subnetwork.resnet18.parameters():
        param.requires_grad = False
    for param in dr_grading_subnetwork.lesion_segmentation_module.parameters():
        param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dr_grading_subnetwork.parameters(), lr=0.001)

    print("here")

    train_augs = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    transforms.RandomRotation(20),  # Randomly rotate images by 20 degrees
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

    val_augs = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join("data_split", 'training'),transform=train_augs),
            batch_size=10, shuffle=True,num_workers=1)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join("data_split", 'testing'),transform=val_augs),
            batch_size=10,num_workers=1)

    print("here1")

    # num_epochs = 10  # Adjust the number of epochs as needed
    # for epoch in range(num_epochs):
    #     dr_grading_subnetwork.train()  # Set the model to training mode
        
    #     # Iterate over the dataset batches
    #     for images, labels in train_loader:  # Assuming you have defined train_loader
    #         # Zero the gradients
    #         optimizer.zero_grad()
            
    #         # Forward pass
    #         outputs = dr_grading_subnetwork(images)
            
    #         # Compute the loss
    #         loss = criterion(outputs, labels)
            
    #         # Backward pass
    #         loss.backward()
            
    #         # Update the weights
    #         optimizer.step()
            


    #     # Print training loss
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    dr_grading_subnetwork.eval()
    input_image = Image.open("data_lesion_detection/1. Original Images/test/IDRiD_55.jpg")  # Load input image using PIL
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match the input size expected by the model
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize image
        mean=[0.485, 0.456, 0.406],   # Mean and standard deviation values used for normalization
        std=[0.229, 0.224, 0.225]
    )
])
    input_tensor = preprocess(input_image)  # Apply preprocessing transformations to input image
    input_tensor = input_tensor.unsqueeze(0)

# Perform inference
    with torch.no_grad():
    # Forward pass your data through the network and obtain predictions
        predictions = dr_grading_subnetwork(input_tensor)

    # Optionally, save the trained model
    torch.save(dr_grading_subnetwork.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    main()
