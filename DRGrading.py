from cgi import test
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch import nn
from PIL import Image
from torch.optim import SGD
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRGradingSubNetwork(nn.Module):
    def __init__(self, resnet18, lesion_segmentation_module, num_classes):
        super(DRGradingSubNetwork, self).__init__()
        self.resnet18 = resnet18
        self.lesion_segmentation_module = lesion_segmentation_module
        self.f1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        
        features_resnet18 = self.resnet18(x)
        features_lesion = self.lesion_segmentation_module.mask_rcnn_model.backbone(x)
        # print(features_lesion.shape,features_resnet18.shape)

        features_lesion = features_lesion.squeeze()
        features_resnet18 = features_resnet18.squeeze()

        # print(features_lesion.shape,features_resnet18.shape)

        combined_features = torch.cat((features_resnet18, features_lesion), dim=1)
        # output = self.fc(self.relu(combined_features))
        # print("hello")
        # print(combined_features.shape)
        # print("hello")

        # print(combined_features)
        self.f1.train()
        logits = self.f1(combined_features)
    
        output = F.softmax(logits, dim=1)
        # print(output)
        return output

def train_classification( loader,num_epochs, dr_grading_subnetwork,lr,device):
        params = [
            {'params': dr_grading_subnetwork.f1.parameters(), 'lr': lr},  # Fixed learning rate
            {'params': dr_grading_subnetwork.resnet18.parameters(), 'lr': 0.0},  # Learning rate set to 0 for fixed parameters
            {'params': dr_grading_subnetwork.lesion_segmentation_module.parameters(), 'lr': 0.0}  # Learning rate set to 0 for fixed parameters
            
        ]
        optimizer = SGD(params, lr, 0.9, weight_decay=0.00001)
        # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for image, label in loader:
                image, label = image.to(device), label.to(device)
                # print(label)
                optimizer.zero_grad()
                pred = dr_grading_subnetwork(image)
                loss = F.cross_entropy(pred, label)
                loss.backward()
                optimizer.step()
                # print("pred: ",pred)
                # print("label: ",label)

                running_loss += loss.item() * image.size(0)
            epoch_loss = running_loss / len(loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            print("DONE TRAINING")

def test_accuracy(loader, model, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during inference
        print("Testing accuracy for grading model")
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            print("pred: ",predicted)
            print("label: ",labels)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))
        
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


    # Define loss function and optimizer

    print("here")
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match the input size expected by the model
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize image
        mean=[0.485, 0.456, 0.406],   # Mean and standard deviation values used for normalization
        std=[0.229, 0.224, 0.225]
    )
])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join("data_split", 'training'),transform=preprocess),
            batch_size=10, shuffle=True,num_workers=1)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join("data_split", 'testing'),transform=preprocess),
            batch_size=10,num_workers=1)

    num_epochs = 1  # Adjust the number of epochs as needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dr_grading_subnetwork.train()

    train_classification(loader =train_loader,num_epochs=num_epochs,dr_grading_subnetwork=dr_grading_subnetwork,lr=0.0001,device=device)

    # Optionally, save the trained model
    # torch.save(dr_grading_subnetwork, 'trained_model.pth')
    # dr_grading_subnetwork = torch.load('trained_model.pth')
    dr_grading_subnetwork.eval()

    test_accuracy(test_loader,dr_grading_subnetwork,device=device)

if __name__ == "__main__":
    main()
