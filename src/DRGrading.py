import torch
import os
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from PIL import ImageFile
import matplotlib.pyplot as plt
import pickle
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRGradingSubNetwork(nn.Module):
    """
    Sub-network for Diabetic Retinopathy grading.

    Args:
        resnet18 (torchvision.models.resnet.ResNet): Pretrained ResNet18 model.
        lesion_segmentation_module (torch.nn.Module): Lesion segmentation module.
        num_classes (int): Number of output classes.

    Attributes:
        resnet18 (torchvision.models.resnet.ResNet): Pretrained ResNet18 model.
        lesion_segmentation_module (torch.nn.Module): Lesion segmentation module.
        f1 (torch.nn.Linear): Fully connected layer for classification.
    """
    def __init__(self, resnet18, lesion_segmentation_module, num_classes):
        super(DRGradingSubNetwork, self).__init__()
        self.resnet18 = resnet18
        self.lesion_segmentation_module = lesion_segmentation_module
        self.f1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        
        features_resnet18 = self.resnet18(x)
        self.lesion_segmentation_module.eval()
        features_lesion = self.lesion_segmentation_module.backbone(x)['pool']
        features_lesion = torch.nn.functional.adaptive_avg_pool2d(features_lesion, (1, 1)).view(features_lesion.size(0), -1)

        features_lesion = features_lesion.squeeze()
        features_resnet18 = features_resnet18.squeeze()
        # print(features_resnet18.shape)
        # print(features_lesion.shape)   


        # print(features_lesion.shape,features_resnet18.shape)
        # print("combined:")
        combined_features = torch.cat((features_resnet18, features_lesion), dim=1)
        # print(combined_features)
        self.f1.train()
        logits = self.f1(combined_features)
    
        output = F.softmax(logits, dim=1)
        # print(output)
        return output

def train_classification( loader,num_epochs, dr_grading_subnetwork,lr,device,test_loader):
    """
    Train the DR grading sub-network.

    Args:
        loader (torch.utils.data.DataLoader): Training data loader.
        num_epochs (int): Number of epochs for training.
        dr_grading_subnetwork (DRGradingSubNetwork): DR grading sub-network.
        lr (float): Learning rate.
        device (torch.device): Device (GPU or CPU) for training.
        test_loader (torch.utils.data.DataLoader): Testing data loader.
    """        
    print("----------TRAINING FINAL DR GRADING----------")
    dr_grading_subnetwork.to(device)
    best_acc = 0.0  # Initialize the best test accuracy
    best_model_wts = copy.deepcopy(dr_grading_subnetwork)
    train_losses, test_accuracies = [], []  # Lists to store metrics for plotting
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
        train_losses.append(epoch_loss)
        dr_grading_subnetwork.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = dr_grading_subnetwork(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Update best model if test accuracy improved
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(dr_grading_subnetwork)

    print(f'Best Test Accuracy: {best_acc:.2f}%')
    torch.save(best_model_wts, './src/DRGrading_trained_model.pth')
    history = {
    'train_loss_history': train_losses,
    'test_acc_history': test_accuracies}

    with open('dr_grading_training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs for DR Grading')
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs for DR Grading')
    plt.legend()

    plt.tight_layout()
    plt.savefig('accuracy_over_time_dr.png')
    # plt.show()
    # plt.close()
    print("----------TRAINING FINAL DR GRADING COMPLETED----------")


def test_accuracy(loader, model, device):
    """
    Calculate the accuracy of the model on the test dataset.

    Args:
        loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        model (torch.nn.Module): Model to be evaluated.
        device (torch.device): Device (GPU or CPU) for evaluation.
    """
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
            # print("pred: ",predicted)
            # print("label: ",labels)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))