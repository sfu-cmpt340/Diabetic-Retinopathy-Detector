# import torchvision.models as models
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import ImageFile
def worker_init_fn(worker_id):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Define transformations for training and validation sets
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


finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 5)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def evaluate_accuracy(data_loader, net, device):
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    print("here6")

    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=  1,
                      param_group=True):
    print("here2")

    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join("data", 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True,num_workers=4)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join("data", 'test'), transform=val_augs),
        batch_size=batch_size,num_workers=4)
    val_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join("data", 'val'), transform=val_augs),
        batch_size=batch_size,num_workers=4)
   
    print("here1")
    devices = d2l.try_all_gpus()
    if not devices:  # If no GPU is found, use CPU
        devices = [torch.device('cpu')]
    loss = nn.CrossEntropyLoss(reduction="none")
    print("here3")

    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    print("here4")

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
    print("here5")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = evaluate_accuracy(val_iter, finetune_net, device)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')



train_fine_tuning(finetune_net, 5e-5)