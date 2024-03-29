# import torchvision.models as models
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torchvision import transforms


# Define transformations for training and validation sets


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
    torch.save(finetune_net.state_dict(), 'finetune_net.pth')

def train_lesion_detection(net, num_epochs, learning_rate,
                           device, param_group=True):
    print("Initializing training process...")

    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join("data_lesion_detection", 'train'), transform=train_augs),
        batch_size=128, shuffle=True,num_workers=0)
    
    val_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join("data_lesion_detection", 'val'), transform=val_augs),
        batch_size=128,num_workers=0)
    print("here 6")
    # Move the model to the specified device (GPU or CPU)
    net.to(device)

    # Loss function
    loss = nn.CrossEntropyLoss()  # Adjust based on your task; this is for classification

    # Optimizer setup
    if param_group:
        # Example: Set different learning rates for different parts of the model
        params_1x = [param for name, param in net.named_parameters()
                     if 'fc' not in name]  # Adjust if your model's layer names differ
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],  # Example adjustment
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    print("here 7")

    # Training loop
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_iter, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            trainer.zero_grad()

            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            trainer.step()

            running_loss += l.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_iter)}')

        # Validation accuracy (or other metric) evaluation
        accuracy = evaluate_accuracy(val_iter, net, device)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%')
   
    print("Training completed.")

# Example usage:
# Assuming lesion_detection_net is your lesion detection model
# and you have defined train_iter and val_iter DataLoaders for your dataset



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

lesion_detection_model =  torchvision.models.resnet18(weights=None)
lesion_detection_model.fc = nn.Linear(lesion_detection_model.fc.in_features, 2) #binary classification

fine_tuned_weights = torch.load('finetune_net.pth')
# Remove the weights for the final layer from this dictionary
# The name of the final layer's weights/bias may vary depending on your architecture
# For a typical ResNet, it's 'fc.weight' and 'fc.bias' for the fully connected layer
del fine_tuned_weights['fc.weight']
del fine_tuned_weights['fc.bias']

# train_fine_tuning(finetune_net, 5e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# finetune_net.load_state_dict(torch.load('finetune_net.pth'))
lesion_detection_model.load_state_dict(fine_tuned_weights, strict=False)

train_lesion_detection(lesion_detection_model, 1, 1e-4, device)
torch.save(lesion_detection_model.state_dict(), 'lesion_detection_model.pth')


