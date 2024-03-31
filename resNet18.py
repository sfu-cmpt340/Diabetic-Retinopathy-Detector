# import torchvision.models as models
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torchvision import transforms
from torchvision.transforms import Compose, Resize, Normalize,ToTensor
import multiLabelClassifier
import Lesion_Detection_Segmentation
import lesionSegmentationDataset


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

def evaluate_multi_label_accuracy(data_loader, net, device, threshold=0.5):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(net(images))  # Apply sigmoid to convert logits to probabilities
            predicted = (outputs > threshold).float()  # Apply threshold
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

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
                           device,train_iter, test_iter, param_group=True):
    print("Initializing training process...")

    # Move the model to the specified device (GPU or CPU)
    net.to(device)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()  # Adjust based on your task; this is for classification

    #  It creates a differential learning rate strategy. 
    # This is common when you're fine-tuning a pre-trained 
    # model and want to update the pre-trained weights slowly while 
    # allowing more substantial updates to the newly added layers
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
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(num_epochs):
        net.train()  # Set the network to training mode
        running_loss = 0.0
        for images, labels in train_iter:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = net(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_iter.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Evaluate after each epoch
        val_accuracy = evaluate_multi_label_accuracy(test_iter, net, device)
        print(f'Validation Multi-Label Accuracy: {val_accuracy:.4f}')
    print("Training completed.")

#using it



def train_lesion_segmentation(num_epochs, optimizer_segmentation, lesion_segmentation_model, train_loader,device):
    lesion_segmentation_module.to(device)
    for epoch in range(num_epochs):
        for images, targets in train_loader:  # Assuming targets now include boxes, labels, and masks
            images = list(image.to(device) for image in images)
            # targets = {k: v.to(device) for k, v in targets.items()}
          
            optimizer_segmentation.zero_grad()
            print("herererere")

            print(type(targets["boxes"]))
            print((targets["boxes"]))

            loss_dict = lesion_segmentation_model(images, targets)
            print("herererere")
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer_segmentation.step()
            
            print(epoch,'loss:', losses.item())
    
        #     images = list(image.to(device) for image in images)
        #     # print(targets)
        #     # break
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Ensure targets are on the correct device

        #     loss_dict = lesion_segmentation_module(images, targets)
        #     losses = sum(loss for loss in loss_dict.values())

        #     optimizer_segmentation.zero_grad()
        #     losses.backward()
        #     optimizer_segmentation.step()

        #     print(f"Epoch {epoch+1}, Loss: {losses.item()}")
        for images, targets in train_loader:
            print(type(targets))  # Should be list or dict
            if isinstance(targets, dict):
                print(targets.keys())  # Should show 'boxes', 'labels', 'masks'
                print(type(targets["boxes"]))
            elif isinstance(targets, list):
                print(type(targets[0]))  # Should be dict
                print(targets[0].keys())  # Should show 'boxes', 'labels', 'masks'
            break






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

ground_truth_dirs_train = {
     'Microaneurysms': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/1. Microaneurysms', 'MA'),
    'Haemorrhages': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/2. Haemorrhages', 'HE'),
    'Hard_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/3. Hard Exudates', 'EX'),
    'Soft_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/4. Soft Exudates', 'SE'),
    'Optic_Disc': ('./data_lesion_detection/2. All Segmentation Groundtruths/train/5. Optic Disc', 'OD')}

ground_truth_dirs_test= {
     'Microaneurysms': ('./data_lesion_detection/2. All Segmentation Groundtruths/test/1. Microaneurysms', 'MA'),
    'Haemorrhages': ('./data_lesion_detection/2. All Segmentation Groundtruths/test/2. Haemorrhages', 'HE'),
    'Hard_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/test/3. Hard Exudates', 'EX'),
    'Soft_Exudates': ('./data_lesion_detection/2. All Segmentation Groundtruths/test/4. Soft Exudates', 'SE'),
    'Optic_Disc': ('./data_lesion_detection/2. All Segmentation Groundtruths/test/5. Optic Disc', 'OD')}

transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset_detection_segmentation = multiLabelClassifier.MultiLabelLesionDataset(images_dir='./data_lesion_detection/1. Original Images/train',
                                  ground_truth_dirs=ground_truth_dirs_train,
                                  transform=transform)
test_dataset_detection_segmentation = multiLabelClassifier.MultiLabelLesionDataset(images_dir='./data_lesion_detection/1. Original Images/test',
                                  ground_truth_dirs=ground_truth_dirs_test,
                                  transform=transform)
train_iter_detection_segmentation = torch.utils.data.DataLoader(train_dataset_detection_segmentation, batch_size=4, shuffle=True)
test_iter_detection_segmentation = torch.utils.data.DataLoader(test_dataset_detection_segmentation, batch_size=4, shuffle=True)


# lesion_detection_model =  torchvision.models.resnet18(weights=None)
# lesion_detection_model.fc = nn.Linear(lesion_detection_model.fc.in_features, 5) #multi labelbinary classification

# fine_tuned_weights = torch.load('finetune_net.pth')
# Remove the weights for the final layer from this dictionary
# The name of the final layer's weights/bias may vary depending on your architecture
# For a typical ResNet, it's 'fc.weight' and 'fc.bias' for the fully connected layer
# del fine_tuned_weights['fc.weight']
# del fine_tuned_weights['fc.bias']

# train_fine_tuning(finetune_net, 5e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# finetune_net.load_state_dict(torch.load('finetune_net.pth'))
# lesion_detection_model.load_state_dict(fine_tuned_weights, strict=False)

# train_lesion_detection(lesion_detection_model, 5, 1e-3, device,train_iter_detection_segmentation,test_iter_detection_segmentation) 
# torch.save(lesion_detection_model.state_dict(), 'lesion_detection_model.pth')

# lesion_detection_model = torch.load('lesion_detection_model.pth')







lesion_detection_model = Lesion_Detection_Segmentation.LesionDetectionModel(num_classes=5, learning_rate=1e-3,device=device)
# lesion_detection_model.train(train_iter_detection_segmentation, test_iter_detection_segmentation, num_epochs=5)
# torch.save(lesion_detection_model.model.state_dict(), 'lesion_detection_model.pth')

# Retrieve the feature extractor
checkpoint = torch.load('lesion_detection_model.pth')

# Assuming 'state_dict' is the key for the model's state_dict in the checkpoint

# Load the model state_dict
lesion_detection_model.model.load_state_dict(checkpoint)

# Now you can access the feature extractor
feature_extractor = lesion_detection_model.get_feature_extractor()
print(feature_extractor)


image_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mask transform only converts mask to tensor without normalization
mask_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
])

segmentation_dataset = lesionSegmentationDataset.MultiLesionSegmentationDataset(images_dir='./data_lesion_detection/1. Original Images/train',
                                         masks_dir=ground_truth_dirs_train,
                                         image_transform=image_transforms,
                                         mask_transform=mask_transforms)
# segmentation_dataset = p.LesionSegMask(root='./data_lesion_detection/')
print(segmentation_dataset)

segmentation_data_loader = torch.utils.data.DataLoader(( segmentation_dataset), batch_size=4, shuffle=True)

lesion_segmentation_module = Lesion_Detection_Segmentation.LesionSegmentationModule(feature_extractor=feature_extractor,num_classes= 5)

criterion_segmentation = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
optimizer_segmentation = torch.optim.Adam(lesion_segmentation_module.parameters(), lr=0.001)

train_lesion_segmentation(1,optimizer_segmentation,lesion_segmentation_module.mask_rcnn_model,segmentation_data_loader,device)





