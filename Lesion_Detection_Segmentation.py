import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
import copy
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN 
from torch.optim import SGD

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


# Define Lesion Detection Module

class LesionDetectionModel():
    def __init__(self, num_classes, learning_rate=1e-3, device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Load your pretrained weights here
        fine_tuned_weights = torch.load('finetune_net.pth')
        del fine_tuned_weights['fc.weight']
        del fine_tuned_weights['fc.bias']
        self.model.load_state_dict(fine_tuned_weights, strict=False)
        self.model.to(self.device)

    def train(self, train_iter, test_iter, num_epochs=5, param_group=True):
        print("Training lesion detection")
        if param_group:
            params_1x = [param for name, param in self.model.named_parameters() if 'fc' not in name]
            optimizer = SGD([
                {'params': params_1x},
                {'params': self.model.fc.parameters(), 'lr': self.learning_rate * 10}
            ], lr=self.learning_rate, weight_decay=0.001)
        else:
            optimizer = SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)

        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        # self.net.to(self.device)
        best_acc = 0.0
        train_losses, val_accuracies = [], []
      

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_iter:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_iter.dataset)
            train_losses.append(epoch_loss)
            # Optionally implement validation accuracy calculation here
            val_accuracy = evaluate_multi_label_accuracy(test_iter, self.model, self.device)
            val_accuracies.append(val_accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Validation Multi-Label Accuracy: {val_accuracy:.4f}')
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())
            print(f"Best Val Accuracy: {best_acc:.4f}")
        print("Training completed.")
        self.model.load_state_dict(best_model_wts)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_epochs), train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.show()
        
        return self.model

    def forward(self, x):
        # Forward pass
        return self.model(x)

    def get_feature_extractor(self):
        # Returns the model without the fully connected layer for feature extraction
        modules = list(self.model.children())[:-1]
        feature_extractor = nn.Sequential(*modules).to(self.device)
        return feature_extractor


class CustomBackboneWithFPN(nn.Module):
    def __init__(self, feature_extractor):
        super(CustomBackboneWithFPN, self).__init__()
        self.feature_extractor = feature_extractor
        # Assuming the feature extractor outputs feature maps of 256 channels.
        # This value should be changed according to your feature extractor's actual output channels.
        self.out_channels = 512
    
    def forward(self, x):
        # print(x)
        return self.feature_extractor(x)

# Define Lesion Segmentation Module
class LesionSegmentationModule(nn.Module):
    def __init__(self, feature_extractor,num_classes=5,trainable_layers=5,):
        # Load MaskRCNN model with ResNet50 backbone and FPN
        # self.mask_rcnn_model = detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        super(LesionSegmentationModule, self).__init__()
        # Create the custom FPN backbone from a pre-trained ResNet model
        backbone = CustomBackboneWithFPN(feature_extractor)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        
        # Initialize the MaskRCNN model with the custom backbone
        self.mask_rcnn_model = MaskRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
        
        for param in backbone.parameters():
            param.requires_grad = False
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("During training, targets must be provided.")
        output = self.mask_rcnn_model(images, targets)
        return output


def train_lesion_segmentation(num_epochs, optimizer_segmentation, lesion_segmentation_module, train_loader,device):
    lesion_segmentation_module.to(device)
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
    epoch_losses = []  # To store sum of losses for each epoch
    print("Training lesion segmentation")
    for epoch in range(num_epochs):
        for images, targets in train_loader:  # Assuming targets now include boxes, labels, and masks
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print("Images shape:", [image.shape for image in images])  # Print shapes of images

          
            optimizer_segmentation.zero_grad()
            loss_dict = lesion_segmentation_module.mask_rcnn_model(images,targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer_segmentation.step()
            
        print(epoch,'loss:', losses.item())
        epoch_loss = losses / len(train_loader)
        epoch_losses.append(epoch_loss.item())  # Append epoch loss
        
        # If this epoch yields the best loss, deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Load best model weights
    lesion_segmentation_module.load_state_dict(best_model_wts)
    torch.save(lesion_segmentation_module.state_dict(), 'lesion_segmentation_model_dict.pth')
    
    # Plot training loss
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs for Lesion Segmentation')
    plt.legend()
    plt.show()

    return lesion_segmentation_module  # Return the model with the best weights

def custom_collate_fn(batch):
    batch_images = [item[0] for item in batch]  # Extract images
    batch_targets = [item[1] for item in batch]  # Extract targets
    
    batched_images = default_collate(batch_images)  # Use PyTorch's default_collate to batch images
    
    # No need to batch targets as they should already be in the correct format (list of dictionaries)
    # print(batch_targets)
    return batched_images, batch_targets
# # Instantiate Lesion Detection Module and Lesion Segmentation Module
# num_classes_detection = 2  # Assuming binary classification (lesion vs. non-lesion)
# num_classes_segmentation = 1  # Assuming single class for lesion segmentation
# # lesion_detection_module = LesionDetectionModule(num_classes_detection)
# lesion_segmentation_module = LesionSegmentationModule(num_classes_segmentation)

# # Define loss function and optimizer
# criterion_detection = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for multi-label classification
# criterion_segmentation = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
# optimizer_detection = torch.optim.Adam(lesion_detection_module.parameters(), lr=0.001)
# optimizer_segmentation = torch.optim.Adam(lesion_segmentation_module.parameters(), lr=0.001)

# # Define dataset and data loader
# num_samples = 1000  # Define the number of samples in your dataset
# batch_size = 32  # Define your desired batch size
# # dataset = PlaceholderDataset(num_samples=num_samples)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Training loop
# num_epochs = 10  # Define the number of epochs







