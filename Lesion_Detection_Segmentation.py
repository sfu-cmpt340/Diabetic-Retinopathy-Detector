

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models.detection as detection
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN 
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
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


class LesionDetectionModel:
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            # Optionally implement validation accuracy calculation here
            val_accuracy = evaluate_multi_label_accuracy(test_iter, self.model, self.device)
            print(f'Validation Multi-Label Accuracy: {val_accuracy:.4f}')
        print("Training completed.")

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







