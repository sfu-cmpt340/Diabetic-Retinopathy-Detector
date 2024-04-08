# load the necessary libraries
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
import copy
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN 
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.ops import misc as misc_nn_ops
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# Function to evaluate multi-label accuracy
def evaluate_multi_label_accuracy(data_loader, net, device, threshold=0.5):
    """
    Function to evaluate multi-label accuracy of a model.

    Args:
        data_loader: DataLoader for the dataset.
        net: Model to be evaluated.
        device: Device to perform computation on (CPU or GPU).
        threshold: Threshold for classification decision.

    Returns:
        accuracy: Multi-label accuracy of the model.
    """
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
    """
    Class for Lesion Detection Model.

    Args:
        num_classes: Number of classes.
        learning_rate: Learning rate for training.
        device: Device to perform computation on (CPU or GPU).

    """
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
        # Prepare optimizer and loss function
        print("----------TRAINING AND TESTING LESION DETECTION----------")
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
      
        # Training loop
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

            # Calculate average training loss for the epoch
            epoch_loss = running_loss / len(train_iter.dataset)
            train_losses.append(epoch_loss)
            # Optionally implement validation accuracy calculation here
            val_accuracy = evaluate_multi_label_accuracy(test_iter, self.model, self.device)
            val_accuracies.append(val_accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Validation Multi-Label Accuracy: {val_accuracy:.4f}')
            # Update best model if validation accuracy improves
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())
            print(f"Best Val Accuracy: {best_acc:.4f}")
        self.model.load_state_dict(best_model_wts)

        print("----------TRAINING AND TESTING LESION DETECTION COMPLETED----------")
        # Plot training curves
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
    def __init__(self, feature_extractor,model,pretrained=True,trainable_layers=3):
        super(CustomBackboneWithFPN, self).__init__()
        self.feature_extractor = feature_extractor
        self.model =   models.resnet18(weights=None)
        d = torch.load('fine_tuned_resnet18_state_dict.pth')
        num_ftrs = model.fc.in_features  # Get the number of input features for the fully connected layer

# Replace the final fully connected layer
        self.model.fc = nn.Linear(num_ftrs, 5)
        self.model.load_state_dict(d)
        self.out_channels = 512
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            self.model, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    
    
    def forward(self, x):        
        # Pass feature maps through the FPN
        x = self.body(x)
        x = self.fpn(x)
        
        return x

# Define Lesion Segmentation Module
class LesionSegmentationModule(nn.Module):
    """
    Lesion Segmentation Module.

    Args:
        feature_extractor: Feature extractor model.
        model: Pretrained model.
        num_classes: Number of classes.
        trainable_layers: Number of trainable layers.

    """
    def __init__(self, feature_extractor,model,num_classes=5,trainable_layers=5,):
        # Load MaskRCNN model with ResNet18 backbone and FPN
        super(LesionSegmentationModule, self).__init__()
        # Create the custom FPN backbone from a pre-trained ResNet model
        anchor_sizes = ((8, 16, 32, 64, 128), )
        aspect_ratios = [(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))]
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        # anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
        
        # Initialize the MaskRCNN model with the custom backbone
        self.mask_rcnn_model = MaskRCNN(CustomBackboneWithFPN(feature_extractor,model), num_classes, 
        # rpn_anchor_generator=rpn_anchor_generator
        )
        
        for param in self.mask_rcnn_model.backbone.parameters():
            param.requires_grad = False
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("During training, targets must be provided.")
        output = self.mask_rcnn_model(images, targets)
        # print(output)
        return output

def train_mask_rcnn_epoch(lesion_segmentation_module, loader,test_loader,device,num_epochs):
    print("-------------TRAINING & TESTING LESION SEGMENTATION-------------")
    optimizer = SGD(lesion_segmentation_module.mask_rcnn_model.parameters(), 0.001, 0.95, weight_decay=0.00001)
    lesion_segmentation_module.mask_rcnn_model.train()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
    # for epoch in range(num_epochs):
    #     process_bar = tqdm(enumerate(loader), total=len(loader))
    #     for batch_cnt, batch in process_bar:
    #         image, label = batch
    #         image = image.permute(0, 3, 1, 2) 
    #         # print(label)
    #         for k, v in label.items():
    #             label[k] = v.squeeze()
    #         image = image.to(device)
    #         for k, v in label.items():
    #             if isinstance(v, torch.Tensor):
    #                 label[k] = label[k].to(device)
    #         # print(label['boxes'].shape)
    #         optimizer.zero_grad()
    #         net_out = lesion_segmentation_module(image, [label])
    #         # print(net_out)
    #         loss = 0
    #         for i in net_out.values():
    #             loss += i  
    #         net_out['loss_sum'] = loss
    #         loss.backward()
    #         optimizer.step()
    #         process_bar.set_description_str(f'loss: {float(loss):.3f}', True)
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
    print("-------------TRAINING & TESTING LESION SEGMENTATION COMPLETED-------------")
    # lesion_segmentation_module.load_state_dict(best_model_wts)
    # torch.save(lesion_segmentation_module.state_dict(), 'lesion_segmentation_model_dict.pth')
    # lesion_segmentation_module.eval()  # Set model to evaluation mode
    s = torch.load('lesion_segmentation_model_dict.pth')
    lesion_segmentation_module.load_state_dict(s)
    test_segmentation(test_loader,lesion_segmentation_module,device)
    
# Train Lesion Segmentation        
def train_lesion_segmentation(num_epochs, optimizer_segmentation, lesion_segmentation_module, train_loader,device,test_loader):
    """
    Function to train Lesion Segmentation.

    Args:
        num_epochs: Number of epochs for training.
        optimizer_segmentation: Optimizer for segmentation.
        lesion_segmentation_module: Lesion segmentation module.
        train_loader: DataLoader for training dataset.
        device: Device to perform computation on (CPU or GPU).
        test_loader: DataLoader for testing dataset.

    Returns:
        Trained lesion segmentation module.
    """
    lesion_segmentation_module.to(device)
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
    epoch_losses = []  # To store sum of losses for each epoch
    print("Training lesion segmentation")
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0  # In
        for images, targets in train_loader:  # Assuming targets now include boxes, labels, and masks
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print("Images shape:", [image.shape for image in images])  # Print shapes of images

          
            optimizer_segmentation.zero_grad()
            loss_dict = lesion_segmentation_module.mask_rcnn_model(images,targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer_segmentation.step()
            running_loss += losses.item() * len(images)
            
        epoch_loss = running_loss / len(train_loader.dataset)  # Calculate average loss for the epoch
        epoch_losses.append(epoch_loss)
        
        # Update best model if current epoch's loss is lower
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(lesion_segmentation_module.state_dict())
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    # Load best model weights
    lesion_segmentation_module.load_state_dict(best_model_wts)
    torch.save(lesion_segmentation_module.state_dict(), 'lesion_segmentation_model_dict.pth')
    lesion_segmentation_module.eval()  # Set model to evaluation mode
    # all_pred, all_target_masks, avg_loss = predict_masks_and_calculate_loss(lesion_segmentation_module=lesion_segmentation_module,test_loader=test_loader,device = device)
    print("Avg loss on testing dataset", avg_loss)
    
    # Plot training loss
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs for Lesion Segmentation')
    plt.legend()
    plt.show()

    return lesion_segmentation_module  # Return the model with the best weights


# Test Segmentation
def test_segmentation(test_loader,model,device):
    """
    Function to test segmentation.

    Args:
        test_loader: DataLoader for testing dataset.
        model: Lesion segmentation model.
        device: Device to perform computation on (CPU or GPU).
    """
    model.eval()  # Set the model to evaluation mode
    total_iou = 0.0
    num_samples = 0

    for images, targets in test_loader:
        print("this is shape")
        print(images.shape)
        with torch.no_grad():
            images = images.permute(0, 3, 1, 2).to(device)
            predictions = model(images)  # Get model predictions
            # Calculate metrics here, e.g., IoU
            for i, prediction in enumerate(predictions):
                pred_mask = prediction['masks']  # Assuming binary mask [N, 1, H, W]
                true_mask = targets['masks']
                print("Predicted mask shape:", pred_mask.shape)
                print("True mask shape:", true_mask.shape)
                print(pred_mask)
                print(true_mask)
                
                iou = calculate_iou(pred_mask, true_mask)  
                total_iou += iou
                num_samples += 1

    average_iou = total_iou / num_samples
    print(f"Average IoU: {average_iou:.4f}")

# Calculate IoU
def calculate_iou(pred_mask, true_mask):
    """
    Function to calculate Intersection over Union (IoU).

    Args:
        pred_mask: Predicted mask.
        true_mask: Ground truth mask.

    Returns:
        IoU score.
    """
    # Ensure the mask tensors are boolean (binary)
    pred_mask_bool = pred_mask.squeeze(1) > 0.5  # Threshold for prediction
    true_mask_bool = true_mask.squeeze(1) > 0.5 if true_mask.dim() == 4 else true_mask > 0.5

    # Intersection: Element-wise AND
    intersection = (pred_mask_bool & true_mask_bool).float().sum((1, 2))  # Shape: [N]

    # Union: Element-wise OR
    union = (pred_mask_bool | true_mask_bool).float().sum((1, 2))  # Shape: [N]

    # Compute IoU and handle division by 0
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero

    # Return the mean IoU score for the batch
    return iou.mean().item()
