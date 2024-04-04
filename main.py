from __future__ import print_function, division
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
import main_fine_tuning as resnetModule
from torchvision.transforms import Compose, Resize, Normalize,ToTensor
from torch.utils.data import DataLoader, WeightedRandomSampler
import multiLabelClassifier
import Lesion_Detection_Segmentation
import lesionSegmentationDataset
import DRGrading
from PIL import ImageFile, Image, UnidentifiedImageError
import alt


ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__': 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dset_loaders,dsets,dset_sizes = alt.BASE_DR()
    
    ### BASE DR MODEL RESNET
    finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 5)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    criterion = nn.CrossEntropyLoss()


    # Run the functions and save the best model in the function model_ft.
    optimizer_ft = optimizer = optim.Adam(params=finetune_net.parameters(), lr=0.001)
    finetune_net = resnetModule.train_model(finetune_net, criterion, optimizer_ft,resnetModule.exp_lr_scheduler,dset_loaders,dset_sizes,
                        num_epochs=1) 
    torch.save(finetune_net.state_dict(), 'fine_tuned_resnet101.pth')

    # data_transforms = {
    # 'training': transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize images to 224x224
    #     transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    #     transforms.RandomRotation(20),  # Randomly rotate images by 20 degrees
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    # ]),
    # 'testing': transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize images to 224x224
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    # ]),
# }
   

  

    ## LESION SEGMENTATION -- THIS USES THE BASE DR MODEL (RESNET)

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
train_dataset_detection = multiLabelClassifier.MultiLabelLesionDataset(images_dir='./data_lesion_detection/1. Original Images/train',
                                  ground_truth_dirs=ground_truth_dirs_train,
                                  transform=transform)
test_dataset_detection = multiLabelClassifier.MultiLabelLesionDataset(images_dir='./data_lesion_detection/1. Original Images/test',
                                  ground_truth_dirs=ground_truth_dirs_test,
                                  transform=transform)
train_iter_detection = torch.utils.data.DataLoader(train_dataset_detection, batch_size=4, shuffle=True)
test_iter_detection = torch.utils.data.DataLoader(test_dataset_detection, batch_size=4, shuffle=True)




lesion_detection_model = Lesion_Detection_Segmentation.LesionDetectionModel(num_classes=5, learning_rate=1e-3,device=device)
lesion_detection_model.train(train_iter_detection,test_iter_detection, 1) 
torch.save(lesion_detection_model.model.state_dict(), 'lesion_detection_model.pth')

# Retrieve the feature extractor
checkpoint = torch.load('lesion_detection_model.pth')

# Load the model state_dict
lesion_detection_model.model.load_state_dict(checkpoint)

# Now you can access the feature extractor
feature_extractor = lesion_detection_model.get_feature_extractor()


#### LESION SEGMENTATION -----------

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


lesion_segmentation_module = Lesion_Detection_Segmentation.LesionSegmentationModule(feature_extractor=feature_extractor,num_classes= 5)
segmentation_data_loader = torch.utils.data.DataLoader( segmentation_dataset, batch_size=4, shuffle=True,collate_fn=Lesion_Detection_Segmentation.custom_collate_fn)

criterion_segmentation = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
optimizer_segmentation = torch.optim.Adam(lesion_segmentation_module.parameters(), lr=0.01)

Lesion_Detection_Segmentation.train_lesion_segmentation(1,optimizer_segmentation,lesion_segmentation_module,segmentation_data_loader,device)

##### FINAL DR GRADING -----------


# Instantiate DR grading sub-network
resNet18 = models.resnet18()
state_dict = torch.load('fine_tuned_resnet101.pth')

    # Remove fully connected layer weights from the state dict

    # Load the modified state dict into the ResNet18 model
resNet18.fc = nn.Linear(resNet18.fc.in_features, 5)
resNet18.load_state_dict(state_dict, strict=False)
resNet18 = torch.nn.Sequential(*(list(resNet18.children())[:-1]))

lesion_segmentation_module = torch.load('lesion_segmentation_model.pth')

dr_grading_subnetwork = DRGrading.DRGradingSubNetwork(resNet18, lesion_segmentation_module, 5)


# Define loss function and optimizer

print("here")

DRGrading.train_classification(loader =dset_loaders["training"],num_epochs=1,dr_grading_subnetwork=dr_grading_subnetwork,lr=0.0001,device=device)

    # Optionally, save the trained model
torch.save(dr_grading_subnetwork, 'DRGrading_trained_model.pth')
dr_grading_subnetwork = torch.load('DRGrading_trained_model.pth')
dr_grading_subnetwork.eval()

DRGrading.test_accuracy(dset_loaders["testing"],dr_grading_subnetwork,device=device)


