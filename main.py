import os
import sys

base_path = os.path.abspath('.')
sys.path.append(os.path.join(base_path, 'src'))

import torch
import torchvision
from torch import nn
import torchvision.models as models
import torch.optim as optim
from PIL import ImageFile
import main_fine_tuning as resnetModule
from torchvision.transforms import Compose, Resize, Normalize,ToTensor
import src.multiLabelClassifier as multiLabelClassifier
import src.Lesion_Detection_Segmentation as Lesion_Detection_Segmentation
import src.lesionSegmentationDataset as lesionSegmentationDataset
import src.DRGrading as DRGrading
from PIL import ImageFile
import src.alt as alt


ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__': 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dset_loaders,dsets,dset_sizes = alt.BASE_DR()
    
    ### BASE DR MODEL RESNET
    finetune_net = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 5)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    criterion = nn.CrossEntropyLoss()


    # Run the functions and save the best model in the function model_ft.
    optimizer_ft = optimizer = optim.Adam(params=finetune_net.parameters(), lr=0.001)
    finetune_net = resnetModule.train_model(finetune_net, criterion, optimizer_ft,resnetModule.exp_lr_scheduler,dset_loaders,dset_sizes,
                        num_epochs=35) 
    torch.save(finetune_net.state_dict(), './src/fine_tuned_resnet18_state_dict.pth')


  

    ## LESION DETECTION -- THIS USES THE BASE DR MODEL (RESNET)

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
lesion_detection_model.train(train_iter_detection,test_iter_detection,35) 
torch.save(lesion_detection_model.model.state_dict(), './src/lesion_detection_model_state_dict.pth')

checkpoint = torch.load('./src/lesion_detection_model_state_dict.pth')

# Load the model state_dict
lesion_detection_model.model.load_state_dict(checkpoint)


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

segmentation_dataset_test = lesionSegmentationDataset.LesionSegMask(images_path="1. Original Images/test", ground_truth_dir=ground_truth_dirs_test, root="./data_lesion_detection")
segmentation_dataset = lesionSegmentationDataset.LesionSegMask(images_path="1. Original Images/train", ground_truth_dir=ground_truth_dirs_train, root="./data_lesion_detection")



lesion_segmentation_module = Lesion_Detection_Segmentation.LesionSegmentationModule(feature_extractor=feature_extractor,model=lesion_detection_model.model,num_classes= 5)
segmentation_data_loader = torch.utils.data.DataLoader( segmentation_dataset, batch_size=1, shuffle=True)
segmentation_data_loader_test = torch.utils.data.DataLoader( segmentation_dataset_test, batch_size=1, shuffle=True)


criterion_segmentation = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
optimizer_segmentation = torch.optim.Adam(lesion_segmentation_module.parameters(), lr=0.01)


Lesion_Detection_Segmentation.train_mask_rcnn_epoch(lesion_segmentation_module,segmentation_data_loader,segmentation_data_loader_test,device,35) 

state_dict = torch.load('./src/lesion_segmentation_model_dict.pth')
lesion_segmentation_module.load_state_dict(state_dict)

##### FINAL DR GRADING -----------


# Instantiate DR grading sub-network
resNet18 = models.resnet18()
state_dict = torch.load('./src/fine_tuned_resnet18_state_dict.pth')

    # Load the modified state dict into the ResNet18 model
resNet18.fc = nn.Linear(resNet18.fc.in_features, 5)
resNet18.load_state_dict(state_dict, strict=False)
resNet18 = torch.nn.Sequential(*(list(resNet18.children())[:-1]))


dr_grading_subnetwork = DRGrading.DRGradingSubNetwork(resNet18, lesion_segmentation_module, 5)


DRGrading.train_classification(loader =dset_loaders["training"],num_epochs=35,dr_grading_subnetwork=dr_grading_subnetwork,lr=0.0001,device=device,test_loader=dset_loaders["testing"])

    # Optionally, save the trained model
dr_grading_subnetwork = torch.load('./src/DRGrading_trained_model.pth')
dr_grading_subnetwork.eval()

DRGrading.test_accuracy(dset_loaders["testing"],dr_grading_subnetwork,device=device)


