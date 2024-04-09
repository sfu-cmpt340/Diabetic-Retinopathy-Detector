import sys
import os

# Add the folder containing DRGrading to sys.path
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torchvision
from torch import nn
import torchvision.models as models
import torch.optim as optim
from PIL import ImageFile
from torchvision.transforms import Compose, Resize, Normalize,ToTensor
import src.multiLabelClassifier as multiLabelClassifier
import src.Lesion_Detection_Segmentation as Lesion_Detection_Segmentation
import src.lesionSegmentationDataset as lesionSegmentationDataset
import src.DRGrading as DRGrading
from PIL import ImageFile
import src.alt as alt
from torchvision.io import read_image
from torch.utils.data import TensorDataset, DataLoader


# Instantiate DR grading sub-network
label4_img = read_image('./sample_data/4.png').float() /255
label0_img = read_image('./sample_data/0.png').float() /255
label3_img = read_image('./sample_data/3.png').float() /255
label2_img = read_image('./sample_data/2.png').float() /255
label1_img = read_image('./sample_data/1.png').float() /255



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resNet18 = models.resnet18()
lesion_detection_model = Lesion_Detection_Segmentation.LesionDetectionModel(num_classes=5, learning_rate=1e-3,device=device)
feature_extractor = lesion_detection_model.get_feature_extractor()
lesion_segmentation_module = Lesion_Detection_Segmentation.LesionSegmentationModule(feature_extractor=feature_extractor,model=lesion_detection_model.model,num_classes= 5)
resNet18 = torch.nn.Sequential(*(list(resNet18.children())[:-1]))

dr_grading_subnetwork =  torch.load('./DRGrading_trained_model.pth')
# dr_grading_subnetwork.load_state_dict(dict)

X_tensor = torch.stack([label4_img, label0_img,label3_img,label2_img,label1_img])  
y_tensor = torch.tensor([4, 0,3,2,1], dtype=torch.long)  

dataset = TensorDataset(X_tensor, y_tensor)


batch_size = 5

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size)

dr_grading_subnetwork.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation during inference
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = dr_grading_subnetwork(images)
            _, predicted = torch.max(outputs, 1)
            print("pred: ",predicted)
            print("label: ",labels)



