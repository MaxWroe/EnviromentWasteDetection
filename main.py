"""
Preconditions :

Date: 03/06/1998
"""
import torchvision
from torchvision import transforms
from dataset_utilities import CocoDetection
import torch
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



#1. Carry out initial data preprocessing, we need to move all images in to folders (run annotations.py)
#2. Download images by running download.py


###########################CREATE DATASET#######################################
#Define path to images and training data
IMAGE_PATH = "data/images"
ANNOTATIONS_PATH = "data/annotations_comp3330.json"


#Create a Coco Detection Dataset for training
def get_transform(): #Define a transformation function that converts inputs to tensors for pytorch
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor()) #ToTensor (Add more below as needed)
    return torchvision.transforms.Compose(custom_transforms)
cocoDataset = CocoDetection(root = IMAGE_PATH,annotation = ANNOTATIONS_PATH, transforms= get_transform())

print('Number of samples : ',len(cocoDataset)) #Print the number of samples
image, label = cocoDataset[1] #Display an image example
img = transforms.ToPILImage()(image)
plt.imshow(img)
plt.show()

###########################CREATE DATA LOADEER#######################################
#Now that the dataset is established, set up a data loader
# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
data_loader = torch.utils.data.DataLoader(cocoDataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)


###########################RUN THE MODEL#######################################
# Select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# 2 classes; Only target class or background
num_classes = 2
num_epochs = 10
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

len_dataloader = len(data_loader)

for epoch in range(num_epochs):
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')