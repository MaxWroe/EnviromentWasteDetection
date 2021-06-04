"""
Preconditions :

Date: 03/06/1998
"""
import torchvision
from torchvision import transforms
from dataset_utilities import CocoDetection
import torch
import matplotlib.pyplot as plt


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
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)

# Select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)