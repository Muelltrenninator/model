
import os
import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np

from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model_architecture import neural_network
from model_functions import train, save_model, load_model

train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    ])



model        = neural_network(5)
img_dir      = os.path.dirname(os.path.realpath(__file__)) + "/data/"
criterion    = nn.CrossEntropyLoss()
optimizer    = optim.Adam(model.parameters(), lr = 0.001)
train_data   = datasets.ImageFolder(root = img_dir, transform = train_transforms, allow_empty = True)
train_loader = DataLoader(train_data, batch_size = 32, shuffle = True, drop_last = True )


device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)
print(str(train_data))
print(str(train_loader))
print(device)

for i in range(1):
    train(train_loader = train_loader, model = model, num_epochs = 60 ,loss_fn = criterion, optimizer = optimizer)
    
    

save_model(model, "/home/julian_hack/Desktop/projects/muelltrenninator/trained_models/model_new.pth")





