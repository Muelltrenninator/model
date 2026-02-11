
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
from torch.utils.data import DataLoader, Dataset
from model_architecture import neural_network
from model_functions import train_model, save_model, load_model, calculate_class_weights
from sklearn.model_selection import train_test_split
from torchvision import models
from Subset import TransformedSubset


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def main():
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    img_dir_large = os.path.dirname(os.path.realpath(__file__)) + "/data_large_classifying/"
    base_data    = datasets.ImageFolder(root = img_dir_large, transform = None, allow_empty = True)
    indices = list(range(len(base_data)))
    train_idx, val_idx = train_test_split(indices, test_size = 0.3, train_size = 0.7, shuffle = True )


    model = None

    match sys.argv[1]:
        case "vit":
            pass
        case "pt":
            model = models.resnet50(weights = "ResNet50_Weights.DEFAULT")
            for param in model.parameters():
                param.requires_grad = False

            for param in model.fc.parameters():
                param.require_grad = True

            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, 5)
)
        
        case "test":
            model = neural_network(5)
        case _:
            print("supply a valid param [vit, pt, test]")

    train_dataset = TransformedSubset(base_data, train_idx, train_transforms)
    val_dataset = TransformedSubset(base_data, val_idx, val_transforms)
    



    train_loader  = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers= 8, pin_memory= True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers= 8, pin_memory= True)



    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)

    print(str(train_loader))

    
    print(device)

    model.to(device)
    optimizer     = optim.Adam(model.parameters(), lr = 0.001)
    train_model(train_loader = train_loader, val_loader = val_loader,  model = model, loss_fn = criterion, optimizer = optimizer)
    save_model(model, os.path.dirname(os.path.realpath(__file__)) +"/trained_models_large/model_transfer_newest.pth")


if __name__ == "__main__":
    main()