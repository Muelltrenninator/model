
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


def main():

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

    model         = neural_network(5)

    img_dir_large = os.path.dirname(os.path.realpath(__file__)) + "/data_large_classifying/"
    
    optimizer     = optim.Adam(model.parameters(), lr = 0.001)

    data_large    = datasets.ImageFolder(root = img_dir_large, transform = train_transforms, allow_empty = True)
    train_data_large, val_data_large = torch.utils.data.random_split(data_large, [0.7, 0.3], generator =  torch.Generator().manual_seed(42))

    train_loader_large  = DataLoader(dataset = train_data_large, batch_size = 32, shuffle = True)
    val_loader_large = DataLoader(dataset = val_data_large, batch_size = 32, shuffle = False)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)

    criterion     = nn.CrossEntropyLoss()

    print(data_large.imgs)
    print(len(train_loader_large))
    print(str(train_loader_large))
    print(device)

    val_loader_large = DataLoader(dataset = train_data_large, batch_size = 32, shuffle = False)
    weights = calculate_class_weights(train_loader_large)
    train_model(train_loader = train_loader_large, val_loader = val_loader_large,  model = model, num_epochs = 1000 ,loss_fn = criterion, optimizer = optimizer)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights))
    

    save_model(model, os.path.dirname(os.path.realpath(__file__)) +"/trained_models_large/model_latest.pth")





if __name__ == "__main__":
    main()