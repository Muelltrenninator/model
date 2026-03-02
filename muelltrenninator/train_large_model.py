
import os
import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from model_architecture import neural_network
from train_functions import train_model, save_model
from utils import calculate_class_weights

from torchvision import models
from utils import get_classes
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from configs.load_configs import configs

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

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def main():
    device       = configs["device"]
    data_root = os.path.dirname(os.path.realpath(__file__)) + configs["split_dir"] 
    classes = get_classes(data_dir = data_root + "/train/")

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
                nn.Linear(512, len(classes))
)
        
        case "test":
            model = neural_network(num_classes = len(classes))
        case _:
            print("supply a valid param [vit, pt, test]")

    train_dataset = ImageFolder(root = data_root + "train/", transform= train_transforms, allow_empty = False)
    val_dataset = ImageFolder(root = data_root + "val/", transform= val_transforms, allow_empty = False)
    test_dataset = ImageFolder(root = data_root + "test/", transform= test_transforms, allow_empty = False)



    train_loader  = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers= 8, pin_memory= True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers= 8, pin_memory= True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False, num_workers = 8, pin_memory = True)


    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)


    
    print(device)

    model.to(device)
    optimizer     = optim.Adam(model.parameters(), lr = configs["learning_rate"])
    train_model(train_loader = train_loader, val_loader = val_loader,  model = model, loss_fn = criterion, optimizer = optimizer)
    save_model(model, os.path.dirname(os.path.realpath(__file__)) +"/trained_models_large/model_transfer_test.pth")


if __name__ == "__main__":
    main()