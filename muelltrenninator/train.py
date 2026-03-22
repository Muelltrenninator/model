
import os
import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import models.model_architectures
import matplotlib.pyplot as plt
import numpy as np
import models.registry
import models.model_template

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from train_functions import train_model, save_model
from utils import calculate_class_weights

from torchvision import models
from utils import get_classes
from ablation_report import generate_report
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from configs.load_configs import configs
from models.registry import MODEL_REGISTRY

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
    if(configs["compare"] == True):
        compare_architectures()
    else:
        train_architecture()
        


def train_architecture():
    
    device       = configs["device_train"]
    data_root = os.path.dirname(os.path.realpath(__file__)) + configs["split_dir"] 
    classes = get_classes(data_dir = data_root + "/train/")

    model = MODEL_REGISTRY["test_nn"](5)

    train_dataset = ImageFolder(root = data_root + "train/", transform= train_transforms, allow_empty = False)
    val_dataset   = ImageFolder(root = data_root + "val/", transform= val_transforms, allow_empty = False)
    test_dataset   = ImageFolder(root = data_root + "test/", transform= test_transforms, allow_empty = False)



    train_loader  = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = configs["num_workers"], pin_memory= True)
    val_loader    = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers = configs["num_workers"] , pin_memory= True)
    test_loader   = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False, num_workers = configs["num_workers"], pin_memory = True)


    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)
    

    model = MODEL_REGISTRY["ResNet50.DEFAULT"](5)
    print(device)

    model.to(device)
    optimizer     = optim.Adam(model.parameters(), lr = configs["learning_rate"])
    train_model(train_loader = train_loader, val_loader = val_loader,  model = model, loss_fn = criterion, optimizer = optimizer, test_loader = test_loader)
    save_model(model, os.path.dirname(os.path.realpath(__file__)) +"/trained_models_large/model_transfer_current.pth")


def compare_architectures():
        
    device       = configs["device_train"]
    data_root = os.path.dirname(os.path.realpath(__file__)) + configs["split_dir"] 
    classes = get_classes(data_dir = data_root + "/train/")

    model = MODEL_REGISTRY["test_nn"](5)

    train_dataset = ImageFolder(root = data_root + "train/", transform= train_transforms, allow_empty = False)
    val_dataset   = ImageFolder(root = data_root + "val/", transform= val_transforms, allow_empty = False)
    test_dataset   = ImageFolder(root = data_root + "test/", transform= test_transforms, allow_empty = False)



    train_loader  = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = configs["num_workers"], pin_memory= True)
    val_loader    = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False, num_workers = configs["num_workers"] , pin_memory= True)
    test_loader   = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False, num_workers = configs["num_workers"], pin_memory = True)


    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)

    
    model_results = {}
    # Train each model config specified in model_architectures and compare them 
    for model_name, model in MODEL_REGISTRY.items():
        model = model(5) 
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = configs["learning_rate"])
        _, loss, accuracy, f1_score =  train_model(train_loader = train_loader, val_loader = val_loader, model = model, loss_fn = criterion, optimizer = optimizer, test_loader = test_loader )
        model_results[model_name] = {"accuracy" : accuracy, "f1_score" : f1_score, "loss" : loss, "params_m" : model.get_num_params() }

    generate_report(model_results)



    """
    print(model_results)
    label_location = np.arange(len(model_names))
    bar_width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for label, value in model_results.items():
        offset = bar_width * multiplier
        rects = ax.bar(label_location + offset, bar_width, label = label)
        ax.bar_label(rects, padding = 3)
        multiplier += 1

    ax.set_ylabel("values")
    ax.set_title("model statistics")
    ax.set_xticks(label_location + bar_width, model_names)
    ax.legend(loc = "upper left", ncols = len(model_names))
    ax.set_ylim(0, 8)

    plt.show()
"""



if __name__ == "__main__":
    main()