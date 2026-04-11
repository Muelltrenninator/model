
import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import models.model_architectures
import numpy as np
import models.registry
import models.model_template


from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train_functions import train_model
from utils import get_classes, fetch_data, calculate_class_weights, split_data, save_model
from ablation_report import generate_report
from configs.load_configs import configs
from models.registry import MODEL_REGISTRY

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    #fetch_data()
    split_data()
    device       = configs["device_train"]
    data_root = os.path.dirname(os.path.realpath(__file__))

    model = MODEL_REGISTRY[configs["model_architecture"]](5)

    print(data_root + configs["split_dir"] + "train")
    train_dataset = ImageFolder(root = data_root + configs["split_dir"] + "train/", transform= train_transforms, allow_empty = False)
    val_dataset   = ImageFolder(root = data_root + configs["split_dir"] + "val/", transform= val_transforms, allow_empty = False)
    test_dataset  = ImageFolder(root = data_root + configs["split_dir"] + "test/", transform= test_transforms, allow_empty = False)



    train_loader  = DataLoader(dataset = train_dataset, batch_size = configs["batch_size"], shuffle = True, num_workers = configs["num_workers"], pin_memory= True)
    val_loader    = DataLoader(dataset = val_dataset, batch_size = configs["batch_size"], shuffle = False, num_workers = configs["num_workers"] , pin_memory= True)
    test_loader   = DataLoader(dataset = test_dataset, batch_size = configs["batch_size"], shuffle = False, num_workers = configs["num_workers"], pin_memory = True)


    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)
    
    model.to(device)
    optimizer     = optim.Adam(model.parameters(), lr = model.learning_rate)
    train_model(train_loader = train_loader, val_loader = val_loader,  model = model, loss_fn = criterion, optimizer = optimizer, test_loader = test_loader)
    save_model(model, os.path.dirname(os.path.realpath(__file__)) +"/trained_models_large/model_transfer_presentation_ready.pth")


def compare_architectures():
    #fetch_data()
    split_data()
    device       = configs["device_train"]
    data_root = os.path.dirname(os.path.realpath(__file__)) + configs["split_dir"] 


    train_dataset = ImageFolder(root = data_root + "train/", transform= train_transforms, allow_empty = False)
    val_dataset   = ImageFolder(root = data_root + "val/", transform= val_transforms, allow_empty = False)
    test_dataset   = ImageFolder(root = data_root + "test/", transform= test_transforms, allow_empty = False)



    train_loader  = DataLoader(dataset = train_dataset, batch_size = configs["batch_size"], shuffle = True, num_workers = configs["num_workers"], pin_memory= True, drop_last = True)
    val_loader    = DataLoader(dataset = val_dataset, batch_size = configs["batch_size"], shuffle = False, num_workers = configs["num_workers"] , pin_memory= True, drop_last = True)
    test_loader   = DataLoader(dataset = test_dataset, batch_size = configs["batch_size"], shuffle = False, num_workers = configs["num_workers"], pin_memory = True, drop_last = True)


    weights = calculate_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device), label_smoothing = 0.1)

    
    model_results = {}
    # Train each model config specified in model_architectures and compare them 
    for model_name, model in MODEL_REGISTRY.items():
        model = model(5) 
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = model.learning_rate)
        _, loss, accuracy, f1_score =  train_model(train_loader = train_loader, val_loader = val_loader, model = model, loss_fn = criterion, optimizer = optimizer, test_loader = test_loader )
        model_results[model_name] = {"accuracy" : accuracy, "f1_score" : f1_score, "loss" : loss, "params_m" : model.get_num_params() }

    generate_report(model_results)





if __name__ == "__main__":
    main()