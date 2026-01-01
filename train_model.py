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
from model_architecture import neural_network
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    )
    ])

classes = ("bio", "elektroschrott", "gelber_sack","papier", "restmuell")

def get_current_versions():
    pass    # Optional mal sehen obs gebraucht wird 

def create_new_model():
    model        = neural_network(5)
    img_dir      = os.path.dirname(os.path.realpath(__file__)) + "/data/"
    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters(), lr = 0.001)
    data         = datasets.ImageFolder(root = img_dir, transform = transform, allow_empty = True)
    train_loader = DataLoader(data, batch_size = 4, shuffle = True, drop_last = True )
    classes      = ("bio", "elektroschrott", "gelber_sack","papier", "restmuell")
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(str(model))
    print(f"[i] Using device {device}")

    model.to(device)


def train(dataloader : DataLoader, model : neural_network, loss_fn , optimizer) -> neural_network:
    """
    Trains the given model.

    Parameters
    ----------
    dataloader : DataLoader
        The Dataloader that should be used for parsing and formatting the training data
    
    model : neural_network
        The model that should be trained
    
    loss_fn : 
        The function for evaluating the loss
    
    optimizer :
        The optimizer used during training
    
    Returns
    -------

    neural_network
        the trained model
    """

    for epoch in range(1):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            np_arr = outputs.detach().cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            print(f"[+] Input: {inputs} Output: {np_arr}")
            if i % 2000 == 1999:
            
                print(f"[i] [{epoch +1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
    
    print("[+] Finished Training")
    return model



def save_model(model : neural_network, model_path : str):
    """
    Saves a supplied model at the given path

    Parameters
    ----------
    model : neural_network
        The model that should be saved
    
    model_path : str
        The path the model should be saved at
    """

    torch.save(model.state_dict(), model_path)
    print(f"[+] Model saved to path: {model_path}")


def load_model(model_path : str) -> neural_network:
    """
    Loads a saved model

    Parameters
    ----------
    model_path : str
        Path of the model to be loaded
    
    Returns
    -------
    neural_network
        The loaded model
    """
    
    loaded_model = neural_network()
    loaded_model.load_state_dict(torch.load(model_path, weights_only= False))
    print("[+] Model loaded successfully")
    return loaded_model

def evalute_input(model : neural_network, image_path : str, image_transforms : transforms = transform) -> str:
    """
    Passes the image to the model

    Parameters
    ----------
    model : neural_network
        The model used for evaluating the image
    
    image_path : str
        The path of the image to be evaluated
    
    image_transforms : transforms, optional
        Transformations that are applied before evaluating
    
    Returns
    -------
    string
        The predicted class name
    """

    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    
    _, predicted = torch.max(output.data, 1)

    return classes[predicted.item()]
