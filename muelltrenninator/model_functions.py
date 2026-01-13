import os
import torch
import torchvision
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import datetime
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from model_architecture import neural_network
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    )
    ])

classes = ("bio", "elektroschrott", "gelber_sack","papier", "restmuell")
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def get_current_versions():
    pass    # Optional




def train(train_loader : DataLoader, model : neural_network, num_epochs : int, loss_fn , optimizer, val_loader : DataLoader = None) -> neural_network:
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
    
    val_loader : DataLoader, optional


    
    Returns
    -------

    neural_network
        the trained model
    """

    timestamp =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.dirname(os.path.realpath(__file__)) + "/logs/"
    writer = SummaryWriter(log_dir + timestamp)
    train_losses = []
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        correct = 0
        total = 0


        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs.to(device)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * train_loader.batch_size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        writer.add_image("Test", inputs[0])
        print(f"[+] epoch_loss: {running_loss/ total:.4f} epoch_acc : {100. * correct/ total:.4f}")
        writer.add_scalar("Loss / train", running_loss/ total, epoch)
        writer.add_scalar("Accuracy / train", correct / total, epoch)
    # TODO implement val_loader if needed
    
    print("[+] Finished Training")
    writer.flush()
    print(f"[+] Created tensorboard summary at {SummaryWriter.get_logdir(writer)}")
    writer.close()
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


def evalute_input(model : neural_network, image_path : str, image_transforms : transforms = transform) -> dict:
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
    JSON string
        A class : probability dictionary sorted descending by probability
    """
    
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image.to(device))
    output = output.to(device)
    print(output)
    probabilities = F.softmax(output, dim = 1)[0].to(device)
    print(probabilities)

    class_prob_pairs = {}

    for i in range(len(classes)):
        class_prob_pairs[classes[i]] = probabilities[i].item()

    sorted_class_prob_pairs = OrderedDict(sorted(class_prob_pairs.items(), key = lambda x: x[1], reverse = True))
    print(class_prob_pairs)
    print(sorted_class_prob_pairs)

    json_string = json.dumps(sorted_class_prob_pairs, indent = 4)

    return json_string