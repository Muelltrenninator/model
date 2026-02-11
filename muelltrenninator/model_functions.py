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
from trash_classifier import trash_pre_detector
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    ])

classes = ("bio", "elektroschrott", "gelber_sack","papier", "restmuell")
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def calculate_class_weights(dataloader : DataLoader):
    
    counter = Counter()
    y = []
    for _, labels in dataloader:
        counter.update(labels.tolist())

    for class_label, count in counter.items():
        y.extend([class_label] * count)
    print(counter)
    print(y)
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
    print(f"Class Weights: {class_weights}")

    return class_weights





def get_current_versions():
    pass    # Optional




def train_model(train_loader : DataLoader, val_loader : DataLoader , model : neural_network, loss_fn , optimizer ) -> neural_network:
    """
    Trains the given model, until val loss doesn't shrink anymore

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
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float("inf")
    min_improve = 1e-3


    model.train()
    epoch = 0
    #for epoch in range(num_epochs):
    while(True):
        model.train()
# --- Training Phase --- #
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, data in enumerate(train_loader, 0):

            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).to(device)

            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted_train = outputs.max(1)
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
            train_correct += predicted_train.eq(labels).sum().item() # comparison sums up true values transform 1d tensor to number with item()
            if(i == 0):
                writer.add_image("Examples train", inputs[0])



# --- Validation Phase --- #
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():

            for i, data in enumerate(val_loader, 0 ):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                
                _, predicted_val = outputs.max(1)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                val_correct += predicted_val.eq(labels).sum().item()
                if(i == 0):
                    writer.add_image("Examples val", inputs[0])

        print(f"[+] epoch_train_loss: {train_loss/ train_total:.4f} epoch_train_acc : {100. * train_correct/ train_total:.4f}")
        print(f"[+] epoch_val_loss: {val_loss/ val_total:.4f} epoch_val_acc : {100. * val_correct/ val_total:.4}")
        writer.add_scalar("Loss / val", val_loss / val_total, epoch)
        writer.add_scalar("Accuracy / val", val_correct / val_total, epoch)

        
        writer.add_scalar("Loss / train", train_loss/ train_total, epoch)
        writer.add_scalar("Accuracy / train", train_correct / train_total, epoch)
        epoch += 1

        if val_loss < best_val_loss - min_improve: # Early Stopping
            best_val_loss = val_loss
            epochs_no_improve = 0
    
        else:
            epochs_no_improve += 1
    
        if epochs_no_improve >= patience:
            print("[+] Finished Training")
            writer.flush()
            print(f"[+] Created tensorboard summary at {SummaryWriter.get_logdir(writer)}")
            writer.close()
            return model
        


    
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
    loaded_model = neural_network
    loaded_model.load_state_dict(torch.load(model_path, weights_only= False))
    print("[+] Model loaded successfully")
    return loaded_model


def evalute_input(model : neural_network, image_path : str, image_transforms : transforms = val_transform, model_small : trash_pre_detector = None) -> dict:
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
    
    model_small : trash_pre_detector, optional
    
    Returns
    -------
    JSON string
        A class : probability dictionary sorted descending by probability the last item is a bool value based on the model_small
    """

    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    if (model_small != None):
        model_small = model_small.eval()
        output_small = model_small(image.to(device))
        _, predicted_small = output_small.max(1)
        print(output_small)

    output = model(image.to(device))
    output = output.to(device)
    print(output)
    probabilities = F.softmax(output, dim = 1)[0]
    probabilities = probabilities.to(device)
    print(probabilities)

    class_prob_pairs = {}
    i = 0
    for i in range(len(classes)):
        class_prob_pairs[classes[i]] = probabilities[i].item()

    sorted_class_prob_pairs = OrderedDict(sorted(class_prob_pairs.items(), key = lambda x: x[1], reverse = True)) # Sort dictionary descending by value
    print(class_prob_pairs)
    print(sorted_class_prob_pairs)

    if(model_small != None and predicted_small == 1):
        sorted_class_prob_pairs["is_trash"] = False
    
    else:
        sorted_class_prob_pairs["is_trash"] = True
    
    json_string = json.dumps(sorted_class_prob_pairs, indent = 4)

    return json_string