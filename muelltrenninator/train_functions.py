import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import datetime
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from utils import get_classes
from torchmetrics.classification import MulticlassConfusionMatrix
from configs.load_configs import configs
from torch.backends import cudnn

data_dir = os.path.dirname(os.path.realpath(__file__)) + configs["temp_dir"]
device       = configs["device"]
if (device == "cuda"):
    cudnn.benchmark = True
    print(f"[+] cudnn benchmark status: {cudnn.benchmark}")

def train_model(train_loader : DataLoader, val_loader : DataLoader , model, loss_fn , optimizer, test_loader = None ):
    """
    Trains the given model, until val loss doesn't shrink anymore

    Parameters
    ----------
    dataloader : DataLoader
        The Dataloader object that should be used for parsing and formatting the training data
    
    val_loader : DataLoader
        The DataLoader object that should be used for parsing the evaluation data.
    
    model : neural_network
        The model that should be trained
    
    loss_fn : 
        The function for evaluating the loss
    
    optimizer :
        The optimizer used during training
    


    Returns
    -------

    model
        the trained model
    """
    classes = get_classes(data_dir= data_dir)
    
    timestamp         =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir           = os.path.dirname(os.path.realpath(__file__)) + "/logs/"
    writer            = SummaryWriter(log_dir + timestamp)
    patience          = 7
    epochs_no_improve = 0
    best_val_loss     = float("inf")
    min_improve       = 1e-3

    confusion_matrix = MulticlassConfusionMatrix(len(classes)).to(device)

    model.train()
    epoch = 0
    #for i in range(2):
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
                
                
                prediction = torch.argmax(input = outputs, dim = 1)

                loss = loss_fn(outputs, labels)
                confusion_matrix.update(prediction, labels)
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

        if (val_loss < best_val_loss + min_improve): # Early Stopping
            best_val_loss = val_loss
            epochs_no_improve = 0
    
        else:
            epochs_no_improve += 1
    
        if (epochs_no_improve >= patience):
            fig, ax = confusion_matrix.plot(labels = classes)
            writer.add_figure("Confusion Matrix", fig)


            # TODO implement confusion matrix
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


def load_model(model_path : str):
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