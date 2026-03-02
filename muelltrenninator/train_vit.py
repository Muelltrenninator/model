import clip
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



classes = ("bio", "elektroschrott", "gelber_sack","papier", "restmuell")
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"




def train_vit(classifier, train_loader : DataLoader, val_loader : DataLoader , vit , loss_fn , optimizer ) -> neural_network:
    """
    Trains the given model, until val loss doesn't shrink anymore

    Parameters
    ----------
    classifier :
    The small classifier that should be trained.

    dataloader : DataLoader
        The Dataloader that should be used for parsing and formatting the training data.
    
    vit :
        The Vision Transformer Backbone.
    
    loss_fn : 
        The function for evaluating the loss.
    
    optimizer :
        The optimizer used during training.
    
    val_loader : DataLoader
        DataLoader Object used for validation.

    Returns
    -------

    vit, classifier 
        vit with the fitting trained classifier
    """

    timestamp =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.dirname(os.path.realpath(__file__)) + "/logs/"
    writer = SummaryWriter(log_dir + timestamp)
    train_losses = []
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float("inf")
    min_improve = 1e-3
    features = None
    epoch = 0
    
    #for epoch in range(num_epochs):
    while(True):
        
# --- Training Phase --- #
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        classifier.train()
        vit.eval()
        for i, data in enumerate(train_loader, 0):
            

            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = vit.encode_image(inputs) * 10
            features = features / features.norm(dim=-1, keepdim=True)
            outputs = classifier(features)

            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted_train = outputs.max(1)
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
            train_correct += predicted_train.eq(labels).sum().item() # comparison sums up true values transform 1d tensor to number with item()
            if( i == 0):
                writer.add_image("Examples train", inputs[0])



# --- Validation Phase --- #
        vit.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():

            for i, data in enumerate(val_loader, 0 ):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                features = vit.encode_image(inputs) * 10
                features = features / features.norm(dim=-1, keepdim=True)
                outputs = classifier(features)
                
                loss = loss_fn(outputs, labels)
                
                _, predicted_val = outputs.max(1)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                val_correct += predicted_val.eq(labels).sum().item()
                if( i == 0):
                    writer.add_image("Examples train", inputs[0])

        
        sum(p.requires_grad for p in vit.parameters())
        print(f"[+] epoch_train_loss: {train_loss/ train_total:.4f} epoch_train_acc : {100. * train_correct/ train_total:.4f}")
        print(f"[+] epoch_val_loss: {val_loss/ val_total:.4f} epoch_val_acc : {100. * val_correct/ val_total:.4}")
        writer.add_scalar("Loss / val", val_loss / val_total, epoch)
        writer.add_scalar("Accuracy / val", val_correct / val_total, epoch)

        
        writer.add_scalar("Loss / train", train_loss/ train_total, epoch)
        writer.add_scalar("Accuracy / train", train_correct / train_total, epoch)
        epoch += 1
        print(classifier.weight.grad.norm())


        print(sum(p.requires_grad for p in vit.parameters()))
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
            return vit, classifier