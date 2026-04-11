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

#from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from utils import get_classes
from configs.load_configs import configs
from torch.backends import cudnn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_dir = os.path.dirname(os.path.realpath(__file__)) + configs["raw_data_dir"] +"images/"
device       = configs["device_train"]
if (device == "cuda"):
    cudnn.benchmark = True
    print(f"[ OK ] cudnn benchmark status: {cudnn.benchmark}")


def train_one_epoch(model, train_loader, loss_fn, optimizer, writer, epoch_idx) -> list:
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_image_example = None
    model.train()
    for i, data in enumerate(train_loader, 0):
        
        optimizer.zero_grad()
        
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).to(device)

        writer.add_image("Train example", inputs[0])

        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted_train = outputs.max(1)

        train_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)
        train_correct += predicted_train.eq(labels).sum().item() 
    
    writer.add_scalar("Train / Accuracy", train_correct / train_total, epoch_idx)
    writer.add_scalar("Train / Loss", train_loss / train_total, epoch_idx)
    return train_loss / train_total, train_correct / train_total, train_image_example


def val_one_epoch(model, val_loader, loss_fn, val_confusion_matrix, writer, epoch_idx):

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_image_example = None

    
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            model.eval()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
        
            writer.add_image("Val examples", inputs[0])
            _, predicted_val = outputs.max(1)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            val_total += labels.size(0)
            val_correct += predicted_val.eq(labels).sum().item()
            val_confusion_matrix.update(predicted_val, labels)
        
        writer.add_scalar("Val / Accuracy", val_correct / val_total, epoch_idx)
        writer.add_scalar("Val / Loss", val_loss / val_total, epoch_idx)

        return val_loss / val_total, val_correct / val_total, val_image_example


def test_model(model, test_loader, loss_fn, test_confusion_matrix, writer):
    model.eval()
    test_loss    = 0.0
    test_total   = 0
    test_correct = 0
    y_true       = []
    y_pred       = []
    score        = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            y_true.append(labels)
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)
            
            writer.add_image("Test examples", inputs[0])
            loss            = loss_fn(outputs, labels)
            _, predicted_test = outputs.max(1)
            y_pred.append(predicted_test)
            test_loss += loss.item() * labels.size(0)
            test_total += labels.size(0)
            test_correct += predicted_test.eq(labels).sum().item()
            test_confusion_matrix.update(predicted_test, labels)

            
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        score = f1_score(y_true = y_true, y_pred = y_pred, average = "macro")
        print(score)
        writer.add_scalar("F1-Score", score)
        writer.add_scalar("Accuracy / Test", test_correct / test_total)
        writer.add_scalar("Loss / Test", test_loss / test_total)
        
        
        return test_loss / test_total, test_correct / test_total, score




def train_model(train_loader : DataLoader, val_loader : DataLoader , model : object, loss_fn : func , optimizer, test_loader : DataLoader) -> list:
    """
    Trains the given model, until val loss doesn't shrink anymore

    Parameters
    ----------
    train_loader : DataLoader
        The DataLoader object that should be used for parsing the training data
    
    val_loader : DataLoader
        The DataLoader object that should be used for parsing the evaluation data.
    
    model : neural_network
        The model that should be trained
    
    loss_fn : 
        The function for evaluating the loss
    
    optimizer :
        The optimizer used during training
    
    test_loader : DataLoader
        The DataLoader object that should be used for parsing the test data 
    

    Returns
    -------
    one list consisting of:

    model
        the trained model
    
    test_loss / test_total
        the loss of the trained model
    
    test_correct / test_total
        the accuracy of the trained model in percent
    
    f1_score
        the macro f1_score of the trained model in percent.
    """
    classes = get_classes(data_dir= data_dir)
    
    timestamp         =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir           = os.path.dirname(os.path.realpath(__file__)) + "/logs/"
    writer            = SummaryWriter(log_dir + timestamp)
    patience          = 7
    epochs_no_improve = 0
    best_val_loss     = float("inf")
    min_improve       = 1e-3

    test_confusion_matrix = MulticlassConfusionMatrix(len(classes)).to(device)
    val_confusion_matrix  = MulticlassConfusionMatrix(len(classes)).to(device)

    epoch = 0
    score = 0
    test_loss = 0.0
    test_total = 0

    
    #for epoch in range(configs["num_epochs"]):
    while(True):
        model.train()

# --- Training Phase --- #

        train_loss_per_epoch, train_accuracy_per_epoch, train_image_example = train_one_epoch(model, train_loader, loss_fn, optimizer, writer, epoch)
        val_loss_per_epoch, val_accuracy_per_epoch, val_image_example = val_one_epoch(model, val_loader, loss_fn, val_confusion_matrix, writer, epoch)

        print(f"[ OK ] epoch_train_loss: {train_loss_per_epoch} epoch_train_acc : {100. * train_accuracy_per_epoch:.4f}")
        print(f"[ OK ] epoch_val_loss: {val_loss_per_epoch} epoch_val_acc : {100. * val_accuracy_per_epoch:.4}")

        epoch += 1

        if (val_loss_per_epoch < best_val_loss + min_improve): # Early Stopping
            best_val_loss = val_loss_per_epoch
            epochs_no_improve = 0
    
        else:
            epochs_no_improve += 1
    
        if (epochs_no_improve >= patience):

            test_loss, test_accuracy, f1_score = test_model(model = model, test_loader = test_loader, loss_fn = loss_fn, test_confusion_matrix = test_confusion_matrix, writer = writer)
            print(f"[ OK ] test_loss: {test_loss:.4f} test_acc : {100. * test_accuracy:.4f}")
            test_fig, test_ax = test_confusion_matrix.plot(labels = classes)
            val_fig, vaL_ax = val_confusion_matrix.plot(labels = classes)
            writer.add_figure("Test Confusion Matrix", test_fig)
            writer.add_figure("Val Confusion Matrix", val_fig)


            print("[ OK ] Finished Training")
            writer.flush()
            print(f"[ OK ] Created tensorboard summary at {SummaryWriter.get_logdir(writer)}")
            writer.close()

            return model, test_loss, 100. * test_accuracy, f1_score
    
    test_loss, test_accuracy, f1_score = test_model(model = model, test_loader = test_loader, loss_fn = loss_fn, test_confusion_matrix = test_confusion_matrix, writer = writer)
    print(f"[ OK ] test_loss: {test_loss:.4f} test_acc : {100. * test_accuracy:.4f}")
    test_fig, test_ax = test_confusion_matrix.plot(labels = classes)
    val_fig, vaL_ax = val_confusion_matrix.plot(labels = classes)
    writer.add_figure("Test Confusion Matrix", test_fig)
    writer.add_figure("Val Confusion Matrix", val_fig)
    print("[ OK ] Finished Training")
    writer.flush()
    print(f"[ OK ] Created tensorboard summary at {SummaryWriter.get_logdir(writer)}")
    writer.close()

    return model, test_loss, 100. * test_accuracy, f1_score
