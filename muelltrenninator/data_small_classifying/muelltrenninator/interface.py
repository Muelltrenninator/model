import gradio as gr 
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
from model_transfer import model_pt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


from model_functions import load_model, evalute_input

curr_model_large_path = os.path.dirname(os.path.realpath(__file__))+ "/trained_models_large/model_transfer_red.pth"
curr_model_large = model_pt
curr_model_large.load_state_dict(torch.load(curr_model_large_path, weights_only= False))
curr_model_large.to("cpu")
# curr_model_small = load_model(os.path.dirname(os.path.realpath(__file__))+ "/trained_models_small/model_test.pth")

def predict(input):
    predicted = evalute_input(curr_model_large,input, model_small = None)


    return predicted


demo = gr.Interface(
    fn= predict,
    inputs = gr.Image(type = "filepath"),
    outputs= gr.Textbox(),
    flagging_mode = "never"


)

demo.launch()