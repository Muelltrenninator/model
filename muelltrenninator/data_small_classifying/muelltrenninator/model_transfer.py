
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


model_pt = models.resnet50(weights = "ResNet50_Weights.DEFAULT")
for param in model_pt.parameters():
    param.requires_grad = False

for param in model_pt.fc.parameters():
    param.require_grad = True
num_ftrs = model_pt.fc.in_features
model_pt.fc = nn.Linear(num_ftrs, 5)