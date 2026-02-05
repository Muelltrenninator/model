import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


class trash_pre_detector(nn.Module):
    def __init__(self, num_final_output = 2):
        super(trash_pre_detector, self). __init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_final_output)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output