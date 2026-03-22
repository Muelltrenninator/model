import torch
import torch.nn as nn 
import torch.nn.functional as F 
import os
import torchvision
import torch.optim as optim
from models.registry import register_model, MODEL_REGISTRY
from models.model_template import model_template
from torchvision import models

@register_model(model_name = "test_nn")
class neural_network(model_template):
    
    def __init__(self, num_final_output = 5):
        super().__init__()
    
        self.layers_analyze = nn.Sequential(
        
            # Layer 1
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )
        
        # Layer 5
        self.layers_combine = nn.Sequential(

            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_final_output)
        )

    def forward(self, input_tensor):
        features = self.layers_analyze(input_tensor)
        output = self.layers_combine(features)
        return output


@register_model(model_name = "ResNet50.DEFAULT")
class resnet50(model_template):
    
    def __init__(self, num_final_output = 5):
        super().__init__()
        self.model = models.resnet50(weights = "ResNet50_Weights.DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_final_output)
)

        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        output = self.model(x)
        return output


@register_model(model_name = "ResNet34.DEFAULT")
class resnet34(model_template):

    def __init__(self, num_final_output = 5):
        super().__init__()
        self.model = models.resnet34(weights = "ResNet34_Weights.DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_final_output)
)

        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        output = self.model(x)
        return output


@register_model(model_name = "ResNet18.DEFAULT")
class resnet18(model_template):

    def __init__(self, num_final_output = 5):
        super().__init__()
        self.model = models.resnet18(weights = "ResNet18_Weights.DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_final_output)
)

        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        output = self.model(x)
        return output

