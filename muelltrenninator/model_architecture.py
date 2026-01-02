import torch
import torch.nn as nn           # parent object for pytorch model
import torch.nn.functional as F # activation function for model
import os
import torch.optim as optim


class neural_network(nn.Module):
    
    def __init__(self, num_final_output = 5):
        super(neural_network, self). __init__()

        # Layer 1: Basic detection (shape, colors)
        # 1. Takes RGB image (3 channels)
        # 2. Applies 32 different 3x3 filters
        # 3. Each filter produces 1 feature map
        # 4. Result: 32 different interpretations of the image

        # Layer 2: Pattern recognition (textures)

        # Layer 3: Details (labels, logos)

        # Layer 4: Advanced Combinations

        # Layer 5: Classifier
        self.layers_analyze = nn.Sequential(
        
            # Layer 1
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Layer 5
        self.layers_combine = nn.Sequential(

            nn.Flatten(),
            nn.Linear(36864, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_final_output)
        )

    def forward(self, input_tensor):
        features = self.layers_analyze(input_tensor)
        output = self.layers_combine(features)
        return output