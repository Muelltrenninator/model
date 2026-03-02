import os
import torch
import torchvision
import json
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np

from torchvision import datasets, transforms
from model_architecture import neural_network
from torchvision import models
from torchvision import datasets, transforms
from trash_classifier import trash_pre_detector
from collections import OrderedDict
from utils import get_classes
from configs.load_configs import configs


val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    ])

device = configs["device"]





def evalute_input(model : neural_network, image_path : str, data_dir : str = None, image_transforms : transforms = val_transform, model_small : trash_pre_detector = None) -> dict:
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
    data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data_large_classifying/"
    classes = get_classes(data_dir)

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