import gradio as gr 
import os
import torch
import torchvision
import torch.nn as nn

from torchvision import datasets, transforms, models
from model_architecture import neural_network
from trash_classifier import trash_pre_detector
from model_functions import load_model, evalute_input

curr_model_large_path = os.path.dirname(os.path.realpath(__file__))+ "/trained_models_large/model_transfer_newest.pth"
curr_model_large = models.resnet50(pretrained = True)
for param in curr_model_large.parameters():
    param.requires_grad = False
curr_model_large.fc = nn.Sequential(
                nn.Linear(curr_model_large.fc.in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, 5)
)
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
