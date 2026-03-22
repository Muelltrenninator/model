import gradio as gr 
import os
import torch
import torchvision
import torch.nn as nn

from torchvision import datasets, transforms, models
from evaluation import evalute_input
from configs.load_configs import configs
from models.registry import MODEL_REGISTRY
import models.model_architectures

curr_model_large_path = os.path.dirname(os.path.realpath(__file__))+ "/trained_models_large/model_transfer_current.pth"

curr_model_large = MODEL_REGISTRY["ResNet50.DEFAULT"](5)

curr_model_large.load_state_dict(torch.load(curr_model_large_path, weights_only= False, map_location= torch.device(configs["device_eval"])))
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