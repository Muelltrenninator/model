import gradio as gr 
import os
import torch
import torchvision
import torch.nn as nn
import models.model_architectures
import threading
import time
import gc

from torchvision import datasets, transforms, models
from evaluation import evalute_input
from configs.load_configs import configs
from models.registry import MODEL_REGISTRY
from threading import Thread
from utils import load_model

curr_model_large = load_model(configs["val_model_architecture"], configs["weights_path"])
curr_model_large.to(configs["device_eval"])
# curr_model_small = load_model(os.path.dirname(os.path.realpath(__file__))+ "/trained_models_small/model_test.pth")
last_runtime = time.time()
runtime_lock = threading.Lock()

def track_usage():
    global curr_model_large
    while(True):
        with runtime_lock:
            if(time.time() - last_runtime >= 1800 and curr_model_large != None):
                curr_model_large = None
                gc.collect()
            
        time.sleep(5)

thread = Thread(target = track_usage)
thread.start()




def predict(input):
    global curr_model_large
    
    with runtime_lock:
        last_runtime = time.time()

    if(curr_model_large == None):
        curr_model_large = load_model(configs["val_model_architecture"])
        curr_model_large.to(configs["device_eval"])

    predicted = evalute_input(curr_model_large,input, model_small = None)
    
    return predicted


demo = gr.Interface(
    fn= predict,
    inputs = gr.Image(type = "filepath"),
    outputs= gr.Textbox(),
    flagging_mode = "never"


)

demo.launch()
