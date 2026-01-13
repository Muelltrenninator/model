import gradio as gr 
import os

from model_functions import load_model, evalute_input
from model_architecture import neural_network


def predict(input):
    curr_model = load_model(os.path.dirname(os.path.realpath(__file__))+ "/trained_models/model_93%.pth")
    predicted = evalute_input(curr_model,input)


    return predicted


demo = gr.Interface(
    fn= predict,
    inputs = gr.Image(type = "filepath"),
    outputs= gr.Textbox(),
    flagging_mode = "never"


)

demo.launch()