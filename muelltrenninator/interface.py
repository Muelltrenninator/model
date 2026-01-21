import gradio as gr 
import os

from model_functions import load_model, evalute_input



def predict(input):
    curr_model_large = load_model(os.path.dirname(os.path.realpath(__file__))+ "/trained_models_large/model_production_ready.pth")
   # curr_model_small = load_model(os.path.dirname(os.path.realpath(__file__))+ "/trained_models_small/model_test.pth")
    predicted = evalute_input(curr_model_large,input, model_small = None)


    return predicted


demo = gr.Interface(
    fn= predict,
    inputs = gr.Image(type = "filepath"),
    outputs= gr.Textbox(),
    flagging_mode = "never"


)

demo.launch()