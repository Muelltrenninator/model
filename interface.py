import gradio as gr 

from train_model import load_model, evalute_input
from model_architecture import neural_network


def predict(input):
    curr_model = load_model("/home/julian_hack/Desktop/projects/muelltrenninator/trained_models/model.pth")
    predicted = evalute_input(curr_model,input)


    return predicted

demo = gr.Interface(
    fn= predict,
    inputs = gr.Image(type = "filepath"),
    outputs= gr.Textbox(),
    flagging_mode = "never"


)

demo.launch()
