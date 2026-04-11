
import os
import requests
import numpy as np
import splitfolders
import torch

from models.registry import MODEL_REGISTRY
from configs.load_configs import configs
from collections import Counter
from sklearn.utils import compute_class_weight
from zipfile import ZipFile
from send2trash import send2trash
from dotenv import load_dotenv, find_dotenv




def calculate_class_weights(dataloader : DataLoader) -> list:
    """
    Calculates the weight of the classes inside the train Dataloader

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader which wheights have to be caluclated.

    Returns
    -------
    class_weights
        A list of class weights indeces are classes and values the weights.

    """

    counter = Counter()
    y = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        counter.update(labels.tolist())

    for class_label, count in counter.items():
        y.extend([class_label] * count)
    print(f"[ OK ] Class Distribution: {counter}")
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
    print(f"[ OK ] Class Weights: {class_weights}")

    return class_weights



def get_current_versions():
    pass    # Optional


def get_classes(data_dir : str) -> list:
    """
    Gets all the available folder inside the data_dir

    Parameters
    ----------
    data_dir : str
        The data directory holding the classes
    
    Returns
    -------
    classes : list 
        An alphabeticlly sorted list of classes as strings inside the specified directory. Only folders, files are being ignored
    
    """
    available_classes = []
    with os.scandir(data_dir) as curr_dir:
        for i in curr_dir:
            if(i.is_dir() == True):
                available_classes.append(i.name)

    return sorted(available_classes, key = str.lower)



def split_data():
    """
    Deletes every old file in the path, gets the data from the raw path and splits it into train, val and test folders with the trash types as subdirectories. 
    """
    full_split_path = os.path.dirname(os.path.realpath(__file__)) + configs["split_dir"]
    if(os.path.exists(full_split_path) == False):
        raise ValueError(f"[ FAILED ] Path does not exist:{full_split_path}")
    
    for item in os.listdir(path = full_split_path):

        file_path = os.path.join(full_split_path, item)
    
        try:
            send2trash(file_path)
            print(f"[ DELETED ] {file_path}")
        except Exception as e:
            print(f"[ FAILED ] {file_path} ({e})")
        

    if(configs["train_ratio"] + configs["val_ratio"] + configs["test_ratio"] != 1):
        raise ValueError(f"[ FAILED ] Split ratios must add up to one train_ratio:{configs["train_ratio"]} val_ratio:{configs["val_ratio"]} test_ratio:{configs["test_ratio"]}")
    
    data_dir = os.path.dirname(os.path.realpath(__file__))
    splitfolders.ratio(input = data_dir + configs["raw_data_dir"] + "/images", output = data_dir + configs["split_dir"], seed = configs["seed"], shuffle = True, ratio = (configs["train_ratio"], configs["val_ratio"], configs["test_ratio"]))


def fetch_data():
    """
    Fetches data from the datly website.
    """
    dotenv_path = find_dotenv()
    print(f"[ OK ] Found environment variables at {dotenv_path}")
    load_dotenv(dotenv_path) 
    api_key = os.getenv("AUTHORIZATION")
    if(api_key == None):
        raise ValueError("API key not found inside os environment variables")
    print("[ OK ] Successfully loaded environment variables")
    response = requests.get('https://datly.con.bz/api/projects/1/submissions/dump', headers = {"Authorization" : api_key}, stream = True)
    response.raise_for_status()

    print("[ OK ] Verified")
    with open("data.zip", "wb") as file:
        file.write(response.content)

    with ZipFile("data.zip", "r") as zip:
        extract_path = os.path.dirname(os.path.realpath(__file__)) + configs["raw_data_dir"]
        zip.extractall(path= extract_path)
        print("[ OK ] data successfully downloaded")
    os.remove("data.zip")
    print("[ DELETED ] data.zip zip file cleaned up")

def load_model(model_name : str, weights_path : str = None) -> object:
    """
    Creates a model object based on given model_name. When supplied with a filepath to a corresponding .pth file it will initialize the weights

    Parameters
    ----------
    model_name : str
        The name of the model architecture

    model_path : str, optional
        Path of the model weights to be loaded
    
    Returns
    -------
    loaded_model : object
        The loaded model for guaranteed object methods consult :py:class:`models.model_template`
    """
    loaded_model = MODEL_REGISTRY[model_name](len(get_classes(os.path.dirname(os.path.realpath(__file__)) + configs["raw_data_dir"] + "images/")))
    if (weights_path != None):
        loaded_model.load_state_dict(torch.load(weights_path, weights_only= False, map_location= torch.device(configs["device_eval"])))
    return loaded_model

def save_model(model : object, model_path : str):
    """
    Saves a supplied model at the given path as a .pth file

    Parameters
    ----------
    model : object
        The model object that should be saved
    
    model_path : str
        The path the model should be saved at
    """

    torch.save(model.state_dict(), model_path)
    print(f"[ OK ] Model saved to path: {model_path}")
