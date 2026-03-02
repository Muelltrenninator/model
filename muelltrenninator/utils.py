
import os
import numpy as np
import splitfolders
from configs.load_configs import configs
from collections import Counter
from sklearn.utils import compute_class_weight




def calculate_class_weights(dataloader : DataLoader):
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
    for _, labels in dataloader:
        counter.update(labels.tolist())

    for class_label, count in counter.items():
        y.extend([class_label] * count)
    print(counter)
    print(y)
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
    print(f"Class Weights: {class_weights}")

    return class_weights



def get_current_versions():
    pass    # Optional


def get_classes(data_dir : str):
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
    if(configs["train_ratio"] + configs["val_ratio"] + configs["test_ratio"] != 1):
        print("Split Ratios must add up to one")
        return

    splitfolders.ratio(input = configs["temp_dir"], output = configs["split_dir"], seed = configs["seed"], shuffle = True, ratio = (configs["train_ratio"], configs["val_ratio"], configs["test_ratio"]))

