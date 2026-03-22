
import yaml
import os

config_path = os.path.dirname(os.path.realpath(__file__))+ "/config.yaml"

with open(config_path, "r") as config_file:
    configs = yaml.safe_load(config_file)
    # TODO implement check for invalid split rates between the datasets
    
