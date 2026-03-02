import os 
import requests
from zipfile import ZipFile
from configs.load_configs import configs


def fetch_data():
    response = requests.get('https://datly.con.bz/api/projects/1/submissions/dump', headers = {"Authorization" : configs["Authentication"]}, stream = True)
    response.raise_for_status()

    if(response.status_code == 200):
        with open("data.zip", "wb") as file:
            file.write(response.content)

    with ZipFile("data.zip", "r") as zip:
        extract_path = os.path.dirname(os.path.realpath(__file__)) + configs["temp_dir"]
        zip.extractall(path= extract_path)

fetch_data()



