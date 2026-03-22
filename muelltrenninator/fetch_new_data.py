import os 
import requests
from zipfile import ZipFile
from configs.load_configs import configs


def fetch_data():
    response = requests.get('https://datly.con.bz/api/projects/1/submissions/dump', headers = {"Authorization" : "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NzM4MjYxNDksImV4cCI6MTc4OTM3ODE0OSwiYXVkIjoiYXV0aCIsInN1YiI6IkVpbmZhY2hudXJKdWxzIiwiaXNzIjoiaHR0cHM6Ly9kYXRseS5jb24uYnoifQ.MdzQE2syRt1Zgd_CXcpHOcWhuKjkCCzDZksQK1qYSvDxi7OYl9-0KBr2yBmHSbgEMXrq-pO9quoa3rtEwl9HVezxj2EuZdMo1E6j2wP1P129S-Q-0tJn6ABUU-s6V2s8skAtedxEVaLeO66UjygcUSk61QR8pFLtyibBw0Dx0m0w3zwi5tqHFUMQU0TEa0ozGoK1nv7J5f-99yLJ8sDIiRdPMqxNwUyiNWzpj5TOqsQyZI0m5AOao4iTe12x4RkXSYOxmM00wkm4vfJAbw-3ZjqyiVBU_G3Mr1XfhYi6lgksvdTT41digoW1s8ZSZahhJ3gI9UTBUKs0eLfddHO7rQ"}, stream = True)
    response.raise_for_status()

    if(response.status_code == 200):
        with open("data.zip", "wb") as file:
            file.write(response.content)

    with ZipFile("data.zip", "r") as zip:
        extract_path = os.path.dirname(os.path.realpath(__file__)) + configs["temp_dir"]
        zip.extractall(path= extract_path)

fetch_data()



