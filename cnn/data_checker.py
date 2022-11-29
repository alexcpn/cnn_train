import os
from PIL import Image

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


if __name__ == "__main__":
    paths = list_files("imagenette2-320/val/dogs50A-val")
    problem_files =[]
    for path in paths:
        with open(path, 'rb') as file:
            try:
                img = Image.open(file).load()
            except Exception as e:
                print(e)
                print(path)
                problem_files.append(path)
    