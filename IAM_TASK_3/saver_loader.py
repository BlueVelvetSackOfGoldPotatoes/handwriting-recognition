import os.path
from os import path
from joblib import dump, load

def saver(model, name):
    if not path.exists("savedModels"):
        print()
        print("Making directory savedModels...")
        print()
        os.mkdir("savedModels")
    dump(model, "savedModels/" + name + '.joblib')
    print("Dumped " + "savedModels/" + name + '.joblib')

def loader(name):
    return load("savedModels/" + name + '.joblib') 