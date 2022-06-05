import sys
import numpy as np
import pickle
import json
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def get_encoded_data():
    i = 0
    for f in sorted(os.listdir("lists")):
        if os.stat('final_data.csv').st_size == 0:
            df = pd.DataFrame()
            df['path_to_image'] = ""
            df['Height'] = ""
            df['Width'] = ""
            df['Perimeter'] = ""
            df['DCT_Norm'] = ""
            df['HOG_Norm'] = ""
            df['IMG_Norm'] = ""
        else:
            df = pd.read_csv('final_data.csv')

        entry = list(np.load("lists/" + f, allow_pickle=True))
        a_series = pd.Series(entry, index = df.columns)
        
        print("-"*20)
        print(f"Series for {f}:")
        print(a_series)
        print("-"*20)
        
        df.loc[i] = entry
        i += 1
        df.to_csv("final_data.csv", index=False)

    return df

def predict(X):

    # Knn
    knn = pickle.load(open("models/X_PERIMETER_KNN_k_selection.sav", 'rb'))
    knn_pred = knn.predict(X)

    # SVM
    # svm = pickle.load(open("models/svm.sav", 'rb'))
    # svm_pred = svm.predict(X)

    return knn_pred