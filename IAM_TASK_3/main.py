"""
Should data be prepared for word-classification, character or sentence classification?
    Regardless, three datasets need to follow:
        1. Word classification -> separate images by words and labels by words based on spaces | remove data points that are under represented / incomplete words / words separate by hyfens / punctuation?
        2. Character classification -> nearly impossible for cursive datasets... 
"""
# Internal libs
import json
import os

from model_metrics import model_comparision
from saver_loader import saver
from OFFLINEgenerate_dictionaries_data import data_processing, save_data_to_json
from sklearn.model_selection import train_test_split

def get_data(dic_path):
    with open(dic_path) as f:
        data = json.load(f)

    return train_test_split(f.keys(), f.values(), test_size = 0.2, random_state = 42)

def main():
    # dic not ml ready (Done)
    # save_data_to_json("data/IAM-data/dic/sentence_no_processing_dictionary.json")

    # data_processing()
    
    # dic vanilla ml ready (Not Done) and dic not ml ready (Done)
    # Img = vanilla img vector | univar label = categorical
	
	# dic img sentence ml ready and dic img sentence not ml ready : univar and multivar
	# Img = preprocessed img sentence vector | univar label = categorical
	# Img = preprocessed img sentence vector | multivar label = categorical
	
	# dic img word ml ready and dic img word not ml ready : univar and multivar
	# Img = preprocessed img word vector | univar label = categorical
	# Img = preprocessed img word vector | multivar label = categorical

    # x_train, x_test, y_train, y_test = data_processing()
    # models = []

    # ------------------------- MODELS
    # models.append(fit_linear_model("sklearnLinearModel_experiment_3", x_train, y_train))
    # models.append(fit_bag("bagging_knn_model", x_train, y_train))
    # models.append(fit_tree("tree_model", x_train, y_train))
    # models.append(fit_extra_trees("extra_forest_model", x_train, y_train))
    # models.append(fit_random_forest("random_forest_model", x_train, y_train))
    
    # model_comparision(models, x_test, y_test)

    # for model in models:
    #     clf = model[0]
    #     name = model[1]
    #     saver(clf, name)

if __name__ == '__main__':
    main()

"""
TODO - 
PROBLEM - INTERNAL BOUNDING BOXES ARE BEING SAVED DESPITE USING HIGHEST HIERARCHY
    1. Build general data pipeline (data loading, processing depending on task - get pixel values from images into vectors and keep that as the input data);
        1.1 Divide sentences into words (both x and y)
        1.2 Extract features from these sentences:
            1.2.1 Geometrical data from each word
            1.2.2 HOG
            1.2.3 DCT
            1.2.4 Length
            1.2.5 Height
            ...
    2. What do svms expect?
    3. What do time series expect?
    4. What do HMM expect?
    5. What do Bayesian techniques expect?
    6. What does knn expect?
    7. Use OCR for character extraction? Or process whole line?
    8. Inform decision with a causal graph that encodes english syntactical rules / bayes?
    9. Decide on size of images before saving them to dictionary
    10. ASSEMBLY - SVM informed by feature vector for clustering with knn or several SVM: so separate the y's

TODO - CHECKING OUTPUT OF DICTIONARY
    1. COMPARE NUMBER OF CHARACTERS WITH AVERAGE SIZE OF X LENGTH WORD
"""