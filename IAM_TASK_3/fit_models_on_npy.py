import os
import sys
import numpy as np
import pickle
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
# KNN
from sklearn.neighbors import KNeighborsClassifier
from ast import literal_eval

# SVM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

# GENERAL ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2

# np.set_printoptions(threshold=sys.maxsize)
folder = "data/IAM-data/lists"

def saver_loader(file, model=None):
    """
    Save or return loaded model.pickle
    
    Parameters:
    -----------
        file: path to model file, string.
        model: object model, else None.
    Returns:
    -----------
        True if saved when model is present in args.
        Model object if model was None
    """
    if model:
        pickle.dump(model, open(file, 'wb'))
        return file
    else:
        loaded_model = pickle.load(open(file, 'rb'))
        return loaded_model

def scatter_plot_classes(df, target):

    number_of_colors = 6732
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

    for type, color in zip(df.encoded_label, colors):
        plt.scatter(df[target][(df.encoded_label == type)],
            color= color,
            label= type)
    plt.xlabel('label')
    plt.ylabel(target)
    plt.legend()
    plt.show()

def visualize_data(feature_vec):
    print("-"*100)
    print(f"File     : {feature_vec[0]}")
    print(f"Label    : {feature_vec[1]}")
    print(f"Height   : {feature_vec[2]}")
    print(f"Width    : {feature_vec[3]}")
    print(f"Perimeter: {feature_vec[4]}")
    print(f"DCT Norm : {feature_vec[5].shape}")
    print(f"HOG Norm : {feature_vec[6].shape}")
    print(f"IMG Norm : {feature_vec[7].shape}")
    print("-"*100)

def label_encoder():
    with open("data/IAM-data/dic/class_data.json", "r") as f:
        data = json.load(f)
    X = list(data.keys())
    labelencoder = LabelEncoder()
    labelencoder.fit_transform(X)
    # encoded = labelencoder.transform(["wish"])
    # decoded = labelencoder.inverse_transform(encoded)
    # print(f"encoded: {encoded}")
    # print(f"decoded: {decoded}")

    return labelencoder

# similarity of two assignments, ignoring permutations: adjusting for random labelling.
def rand_index(labels_true, labels_pred):
    print(metrics.adjusted_rand_score(labels_true, labels_pred))
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def supervised_knn(X, y, name, k=0):

    # splitting the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    if k == 0:
        error1= []
        error2= []
        # 6732 total classes
        for k in range(1, 6000, 100):
            clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)) 
            clf.fit(X_train,y_train)
            # testing the model
                
            # stroring the errors
            y_pred1= clf.predict(X_train)
            error1.append(np.mean(y_train!= y_pred1))
            y_pred2 = clf.predict(X_test)
            error2.append(np.mean(y_test != y_pred2))

            print("-"*20)
            print(f"At {k} K-clusters: accuracy on TEST set {accuracy_score(y_test,y_pred2)}, accuracy on TRAINING set {accuracy_score(y_train,y_pred1)}")
            print("-"*20)

        # ploting the graphs for testing and training 
        plt.plot(range(1,6000), error1, label="Train")
        plt.plot(range(1,6000), error2, label="Test")
        plt.xlabel('k Value')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
        
    else:
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)) 
        clf.fit(X_train,y_train)
        # testing the model
            
        # stroring the errors
        y_pred1= clf.predict(X_train)
        y_pred2 = clf.predict(X_test)

        # print("-"*20)
        # print(f"At {k} K-clusters: accuracy on TEST set {accuracy_score(y_test,y_pred2)}, accuracy on TRAINING set {accuracy_score(y_test,y_pred1)}")
        # print("-"*20)

        pickle.dump(clf, open(name + ".sav", 'wb'))

        with open(name + ".txt", "w") as f:
            f.write("Parameters: " + str(clf.get_params()) + "\n---------------------\nMean Accuracy on the test set: " + str(clf.score(X_test, y_test)) + "\nRand Index: " + str(rand_index(y_test,clf.predict(X_test))))

def SGD_svm(X, y, name, feature_selection=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    if feature_selection:
        clf = Pipeline(
            [
                ("anova", SelectPercentile(chi2)),
                ("scaler", StandardScaler()),
                ("svc", SGDClassifier(max_iter=1000, tol=1e-3)),
            ])

        clf.fit(X_train, y_train)

        # score_means = list()
        # score_stds = list()
        # percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

        # for percentile in percentiles:
        #     clf.set_params(anova__percentile=percentile)
        #     this_scores = cross_val_score(clf, X_test, y_test)
        #     score_means.append(this_scores.mean())
        #     score_stds.append(this_scores.std())

        # plt.errorbar(percentiles, score_means, np.array(score_stds))
        # plt.title("Performance of the SVM-Anova varying the percentile of features selected")
        # plt.xticks(np.linspace(0, 100, 11, endpoint=True))
        # plt.xlabel("Percentile")
        # plt.ylabel("Accuracy Score")
        # plt.axis("tight")
        # plt.show()

        with open(name + ".txt", "w") as f:
            f.write("Parameters: " + str(clf.get_params()) + "\n---------------------\nMean Accuracy on the test set: " + str(clf.score(X_test, y_test)) + "\nRand Index: " + str(rand_index(y_test,clf.predict(X_test))))
        pickle.dump(clf, open(name, 'wb'))
    else:
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X_train, y_train)
    
    with open(name + ".txt", "w") as f:
        f.write("Parameters: " + str(clf.get_params()) + "\n---------------------\nMean Accuracy on the test set: " + str(clf.score(X_test, y_test)) + "\nRand Index: " + str(rand_index(y_test,clf.predict(X_test))))

    pickle.dump(clf, open(name + ".sav", 'wb'))

def linear_svm(X, y, name, feature_selection=0):

    # splitting the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    if feature_selection:
        clf = Pipeline(
            [
                ("anova", SelectPercentile(chi2)),
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(random_state=42, tol=1e-5)),
            ])

        clf.fit(X_train, y_train)

        # score_means = list()
        # score_stds = list()
        # percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

        # for percentile in percentiles:
        #     clf.set_params(anova__percentile=percentile)
        #     this_scores = cross_val_score(clf, X_test, y_test)
        #     score_means.append(this_scores.mean())
        #     score_stds.append(this_scores.std())

        # plt.errorbar(percentiles, score_means, np.array(score_stds))
        # plt.title("Performance of the SVM-Anova varying the percentile of features selected")
        # plt.xticks(np.linspace(0, 100, 11, endpoint=True))
        # plt.xlabel("Percentile")
        # plt.ylabel("Accuracy Score")
        # plt.axis("tight")
        # plt.show()

        with open(name + ".txt", "w") as f:
            f.write("Parameters: " + str(clf.get_params()) + "\n---------------------\nMean Accuracy on the test set: " + str(clf.score(X_test, y_test)) + "\nRand Index: " + str(rand_index(y_test,clf.predict(X_test))))

        pickle.dump(clf, open(name, 'wb'))
        
    else:
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42, tol=1e-5))
        clf.fit(X_train, y_train)
        
        with open(name + ".txt", "w") as f:
            f.write("Parameters: " + str(clf.get_params()) + "\n---------------------\nMean Accuracy on the test set: " + str(clf.score(X_test, y_test)) + "\nRand Index: " + str(rand_index(y_test,clf.predict(X_test))))

    pickle.dump(clf, open(name + ".sav", 'wb'))

def main():

    # ------------------------------------------------------------- SAVE DATA PER CLASS V
    # with open("data/IAM-data/dic/word_label.json", "r") as f:
    #     data = json.load(f)

    # dic_classes = {}
    # Lists and image names are the same.
    # for key, value in zip(data.keys(), data.values()):
    #     if value in dic_classes:
    #         dic_classes[value].append(key)
    #     else:
    #         dic_classes[value] = [key]

    # with open("data/IAM-data/dic/class_data.json", "w") as f:
    #     json.dump(dic_classes,f)
    # ------------------------------------------------------------- SAVE DATA PER CLASS ^

    # ------------------------------------------------------------- SAVE DATA WITH ENCODED LABELS v

    # enc = label_encoder()

    # with open("data/IAM-data/dic/word_label.json", "r") as f:
    #     data_words = json.load(f)

    # encoded_dic = {}
    # for key, value in zip(data_words.keys(), data_words.values()):
    #     new_val = enc.transform([value])
    #     encoded_dic[key] = int(new_val[0])
    #     print(f"Encoded: {value} to {new_val[0]} for {key}")

    # with open("data/IAM-data/dic/word_label_encoded.json", "w") as f:
    #     json.dump(encoded_dic, f)
    # ------------------------------------------------------------- SAVE DATA WITH ENCODED LABELS ^

    # ------------------------------------------------------------- SAVE FULL ML READY DATASET v
    # NEED TO RERUN THIS AND NOT SAVE SHIT AS STRINGS
    # print(df.head())
    # enc = label_encoder()
    # i = 0
    # for f in sorted(os.listdir("data/IAM-data/lists")):
    #     if os.stat('final_data.csv').st_size == 0:
    #         df = pd.DataFrame()
    #         df['path_to_image'] = ""
    #         df['label'] = ""
    #         df['encoded_label'] = []
    #         df['Height'] = []
    #         df['Width'] = []
    #         df['Perimeter'] = []
    #         df['DCT_Norm'] = []
    #         df['HOG_Norm'] = []
    #         df['IMG_Norm'] = []
    #     else:
    #         df = pd.read_csv('final_data.csv')

    #     # df = pd.DataFrame(df).convert_dtypes()
    #     entry = list(np.load("data/IAM-data/lists/" + f, allow_pickle=True))
    #     # print(entry)
    #     entry[5] = entry[5].astype(float)
    #     entry[6] = entry[6].astype(float)
    #     entry[7] = entry[7].astype(float)
    #     entry.insert(2, enc.transform([entry[1]])[0])
    #     # print(entry)
    #     # a_series = pd.Series(entry, index = df.columns, dtype='float')
    #     # print(a_series)
    #     df.loc[i] = entry
    #     # df.append(pd.DataFrame(a_series), ignore_index=True)
    #     # print(df.tail())
    #     i += 1
    #     print(i)
    #     df.to_csv("final_data.csv", index=False)
    #     del df

    # STOPPED AT 34435
    # ------------------------------------------------------------- SAVE FULL ML READY DATASET ^

    # ------------------------------------------------------------- FIT MODELS V
    df = pd.read_csv('final_data.csv')
    
    Y = df['encoded_label'].tolist()

    X_HEIGHT = df['Height'].tolist()
    height_list = []
    for x in X_HEIGHT:
        height_list.append(int(x))
    X_HEIGHT = np.array(height_list)
    height_list = X_HEIGHT.reshape(-1, 1)

    X_WIDTH = df['Width'].tolist()
    width_list = []
    for x in X_WIDTH:
        width_list.append(int(x))
    X_WIDTH = np.array(width_list)
    width_list = X_WIDTH.reshape(-1, 1)

    X_PERIMETER = df['Perimeter'].tolist()
    perimeter_list = []
    for x in X_PERIMETER:
        perimeter_list.append(int(x))
    X_PERIMETER = np.array(perimeter_list)
    perimeter_list = X_PERIMETER.reshape(-1, 1)

    X_DCT = df['DCT_Norm'].tolist()
    DCT_list = []
    for x in X_DCT:
        x = x.replace("[", " ")
        x = x.replace("]", " ")
        DCT_list.append([float(item) for item in x.split()])
    X_DCT = np.array(DCT_list)
    DCT_list = X_DCT.reshape(-1, 1)

    # X_IMG = df['IMG_Norm'].tolist()
    # IMG_list = []
    # for x in X_IMG:
    #     print(x)
    #     x = x.replace("[", " ")
    #     x = x.replace("]", " ")
    #     IMG_list.append([float(item) for item in x.split()])
    # X_IMG = np.array(IMG_list)
    # IMG_list = X_IMG.reshape(-1, 1)
    # ------------------------------------------------------------- FIT SVM V

    # Features need to be the same length and <= 2 dimensions
    # X = np.array(X)
    # nsamples, nx, ny = X.shape
    # reshaped_data = X.reshape((nsamples,nx*ny))

    X = np.array([height_list,width_list,perimeter_list])
    X = X.transpose()
    # linear_svm(np.squeeze(X), Y, "X_HEIGHT_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(X_WIDTH, Y, "X_WIDTH_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(X_PERIMETER, Y, "X_PERIMETER_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(X_DCT, Y, "X_DCT_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(X_HOG, Y, "X_HOG_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(X_IMG, Y, "X_IMG_LINEAR_SVM_no_feature_selection", 0)
    # linear_svm(DCT_list, Y, "X_DCT_LINEAR_SVM_no_feature_selection", 0)
    
    # SGD_svm(X_HEIGHT, Y, "X_HEIGHT_SGDSVM_no_feature_selection", 0)
    # SGD_svm(X_WIDTH, Y, "X_WIDTH_SGDSVM_no_feature_selection", 0)
    # SGD_svm(X_PERIMETER, Y, "X_PERIMETER_SGDSVM_no_feature_selection", 0)
    # SGD_svm(X_DCT, Y, "X_DCT_SGDSVM_no_feature_selection", 0)
    # SGD_svm(X_HOG, Y, "X_HOG_SGDSVM_no_feature_selection", 0)
    # SGD_svm(X_IMG, Y, "X_IMG_SGDSVM_no_feature_selection", 0)
    # ------------------------------------------------------------- FIT SVM ^
    # ------------------------------------------------------------- FIT KNN V
    # FIT KNN ON HOG, HEIGHT, WIDTH AND IMAGE PCA NORM ARRAYS
    # For single feature arrays.
    # X = np.array(X)
    # X = X.reshape(-1, 1)

    supervised_knn(np.squeeze(X) , Y, "X_HEIGHT_KNN_k_selection", 100)
    # supervised_knn(X_WIDTH, Y, "X_WIDTH_KNN_k_selection", 0)
    # supervised_knn(X_PERIMETER, Y, "X_PERIMETER_KNN_k_selection", 0)

    # supervised_knn(X_DCT, Y, "X_DCT_KNN_k_selection", 0)
    # supervised_knn(X_HOG, Y, "X_HOG_KNN_k_selection", 0)
    # supervised_knn([X_IMG], Y, "X_IMG_KNN_k_selection", 0)
    # supervised_knn(DCT_list, Y, "X_DCT_KNN_k_selection", 0)
    # ------------------------------------------------------------- FIT KNN ^
    # ------------------------------------------------------------- FIT MODELS ^

if __name__ == '__main__':
    main()