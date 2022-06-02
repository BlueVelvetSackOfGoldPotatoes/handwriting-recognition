"""
This file is responsible for reading the data image files and labels into a dictionary, saving it as a key:value json file where keys are the image paths and values are the labels for these images. These two lists are then split into training set and testing set using sklearn train_test_split. The testing set consists of 20% of the overall data and the random state is 42 for reproducibility sake.
"""

from tracemalloc import stop
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import sys
import re
import string
import json
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

# Paths
path_to_img_files = "data/IAM-data/img/"
path_to_input_text_file = "data/IAM-data/iam_lines_gt.txt"
path_to_output_text_file = "output.txt"
path_to_json = "data/IAM-data/dic/sentence_no_processing_dictionary.json"

# Global preprocessing vars
# Img size: to test - 28, 32, 64, 128
img_size = 28
# Word vs character segmentation
segment = "word"
        
def read_json(path):
    print("----------------")
    print("Reading Json ...")
    print("----------------")
    print()
    with open(path) as f:
        data = json.load(f)
    
    # Order is conserved
    return list(data.keys()), list(data.values())

def count_chars(data):
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    a_chars = 0
    a_punct = 0

    for s in data:
        a_chars = a_chars + count(s, string.ascii_letters)
        a_punct = a_punct + count(s, string.punctuation)

    print(a_chars)
    print(a_punct)

def generate_binary_nonbinary_label_word_data(y):
    """
    On the labels, Y:
        One-hot encoding - result is a categorical dataset.
        Or
        Label binarizer for each word.

        PARAMATERS:
            y : list of sentence label strings (well mappoed to sentence image labels).
        -----------
        
        Returns:
            word_labels : list of word label strings (well mapped to word image labels)
            word_labels_binarized : list of word label vectors (well mapped to word image labels)
        -----------
    """

    print("----------------------")
    print("Generating Y labels...")
    print("----------------------")
    print()

    unique_words = []
    word_labels = []

    # Get each individual word into solo_words and all strings into word_labels so as to match the img word data
    words_count = []
    for words in y:
        for word in words.split(" "):
            word_labels.append(word)
            if word not in unique_words:
                unique_words.append(word)
                words_count.append(word_labels.count(word))
        print(f"Word count entry: {word} : {word_labels.count(word)}") 

    # Y_words = OneHotEncoder(handle_unknown='ignore')
    # Y_words.fit(words_count)

    encoder = LabelBinarizer()
    encoded_labels = encoder.fit_transform(unique_words)
    print("Encoded labels size: {} and unique words: {}".format(len(encoded_labels), len(unique_words)))

    with open("data/IAM-data/data_count.txt", "a") as f:
        for s in range(len(unique_words)):
            print("writing '{}' to file".format(unique_words[s]))
            f.write("Word:{0} | Count:{1}\n".format(unique_words[s], words_count[s]) + "--------------\n")
        
    label_to_bin = []
    
    for word in word_labels:
        print("Encoding '{}' to {}".format(word, encoded_labels[unique_words.index(word)]))
        label_to_bin.append(encoded_labels[unique_words.index(word)])

    return label_to_bin, word_labels

def binarize_img_keys(dic):
    """
    Binarizes images using PIL and numpy. 

    PARAMETERS: 
        dic: an img (path) : label (binarized) dictionary
    -----------

    Returns:
        bin_img: an img (binary numpy vector) list
    """
    print("-----------------------")
    print("Binarizing img keys ...")
    print("-----------------------")
    print()
    bin_img = []
    for key in dic:
        print("Binarizing {}".format(key))
        img = Image.open(key) # create PIL image object
        img_array = np.array(img) # Map PIL img object to numpy array
        print("Saving {}".format(img_array.tolist()))
        bin_img.append(img_array.tolist())
    return bin_img

def save_dictionary(X, Y, output_path):
    print("---------------------")
    print("Saving Dictionary ...")
    print("---------------------")
    print()
    dic = {}
    for x, y in zip(X, Y):
        print("Saving pair -> {} : {}".format(x, y))
        dic[x] = y

    with open(output_path, "w") as f:
        json.dump(dic,f)

# def generate_binary_feature_labels(x, y):

def get_img(path):
    print("-----------------------")
    print("Saving imgs to list ...")
    print("-----------------------")
    print()
    img_list = []
    for img in os.listdir(path):
        img_path = path + img
        img_list.append(img_path)
    return img_list

# Generate dictionary for img:label data
def save_data_to_json(output_path):
    dic = {}
    lines = []
    x = []
    y = []
    with open(path_to_input_text_file) as f:
        for line in f:
            line = line[:-1]
            if line != "":
                lines.append(line)
        # Correcting for the removal of the last character in the last label (removed as a consequence of avoiding new line character)
        lines[-1] = lines[-1] + "l"

    # Place image path in x and labels in y
    for i in range(1, len(lines), 2):
        if i % 3 == 0:
            continue
        x.append(path_to_img_files + lines[i-1])
        y.append(lines[i])
        dic[path_to_img_files + lines[i-1]] = lines[i]

    with open(output_path, "w") as f:
        json.dump(dic,f)
    return dic

# USE THIS TO RESIZE TO A GOOD ASPECT RATIO AND REMOVE UNDESIRABLE IMAGES DUE TO CLASS SIZE BEING TOO SMALL
def rename_img(path):
    with open(path, "r") as f:
        dic = json.load(f)

    for img, label in zip(dic.keys(), dic.values()):
        img = img.astype('float64')

        normalized_img *= 255.0/img.max()

        exit()

def attempt_dilations(list_cut_words, img, median_img, thresholded_img, words, alpha, n_contours):
    
    # Higher threshold selects for loose parts of word, e.g. the dot in 'i', or an 'hyphen'
    kernel = np.ones((alpha,alpha),np.uint8)
    dilated_img = cv2.dilate(median_img,kernel,iterations = 3)

    ret, thresh = cv2.threshold(dilated_img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_contours = len(contours)
    print(f"N_contours:{n_contours}")

    if n_contours == words:
        i = 0
        for cnt, c in zip(contours, range(len(contours))):
            x,y,w,h = cv2.boundingRect(cnt)
            rect_img = cv2.rectangle(median_img,(x,y),(x+w,y+h),(0,255,0),2)
            #  FOR REPORT WRITING
            # cv2.imwrite("preprocessing_imgs/contour" + str(i) + ".jpg", img)

            mask = np.zeros_like(rect_img) # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, contours, c, 255, -1) # Draw filled contour in mask
            out = np.zeros_like(rect_img) # Extract out the object and place into output image
            out[mask == 255] = rect_img[mask == 255]

            # Now crop
            (y, x) = np.where(mask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out = out[topy:bottomy+1, topx:bottomx+1]
            
            _, tail = os.path.split(img)
            path = "data/IAM-data/word_img/" + tail[:-4] + "_word_" + str(i) + ".png"
            cv2.imwrite(path, out)
            i = i + 1
            list_cut_words.append(path)
        return list_cut_words

    elif n_contours > words and alpha <= 25:
        print(f"Alpha: {alpha}")
        attempt_dilations(list_cut_words, img, median_img, thresholded_img, words, alpha+1, n_contours)
    # exit()
    else:
        return []
            
def generate_img_word_data(dic):
    """
        PARAMETERS
        ----------
            x: list of path to img
            y: list of strings

        RETURNS
            X: processed list of paths for img per word
        ----------

    """
    # Following from  "Handwritten Digits Recognition Using SVM, KNN, RF, and Deep Learning Neural Networks" by Chychkarov, Serhiienko, Syrmamiikh, Kargin.
    # On the input data, X:
    #     1. Greyscaling and filtering to reduce noise: gaussian filter - cv2.GaussianBlur
    #     2. Binarization to cut off noise: cv2.threshold (parameters to be adapted to use)
    #     3. Highlight contrasting border: Canny edge detector - cv2.Canny (maximum and minimum valuesof the gradient must be preliminarily selected)
    #     4. Reduce image heterogeneity (median filter using cv2.medianBlur)
    #     5. Image binarization (cv2.threshold)
    #     6. Morphological transformation (dilatation - cv2.dilate)
    #     7. Select contours and their sorting (cv2.findContours)
    #     8. Image segmentation: cv2.boundingRect
    #         Slice images: count words in label and slice image into that many images. Slice where the distance is "significant" between pixels.
    #         Delete under represented words - make histogram of words first and then decide

    #     9. 28*28 - last step is image normalization (why last?)
    #     10. convert dataset to float32 and divide by 255.0, making each feature [0.0, 1.0]
    # On the labels, Y:
    #     One-hot encoding - result is a categorical dataset.

    # Features to be selected (from Handwriting Word Recognition Based on SVM Classifier by Mustafa and Alia) :
    #     1. DCT - Discrete Cosine Transform converts pixel values into its elementary frequency components. Arrange DCT array by largest to smallest and select the first 50 components.
    #     2. HOG - Histogram of Oriented Gradient: counts occurrences of gradient orientation in part of an image.

    #     Both features are normalized following this formula: A' = (A - Min(A))/(Max(A) - Min(A))
    # Remove punctuation and spaces within words
    # for i in range(len(y)):
    #     y[i] = re.sub(r'[^\w\s]', '', y[i])
    #     y[i] = re.sub(' {2,}', ' ', y[i])

    word_dic = {}

    for img, label in zip(dic.keys(), dic.values()):
        # Label processing
        words = label.split(" ")
        spaces = len(words) - 1
        print("-"*20)
        print(img)
        print("words: {}".format(words))
        print("count: {}".format(len(words)))
        print(label)
        print("-"*20)

        # Img processing
        cv2_img = cv2.imread(img)
        grey_scale = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        # Can be bumped by x+10, y*2
        thresholded_img = cv2.adaptiveThreshold(grey_scale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,31,6)

        blur_img = cv2.GaussianBlur(thresholded_img,(5,5),0)

        plt.imshow(blur_img)
        plt.show()
        exit()

        edge_img = cv2.Canny(thresholded_img,100,200)
        
        median_img = cv2.medianBlur(edge_img,1)

        #  FOR REPORT WRITING
        # thresholded_img2 = cv2.adaptiveThreshold(median_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    # cv2.THRESH_BINARY,31,6)
        list_cut_words = []

        attempt_dilations(list_cut_words, img, median_img, thresholded_img, len(words), 1, 0)
        
        print(list_cut_words)
        
        for path, word_label in zip(list_cut_words, words):
            print("WRITING")
            word_dic[path] = word_label
            print(f"Key -> {path} : label -> {word_label}")
            
        #  FOR REPORT WRITING
        # cv2.imwrite("preprocessing_imgs/cv2_img.jpg", cv2_img)
        # cv2.imwrite("preprocessing_imgs/blur_img.jpg", blur_img)
        # cv2.imwrite("preprocessing_imgs/thresholded_img.jpg", thresholded_img)
        # cv2.imwrite("preprocessing_imgs/edge_img.jpg", edge_img)
        # cv2.imwrite("preprocessing_imgs/median_img.jpg", median_img)
        # cv2.imwrite("preprocessing_imgs/thresholded_img2.jpg", thresholded_img2)
        # cv2.imwrite("preprocessing_imgs/dilated_img.jpg", dilated_img)

    return word_dic

def data_processing():

    # Build img : label dictionary
    # save_data_to_json("data/IAM-data/dic/sentence_no_processing_dictionary.json")
    # x, y = read_json("data/IAM-data/dic/sentence_no_processing_dictionary.json")
    with open('data/IAM-data/dic/sentence_no_processing_dictionary.json') as json_file:
        unprocessed_dic = json.load(json_file)

    word_dic = generate_img_word_data(unprocessed_dic) # GENERATES THE WORD IMG DATA - RETURNS FILE PATHS
    
    with open("data/IAM-data/dic/unserialized_word_data.json", "w") as f:
        json.dump(word_dic,f)
    
    # X_words = get_img("data/IAM-data/word_img/")
    exit()

    Y_words_binarized, Y_words = generate_binary_nonbinary_label_word_data(y)

    # Save dictionary word img path : string word data
    save_dictionary(X_words, Y_words, "data/IAM-data/dic/unserialized_word_data.json")

    # Save dictionary word binarized img : binarized word data
    X_words_binarized = binarize_img_keys(X_words)
    save_dictionary(X_words_binarized, Y_words_binarized, "data/IAM-data/dic/serialized_word_data.json")

"""
TODO
2. Keep track of ML-ready mapping and its original data set via linked dictionaries (a list of two dictionaries where the first is the vanilla one and the second is the ML-ready or processed one)
3. Generate these dictionaries: the way to do this, for replicability purposes, is by method independence -that is, input dictionary - singular method - output dictionary. Avoid interdependency at all costs.
4. Add the class size in the feature list
5. change image size based on 
"""