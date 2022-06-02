import cv2
import numpy as np
import re
import os
import json
from PIL import Image
from scipy.fftpack import fft, dct
from matplotlib import pyplot as plt
from skimage import feature
from skimage import exposure

def get_words(img_path, word_labels):
    """
    Get gray tones image, apply denoising and threshold that. Generate contours from dilated image using a 20,20 kernel. Return images generated from cut contours.

    Parameters
    ----------
        image: a cv2 loaded image object.

    Returns
    ----------
        words: a list of cv2 loaded image objects.
    """
    image = cv2.imread(img_path)

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Remove Salt and pepper noise
    saltpep = cv2.fastNlMeansDenoising(gray,None,21,15)

    thresh = cv2.threshold(saltpep,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #dilation
    # Dilation mask could follow from average distance between pixels - the larger the distance, the thicker the mask.
    kernel = np.ones((20,20), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by caught word:
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in ctrs]
    (ctrs, boundingBoxes) = zip(*sorted(zip(ctrs, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    words = []

    word_labels = [c.strip() for c in re.split('(\W+)', word_labels) if c.strip() != '']

    # print(word_labels)

    # get word labels
    # for i in range(1, len(word_labels)):
    #     prev = word_labels[i-1]
    #     if "i" in prev or "j" in prev:
    #         # print(f"Comparing: {word_labels[i]} and {prev}")
    #         word_labels.insert(i,"dot")

    i = 1

    for label, ctr in zip(word_labels, ctrs):
        # print("Word: ", label) # remove
        # print(f"-"*20)
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Features height and width
        height = h
        width = w
        # print(f"width: {w} and height: {h}")

        # Getting ROI
        roi = thresh[y:y+h, x:x+w]

        # inter_nearest is the best so far, inter_area
        im = cv2.resize(roi, None, fx=3, fy=3, interpolation = cv2.INTER_NEAREST)

        # FEATURE - binary vector
        img_array = np.asarray(im) # Map PIL img object to numpy array
        # Normalize the image
        img_array_normalized = img_array.astype('float32')
        img_array_normalized /= 255.0
        # print(f"img_array_normalized: {img_array_normalized}")
        # Visualize ones and zeros count

        # FEATURE - Discrete Cosine Transform
        dct_data = dct(img_array_normalized,1)

        # Get first descending 50 coefficients of the dct
        dct_data_1d = dct_data.ravel()
        dct_data_1d_sorted = -np.sort(-dct_data_1d) # Descending
        dct_data_1d_normalized = dct_data_1d_sorted[:50] / np.sqrt(np.sum(dct_data_1d_sorted[:50] **2))   
        # print(f"dct_data_1d_normalized: {dct_data_1d_normalized}")

        # FEATURE - perimeter
        perimeter = round(cv2.arcLength(ctr,True), 2)
        # print(f"perimeter: {perimeter}")

        # FEATURE - Histogram of oriented gradient
        Hog = feature.hog(im, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
        Hog_normalized = Hog / np.sqrt(np.sum(Hog**2))
        # print(f"hog: {Hog_normalized}")

        # New word image path
        _, tail = os.path.split(img_path)
        word_path = "data/IAM-data/word_img/" + tail[:-4] + "_word_" + str(i) + ".png"
        # cv2.imwrite(word_path, im)

        words.append([word_path, label, height, width, img_array_normalized, dct_data_1d_normalized, Hog_normalized, perimeter])
        i = i + 1 # word label
        # print(f"At img: {tail}, word: {i}")
    # exit()
    return words

def main():
    with open('data/IAM-data/dic/sentence_no_processing_dictionary.json') as json_file:
        dic = json.load(json_file)

    first_run = 1

    for img_path, label in sorted(zip(dic.keys(), dic.values())):
        words = get_words(img_path, label)
        print(f"img_path: {img_path} and label: {label}")

        if words == 0:
            continue
        
        # If first run then the files need to be created
        if first_run:
            # word_label_dic = {}
            final_data = []

            for word in words:
                # word_label_dic[word[0]] = word[1] # Save path to word image and label for word in dic
                np.append(final_data,word) # Save complete list with features, image, label, etc

            first_run = 0

        # Files already exist
        else:
            # word_label_f = open("word_label.json", "r")
            final_data_lists_f = open("final_data_lists.npy", "rb")

            # word_label_dic = json.load(word_label_f)
            final_data = np.load(final_data_lists_f, allow_pickle=True)
            final_data_lists_f.close()
            # word_label_f.close()

            for word in words:
                # word_label_dic[word[0]] = word[1] # Save path to word image and label for word in dic
                np.append(final_data,word) # Save complete list with features, image, label, etc

        # word_label_f = open("word_label.json", "w")
        final_data_lists_f = open("final_data_lists.npy", "wb")

        # Rewrite the files
        # json.dump(word_label_dic, word_label_f, sort_keys=True)
        np.save(final_data_lists_f, final_data)

        # Close files
        final_data_lists_f.close()
        # word_label_f.close()

        # Free space from variables to avoid killing process
        del word_label_dic
        # del final_data

if __name__ == '__main__':
    main()

# TODO 1. grab all contours of hierarchy 0, ignore the largest one (assuming it is the bounding box for the window), make a mask for everything that is not the 2nd largest and fill it with black thus elliminating the blobs outside of the word. 2. Clean bad data after feature vector (threshold against average length/height/pixel count/etc)