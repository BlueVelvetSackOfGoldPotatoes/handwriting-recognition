import cv2
import os
import ntpath
import numpy as np
import skimage.io as io

from PIL import Image
from skimage import feature
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from skimage.measure import regionprops

def visualize_data(feature_vec):
    """
    Example:
    ----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------
    File     : data/IAM-data/word_img/b06-059-09_word_5.png
    Label    : the
    Height   : 123
    Width    : 636
    Perimeter: 3234.19
    DCT Norm : (50,)
    HOG Norm : (128, 128)
    IMG Norm : (128, 128)
    ----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------
    File     : data/IAM-data/word_img/r06-044-10_word_8.png
    Label    : point
    Height   : 390
    Width    : 642
    Perimeter: 4791.61
    DCT Norm : (50,)
    HOG Norm : (128, 128)
    IMG Norm : (128, 128)
    ----------------------------------------------------------------------------------------------------

    """
    print("-"*100)
    print(f"(index: 0) File     : {feature_vec[0]}")
    print(f"(index: 1) Height   : {feature_vec[1]}")
    print(f"(index: 2) Width    : {feature_vec[2]}")
    print(f"(index: 3) Perimeter: {feature_vec[3]}")
    print(f"(index: 4) DCT Norm : {feature_vec[4].shape}")
    print(f"(index: 5) HOG Norm : {feature_vec[5].shape}")
    print(f"(index: 6) IMG Norm : {feature_vec[6].shape}")
    print("-"*100)

def pca_deployment(img):
    pca = PCA(10) # Yields an avg > 90% 
    img_transformed = pca.fit_transform(img)
    img_inverted = pca.inverse_transform(img_transformed)
    return img_inverted

def gen_feature_vector(img):

    image_object_boxer = Image.open(img)
    # im = np.asarray(image_object)

    image_object = cv2.imread(img)
    gray = cv2.cvtColor(image_object,cv2.COLOR_BGR2GRAY)
    saltpep = cv2.fastNlMeansDenoising(gray,None,21,15)
    thresh = cv2.threshold(saltpep,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    image_resized = cv2.resize(thresh, (128,128), interpolation = cv2.INTER_AREA)

    # PCA reduction
    img_inverted = pca_deployment(image_resized)
    image_resized_array = img_inverted.astype(np.uint8)
    image_resized_array_normalized = image_resized_array / 255

    bbox = image_object_boxer.getbbox()

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # FEATURE - perimeter
    image_ = io.imread(img)
    regions = regionprops(image_.astype(int))
    perimeter = round(regions[0].perimeter,2)

    # FEATURE - Discrete Cosine Transform
    dct_data = dct(image_resized_array_normalized,1)

    # Get first descending 50 coefficients of the dct
    dct_data_1d = dct_data.ravel()
    dct_data_1d_sorted = -np.sort(-dct_data_1d) # Descending
    dct_data_1d_normalized = dct_data_1d_sorted[:50] / np.sqrt(np.sum(dct_data_1d_sorted[:50] **2))

    # FEATURE - Histogram of oriented gradient
    _, Hog = feature.hog(image_object, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), visualize=True, multichannel=True)
    Hog_resized = cv2.resize(Hog, (128,128), interpolation = cv2.INTER_AREA)
    Hog_normalized = Hog_resized / np.sqrt(np.sum(Hog_resized**2))

    _, tail = ntpath.split(img)

    with open('lists/' + tail[:-4] + ".npy", 'wb') as f:
        np.save(f, np.array([img, height, width, perimeter, dct_data_1d_normalized, Hog_normalized, image_resized_array_normalized], dtype=object))

    return np.array([img, height, width, perimeter, dct_data_1d_normalized, Hog_normalized, image_resized_array_normalized], dtype=object)

def get_words(img_path):
    """
    Get gray tones image, apply denoising and threshold that. Generate contours from dilated image using a 20,20 kernel. Return images generated from cut contours.

    Parameters
    ----------
        image: a path to an image

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

    i = 1

    for ctr in ctrs:
        # print("Word: ", label) # remove
        # print(f"-"*20)
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = thresh[y:y+h, x:x+w]

        # inter_nearest is the best so far, inter_area
        im = cv2.resize(roi, None, fx=3, fy=3, interpolation = cv2.INTER_NEAREST)

        # New word image path
        _, tail = os.path.split(img_path)
        word_path = "processed_images/" + tail[:-4] + "_word_" + str(i) + ".png"
        cv2.imwrite(word_path, im)

        words.append(gen_feature_vector(word_path))
        i = i + 1 # word label
    return words