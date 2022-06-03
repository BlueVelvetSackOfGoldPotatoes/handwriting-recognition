import numpy as np
import json
import skimage.io as io
import pandas as pd
import cv2
import pickle
import sys
import ntpath

from PIL import Image
from scipy.fftpack import dct
from matplotlib import pyplot as plt
from skimage import feature
from skimage.measure import regionprops
from sklearn.decomposition import PCA

np.set_printoptions(threshold=sys.maxsize)

# pca = pickle.load(open("pca.pkl",'rb'))

# PCA only for distance measure

# 90.0068716076899 explained variance ratio
# print(np.cumsum(pca.explained_variance_ratio_ * 100)[-1])

# 41.762 WORD IMAGES = 14 BATCHES OF 2983 IMAGES EACH

# data_features[0][0] -> word img path, string
# data_features[0][1] -> label, string
# data_features[0][2] -> height, int
# data_features[0][3] -> width, ing
# data_features[0][4] -> perimeter, float
# data_features[0][5] -> dct_data_1d_normalized, list (50) FED image_resized_array_normalized
# data_features[0][6] -> Hog_normalized, list (105300) FED image_object
# data_features[0][7] -> img_array_normalized, list (125, 125) FED PCA

# USE THIS FIRST TO CHECK APPROPRIATE DIMENSION TO REDUCE DATA OF FEATURE VECTOR BEFORE SAVING NPY
def pca_explained_variance_ratio(data):
    pca = PCA(n_components=0.9).fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    return pca

def pca_plot_scatter(data):
    pca = PCA(2)
    X_new = pca.inverse_transform(data)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show(0)

def pca_plot_components(data):
    # NEED TO CHANGE DATA TO DATA[N] WHERE N IS THE INDEX OF THE FEATURE
    pca = PCA(2)
    projected = pca.fit_transform(data)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=data[1], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

def generate_pca_data():
    with open("data/IAM-data/dic/word_label.json", "r") as f:
        data = json.load(f)
    img_n = 5000
    pca_data = []
    for img, label in zip(data.keys(), data.values()):
        if img_n == 0:
            return pca_data
        pca_image = cv2.imread(img)

        gray = cv2.cvtColor(pca_image,cv2.COLOR_BGR2GRAY)
        saltpep = cv2.fastNlMeansDenoising(gray,None,21,15)
        thresh = cv2.threshold(saltpep,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        pca_image = cv2.resize(thresh, (128,128), interpolation = cv2.INTER_AREA)
        pca_image = pca_image.astype(np.uint8)
        pca_image_normalized = pca_image / 255
        pca_image = pd.Series(pca_image_normalized.flatten(), name=img)
        pca_data.append(pca_image)
        img_n -= 1
    return pca_data

def pca_reduce_img(img):
    pca = PCA(n_components=100)
    pca_fitted = pca.fit_transform(img)
    print(pca_fitted.shape)
    pca_recovered = pca.inverse_transform(pca_fitted)

    image = pca_recovered[1,:].reshape([128,128])

    print(np.cumsum(pca_fitted.explained_variance_ratio_ * 100)[-1])
    plt.imshow(image, cmap='gray_r')
    print(image.shape)
    print("-------------------------------------------")
    return pca_fitted

def gen_feature_vector():
    with open("data/IAM-data/dic/word_label.json", "r") as f:
        data = json.load(f)

    pca = PCA(10) # Yields an avg > 90% 
    pca_sum = 0
    for img, label in zip(data.keys(), data.values()):
        image_object_boxer = Image.open(img)
        # im = np.asarray(image_object)

        image_object = cv2.imread(img)
        gray = cv2.cvtColor(image_object,cv2.COLOR_BGR2GRAY)
        saltpep = cv2.fastNlMeansDenoising(gray,None,21,15)
        thresh = cv2.threshold(saltpep,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        image_resized = cv2.resize(thresh, (128,128), interpolation = cv2.INTER_AREA)

        # PCA
        img_transformed = pca.fit_transform(image_resized)
        img_inverted = pca.inverse_transform(img_transformed)

        pca_sum += (np.cumsum(pca.explained_variance_ratio_ * 100)[-1])
        # print(img_inverted.shape)
        # plt.imshow(img_inverted)
        # plt.show()

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
        Hog = feature.hog(image_object, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
        Hog_normalized = Hog / np.sqrt(np.sum(Hog**2))
        # print(Hog_normalized.shape)

        _, tail = ntpath.split(img)

        with open('data/IAM-data/lists/' + tail[:-4] + ".npy", 'wb') as f:
            np.save(f, np.array([img, label, height, width, perimeter, dct_data_1d_normalized, Hog_normalized, image_resized_array_normalized], dtype=object))
    print(" Explained variance ratio %{}".format(pca_sum/41762))
def main():
    # FITTING PCA FIRST -----------------------------------------
    # pca_data = generate_pca_data()
    # pca = pca_explained_variance_ratio(pca_data)
    # pickle.dump(pca, open("pca.pkl","wb"))

    gen_feature_vector()


if __name__ == '__main__':
    main()