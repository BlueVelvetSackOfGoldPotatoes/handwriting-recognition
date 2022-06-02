import cv2
import numpy as np
import re
import os
import json
import skimage.io as io
from PIL import Image
from scipy.fftpack import fft, dct
from matplotlib import pyplot as plt
from skimage import feature
from skimage import exposure
from skimage.measure import label, regionprops

from sklearn.decomposition import PCA

# USE THIS FIRST TO CHECK APPROPRIATE DIMENSION TO REDUCE DATA OF FEATURE VECTOR BEFORE SAVING NPY
def pca_explained_variance_ratio(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


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


def main():

    with open("data/IAM-data/dic/word_label.json", "r") as f:
        data = json.load(f)

    data_feature = []

    for img, label in zip(data.keys(), data.values()):
        # print(f"image: {img} and label: {label}")

        # Load image
        image_object = Image.open(img)
        im = np.asarray(image_object)

        bbox = image_object.getbbox()

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Normalize the image
        img_array_normalized = im.astype('float32')
        img_array_normalized /= 255.0

        # FEATURE - Discrete Cosine Transform
        dct_data = dct(img_array_normalized,1)

        # Get first descending 50 coefficients of the dct
        dct_data_1d = dct_data.ravel()
        dct_data_1d_sorted = -np.sort(-dct_data_1d) # Descending
        dct_data_1d_normalized = dct_data_1d_sorted[:50] / np.sqrt(np.sum(dct_data_1d_sorted[:50] **2))

        # FEATURE - perimeter
        image_ = io.imread(img)
        regions = regionprops(image_.astype(int))
        perimeter = regions[0].perimeter

        # FEATURE - Histogram of oriented gradient
        Hog = feature.hog(image_object, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
        Hog_normalized = Hog / np.sqrt(np.sum(Hog**2))

        data_feature.append([img, label, height, width, img_array_normalized, dct_data_1d_normalized, Hog_normalized, perimeter])

        # with open('data/IAM-data/lists/final_data.npy', 'ab') as f:
        #     np.save(f, [img, label, height, width, img_array_normalized, dct_data_1d_normalized, Hog_normalized, perimeter])
    
    

if __name__ == '__main__':
    main()