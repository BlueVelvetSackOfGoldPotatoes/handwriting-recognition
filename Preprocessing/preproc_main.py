import rpy2.robjects as robjects
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

rootdir = './Outputs/'
rootdir2 = './Outputs/combined'
os.makedirs(rootdir, exist_ok=True)
os.makedirs(rootdir2, exist_ok=True)


r = robjects.r
r.source('A-star.R')  

for root, dirs, files in os.walk(rootdir):
    outpath = (f"{root}/Cropped/")
    if "combined" in dirs:
        dirs.remove("combined")
    if "Cropped" in dirs:
        dirs.remove("Cropped")
    for file in files:
        os.makedirs(outpath, exist_ok=True)
        print(f"Dealing with file {root}/{file}")
        img_directory = (f"{root}/{file}")
        img = cv2.imread(str(img_directory))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255*(gray < 50).astype(np.uint8)  # To invert the text to white
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))  # Perform noise filtering
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        # Crop the image - note we do this on the original image
        rect = img[y:y+h, x:x+w]
        res = cv2.resize(rect, dsize=(509,60), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(outpath,file),res[:, :, 1])