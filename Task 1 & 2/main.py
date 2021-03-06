import sys, os

import argparse
import glob
import shutil
import time

import rpy2.robjects as robjects
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from difflib import SequenceMatcher


#################
### Arguments ###
#################

# Parse arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-d", "--debug", help="run program in debug mode",action="store_true")
group.add_argument("-f", "--fast",  help="skip over intermediary results (used to speed-up the program)",action="store_true")
parser.add_argument("-e", "--extension", type=str, default="jpg",help="specify the extension (default=.jpg)",)
parser.add_argument("-o", "--output", type=str, default="Output",help="specify the output directory",)
parser.add_argument("input_dir",help="Specify the input directory")
args = parser.parse_args()

# Process parsed arguments
if args.debug:
    runmode = 2  # Show debug messages and intermediary steps
    runmode_str = "DEBUG"
elif args.fast:
    runmode = 0  # Do not show any debug messages or intermediary steps
    runmode_str = "FAST"
else:
    runmode = 1  # Default behaviour: show intermediary steps, but no debug
    runmode_str = "NORMAL"

# Set I/O directory names
input_directory = args.input_dir
extension = "*" + args.extension
files_directory = os.path.join(os.path.abspath(input_directory), extension)
files = glob.glob(files_directory)

output_directory = args.output
#ensure_directory(output_directory) # No need to remove the output folder first

# Handle empty directory
if (not files):
    print("ERROR: Directory " + input_directory + " is empty!")
    sys.exit()


# Prepare final directories
rootdir = './Preproc_Outputs/'
preproc_dir = '/Preproc/'
rootdir2 = './Preproc_Outputs/combined'
dst_dir = './DSS_Binarized'
os.makedirs(rootdir, exist_ok=True)
os.makedirs(rootdir2, exist_ok=True)
os.makedirs(dst_dir, exist_ok=True)

# Copy images from input directory to local directory (necessary for R-script)
for jpgfile in glob.iglob(os.path.join(input_directory, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)
    

# Loading and running the R script containing A* algorithm for line segmentation    
r = robjects.r
r.source('A-star.R')  

# Iterate over lines segmented in previous step and prepare them for the model
for root, dirs, files in os.walk(rootdir):
    outpath = (f"{root}/Cropped/")
    if "combined" in dirs:
        dirs.remove("combined")
    if "Cropped" in dirs:
        dirs.remove("Cropped")
    for file in files:
        # Preparing output directory
        os.makedirs(outpath, exist_ok=True)
        print(f"Dealing with file {root}/{file}")
        img_directory = (f"{root}/{file}")
        # Loading image
        img = cv2.imread(str(img_directory))
        # Creating a grayscale image to find bounding boxes to crop by
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # To invert the text to white
        gray = 255*(gray < 50).astype(np.uint8)
        # Perform noise filtering 
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) 
        # Find all non-zero points (text) 
        coords = cv2.findNonZero(gray) 
        # Find minimum spanning bounding box 
        x, y, w, h = cv2.boundingRect(coords)  
        # Crop the image - we do this on the original image
        rect = img[y:y+h, x:x+w]
        # Making sure image is saved in desired size
        res = cv2.resize(rect, dsize=(499,60), interpolation=cv2.INTER_LANCZOS4)
        # Saving image (grayscale)
        cv2.imwrite(os.path.join(outpath,file),res[:, :, 1])
        
        
        
# Standard vocab to ensure correct encoding
vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "+", " "]

# Batch size for training and validation
batch_size = 1

# Desired image dimensions
img_width = 499
img_height = 60

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Preprocessing ----------------------------------------------------------
# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=vocab, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and make sure image is grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Making sure that the image is in the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

# Loading the model
model = keras.models.load_model('trained_model')
# As max length is based on labels (which in this case are not relevant) we set it to a high nr so predictions can be made for lines of all lengths
max_length = 50
# Inference ----------------------------------
# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
#prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text



# Create final output directory for txt files
rootdir23 = 'results'
os.makedirs(rootdir23, exist_ok=True)
# Creating path to save images at
save_path = os.getcwd()
fin_out_complete = os.path.join(save_path, rootdir23)  


# Iterate over the preprocess images (that is cropped lines)
for root, dirs, files in os.walk(rootdir):
    temp_lines = []
    if "combined" in dirs:
        dirs.remove("combined")
    if "Cropped" in dirs:
        # Get unique id as folder name shows name of image lines were extracted from
        folderName = os.path.split(root)[1]
        data_dir = Path(f"{root}/Cropped/")
        # Read images in jpg format
        images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
        labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
        # Create tensorflow object out of these images
        test_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        test_dataset = (test_dataset.map(encode_single_sample).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        print("Predicted labels")
        print(folderName)
        
        # Making predictions
        for batch in test_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            # Predicting each line
            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)
            new_pred = list()
            for i in range(len(pred_texts)):
                # Removing unknown strings which were a result of the high max_length
                newstr = pred_texts[i].replace("[UNK]", "")
                new_pred += [newstr]
            
            # Transcribing predictions for each line
            for i in range(len(new_pred)):
                # Mapping English characters to Hebrew 
                line = new_pred[i].replace("a", "??")
                line = line.replace("b", "??")
                line = line.replace("c", "??")
                line = line.replace("d", "??")
                line = line.replace("e", "??")
                line = line.replace("g", "??")
                line = line.replace("f", "??")
                line = line.replace("i", "??")
                line = line.replace("h", "??")
                line = line.replace("j", "??")
                line = line.replace("l", "??")
                line = line.replace("k", "??")
                line = line.replace("m", "??")
                line = line.replace("n", "??")
                line = line.replace("p", "??")
                line = line.replace("o", "??")
                line = line.replace("q", "??")
                line = line.replace("r", "??")
                line = line.replace("s", "??")
                line = line.replace("t", "??")
                line = line.replace("u", "??")
                line = line.replace("v", "??")
                line = line.replace("w", "??")
                line = line.replace("x", "??")
                line = line.replace("y", "??")
                line = line.replace("z", "??")
                line = line[::-1]
                temp_lines += [line]
        
                # Get max length of predicted lines for padding output
                max_len = max([len(K) for K in temp_lines])
                name_of_file = str(folderName)
                completeName = os.path.join(fin_out_complete, name_of_file+".txt")         
                fo = open(completeName, 'w', encoding='utf-8')
                for z in temp_lines:
                    # Each line is padded with the maximum length
                    fo.write(z.rjust(max_len) + "\n")
                fo.close()
        print(temp_lines)