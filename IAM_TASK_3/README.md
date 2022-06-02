## IAM Task 3
. Download the IAM data from here (338 MB):
https://unishare.nl/index.php/s/jtrXypbkrp69iyz

The input to your program should be a path to the directory that contains the line images. Your code should write the recognized text for each line to a txt file with the same structure as we used to provide the ground-truth data (two lines per prediction: one for the image name, one for the text prediction, where predictions are separated by two lines). To estimate the performance, character-level and word-level accuracy will be measured.

Line recognition (IAM dataset):
You will perform text recognition on line images from the IAM dataset. Given an input image of a handwritten line of text, the task is to transcribe the text in the image. As an example,see Fig. 2. The expected output would be “Susan Hayward plays the wife sharply”. Rather than segmenting words or characters as a preprocessing step, you may want to consider an end-to-end deep learning solution without initial segmentation. Modern deep learningarchitectures for handwriting recognition can deal with varying image sizes, e.g., using an attention mechanism. It is your job to find the most suitable solution for the task. Since deep learning methods thrive on large volumes of data, you may again want to consider using data augmentation methods to increase the number of training samples. Key words: encoder-decoder, CNN, RNN, Transformer, CTC loss, attention, Word  Error Rate, Character Error Rate, etc.

Milan is looking into cnn rnn without input transformation.

# TODO
Read images and labels

