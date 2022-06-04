# handwriting-recognition
Course and projects for handwriting recognition course AI MSc 2022

REMEMBER, DO NOT ADD DATA/IMG FILES/FOLDERS TO THE REPO! 

# OTHER README FILES FOR SUBMISSION
Make sure to include a README.md file in your submission with clear instructions on how to set up dependencies and compile and run your code. Ideally, you should provide a list of commands that we can run in a Linux environment to install all the dependencies. Your list of dependencies should
be appropriately updated once you are finished with the project. If necessary, test your code on the system of one of your group members to make sure it works on systems besides your own. If you use Python, it is highly recommended to work with a virtual environment for managing packages. If you want, you can also use a Docker container to manage your dependencies. In that case, please
describe clearly in the README how we can start your container and give it access to the input and output directories. After your code is set up, we should be able to run each method by executing a command and providing paths to the input and output directories as command-line arguments. Again, make sure
you describe how to do this in your README and avoid using hard-coded paths.

 
## Dead Sea Scrolls Task 1 & 2

Preparatory steps:
- Python version == 3.8.13
- Make sure you have a working installation of R (https://cran.r-project.org/bin/windows/base/)
- Install all the required libraries using pip install -r requirements.txt
- The model was compiled using cuDNN 8.1.0 and Cuda 11.2 - make sure you have these installed as well (conda example for installation: conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0)

Running the model:
- Run main.py in the format:
	- python main.py c:\users\username\desktop\folder 
	- where folder contains jpg images of binarized Dead Sea Scrolls
- The console will ask you whether you wish to use a personal library. Please type yes to continue  
-  Final txt files can be found in Final_output

## IAM Task 3
. Download the IAM data from here (338 MB):
https://unishare.nl/index.php/s/jtrXypbkrp69iyz

The input to your program should be a path to the directory that contains the line images. Your code should write the recognized text for each line to a txt file with the same structure as we used to provide the ground-truth data (two lines per prediction: one for the image name, one for the text prediction, where predictions are separated by two lines). To estimate the performance, character-level and word-level accuracy will be measured.

# TODO

1. Preprocessing and character segmentation (DSS dataset):
In order to recognize each character, the first step is to segment them from the test images. The output of the segmentation should be similar to the ones in the training set. The performance can be measured by the overlap (Intersection over Union IoU) between your box and the ground-truth. (one example can be found in page-8 of this paper). In this project, we will not make any evaluation on the bounding box, this is just for your own knowledge of how things work. Keywords for literature: line segmentation, center of gravity (for character segmentation), horizontal/vertical bound, etc.

2. Character recognition (DSS dataset):
(The reading direction for Hebrew is from right to left.) Your recognizer should recognize all the characters segmented from the first step. You can use
the printed characters (font/s) to pre-train your model and then fine-tune using the provided training data set. You can try different models. Note that the total number of characters in actual test images is unknown (for now!) but will be similar to the sample images. If you do not have enough training samples, some data augment methods might be used. You can use the image-morph tool from Lambert: https://github.com/GrHound/imagemorph.c
Key words: Character morphing, Elastic morphing.

You can also train a character recognizer (step 2) and apply it to detect the character box (step 1) using a sliding window strategy. Your main task is the design a character recognizer irrespective of the way you want to do that. In case you want to integrate the linguistic context, we are providing here the n-grams (bi-gram, tri-gram etc.): 
https://unishare.nl/index.php/s/xDFgy2HiNgqsxR7

(The easiest way to explain these bi-grams or tri-grams are as follows:
If there are three instances of the word CAT, five instances of CAPE, and two instances of CANADA in a document, the count will be as follows:
CAT = 3
CAP = 5
CA = 2
You can see here that ’when it is possible’ to count the tri-grams or more, it has been done following ’CA-’, else it was taken as bi-grams of ’CA’. It is obvious that you can find CA a total of ten times; that is true. However, this is not how the n-grams were calculated on the provided list. For instance: in the case of Kaf Waw, you can find the combination in both Kaf Waw and Kaf Waw Lamed (so the total number would be the sum of both the counts).) Anticipatory buzzwords: Binarization, OTSU, HMM, CNN, LSTM. Packages: Tensorflow, Theano, Keras, Caffe etc.

3. Line recognition (IAM dataset):
You will perform text recognition on line images from the IAM dataset. Given an input image of a handwritten line of text, the task is to transcribe the text in the image. As an example,see Fig. 2. The expected output would be “Susan Hayward plays the wife sharply”. Rather than segmenting words or characters as a preprocessing step, you may want to consider an end-to-end deep learning solution without initial segmentation. Modern deep learningarchitectures for handwriting recognition can deal with varying image sizes, e.g., using an attention mechanism. It is your job to find the most suitable solution for the task. Since deep learning methods thrive on large volumes of data, you may again want to consider using data augmentation methods to increase the number of training samples. Key words: encoder-decoder, CNN, RNN, Transformer, CTC loss, attention, Word  Error Rate, Character Error Rate, etc.

4. Presentations
	1. Overview of articles you have read (including a total number read so far);
	2. Amount of text written for literature review & methodology section of the paper;
	3. Programming progress (how many modules will the system have, how far is each module
	completed?);
	4. Empirical evaluation progress (both technical and theoretical);
	5. Is the overall progress on schedule?

Appoint a person for maintaining the progress and updating PPT slides; a person overlooking overall system architecture, a person designing the empirical evaluation (test scripts), etc. Each group will have 5-6 minutes for presentation and 2 minutes for Q&A. The presentations will be graded and count for 10% of the final course grade.

5. Report

The report needs to be a scientific paper about the handwriting recognition system(s) that you built during the course. It should be around 4000-5000 words long, written in English (roughly 8-10 pages). The report should be structured like a scientific article and consist of the following sections:

	• Title + Group number and members’ names & student numbers
	• Abstract
	• Introduction and Literature review
	• Methods
	• Results and detailed descriptions
	• Discussion and conclusions (submit individually)
	• References
	• Appendix, if any
	• Note on individual contribution in the project
	
The report should be submitted as a group except for the ’Discussions and conclusions’ section. Every member should contribute extensively to the group report. Coordinate the distribution of focus points among your team and describe which group members contribute to which topic. The ’Discussions and conclusions’ should be written and handed in individually (please mention
your name, student number, and group number at the top). It should be a separate (PDF) document of around 800-1200 words ( 2 pages) where individual students reflect and discuss the project work they did as a group. You can use the discussion to demonstrate that you understand the project and to reflect on the methods that you used and the results you obtained. You can also compare the methods you used for both datasets and their preconditions, advantages, and disadvantages. Your discussion can contain a small bibliography if necessary. 

# IDEAS

Make a dynamic system that can be updated through transferlearning / saves model states after training / produces evaluative statistics automatically etc.
