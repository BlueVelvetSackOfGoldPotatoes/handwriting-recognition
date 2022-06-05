# HELLO, START HERE V
Install packages contained in environment.yml please.
    conda env create -f envrionment.yml
    conda activate hwr_clustering

    If this does not work, attempt:
        pip install -r requirements.txt

    If this also does not work, then install each of the libraries listed in requirements.txt by hand, like so:
        pip install libname

## EXECUTE TO RUN
    python main.py

## PROGRAM FLOW
(0) In sum, main.py will take in your folder path and iteratively process each image, generating the contents of processed_images - the segmented word images, as well as the npy files storing the feature vectors from these segmented images, stored in "lists";
(1) preprocessing_pipeline.py will processed the input images, produce individual word images and collect features from these, saving them in a npy file and later in a pandas dataframe;
(2) deploy_models.py will produce the output classification using the trained models - SVM and KNN;
(3) main.py saves the output of this classification (SVM and KNN) to output.txt;

## FOLDERS:
(i) In lists you'll find the feature vectors generated for each segmented word;
(ii) In models you will find the trained models, in this case an SVM and a KNN;
(iii) In processed_images you will find the image files for the segmented images, generated from your output.