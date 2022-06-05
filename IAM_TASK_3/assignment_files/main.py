"""
The input to your program should be a path to the directory that contains the line images. Your code should write the recognized text for each line to a txt file with the same structure as we used to provide the ground-truth data (two lines per prediction: one for the image name, one for the text prediction, where predictions are separated by two lines). To estimate the performance, character-level and word-level accuracy will be measured.
"""
import os
import time

# Internal libs
from deploy_models import *
from preprocessing_pipeline import get_words
from preprocessing_pipeline import visualize_data
from deploy_models import get_encoded_data, predict

def label_encoder():
    with open("class_data.json", "r") as f:
        data = json.load(f)
    
    labelencoder = LabelEncoder()
    labelencoder.fit_transform(list(data.keys()))
    # encoded = labelencoder.transform(["wish"])
    # decoded = labelencoder.inverse_transform(encoded)
    # print(f"encoded: {encoded}")
    # print(f"decoded: {decoded}")

    return labelencoder

def main():
    print("Installing dependencies ...")
    # ret_value = os.system("conda env create -f environment.yml")
    # if ret_value != 0:
    #     print(f"Error. Command returned {ret_value}")
    #     print("Attempting pip ...")
    #     ret_value = os.system("pip install -r requirements.txt")
    #     if ret_value != 0:
    #         print("Pip and conda failed ... You have some checking to do!")
    #         exit()
    # else:
    #     os.system("conda activate hwr_clustering")
    
    print("Dependencies installed successfully!")

    print("Getting encoder to decode predictions ...")
    enc = label_encoder()

    os.system('clear')
    eval_set_path = input("Input the path to the evaluation set please ...\n:")
    # Read path
    if os.path.exists(eval_set_path):
        print(f"path : '{eval_set_path}' is read!")
    else:
        print(f"Could not find path to '{eval_set_path}'. Is the path accessible from this script? Rerunning script in 5 seconds, press 'CTRL + C' to quit ...")
        time.sleep(3)
        os.system("python main.py")

    print("Reading files ...")
    time.sleep(1)

    # input_files = [] # list with paths to input images

    # Iterate input files from eval_set_path
    for filename in os.listdir(eval_set_path):
        path = eval_set_path + "/" + filename
        # input_files.append(path)
        print(f"Reading: {path}")
    # print(f"Read {len(input_files)} image files!")
        images = get_words(path)
        print("#"*50)
        print("Processed Image(s) and feature vector contents:")
        print("#"*50)
        svm_final_pred = ""
        knn_final_pred = ""
        for vector in images:
            visualize_data(vector)
            df = get_encoded_data()
            X_PERIMETER = df['Perimeter'].tolist()
            X_PERIMETER = np.array(X_PERIMETER)
            X_PERIMETER = X_PERIMETER.reshape(-1, 1)

            knn_pred = predict(X_PERIMETER)

            # svm_final_pred = svm_final_pred + " " + enc.inverse_transform(svm_pred)
            knn_final_pred = knn_final_pred + " " + str(enc.inverse_transform(knn_pred))

        print(f"-- Final prediction for {path} ---------------------------------------")
        print(f"Final prediction from KNN: '{knn_final_pred}'")
        # print(f"Final prediction from SVM: '{svm_final_pred}'")
        print("Writting to output.txt ...")

        with open("output.txt", "a") as f:
            f.write(path + "\n")
            f.write("KNN: " + str(knn_final_pred) + "\n")
            # f.write("SVM: " + str(svm_final_pred) + "\n")
            f.write("\n")
            f.write("\n")
            
    # for filename in input_files:
    #     images = get_words(filename)
    #     print("#"*50)
    #     print("Processed Image(s) and feature vector contents:")
    #     print("#"*50)
        
        # for vector in images:
        #     visualize_data(vector)

if __name__ == '__main__':
    main()