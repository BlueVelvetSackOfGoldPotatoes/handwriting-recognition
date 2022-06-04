# Handwriting Recognition

## Dead Sea Scrolls Task 1 & 2

Preparatory steps:
- Python version == 3.8.13
- Make sure you have a working installation of the latest R (https://cran.r-project.org/bin/windows/base/) - an R version of at least 4.1.3 is required for running the packages used
- Install all the required libraries using pip install -r requirements.txt
- The model was compiled using cuDNN 8.1.0 and Cuda 11.2 - make sure you have these installed as well if you wish to run tensorflow from GPU (conda example for installation: conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0)

Running the model:
- Run main.py in the format:
	- python main.py c:\users\username\desktop\folder 
	- where folder contains jpg images of binarized Dead Sea Scrolls
- The console will ask you whether you wish to use a personal library. Please type yes to continue  
- Final txt files can be found in Final_output

## IAM Task 3

### Installation

1. Since you are reading this README, we will assume that you have managed to download the content locally. If not, please download/clone it.
2. Change directory to the root directory of this project. On linux, this can be done through:
```bash
cd path/to/folder/located/somewhere
```
3. List the current files in the directory, and make sure you see files such as `create_lmdb_dataset.py`, `demo.py`, `run.py`, ..., etc. If not, please retry step 2. On linux, this can be done through:
```bash
ls -A
```
4. Ensure you have python(3) installed on your system.
5. Create the virtual environment as follows:
``` bash
python3 -m venv venv # creates a virtualenv (venv) where dependencies will be installed
source venv/bin/activate # activates the virtualenv
pip3 install -r requirements.txt # installs the dependencies
```
> **_NOTE:_** The current requirements file will install the CPU version of PyTorch, since not all systems are CUDA-enabled, and the demo will run just fine on the CPU. If you prefer to install a CUDA-enabled version of PyTorch, remove these entries from the requirements file. After which you can follow the installation instructions [here](https://pytorch.org/).

### Prerequisites
1. You have the test images located inside of a folder, e.g.:
```
- <root-project-directory>/
    - <test-images-directory>/
        - <image-1>.png
        - <image-2>.png
        - ...
        - <image-N>.png
```
2. You have followed the installation instructions, and are still located (terminal) in the root directory of the project.
3. You have noted the name of the directory which holds your test images.

### Run instructions
1. Ensure you meet the prerequisites described above.
2. To run the demo, execute (replacing the test image directory to your specification):
```bash
python3 run.py --folder <test-images-directory>
```
3. The predictions will have been written to `results/*.txt`.

### Extra scripts
The folder `additional_code/` includes some of the code that was written in the process of investigating the problem, trying out different pipelines and models, or simply for convenience. These were not used in the final pipeline, but may nice to take a look at. The additional code includes processes such as binarization, smart padding and resizing, encoding and decoding labels, splitting up the original training data (also for k-fold cross-validation), defining a CRNN, creating a PyTorch dataset out of the images and ground-truths.

### Citation
Codebase partially adapted from:
```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  note={to appear},
  pubstate={published},
  tppubtype={inproceedings}
}
```

### License
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
