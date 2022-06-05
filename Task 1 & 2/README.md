# Handwriting Recognition

## Dead Sea Scrolls Task 1 & 2

- The code was built in Python version 3.8.13

## Preparatory steps
- Please follow these steps in the specified order
- Make sure you have a working installation of the latest R (https://cran.r-project.org/bin/windows/base/) - an R version of at least 4.1.3 is required for installing and running the packages used
- Ensure you have python(3) installed on your system.
- Change directory to the root directory of this project. On linux, this can be done through:
```bash
cd path/to/folder_with_task1&2/located/somewhere
```
- Create the virtual environment as follows:
``` bash
python3 -m venv venv # creates a virtualenv (venv) where dependencies will be installed
source venv/bin/activate # activates the virtualenv
pip install -r requirements.txt # installs the dependencies
```
- Executing the code above will install the Tensorflow package on your system. If you wish to run Tensorflow from GPU: the model was compiled using cuDNN 8.1.0 and Cuda 11.2, so please make sure you have these installed as well (conda example for installation: conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0)
- Please make sure that you have the test images located inside of a folder

## Running the model
- Please make sure that you're still located (terminal) in the root directory of Task 1 & 2.
- Run main.py in the format:
```bash
python main.py path/to/folder_with_jpg/located/somewhere
```
- The folder should contain jpg images of binarized Dead Sea Scrolls
- Final txt files can be found in the results folder

### Extra scripts
The folder `additional_code/` includes some of the code that was written in the process of investigating the problem, trying out different pipelines and models, or simply for convenience. These were not used in the final pipeline, but may be nice to take a look at. 

### Citation
Line segmentation adapted from:
```
@inproceedings{Surinta2014Astar,
	author = {Surinta, Olarik and Holtkamp, Michiel and Karaaba, Mahir and Oosten, Jean-Paul and Schomaker, Lambert and Wiering, Marco},
	year = {2014},
	month = {05},
	pages = {},
	title = {A* Path Planning for Line Segmentation of Handwritten Documents},
	volume = {2014},
	journal = {Proceedings of International Conference on Frontiers in Handwriting Recognition, ICFHR},
	doi = {10.1109/ICFHR.2014.37}
}
```
Model adapted from
```
https://keras.io/examples/vision/captcha_ocr/
```
