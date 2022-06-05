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

