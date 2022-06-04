# Handwriting Recognition - Task 3

## Installation

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

## Prerequisites
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

## Run instructions
1. Ensure you meet the prerequisites described above.
2. To run the demo, execute (replacing the test image directory to your specification):
```bash
python3 run.py --folder <test-images-directory>
```
3. The predictions will have been written to `results/*.txt`.

## Citation
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

## License
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
