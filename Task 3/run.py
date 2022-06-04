import os
import argparse
from typing import List, Tuple

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='path to folder containing gts and images')
    opt = parser.parse_args()

    if opt.folder[-1] == '/':
        opt.folder = opt.folder[:-1]

    os.system(f'python3 create_lmdb_dataset.py --test --inputPath {opt.folder} --outputPath demo/')
    os.system(f'python3 demo.py --eval_data demo --saved_model saved_models/VGG-BiLSTM-CTC/best_norm_ED.pth')
