import os
import argparse
from typing import List, Tuple

def read_images_names_and_gts(folder: str) -> Tuple[List[str], List[str]]:
    """Read in image names and ground truths."""

    with open(f'{folder}/iam_lines_gt.txt', 'r') as f:
        # Read lines and filter out empty ones
        lines: List[str] = [line.rstrip('\n') for line in f.readlines()]
        lines = [line for line in lines if line]

        # There should be an even number of lines (image name + ground truth)
        assert lines and len(lines) % 2 == 0

    # Initialize lists for image_names and ground_truths
    image_names: List[str] = []
    ground_truths: List[str] = []

    # Go over each line pair
    for i in range(int(len(lines) / 2)):
        image_name = lines[i * 2 + 0].strip()
        ground_truth = lines[i * 2 + 1].strip()

        # Add image name and ground truth only if the image actually exists
        if os.path.isfile(os.path.join(f'{folder}/lines/', image_name)):
            image_names.append(image_name)
            ground_truths.append(ground_truth)

    # There should be an equal amount of image names as there are ground truths
    assert len(image_names) == len(ground_truths)
    return image_names, ground_truths


def format_iam_lines_gt(folder: str) -> None:
    """Read in image names and ground truths, then write into format required."""
    image_names, ground_truths = read_images_names_and_gts(folder)
    print(f'Found {len(image_names)} images and corresponding ground truths\n')

    # Write to file in correct format
    with open(f'{folder}/gt.txt', 'w') as f:
        for i in range(len(ground_truths)):
            f.write(image_names[i] + '\t' + ground_truths[i] + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='path to folder containing gts and images')
    opt = parser.parse_args()

    format_iam_lines_gt(opt.folder)
