import os
import math
import re
import sys
import string
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import pad, resize


class IAMDataset(Dataset):
    """Defines the IAM PyTorch data-set."""

    DESIRED_IMAGE_WIDTH = 960
    DESIRED_IMAGE_HEIGHT = 64
    DESIRED_HW_RATIO = 15
    INTENSITY_THRESHOLD = 215
    LABEL_LENGTH = 128

    def __init__(
        self,
        transform: Optional[T.Compose] = None,
        device: Optional[torch.device] = None,
        image_directory_path: str = 'data-unversioned/lines/',
        annotations_path: str = 'data-unversioned/iam_lines_gt.txt',
        enable_custom_transform: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize data-set."""
        # Set member variables
        self.__transform: Optional[T.Compose] = transform
        self.__device: Optional[torch.device] = device
        self.__enable_custom_transform: bool = enable_custom_transform
        self.__image_directory_path: str = image_directory_path
        self.__annotations_path: str = annotations_path
        self.__n_classes = 0
        self.__min_gt_length: int = sys.maxsize
        self.__max_gt_length: int = -sys.maxsize
        self.__min_gt: Optional[str] = None
        self.__max_gt: Optional[str] = None

        # Select a device if not given explicitly
        if not self.__device:
            self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generating encode/decode dictionaries
        self.__char_encode_dict: Dict[str, int] = {}
        self.__char_decode_dict: Dict[int, str] = {}
        self.__generate_encode_decode_maps()

        # Read in the image names and ground truths
        self.__image_names: List[str] = []
        self.__ground_truths: List[str] = []
        self.__ground_truths_encoded: List[torch.IntTensor] = []
        self.__read_image_names_and_ground_truths()
        if self.__enable_custom_transform:
            self.__encode_ground_truths()

        if verbose:
            print('[DATASET] Created dataset with:')
            print(f'\t- {len(self)} images/labels')
            print(f'\t- {self.__n_classes} classes (after reduction)')
            print(f'\t- shortest label with length {self.__min_gt_length} is: "{self.__min_gt}"')
            print(f'\t- longest label with length {self.__max_gt_length} is: "{self.__max_gt}"\n')

    def __len__(self) -> int:
        """Return the length of the data-set."""
        return len(self.__image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.IntTensor, List[str]]]:
        """Retrieve and transform single data-item."""
        # Read in image as grayscale
        image_path = os.path.join(self.__image_directory_path, self.__image_names[idx])
        image = read_image(image_path, ImageReadMode.GRAY)

        # Transform image according to pre-set transform
        if self.__transform:
            image = self.__transform(image)

        # Apply custom transformations for data-set consistency
        if self.__enable_custom_transform:
            # Pad and resize image tensor
            image = self.__pad_image_to_hw_ratio(image)
            image = self.__resize_to_desired_size(image)

            # Convert image to binary where all is 0 except written text
            self.__convert_to_binary_image(image)

        # Retrieve ground truth (label)
        gt = self.__ground_truths_encoded[idx] if self.__enable_custom_transform else self.__ground_truths[idx]

        # Return image with ground truth (label)
        return image, gt

    def get_n_classes(self) -> int:
        """Return number of classes found in labels."""
        return self.__n_classes

    def __pad_image_to_hw_ratio(self, image: torch.Tensor) -> torch.Tensor:
        # Determine current h-w ratio
        _, height, width = image.size()
        cur_hw_ratio = width / height

        # Determine required pad to meet desired h-w ratio
        pad_w = pad_h = 0
        if cur_hw_ratio > self.DESIRED_HW_RATIO:
            # Height should be padded
            pad_h = int(((width / self.DESIRED_HW_RATIO) - height) / 2)
        elif cur_hw_ratio < self.DESIRED_HW_RATIO:
            # Width should be padded
            pad_w = int(((height * self.DESIRED_HW_RATIO) - width) / 2)

        # Pad the image to attain (approximately) the desired h-w ratio
        return pad(image, [pad_w, pad_h], fill=255, padding_mode='constant')

    def __resize_to_desired_size(self, image: torch.Tensor) -> torch.Tensor:
        # Resize to the desired scale without warping text by padding beforehand
        return resize(image, size=(self.DESIRED_IMAGE_HEIGHT, self.DESIRED_IMAGE_WIDTH), antialias=True)

    def __convert_to_binary_image(self, image: torch.Tensor) -> None:
        # Convert to 0 if above threshold intensity, else to 1
        image.apply_(lambda x: 0 if x > self.INTENSITY_THRESHOLD else 1)

    def __read_image_names_and_ground_truths(self) -> None:
        """Read in image names and ground truths.

        Reads in the image names and ground truths.
        These are defined in the annotation file as pairs.
        """
        with open(self.__annotations_path, 'r') as f:
            # Read lines and filter out empty ones
            lines: List[str] = [line.rstrip('\n') for line in f.readlines()]
            lines = [line for line in lines if line]

            # There should be an even number of lines (image name + ground truth)
            assert lines and len(lines) % 2 == 0, 'Expected an even amount of lines (image + label pairs)'

        # Initialize lists for image_names and ground_truths
        image_names: List[str] = []
        ground_truths: List[str] = []

        # Go over each line pair
        for i in range(int(len(lines) / 2)):
            image_name = lines[i * 2 + 0].strip()
            ground_truth = re.sub(r'([^\w ]|\n)', '', lines[i * 2 + 1].strip().lower())
            gt_len = len(ground_truth)

            # Do not add image name and ground truth if the image does not exists or label is empty
            if gt_len == 0 or not os.path.isfile(os.path.join(self.__image_directory_path, image_name)):
                continue

            # Keep track of shortest label (length) encountered
            if gt_len < self.__min_gt_length:
                self.__min_gt_length = gt_len
                self.__min_gt = ground_truth

            # Keep track of shortest label (length) encountered
            if gt_len > self.__max_gt_length:
                self.__max_gt_length = gt_len
                self.__max_gt = ground_truth

            # Pad label to fixed length
            pad_left = math.floor((self.LABEL_LENGTH - len(ground_truth)) / 2)
            pad_right = math.ceil((self.LABEL_LENGTH - len(ground_truth)) / 2)
            ground_truth = ' ' * pad_left + ground_truth + ' ' * pad_right

            # Add the image and ground truth to the dataset
            image_names.append(image_name)
            ground_truths.append(ground_truth)

        # There should be an equal amount of image names as there are ground truths
        assert len(image_names) == len(ground_truths), 'Length of Images and Labels must match!'

        # Store the retrieved image names and ground truths
        self.__image_names = image_names
        self.__ground_truths = ground_truths

    def __generate_encode_decode_maps(self) -> None:
        """Generate dictionaries for mapping characters to integers and the other way around."""
        numbers: Dict[str, int] = {
            str(n): n
            for n in range(10)
        }
        alphabet: Dict[str, int] = {
            char: i + 10
            for char, i in zip(string.ascii_lowercase, range(26))
        }
        self.__char_encode_dict = numbers | alphabet
        self.__char_encode_dict = self.__char_encode_dict | {' ': len(self.__char_encode_dict)}
        self.__char_encode_dict = self.__char_encode_dict | {'_': len(self.__char_encode_dict)}
        self.__char_decode_dict = {v: k for k, v in self.__char_encode_dict.items()}
        self.__n_classes = len(self.__char_encode_dict)

    def __encode_ground_truths(self) -> None:
        """Encode labels into a list of integer tensors."""
        self.__ground_truths_encoded = [
            torch.IntTensor([self.__char_encode_dict.get(char) for char in gt])
            for gt in self.__ground_truths
        ]
