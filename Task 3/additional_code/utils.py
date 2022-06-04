import os
from typing import List

from sklearn.model_selection import KFold


def split_iam_lines_gt_into_folds(K: int = 5) -> None:
    """Read in image names and ground truths, then split into K folds.

    Reads in the image names and ground truths.
    These are defined in the annotation file as pairs.
    Splits them up into K folds, written to separate files.
    """
    with open('data-unversioned/iam_lines_gt.txt', 'r') as f:
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
        if os.path.isfile(os.path.join('data-unversioned/lines/', image_name)):
            image_names.append(image_name)
            ground_truths.append(ground_truth)

    # There should be an equal amount of image names as there are ground truths
    assert len(image_names) == len(ground_truths)
    print(f'[DATASET TOTAL] Found {len(image_names)} images '
          'and corresponding ground truths\n')

    # Initialize variables used for splitting the data
    kf = KFold(n_splits=K, shuffle=True)
    i: int = 1

    # Fold into separate sets
    for train_idxs, test_idxs in kf.split(image_names, ground_truths):
        print(f'[DATASET FOLD {i}] Train Length: {len(train_idxs)}, Test Length: {len(test_idxs)}')

        # Write training selection to file
        with open(f'data-unversioned/iam_split_{i}_train.txt', 'w') as f:
            for train_idx in train_idxs:
                f.write(image_names[train_idx] + '\n')
                f.write(ground_truths[train_idx] + '\n\n')
        # Write test selection to file
        with open(f'data-unversioned/iam_split_{i}_test.txt', 'w') as f:
            for test_idx in test_idxs:
                f.write(image_names[test_idx] + '\n')
                f.write(ground_truths[test_idx] + '\n\n')
        i += 1


def split_iam_lines_gt_into_lmdb() -> None:
    """Read in image names and ground truths, then lmdb sets.

    Reads in the image names and ground truths.
    These are defined in the annotation file as pairs.
    Splits them up into lmdb sets, written to separate files.
    """
    with open('data-unversioned/iam_lines_gt.txt', 'r') as f:
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
        if os.path.isfile(os.path.join('data-unversioned/lines/', image_name)):
            image_names.append(image_name)
            ground_truths.append(ground_truth)

    # There should be an equal amount of image names as there are ground truths
    assert len(image_names) == len(ground_truths)
    print(f'[DATASET TOTAL] Found {len(image_names)} images '
          'and corresponding ground truths\n')

    # Initialize variables used for splitting the data
    kf = KFold(n_splits=5, shuffle=True)

    # Fold into separate sets
    train_idxs, test_idxs = next(kf.split(image_names, ground_truths))
    test_idxs, val_idxs = test_idxs[:int(len(test_idxs) / 2)], test_idxs[int(len(test_idxs) / 2):]
    print('[DATASET LMDB] '
          f'Train Length: {len(train_idxs)}, '
          f'Test Length: {len(test_idxs)}, '
          f'Val Length: {len(val_idxs)}')

    # Write training selection to file
    with open('data-lmdb/iam_lmdb_train.txt', 'w') as f:
        for train_idx in train_idxs:
            f.write(image_names[train_idx] + '\t' + ground_truths[train_idx] + '\n')

    # Write test selection to file
    with open('data-lmdb/iam_lmdb_test.txt', 'w') as f:
        for test_idx in test_idxs:
            f.write(image_names[test_idx] + '\t' + ground_truths[test_idx] + '\n')

    # Write validation selection to file
    with open('data-lmdb/iam_lmdb_val.txt', 'w') as f:
        for val_idx in val_idxs:
            f.write(image_names[val_idx] + '\t' + ground_truths[val_idx] + '\n')
