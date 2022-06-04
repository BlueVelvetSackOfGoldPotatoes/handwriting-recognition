from dataset import IAMDataset

from model import CRNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T

from utils import split_iam_lines_gt_into_folds

print('\n--- Imported libraries\n')

# Split full data-set into K folds
split_iam_lines_gt_into_folds(K=5)

print('\n--- Split data into K folds\n')

# Define transform
transform_augment = T.Compose([
    T.RandomAffine(2.5, shear=2.5, fill=255),
    T.GaussianBlur((3, 45), sigma=0.25),
])

# Define train and validation data
train = IAMDataset(transform=transform_augment, annotations_path='data-unversioned/iam_split_1_train.txt')
val = IAMDataset(annotations_path='data-unversioned/iam_split_1_test.txt')

# Define train and validation data loaders
train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=1)
val_loader = DataLoader(val, batch_size=len(val), shuffle=True, num_workers=1)

print('--- Instantiated data loaders\n')

# Define model, criterion and optimizer
model = CRNN(num_classes=train.get_n_classes())
criterion = nn.CTCLoss(blank=train.get_n_classes() - 1, reduction='mean', zero_infinity=False)
if torch.cuda.is_available():
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
print('\n--- Defined model, criterion and optimizer\n')

example_xy = next(iter(train_loader))
print(f'Example batch size (images): {example_xy[0].size()}')
print(f'Example batch size (labels): {example_xy[1].size()}')
with torch.no_grad():
    pred = F.log_softmax(model(example_xy[0].float()), 2)
    pred = torch.permute(pred, (1, 0, 2))
    print(f'Example prediction size: {pred.size()}')
    loss = criterion(pred, example_xy[1], 239, 128)
    print(f'Example loss size: {loss}')

print('\n--- Done an example prediction to test model\n')

print('\n--- Training model...\n')
