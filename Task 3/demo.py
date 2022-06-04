import os
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import CTCLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_prediction(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    os.makedirs('results', exist_ok=True)
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred).log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

        # write prediction
        for img_name, pred in zip(labels, preds_str):
            output_path = os.path.join('results', img_name.replace('.png', '_characters.txt'))
            print(f'Writing prediction for image {img_name} to {output_path}')
            with open(output_path, 'w+') as f:
                f.write(pred)


def demo(opt):
    """ model configuration """
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.characters)
    opt.input_channel = 1

    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    print(model, '\n')

    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=len(eval_data),
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation,
            pin_memory=True
        )
        make_prediction(model, criterion, evaluation_loader, converter, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=128, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=480, help='the width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()\'\",.\*\+\-#$!@%;\\\/?:&[] ', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
