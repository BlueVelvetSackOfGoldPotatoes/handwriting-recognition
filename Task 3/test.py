import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        preds = model(image, text_for_pred).log_softmax(2)
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # permute 'preds' to use CTCloss format
        cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
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
    model.load_state_dict(torch.load(opt.saved_model))
    opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)

        print(accuracy_by_best_model)
        with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'a') as log:
            log.write(str(accuracy_by_best_model) + '\n')


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

    test(opt)
