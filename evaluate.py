import argparse
import itertools 
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import dataloaders.dataloader as dataloader
from model.deeplab import *
from utils.loss import loss_fns
from utils.metrics import Evaluator
import utils.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--model_type', default='deeplab',
                    help="Type of deeplab")
parser.add_argument('--num_classes', default=21,
                    help="Numbers of classes")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def evaluate(model, dataloader, loss_fns, evaluator, params):
    model.eval()
    evaluator.reset()
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for sample in dataloader:
            with torch.no_grad():
                data_batch, labels_batch = sample['image'], sample['label']
                if params.cuda:
                    data_batch, labels_batch = data_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)

                labels_batch=labels_batch.long()
                
                output_batch = model(data_batch).float()

                loss = loss_fns['CrossEntropy'](output_batch, labels_batch)

                output_batch = output_batch.data.cpu().numpy()
                data_batch = data_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                evaluator.add_batch(labels_batch, output_batch)
                # update the average loss
                loss_avg.update(loss.item())
                
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()
    metrics_mean = {'mIOU': evaluator.Mean_Intersection_over_Union(), 'loss': loss_avg()}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    params.batch_size = 1

    torch.manual_seed(1)
    if params.cuda:
        torch.cuda.manual_seed(1)

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    dl = dataloader.fetch_dataloader(['val'], params)

    test_dl = dl['val']

    logging.info("-done")

    if params.cuda:
        model = DeepLab(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False).cuda()
    else:
        model = DeepLab(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

    logging.info("-Model Type: {}".format(args.model_type))

    loss_fns = loss_fns

    evaluator = Evaluator(20+1)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.model_type + '_' + args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, test_dl, loss_fns, evaluator, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    logging.info("- done.")