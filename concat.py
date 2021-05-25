from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy.distutils.fcompiler import str2bool
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from ancor.head_factory import HEADS
from eval.meta_eval import loop_get_features, classify
from eval.util import mean_confidence_interval, get_eval_datasets
from models.util import create_model, load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # load pretrained model
    parser.add_argument('--model', type=str, nargs='*', default='resnet12', help='Multiple model for ensemble')
    parser.add_argument('--model_path', type=str, nargs='*', default=None, help='absolute path to .pth model')
    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tiered', 'dota', 'cifar100', 'entity13', 'living17', 'nonliving26', 'entity30',
                                 'aircraft'])
    # specify data_root
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--mode', type=str, required=True, choices=['coarse', 'fine'])
    parser.add_argument('--partition', type=str, required=True, choices=['train', 'test', 'validation'])
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--only-base', nargs='*', type=str2bool,
                        help='same as only-base but array because multiple models')
    parser.add_argument('--dim', type=int, default=128,
                        help="network output dim. only necessary when only-base is false and using fork or seq heads")
    parser.add_argument('--head', required=True, choices=HEADS)
    parser.add_argument('--fg', action='store_true',
                        help="Whether 2nd meta test wil be intra-fine (if true) or all-way (if false)")
    parser.add_argument('--cls', choices=['Cosine', 'LR', 'NN'], default='Cosine')
    args = parser.parse_args()

    return args


def main():
    args = parse_option()
    args = args

    meta_fg_val_dataset, meta_val_dataset, n_cls = get_eval_datasets(args)

    meta_valloader = DataLoader(meta_val_dataset,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    meta_fg_valloader = DataLoader(meta_fg_val_dataset,
                                   batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)
    # for simplicity, assume only-base=True for all models (it performs better)
    only_base = True
    models = [create_model(model, n_cls, only_base, args.head, args.dim) for model in args.model]
    [load_model(model, model_path, not only_base) for model, model_path in zip(models, args.model_path)]

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        models = [model.cuda() for model in models]
        cudnn.benchmark = True

    # evalation
    evaluate(meta_valloader, models, args, "Regular")
    if args.fg:
        evaluate(meta_fg_valloader, models, args, 'Fine-Grained')
    else:
        evaluate(meta_fg_valloader, models, args, 'All-Way')


def concat_meta_test(nets, testloader, only_base=True, is_norm=True, classifier='LR'):
    acc1 = []
    nets = [net.eval() for net in nets]
    for idx, data in tqdm(enumerate(testloader)):
        support_xs, support_ys, query_xs, query_ys = data
        support_features_list, support_ys_list = map(list, zip(
            *[loop_get_features(net, support_xs, support_ys, only_base, is_norm,
                                batch_size=1024) for net in nets]))
        query_features_list, query_ys_list = map(list,
                                                 zip(*[loop_get_features(net, query_xs, query_ys, only_base, is_norm,
                                                                         batch_size=1024) for net in nets]))
        support_features = np.concatenate(support_features_list, axis=-1)
        query_features = np.concatenate(query_features_list, axis=-1)
        query_ys_pred = classify(classifier, query_features, support_features, support_ys_list[0])
        acc1.append(metrics.accuracy_score(query_ys_list[0], query_ys_pred))

    return mean_confidence_interval(acc1)


def evaluate(meta_valloader, models, args, mode):
    start = time.time()
    val_acc1, val_std1 = concat_meta_test(models, meta_valloader, classifier=args.cls, is_norm=True)
    val_time = time.time() - start
    print(f'Mode: ' + mode)
    print(
        f'Partition: {args.partition} Accuracy: {round(val_acc1 * 100, 2)}' + u" \u00B1 " + f'{round(val_std1 * 100, 2)}, Time: {val_time}')


def get_coarse_class_for_fine_gt(coarse2fine, y):
    for c, f in coarse2fine.items():
        if y in f:
            return c


if __name__ == '__main__':
    main()
