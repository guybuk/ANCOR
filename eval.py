from __future__ import print_function

import argparse
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ancor.head_factory import HEADS
from eval.meta_eval import meta_test
from eval.util import get_eval_datasets
from models.util import create_model, load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='resnet12')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
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
    parser.add_argument('--only-base', action='store_true')
    parser.add_argument('--partition', type=str, required=True, choices=['train', 'test', 'validation'])
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--dim', type=int, default=128,
                        help="network output dim. only necessary when only-base is false and using fork or seq heads")
    parser.add_argument('--head', required=True, choices=HEADS)
    parser.add_argument('--fg', action='store_true',
                        help="Whether 2nd meta test wil be intra-fine (if true) or all-way (if false)")
    parser.add_argument('--cls', choices=['Cosine', 'LR', 'NN'], default='LR')
    parser.add_argument('--split', choices=['rand', 'good', 'bad'], default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    meta_fg_val_dataset, meta_val_dataset, n_cls = get_eval_datasets(args)

    meta_valloader = DataLoader(meta_val_dataset,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    meta_fg_valloader = DataLoader(meta_fg_val_dataset,
                                   batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)

    model = create_model(args.model, n_cls, args.only_base, args.head, args.dim)
    load_model(model, args.model_path, not args.only_base)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        cudnn.benchmark = True

    evaluate(meta_valloader, model, args, "N-Way")
    if args.fg:
        evaluate(meta_fg_valloader, model, args, 'Fine-Grained')
    else:
        evaluate(meta_fg_valloader, model, args, 'All-Way')


def evaluate(meta_valloader, model, args, mode):
    start = time.time()
    val_acc1, val_std1 = meta_test(model, meta_valloader, only_base=args.only_base, classifier=args.cls,
                                   is_norm=True)
    val_time = time.time() - start
    print(f'Mode: ' + mode)
    print(
        f'Partition: {args.partition} Accuracy: {round(val_acc1 * 100, 2)}' + u" \u00B1 " + f'{round(val_std1 * 100, 2)}, Time: {val_time}')


if __name__ == '__main__':
    main()
