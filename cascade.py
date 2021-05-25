from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from ancor.head_factory import HEADS
from eval.meta_eval import loop_get_features, classify
from eval.util import mean_confidence_interval, get_eval_datasets
from models.util import create_model, load_model
from tsne import loop_get_features_with_class


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, nargs='*', default='resnet12')
    parser.add_argument('--model_path', type=str, nargs='*', default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tiered', 'dota', 'cifar100', 'entity13', 'living17', 'nonliving26', 'entity30',
                                 'aircraft'])
    # parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

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
    parser.add_argument('--only-base', action='store_true')
    parser.add_argument('--partition', type=str, required=True, choices=['train', 'test', 'validation'])
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
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

    meta_fg_val_dataset, meta_val_dataset, n_cls = get_eval_datasets(args)

    meta_valloader = DataLoader(meta_val_dataset,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    meta_fg_valloader = DataLoader(meta_fg_val_dataset,
                                   batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)
    # Assume model #1 is the first (coarse cls) model, only-base = False so the logits layer will be loaded
    # Assume model #2 is the second (intra-class cls) model, only-base = True for maximum performance
    only_bases = [False, True]
    mlps = [False, True]
    models = [create_model(model, n_cls, only_base, args.head, args.dim, mlp) for model, only_base, mlp in
              zip(args.model, only_bases, mlps)]
    [load_model(model, model_path, not only_base) for model, model_path, only_base in
     zip(models, args.model_path, only_bases)]

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


def ensemble_meta_test(nets, testloader, only_base=True, is_norm=True, classifier='LR', num_workers=None):
    acc = []
    acc1 = []
    acc5 = []
    nets = [net.eval() for net in nets]
    num_workers = num_workers if num_workers > 0 else 1
    for idx, data in tqdm(enumerate(testloader)):
        support_xs, support_ys, query_xs, query_ys = data
        support_features_list, support_ys_list = map(list, zip(
            *[loop_get_features(net, support_xs, support_ys, only_base, is_norm,
                                batch_size=1024) for net in nets]))
        query_features_list, query_ys_list = map(list,
                                                 zip(*[loop_get_features(net, query_xs, query_ys, only_base, is_norm,
                                                                         batch_size=1024) for net in nets]))
        classifiers = [LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                          multi_class='multinomial', n_jobs=num_workers).fit(support_features_list[i],
                                                                                             support_ys_list[i])
                       for i in range(len(support_features_list))]
        probas = np.array(
            [classifier.predict_proba(query_features_list[i]) for i, classifier in enumerate(classifiers)])
        log_probas = np.log(probas)
        gmean = np.exp(log_probas.sum(axis=0) / log_probas.shape[0])
        query_ys_pred = np.argmax(gmean.squeeze(), axis=1)
        acc1.append(metrics.accuracy_score(query_ys_list[0], query_ys_pred))

    return mean_confidence_interval(acc1)


def cascade_meta_test(nets, testloader, only_base=True, is_norm=True, classifier='LR'):
    coarse_net = nets[0]
    fine_net = nets[1]
    coarse_net = coarse_net.eval()
    fine_net = fine_net.eval()
    acc1 = []
    for idx, data in tqdm(enumerate(testloader)):
        support_xs, support_ys, query_xs, query_ys = data
        _, _, support_coarse_preds = loop_get_features_with_class(coarse_net, support_xs, support_ys, is_norm,
                                                                  batch_size=1024)
        _, _, query_coarse_preds = loop_get_features_with_class(coarse_net, query_xs, query_ys, is_norm,
                                                                batch_size=1024)
        support_features, support_ys = loop_get_features(fine_net, support_xs, support_ys, only_base, is_norm,
                                                         batch_size=1024)
        query_features, query_ys = loop_get_features(fine_net, query_xs, query_ys, only_base, is_norm, batch_size=1024)

        dict_query = {i: None for i in range(len(query_ys))}
        for i in range(len(query_coarse_preds)):
            for j in range(len(support_coarse_preds)):
                if query_coarse_preds[i] == support_coarse_preds[j]:
                    if dict_query[i] is None:
                        dict_query[i] = [j]
                    else:
                        dict_query[i].append(j)
        preds = [None] * len(query_coarse_preds)
        classified = []
        # predict on overlapping coarse classes
        for i, v in dict_query.items():
            if v is not None:
                preds[i] = classify(classifier, query_features[i], support_features[v], support_ys[v])
                classified.append(i)
        for i in dict_query:
            if i in classified:
                continue
            preds[i] = classify(classifier, query_features[i], support_features, support_ys)

        query_ys_pred = np.array(preds)

        acc1.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc1)


def evaluate(meta_valloader, models, args, mode):
    start = time.time()
    val_acc1, val_std1 = cascade_meta_test(models, meta_valloader, classifier=args.cls, is_norm=True)
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
