from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from box import Box
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from ancor.head_factory import HEADS
from datasets.breeds import BREEDSFactory
from datasets.cifar100 import MetaFGCifar100, MetaCifar100
from datasets.meta_dataset import MetaDataset
from datasets.tiered_imagenet import MetaTieredImageNet, MetaFGTieredImageNet
from eval.meta_eval import get_features
from eval.util import normalize
from models.util import create_model, AUGS, load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

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

    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    meta_fg_val_dataset, meta_val_dataset, n_cls = get_tsne_datasets(args)

    meta_valloader = DataLoader(meta_val_dataset,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False,
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

    is_norm = True
    all_features, all_ys, all_preds = get_features_and_preds_from_dataloader(model, iter(meta_valloader),
                                                                             is_norm=is_norm)

    all_coarse_gt = np.array([get_coarse_class_for_fine_gt(meta_val_dataset.coarse2fine, y) for y in all_ys])
    print(f"Accuracy: {float(sum(all_preds == all_coarse_gt)) / len(all_preds)}")
    if isinstance(model.fc, torch.nn.Sequential):
        if args.head in ['fork', 'seq']:
            if is_norm:
                class_weights = F.normalize(model.fc[2].fc2.weight).detach().cpu().numpy()
            else:
                class_weights = model.fc[2].fc2.weight.detach().cpu().numpy()
        else:
            if is_norm:
                class_weights = F.normalize(model.fc[2].weight).detach().cpu().numpy()
            else:
                class_weights = model.fc[2].weight.detach().cpu().numpy()
    else:
        class_weights = F.normalize(model.fc.weight).detach().cpu().numpy()
    model_name = os.path.basename(os.path.dirname(args.model_path))
    tsne_with_weights(all_features, all_coarse_gt, class_weights, f"{model_name}_{args.dataset}_coarse")
    intra_features, intra_ys, intra_preds = get_features_and_preds_from_dataloader(model, iter(meta_fg_valloader),
                                                                                   is_norm=is_norm)
    bincount = np.bincount(intra_preds)
    coarse_class = np.argmax(bincount)
    print(f"{bincount} => Class: {coarse_class}")
    tsne_plot_with_arrows(intra_features, intra_ys, class_weights[coarse_class],
                          title=f"{model_name}_{args.dataset}_fine")


def get_features_and_preds_from_dataloader(net, meta_loader, is_norm):
    net = net.eval()
    with torch.no_grad():
        data = next(meta_loader)
        support_xs, support_ys, query_xs, query_ys = data
        support_features, support_ys, support_coarse_preds = loop_get_features_with_class(net, support_xs, support_ys,
                                                                                          is_norm, batch_size=1024)
        query_features, query_ys, query_coarse_preds = loop_get_features_with_class(net, query_xs, query_ys, is_norm,
                                                                                    batch_size=1024)
        all_coarse_preds = np.concatenate([support_coarse_preds, query_coarse_preds])
        all_features = np.concatenate([support_features, query_features])
        all_ys = np.concatenate([support_ys, query_ys])
    return all_features, all_ys, all_coarse_preds


def tsne_with_weights(all_features, all_ys, weights, title=""):
    features_and_weights = np.concatenate([all_features, weights])
    all_transformed = TSNE(n_jobs=20, metric='cosine').fit_transform(features_and_weights)
    num_features = len(all_features)
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(all_transformed[:num_features, 0], all_transformed[:num_features, 1], c=all_ys, cmap="tab20",
                s=10)
    if num_features != len(all_transformed):
        plt.scatter(all_transformed[num_features:, 0], all_transformed[num_features:, 1], marker="*",
                    edgecolors="k", s=200)
    plt.title(title)
    plt.savefig(f"{title}.png")
    # plt.show()


def tsne_plot_with_arrows(all_features, all_ys, weight, title=""):
    all_transformed = TSNE(n_jobs=20, metric='cosine').fit_transform(
        np.concatenate([all_features, weight.reshape(1, -1)]))
    transformed_features = all_transformed[:len(all_features)]
    transformed_weight = all_transformed[-1]
    medians = np.array([np.median(transformed_features[all_ys == ys], axis=0) for ys in np.unique(all_ys)])

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=all_ys, cmap='tab20', s=10)
    plt.scatter(transformed_weight[0], transformed_weight[1], marker="*", edgecolors="k", s=200)
    plt.quiver([transformed_weight[0]] * len(medians), [transformed_weight[1]] * len(medians),
               medians[:, 0] - transformed_weight[0], medians[:, 1] - transformed_weight[1], angles='xy',
               scale_units='xy', scale=1)
    plt.title(title)
    plt.savefig(f"{title}.png")
    # plt.show()


def loop_get_features_with_class(net, x, y, is_norm, batch_size):
    _, _, height, width, channel = x.size()
    x = x.view(-1, height, width, channel)
    y = y.view(-1)
    batches = int(abs(len(x) / batch_size)) + 1
    all_features, all_ys, all_preds = [], [], []
    for b in range(batches):
        features, ys, coarse_preds = get_features_and_coarse_pred(net, x[b * batch_size:b * batch_size + batch_size],
                                                                  y[b * batch_size:b * batch_size + batch_size],
                                                                  is_norm)
        all_features.append(features)
        all_ys.append(ys)
        all_preds.append(coarse_preds)
    all_features = np.concatenate(all_features)
    all_ys = np.concatenate(all_ys)
    all_preds = np.concatenate(all_preds)
    return all_features, all_ys, all_preds


def get_features_and_coarse_pred(net, x, y, is_norm=True):
    x = x.cuda()
    try:
        feature, coarse_logits = net(x)
        feature = feature.view(feature.size(0), -1)
        y = y.view(-1).numpy()
        if is_norm:
            feature = normalize(feature)

        coarse_preds = torch.argmax(coarse_logits, dim=1)
        feature = feature.detach().cpu().numpy()
        coarse_preds = coarse_preds.detach().cpu().numpy()
    except ValueError as e:
        # if a coarse baseline, does not return 2 outputs, only 1
        feature, y = get_features(net, x, y, True, is_norm)
        if isinstance(net.fc, torch.nn.Sequential):
            feature = net.fc[1](net.fc[0](torch.tensor(feature).cuda()))
        if is_norm:
            feature = normalize(torch.tensor(feature))
        feature = feature.detach().cpu().numpy()
        coarse_preds = torch.argmax(net(x), dim=1).cpu().numpy()

    return feature, y, coarse_preds


def get_tsne_datasets(args):
    if args.dataset == 'tiered':
        meta_fg_val_dataset = MetaFGTieredImageNet(args=args, partition=args.partition,
                                                   train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                                                   test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                                                   fix_seed=False)

        meta_val_dataset = MetaTieredImageNet(args=args, partition=args.partition,
                                              train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                                              test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                                              fix_seed=False)
        n_cls = 351
    elif args.dataset == 'cifar100':
        meta_fg_val_dataset = MetaFGCifar100(args=dict(
            data_root=args.data_root,
            n_ways=args.n_ways,
            n_shots=15,
            n_queries=35,
            n_test_runs=args.n_test_runs,
            n_aug_support_samples=0,
            mode=args.mode
        ), partition=args.partition, train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fix_seed=False)
        meta_val_dataset = MetaCifar100(args=dict(
            data_root=args.data_root,
            n_ways=args.n_ways,
            n_shots=args.n_shots,
            n_queries=args.n_queries,
            n_test_runs=args.n_test_runs,
            n_aug_support_samples=0,
            mode=args.mode
        ), partition=args.partition, train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fix_seed=False)
        n_cls = 20
    elif args.dataset in ['living17', 'nonliving26', 'entity30', 'entity13']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.data_root, "BREEDS"),
                                       data_dir=os.path.join(args.data_root, "Data", "CLS-LOC"))
        meta_val_dataset = MetaDataset(
            args=Box(
                n_ways=args.n_ways,
                n_shots=args.n_shots,
                n_queries=args.n_queries,
                n_test_runs=args.n_test_runs,
                n_aug_support_samples=0,
            ),
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition=args.partition,
                mode=args.mode,
                transforms=None,
            ),
            fg=False,
            train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )

        meta_fg_val_dataset = MetaDataset(
            args=Box(
                n_ways=args.n_ways,
                n_shots=15,
                n_queries=35,
                n_test_runs=args.n_test_runs,
                n_aug_support_samples=0,
            ),
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition=args.partition,
                mode=args.mode,
                transforms=None,
            ),
            fg=True,
            train_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
        n_cls = int(args.dataset[-2:])
    else:
        raise NotImplementedError(args.dataset)
    return meta_fg_val_dataset, meta_val_dataset, n_cls


def get_coarse_class_for_fine_gt(coarse2fine, y):
    for c, f in coarse2fine.items():
        if y in f:
            return c


if __name__ == '__main__':
    main()
