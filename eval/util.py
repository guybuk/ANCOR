import os

import scipy
from box import Box
from scipy.stats import t
import torch
import numpy as np
from torchvision import transforms

from datasets.breeds import BREEDSFactory
from datasets.cifar100 import MetaCifar100, MetaFGCifar100
from datasets.meta_dataset import MetaDataset
from datasets.tiered_imagenet import MetaFGTieredImageNet, MetaTieredImageNet
from models.util import AUGS


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        output_torch = torch.tensor(output)
        target_torch = torch.tensor(target)
        maxk = max(topk)
        batch_size = target_torch.size(0)

        _, pred = output_torch.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_torch.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def get_eval_datasets(args):
    if args.dataset == 'tiered':
        if args.fg:
            meta_fg_val_dataset = MetaFGTieredImageNet(
                args=Box(
                    data_root=args.data_root,
                    mode='fine',
                    n_ways=args.n_ways,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples
                ),
                partition=args.partition,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True
            )
        else:
            meta_fg_val_dataset = MetaTieredImageNet(
                args=Box(
                    data_root=args.data_root,
                    mode='fine',
                    n_ways=1000,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples
                ),
                partition=args.partition,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True
            )

        meta_val_dataset = MetaTieredImageNet(
            args=Box(
                data_root=args.data_root,
                mode='fine',
                n_ways=args.n_ways,
                n_shots=args.n_shots,
                n_queries=args.n_queries,
                n_test_runs=args.n_test_runs,
                n_aug_support_samples=args.n_aug_support_samples
            ),
            partition=args.partition,
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fix_seed=True
        )
        n_cls = 20
    elif args.dataset == 'cifar100':
        meta_val_dataset = MetaCifar100(
            args=Box(
                data_root=args.data_root,
                mode='fine',
                n_ways=args.n_ways,
                n_shots=args.n_shots,
                n_queries=args.n_queries,
                n_test_runs=args.n_test_runs,
                n_aug_support_samples=args.n_aug_support_samples
            ),
            partition=args.partition,
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fix_seed=True
        )

        if args.fg:
            meta_fg_val_dataset = MetaFGCifar100(
                args=Box(
                    data_root=args.data_root,
                    mode='fine',
                    n_ways=args.n_ways,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples
                ),
                partition=args.partition,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True
            )

        else:
            meta_fg_val_dataset = MetaCifar100(
                args=Box(
                    data_root=args.data_root,
                    mode='fine',
                    n_ways=100,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples
                ),
                partition=args.partition,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True
            )

        n_cls = 100
    elif args.dataset in ['living17', 'nonliving26', 'entity30', 'entity13']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.data_root, "BREEDS"),
                                       data_dir=os.path.join(args.data_root, "Data", "CLS-LOC"))
        meta_val_dataset = MetaDataset(
            args=Box(
                n_ways=args.n_ways,
                n_shots=args.n_shots,
                n_queries=args.n_queries,
                n_test_runs=args.n_test_runs,
                n_aug_support_samples=args.n_aug_support_samples,
            ),
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition=args.partition,
                mode=args.mode,
                transforms=None,
                split=args.split
            ),
            fg=False,
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fix_seed=True
        )
        if args.fg:
            meta_fg_val_dataset = MetaDataset(
                args=Box(
                    n_ways=args.n_ways,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples,
                ),
                dataset=breeds_factory.get_breeds(
                    ds_name=args.dataset,
                    partition=args.partition,
                    mode=args.mode,
                    transforms=None,
                    split=args.split
                ),
                fg=True,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True
            )
        else:
            meta_fg_val_dataset = MetaDataset(
                args=Box(
                    n_ways=1000,
                    n_shots=args.n_shots,
                    n_queries=args.n_queries,
                    n_test_runs=args.n_test_runs,
                    n_aug_support_samples=args.n_aug_support_samples,
                ),
                dataset=breeds_factory.get_breeds(
                    ds_name=args.dataset,
                    partition=args.partition,
                    mode=args.mode,
                    transforms=None,
                    split=args.split
                ),
                fg=False,
                train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
                test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
                fix_seed=True

            )

        n_cls = int(args.dataset[-2:])
    else:
        raise NotImplementedError(args.dataset)
    return meta_fg_val_dataset, meta_val_dataset, n_cls
