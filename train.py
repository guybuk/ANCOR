#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import io
import math
import os
import pprint
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from box import Box

import ancor.loader
from ancor.ancor_model_generator import ANCORModelGenerator
from ancor.calculator_factory import CALCULATORS, METRICS
from ancor.head_factory import HEADS
from ancor.logger import get_logger
from ancor.loss_factory import LOSSES
from ancor.queue_factory import QUEUES
from datasets.breeds import BREEDSFactory
from datasets.cifar100 import Cifar100, MetaCifar100, MetaFGCifar100
from datasets.meta_dataset import MetaDataset
from datasets.tiered_imagenet import TieredImageNet, MetaTieredImageNet, MetaFGTieredImageNet
from eval.meta_eval import meta_test
from models import model_pool
from models.util import AUGS

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                         ' | '.join(model_pool.keys()) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:2375', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for ei ther single node or '
                         'multi node data parallel training')
# queue specific configs:
parser.add_argument('--cst-dim', default=128, type=int,
                    help='feature dimension')
parser.add_argument('--queue-k', default=65536, type=int,
                    help='queue size; number of negative keys (per class or all depending on queue type)')
parser.add_argument('--encoder-m', default=0.999, type=float,
                    help='momentum of updating key encoder')
parser.add_argument('--cst-t', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--num-cycles', default=10, type=int)

# options for ANCOR
parser.add_argument('--mode', type=str, required=True,
                    choices=['fine', 'coarse'],
                    help="Whether to train with coarse or fine labels")
parser.add_argument('--queue', type=str, choices=QUEUES, default='multi',
                    help="Queue type")
parser.add_argument('--metric', type=str, choices=METRICS, default='angular',
                    help='Which metric to apply before calculating contrastive dot products')
parser.add_argument('--calc-types', nargs='*', type=str, choices=CALCULATORS,
                    default=['cls', 'cst_by_class'],
                    help='List of loss calculators to be used in training')
parser.add_argument('--loss-types', nargs='*', type=str, choices=LOSSES,
                    default=['ce', 'ce'],
                    help='List of loss methods that receive the logits and labels as inputs')
parser.add_argument('--head', type=str, default='seq', choices=HEADS)
parser.add_argument('-s', '--save-freq', default=1, type=int,
                    help='save once in how many epochs')
parser.add_argument('--dataset', default='living17',
                    choices=['tiered', 'cifar100', 'living17', 'entity13', 'nonliving26', 'entity30'])
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multistep learning rate scheduler factor.')
parser.add_argument('--keep-epochs', default=[59, 99, 159, 199], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--split', default=None, type=str, choices=['good', 'bad'])


def main():
    args = parser.parse_args()
    s = pprint.pformat(vars(args))
    with open("args.log", "w") as f:
        f.write(s)
    with open("train.sh", "w") as f:
        f.write(f"python {' '.join(sys.argv)}")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    logger = get_logger(name='log',
                        log_dir='.')

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.dataset == 'tiered':
        train_dataset = TieredImageNet(
            root=args.data,
            partition='train',
            mode=args.mode,
            transform=ancor.loader.TwoCropsTransform(transforms.Compose(AUGS[f"train_{args.dataset}"]))
        )
        val_dataset = MetaTieredImageNet(
            args=Box(
                data_root=args.data,
                mode='fine',
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5
            ),
            partition='validation',
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
        fg_val_dataset = MetaFGTieredImageNet(
            args=Box(
                data_root=args.data,
                mode='fine',
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5
            ),
            partition='validation',
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
    elif args.dataset == 'cifar100':
        train_transforms = transforms.Compose(AUGS[f"train_{args.dataset}"][1:])
        train_dataset = Cifar100(
            root=args.data,
            train=True,
            mode=args.mode,
            transform=ancor.loader.TwoCropsTransform(train_transforms))
        val_dataset = MetaCifar100(
            args=Box(
                data_root=args.data,
                mode='fine',
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5,
            ),
            partition='test',
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
        fg_val_dataset = MetaFGCifar100(
            args=Box(
                data_root=args.data,
                mode='fine',
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5,
            ),
            partition='test',
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.data, "BREEDS"),
                                       data_dir=os.path.join(args.data, "Data", "CLS-LOC"))
        train_dataset = breeds_factory.get_breeds(
            ds_name=args.dataset,
            partition='train',
            mode=args.mode,
            transforms=ancor.loader.TwoCropsTransform(transforms.Compose(AUGS[f"train_{args.dataset}"])),
            split=args.split
        )
        val_dataset = MetaDataset(
            args=Box(
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5,
            ),
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition='val',
                mode='fine',
                transforms=None,
                split=args.split
            ),
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"])
        )
        fg_val_dataset = MetaDataset(
            args=Box(
                n_ways=5,
                n_shots=1,
                n_queries=15,
                n_test_runs=200,
                n_aug_support_samples=5,
            ),
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition='val',
                mode='fine',
                transforms=None,
                split=args.split
            ),
            train_transform=transforms.Compose(AUGS[f"meta_test_{args.dataset}"]),
            test_transform=transforms.Compose(AUGS[f"test_{args.dataset}"]),
            fg=True
        )
    else:
        raise NotImplementedError
    # create model
    model, criterions = ANCORModelGenerator().generate_ancor_model(arch=args.arch, head_type=args.head,
                                                                   dim=args.cst_dim,
                                                                   K=args.queue_k, m=args.encoder_m, T=args.cst_t,
                                                                   mlp=args.mlp,
                                                                   num_classes=train_dataset.num_classes,
                                                                   queue_type=args.queue,
                                                                   metric=args.metric, calc_types=args.calc_types,
                                                                   loss_types=args.loss_types, gpu=args.gpu)
    log(args.rank, logger, "loaded model")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log(args.rank, logger, "=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            msg = model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_accs' in checkpoint:
                best_accs = checkpoint['best_accs']
            else:
                best_accs = [0.]
                log(args.rank, logger, " WARNING: BACKWARDS COMPATIBLE RESUME. NO BEST MODEL CHECKPOINT")
            log(args.rank, logger, "=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            log(args.rank, logger, "=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError()
    else:
        best_accs = [0.]

    cudnn.benchmark = True

    # Data loading code

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    fg_val_loader = torch.utils.data.DataLoader(
        fg_val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    for epoch in range(args.start_epoch, args.epochs):
        best_flag = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterions, optimizer, epoch, logger, args)
        if args.rank % ngpus_per_node == 0:
            if (epoch + 1) % args.save_freq == 0 and val_dataset is not None:
                val_acc, val_std = meta_test(model.module.encoder_q, val_loader, only_base=True, is_norm=True,
                                             classifier="Cosine")
                if best_accs[-1] < val_acc:
                    best_accs.append(val_acc)
                    with open("best_accs.log", 'a') as f:
                        f.write(
                            f"EPOCH {epoch}: Validation Accuracy: {round(val_acc * 100, 2)}+-{round(val_std * 100, 2)}\n")
                    best_flag = True
                log(args.rank, logger,
                    f"EPOCH {epoch}: Validation Accuracy: {round(val_acc * 100, 2)}+-{round(val_std * 100, 2)}")
            if (epoch + 1) % args.save_freq == 0 and fg_val_dataset is not None:
                val_acc, val_std = meta_test(model.module.encoder_q, fg_val_loader, only_base=True, is_norm=True,
                                             classifier="Cosine")
                log(args.rank, logger,
                    f"EPOCH {epoch}: Validation FG - Accuracy: {round(val_acc * 100, 2)}+-{round(val_std * 100, 2)}")

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq == 0 or best_flag:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_accs': best_accs
                }, is_best=best_flag, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
    if args.rank % ngpus_per_node == 0:
        remove_excess_epochs(args.keep_epochs)


def remove_excess_epochs(epochs_to_keep):
    files_to_keep = ['checkpoint_{:04d}.pth.tar'.format(epoch) for epoch in epochs_to_keep]
    for file in os.listdir('.'):
        if (not file in files_to_keep) and 'checkpoint_' in file:
            os.remove(file)


def train(train_loader, model, criterions, optimizer, epoch, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meters = []
    top1_meters = []
    top5_meters = []
    for loss_name in criterions.keys():
        loss_meters.append(AverageMeter(f'{loss_name}', ':.4e'))
        top1_meters.append(AverageMeter(f'{loss_name} Acc@1', ':6.2f'))
        top5_meters.append(AverageMeter(f'{loss_name} Acc@5', ':6.2f'))
    meters = []
    for loss_meter, top1_meter, top5_meter in zip(loss_meters, top1_meters, top5_meters):
        meters.append(loss_meter)
        meters.append(top1_meter)
        meters.append(top5_meter)

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, *meters],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, cls_labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            cls_labels = cls_labels.cuda(args.gpu, non_blocking=True).long()
        logits_and_labels = model(im_q=images[0], im_k=images[1], cls_labels=cls_labels)
        losses = []
        for criterion, (logits, labels) in zip(criterions.values(), logits_and_labels):
            if logits is None or labels is None:
                continue
            losses.append(criterion(logits, labels))
        total_loss = sum(losses)

        # log updates
        for loss_meter, top1_meter, top5_meter, (logits, labels), loss in zip(loss_meters, top1_meters, top5_meters,
                                                                              logits_and_labels, losses):
            if logits is None or labels is None:
                continue
            loss_meter.update(loss.item(), labels.size(0))
            if len(logits.shape) > len(labels.shape):
                # do accuracies for all except distill...
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                top1_meter.update(acc1[0], labels.size(0))
                top5_meter.update(acc5[0], labels.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log(args.rank, logger, progress.display(i))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr

    if args.cos:  # cosine lr schedule
        args.epochs_per_cycle = math.floor(args.epochs / args.num_cycles)
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch % args.epochs_per_cycle) / args.epochs_per_cycle))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= args.gamma if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def unpack_bytes_png(img):
    return Image.open(io.BytesIO(img))


def log(rank, logger, msg):
    if rank == 0:
        logger.info(msg)


if __name__ == '__main__':
    main()
