import io
import os
import pickle

import numpy as np
import torch
from PIL import Image
from learn2learn.vision.datasets import TieredImagenet


class TieredImageNet(TieredImagenet):
    def __init__(self, root, partition="train", mode='coarse', transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        tiered_imaganet_path = os.path.join(self.root, 'tiered-imagenet')
        short_partition = 'val' if partition == 'validation' else partition
        labels_path = os.path.join(tiered_imaganet_path, short_partition + '_labels.pkl')
        images_path = os.path.join(tiered_imaganet_path, short_partition + '_images_png.pkl')
        with open(images_path, 'rb') as images_file:
            self.images = pickle.load(images_file)
        with open(labels_path, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.coarse2fine = {}
            for c, f in zip(self.labels['label_general'], self.labels['label_specific']):
                if c in self.coarse2fine:
                    if f not in self.coarse2fine[c]:
                        self.coarse2fine[c].append(f)
                else:
                    self.coarse2fine[c] = [f]
            if self.mode == 'coarse':
                self.labels = self.labels['label_general']
            elif self.mode == 'fine':
                self.labels = self.labels['label_specific']
            else:
                raise NotImplementedError

    @property
    def num_classes(self):
        return len(np.unique(self.labels))


class MetaTieredImageNet(TieredImageNet):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaTieredImageNet, self).__init__(
            root=args.data_root,
            partition=partition,
            mode=args.mode)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.data = {}
        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        if len(self.classes) > self.n_ways:
            cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        else:
            cls_sampled = np.array(self.classes) if self.classes is not np.ndarray else self.classes
        # cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls])
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])

        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way))
        support_xs = support_xs.reshape(-1)
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples))
            support_ys = np.tile(support_ys.reshape(-1), self.n_aug_support_samples)
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        support_xs = torch.stack(list(map(lambda x: self.train_transform(self._load_png_byte(x[0])), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(self._load_png_byte(x[0])), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def _load_png_byte(self, bytes):
        return Image.open(io.BytesIO(bytes))

    def __len__(self):
        return self.n_test_runs


class MetaFGTieredImageNet(MetaTieredImageNet):
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        coarse_sampled = np.random.choice(list(self.coarse2fine.keys()), 1, False)[0]
        cls_sampled = np.random.choice(self.coarse2fine[coarse_sampled], self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls])
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])

        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way))
        support_xs = support_xs.reshape(-1)
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples))
            support_ys = np.tile(support_ys.reshape(-1), self.n_aug_support_samples)
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        support_xs = torch.stack(list(map(lambda x: self.train_transform(self._load_png_byte(x[0])), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(self._load_png_byte(x[0])), query_xs)))

        return support_xs, support_ys, query_xs, query_ys
