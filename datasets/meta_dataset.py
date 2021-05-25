import numpy as np
import torch
from torch.utils.data import Dataset


class MetaDataset(Dataset):
    def __init__(self, args, dataset, train_transform=None, test_transform=None, fix_seed=False, fg=False):
        super(Dataset, self).__init__()
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data = {}
        self.fg = fg
        self.loader = dataset.loader
        self.coarse2fine = dataset.coarse2fine

        if hasattr(dataset, "samples"):
            self.images = [s[0] for s in dataset.samples]
        elif hasattr(dataset, "images"):
            self.images = dataset.images
        if hasattr(dataset, "targets"):
            self.labels = dataset.targets
        elif hasattr(dataset, "labels"):
            self.labels = dataset.labels

        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = sorted(list(self.data.keys()))

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        if self.fg:
            classes = self.coarse2fine[np.random.choice(list(self.coarse2fine.keys()), 1, False)[0]]
        else:
            classes = self.classes
        if len(classes) > self.n_ways:
            cls_sampled = np.random.choice(classes, self.n_ways, False)
        else:
            cls_sampled = np.array(classes) if classes is not np.ndarray else classes
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
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, 1))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, 1))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, 1))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(self.loader(x.squeeze(0)[0])), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(self.loader(x.squeeze(0)[0])), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs