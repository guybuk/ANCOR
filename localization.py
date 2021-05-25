from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.breeds import BREEDSFactory
from models.util import create_model, load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet'
                        )
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
    parser.add_argument('-b', dest='batch_size', type=int)
    parser.add_argument('--mode', type=str, required=True, choices=['coarse', 'fine'])
    parser.add_argument('--only-base', action='store_true')
    parser.add_argument('--partition', type=str, required=True, choices=['train', 'test', 'validation'])
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    # ===========IRRELEVANT===============
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--head', default=None)
    parser.add_argument('--fg', action='store_true')
    parser.add_argument('--simclr', action='store_true')
    parser.add_argument('--cascade', action='store_true')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    opt.data_aug = True

    return opt


def main():
    args = parse_option()

    train_dataset, n_cls = get_datasets(args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)

    model = create_model(args.model, n_cls, args.only_base, args.head, args.dim)
    load_model(model, args.model_path, not args.only_base)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        cudnn.benchmark = True

    for i, (images, labels) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu)

        def attention_forward(encoder, imgs):
            # hard-coded forward because we need the feature-map and not the finalized feature
            x = encoder.conv1(imgs)
            x = encoder.bn1(x)
            x = encoder.relu(x)
            x = encoder.maxpool(x)
            x = encoder.layer1(x)
            x = encoder.layer2(x)
            x = encoder.layer3(x)
            feats = encoder.layer4(x)
            feats_as_batch = feats.permute((0, 2, 3, 1)).contiguous().view((-1, feats.shape[1]))
            # reminder: "fc" layer outputs: (feature, class logits)
            feats_as_batch = encoder.fc(feats_as_batch)[0]
            feats_as_batch = feats_as_batch.view(
                (feats.shape[0], feats.shape[2], feats.shape[3], feats_as_batch.shape[1]))
            feats_as_batch = feats_as_batch.permute((0, 3, 1, 2))
            return feats_as_batch

        f_q = attention_forward(model, images)
        localization(images, f_q, args.batch_size, batch_id=i, img_size=448)
        if i == 10:
            break


def get_datasets(args):
    augs = [
        transforms.RandomResizedCrop(448, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ]
    if args.dataset in ['living17', 'nonliving26', 'entity30', 'entity13']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.data_root, "BREEDS"),
                                       data_dir=os.path.join(args.data_root, "Data", "CLS-LOC"))
        train_dataset = breeds_factory.get_breeds(ds_name=args.dataset, partition=args.partition, mode=args.mode,
                                                  transforms=transforms.Compose(augs))
        n_cls = int(args.dataset[-2:])
    else:
        raise NotImplementedError(args.dataset)
    return train_dataset, n_cls


def localization(im_q, f_q, batch_size, batch_id, img_size):
    os.makedirs('imgs', exist_ok=True)
    for idd in range(batch_size):
        aa = torch.norm(f_q, dim=1)
        imgg = im_q[idd] * torch.Tensor([[[0.229, 0.224, 0.225]]]).view(
            (1, 3, 1, 1)).cuda() + torch.Tensor(
            [[[0.485, 0.456, 0.406]]]).view((1, 3, 1, 1)).cuda()
        heatmap = F.interpolate((aa[idd] / aa[0].max()).detach().unsqueeze(0).unsqueeze(0).repeat((1, 3, 1, 1)),
                                [img_size, img_size])
        thresh = 0
        heatmap[heatmap < thresh] = 0
        plt.imsave(f'imgs/bImg_{idd}_batch_{batch_id}.png',
                   torch.cat((imgg, heatmap * imgg), dim=3).squeeze(0).cpu().permute(
                       (1, 2, 0)).clamp(0, 1).numpy().astype(float))


if __name__ == '__main__':
    main()
