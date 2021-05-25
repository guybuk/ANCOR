import warnings
from pathlib import Path

import torch
from torchvision import transforms
import ancor.loader
from ancor.ancor_model_generator import ANCORModelGenerator
from ancor.head_factory import HEADS
from models import model_pool


def load_model(model, pretrained, with_mlp=False):
    if not Path.is_file(Path(pretrained)):
        raise ValueError("=> no checkpoint found at '{}'".format(pretrained))

    for name, param in model.named_parameters():
        param.requires_grad = False

    print("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        load_encoder_q_state_dict(model, state_dict, with_mlp)
    elif 'module.prototypes.weight' in checkpoint:
        load_swav_resnet_state_dict(model, checkpoint)
    else:
        raise NotImplementedError


def load_swav_resnet_state_dict(model, state_dict):
    for k in list(state_dict.keys()):
        if not (k.startswith('module.projection') or k.startswith('module.prototypes')):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    msg = model.load_state_dict(state_dict, False)
    assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}


def load_encoder_q_state_dict(model, state_dict, with_mlp=False):
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q.'):
            # remove prefix
            if not with_mlp and k.startswith('module.encoder_q.fc'):
                continue
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if not with_mlp:
        # assume vanilla resnet model
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
    else:
        msg = model.load_state_dict(state_dict)


def create_model(name, n_cls, only_base=True, head='fork', dim=128, mlp=True):
    if only_base:
        if name == 'resnet12':
            model = model_pool[name](num_classes=n_cls)
        elif name == "resnet50":
            model = model_pool[name](num_classes=n_cls)
        elif name=='resnet12forcifar':
            model = model_pool[name](num_classes=n_cls)
        else:
            raise NotImplementedError
    elif head in HEADS:
        model, _ = ANCORModelGenerator().generate_ancor_model(
            arch=name, head_type=head, dim=dim, mlp=mlp, num_classes=n_cls
        )
        model = model.encoder_q
    else:
        raise NotImplementedError
    return model


AUGS = {
    "train_tiered": [
        transforms.RandomResizedCrop(84, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                             std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
    ],
    "meta_test_tiered": [
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                             std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
    ],
    "test_tiered": [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                             std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
    ],
    "train_cifar100": [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([ancor.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ],
    "meta_test_cifar100": [
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ],
    "test_cifar100": [
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ],
    "train_living17": [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ancor.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "meta_test_living17": [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "test_living17": [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "train_entity13": [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ancor.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "meta_test_entity13": [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "test_entity13": [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "train_nonliving26": [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ancor.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "meta_test_nonliving26": [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "test_nonliving26": [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])

    ],
    "train_entity30": [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ancor.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "meta_test_entity30": [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
    "test_entity30": [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ],
}
