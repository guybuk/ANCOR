import torch
from torchvision.models import ResNet as ResNetTorchvision

from models.resnet_few_shot import ResNet as ResNetFewShot


def few_shot_resnet_forward(encoder, x):
    x = encoder.layer1(x)
    x = encoder.layer2(x)
    x = encoder.layer3(x)
    x = encoder.layer4(x)
    if encoder.keep_avg_pool:
        x = encoder.avgpool(x)
    x = x.view(x.size(0), -1)
    return x


def torchvision_resnet_forward(encoder, imgs):
    x = encoder.conv1(imgs)
    x = encoder.bn1(x)
    x = encoder.relu(x)
    x = encoder.maxpool(x)
    x = encoder.layer1(x)
    x = encoder.layer2(x)
    x = encoder.layer3(x)
    x = encoder.layer4(x)
    x = encoder.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def forward(net, x):
    if type(net) is torch.nn.Sequential:
        x = net(x).view(x.size(0), -1)
    elif type(net) is ResNetTorchvision:
        x = torchvision_resnet_forward(net, x).view(x.size(0), -1)
    elif type(net) is ResNetFewShot:
        x = few_shot_resnet_forward(net, x).view(x.size(0), -1)
    else:
        raise NotImplementedError
    return x
