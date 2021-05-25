from torchvision.models import resnet50

from models.resnet_few_shot import resnet12

model_pool = {
    "resnet50": resnet50,
    "resnet12": lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=num_classes),
    "resnet12forcifar": lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2,
                                                       num_classes=num_classes)
}