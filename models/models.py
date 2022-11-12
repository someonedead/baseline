from torch import nn
from torchvision import models
from .inceptionnet import InceptionResnetV1
from .densenet import DenseNet
from .efficientnet import EfficientNet
from .arcface_model import WhalesNet

# class CustomModel(nn.Module):

#     def __init__(self, backbone, classify=True):
#         self.backbone = backbone


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    s = float(config.model.s)
    m = float(config.model.m)
    num_classes = config.dataset.num_of_classes
    dropout = config.model.dropout
    model = WhalesNet(channel_size=512,
                      backbone=arch,
                      s=s,
                      m=m,
                      dropout=dropout,
                      out_feature=num_classes)
    # if arch == "efficientnet":
    #     model = EfficientNet(channel_size=512, out_feature=num_classes)
    # elif arch == "densenet":
    #     model = DenseNet(
    #         channel_size=512, out_feature=num_classes, s=config.train.arcface.s, m=config.train.arcface.m
    #     )
    # elif arch == 'inception_resnetv1':
    #     model = InceptionResnetV1(classify=True, num_classes=num_classes)
    # elif arch.startswith('resnet'):
    #     model = models.__dict__[arch](pretrained=True)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = SHOPEEDenseNet(channel_size=512, out_feature=num_classes)

    # model = InceptionResnetV1(classify=True, num_classes=num_classes)
    # if arch.startswith('resnet'):
    #     model = models.__dict__[arch](pretrained=True)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.fc =  nn.Sequential(
    #     nn.Linear(model.fc.in_features, 512, bias=False),
    #     nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True),
    #     # nn.ReLU(),
    #     # nn.Dropout(0.1),
    #     nn.Linear(512, num_classes),
    # )
    # else:
    #     raise Exception('model type is not supported:', arch)
    model.to("cuda")
    return model
