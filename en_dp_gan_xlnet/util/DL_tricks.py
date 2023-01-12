"""
Author: Lynx Zhang
Date: 2020.
Email: zzlynx@outlook.com
"""
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def log(x):
    return torch.log(x + 1e-8)
