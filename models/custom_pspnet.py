"""
https://arxiv.org/abs/1612.01105
https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_stage(features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [nn.functional.interpolate(
            input=stage(feats), size=(h, w), mode='bilinear', align_corners=True
        ) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = nn.functional.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=1, sizes=(1, 2, 3, 6, 8, 12), psp_size=128, pretrained=True):
        super().__init__()
        model = ptcv_get_model("resnet18", pretrained=False)
        self.feats = nn.Sequential(*list(model.features.children())[:-3])
        """ No pretrained models allowed
        for param in self.feats.parameters():  # Frost encoder
            param.requires_grad = False
        """
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward(self, x):
        f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)


def custom_psp_model_selector(model_name, n_classes):
    pretrained = False
    if "pretrained" in model_name:
        pretrained = True
    model = PSPNet(n_classes=n_classes, psp_size=128, pretrained=pretrained)
    return model.cuda()
