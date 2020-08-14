"""
https://arxiv.org/abs/1612.01105
"""
import torch
from torch import nn
from torch.nn import functional as F

import models.extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
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
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=4, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',
                 pretrained=False, classification=False, in_channels=1):
        super().__init__()
        self.classification = classification
        self.feats = getattr(models.extractors, backend)(pretrained)

        if classification:
            self.linear = nn.Linear(256*3*3, n_classes)
        else:  # Segmentation
            self.psp = PSPModule(psp_size, 1024, sizes)
            self.drop_1 = nn.Dropout2d(p=0.3)

            self.up_1 = PSPUpsample(1024, 256)
            self.up_2 = PSPUpsample(256, 64)
            self.up_3 = PSPUpsample(64, 64)

            self.drop_2 = nn.Dropout2d(p=0.15)
            self.final = nn.Sequential(
                nn.Conv2d(64, n_classes, kernel_size=1),
            )

    def forward(self, x):
        f, class_f = self.feats(x)
        if self.classification:
            out = F.avg_pool2d(class_f, 8)  # (batch, 256, 28, 28) -> (batch, 256, 3, 3)
            out = out.view(out.size(0), -1)  # (batch, 256, 3, 3) -> (batch, 2304)
            out = self.linear(out)
            return out

        # Segmentation case
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)


def psp_model_selector(model_name, n_classes, classification, in_channels=1):
    if in_channels != 3: assert False, "PSPNet with != 3 ToDo"
    if "resnet34_pspnet" in model_name:
        model = PSPNet(n_classes=n_classes, backend="resnet34", classification=classification, in_channels=in_channels)
    elif "resnet18_pspnet" in model_name:
        model = PSPNet(n_classes=n_classes, backend="resnet18", classification=classification, in_channels=in_channels)
    else:
        assert False, "Unknown '{}' in pspnet models!".format(model_name)

    return model.cuda()
