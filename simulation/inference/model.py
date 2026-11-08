"""ResNet-18 backbone with three early-exit heads, following the
BranchyNet-style design named in the AeroMesh manuscript (Section on
Methods: "a lightweight backbone ... with early-exit heads attached at
three depths"). Exit 1 taps layer2, exit 2 taps layer3, exit 3 is the
standard final-layer ResNet-18 head.
"""
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 4  # no-damage, minor-damage, major-damage, destroyed


class EarlyExitResNet18(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.exit1_head = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))
        self.exit2_head = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))
        self.exit3_head = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        e1 = self.exit1_head(self.pool(x))
        x = self.layer3(x)
        e2 = self.exit2_head(self.pool(x))
        x = self.layer4(x)
        e3 = self.exit3_head(self.pool(x))
        return e1, e2, e3
